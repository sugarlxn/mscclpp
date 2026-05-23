# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""MT-MSCCL++ tenant abstractions and host-side QoS scheduler.

See doc/design.md §3.5, §5, §6 for full design rationale.

Iteration 1 (this module) keeps the MSCCL++ device-side ABI untouched and
implements tenant-aware scheduling at the host/Python layer:

  - TenantContext / PolicyTable: data model (mirrors C++ header)
  - TokenBucket: per-tenant byte-rate rate limiter
  - TenantScheduler: WFQ (Deficit Round Robin), Strict Priority, Hybrid
  - single-tenant bypass: zero overhead when only one tenant is active

Iteration 2 will lower scheduling into ProxyService and use 5-bit tenant_id in
ProxyTrigger (see design.md §5.2, §7.1, §8 Phase 2/3).
"""

from __future__ import annotations

import enum
import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional


MAX_TENANTS = 32
DEFAULT_TENANT = 0
DEFAULT_QUANTUM_BYTES = 64 * 1024     # WFQ service quantum (design.md §5.3.1)
DEFAULT_AGING_MS = 100                # Priority aging threshold (design.md §5.4.1)
DEFAULT_POLICY_SHM = "/dev/shm/mtccl_policy_table.json"


class QoSClass(enum.IntEnum):
    BEST_EFFORT = 0
    STANDARD    = 1
    PREMIUM     = 2
    REALTIME    = 3


class PolicyMode(enum.IntEnum):
    SINGLE_PASSTHROUGH = 0
    FAIR               = 1
    STRICT_PRIORITY    = 2
    HYBRID             = 3


@dataclass
class TenantContext:
    tenant_id: int
    qos_class: QoSClass = QoSClass.BEST_EFFORT
    weight: int = 1
    sla_latency_ns: int = 0
    bandwidth_min_bps: int = 0     # 0 = no floor
    bandwidth_max_bps: int = 0     # 0 = unlimited
    create_ts_ns: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["qos_class"] = int(self.qos_class)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TenantContext":
        d = dict(d)
        d["qos_class"] = QoSClass(d.get("qos_class", 0))
        return cls(**d)


@dataclass
class BandwidthBudget:
    bytes_per_second: int = 0
    burst_bytes: int = 0
    last_refill_ts_ns: int = 0


@dataclass
class PolicyTable:
    mode: PolicyMode = PolicyMode.SINGLE_PASSTHROUGH
    tenants: dict[int, TenantContext] = field(default_factory=dict)
    budgets: dict[int, BandwidthBudget] = field(default_factory=dict)
    version: int = 0

    def active_tenant_ids(self) -> list[int]:
        return sorted(self.tenants.keys())

    def num_active(self) -> int:
        return len(self.tenants)

    def to_json(self) -> str:
        return json.dumps({
            "mode": int(self.mode),
            "version": self.version,
            "tenants": {str(k): v.to_dict() for k, v in self.tenants.items()},
            "budgets": {str(k): asdict(v) for k, v in self.budgets.items()},
        })

    @classmethod
    def from_json(cls, s: str) -> "PolicyTable":
        raw = json.loads(s)
        tbl = cls(
            mode=PolicyMode(raw.get("mode", 0)),
            version=int(raw.get("version", 0)),
        )
        for k, v in raw.get("tenants", {}).items():
            tbl.tenants[int(k)] = TenantContext.from_dict(v)
        for k, v in raw.get("budgets", {}).items():
            tbl.budgets[int(k)] = BandwidthBudget(**v)
        return tbl


# ---------------------------------------------------------------------------
# Token-bucket rate limiter (design.md §6.3.1)
# ---------------------------------------------------------------------------

class TokenBucket:
    """Byte-rate token bucket. Thread-safe.

    consume(bytes) returns immediately with (granted, wait_ns):
      granted=True  → bucket was charged, caller may proceed.
      granted=False → caller should sleep `wait_ns` and retry.
    """

    def __init__(self, refill_bps: int, burst_bytes: int):
        self.refill_bps = int(refill_bps) if refill_bps > 0 else 0
        self.burst_bytes = int(burst_bytes) if burst_bytes > 0 else int(refill_bps)
        self.tokens = float(self.burst_bytes)
        self.last_ts = time.perf_counter_ns()
        self.lock = threading.Lock()

    def _refill_locked(self, now_ns: int) -> None:
        if self.refill_bps == 0:
            self.tokens = float(self.burst_bytes)
            self.last_ts = now_ns
            return
        elapsed_s = (now_ns - self.last_ts) / 1e9
        if elapsed_s <= 0:
            return
        self.tokens = min(float(self.burst_bytes),
                          self.tokens + elapsed_s * self.refill_bps)
        self.last_ts = now_ns

    def consume(self, nbytes: int) -> tuple[bool, int]:
        with self.lock:
            now = time.perf_counter_ns()
            self._refill_locked(now)
            if self.refill_bps == 0:
                return True, 0
            if self.tokens >= nbytes:
                self.tokens -= nbytes
                return True, 0
            deficit = nbytes - self.tokens
            wait_ns = int(deficit / self.refill_bps * 1e9)
            return False, wait_ns


# ---------------------------------------------------------------------------
# Scheduler — operates on logical work items (one per collective invocation)
# ---------------------------------------------------------------------------

@dataclass
class WorkItem:
    tenant_id: int
    nbytes: int
    enqueue_ts_ns: int
    callback: Callable[[], None]     # actual collective invocation
    priority_boost: int = 0          # incremented by aging


class TenantScheduler:
    """Host-side WFQ / Strict Priority scheduler for multi-tenant collectives.

    Usage:
        sched = TenantScheduler(policy_table, mode=PolicyMode.FAIR)
        sched.start()
        sched.submit(WorkItem(tenant_id=k, nbytes=size, ..., callback=do_allreduce))
        sched.wait_idle()
        sched.stop()

    Single-tenant bypass: when exactly one tenant is registered AND
    the scheduler's queue is empty, submit() invokes the callback inline
    (zero overhead path, design.md §5.5).
    """

    def __init__(self,
                 policy: PolicyTable,
                 mode: PolicyMode | None = None,
                 quantum_bytes: int = DEFAULT_QUANTUM_BYTES,
                 aging_ms: int = DEFAULT_AGING_MS,
                 enforce_rate_limit: bool = True):
        self.policy = policy
        self.mode = mode if mode is not None else policy.mode
        self.quantum = quantum_bytes
        self.aging_ns = aging_ms * 1_000_000
        self.enforce_rate_limit = enforce_rate_limit

        self.queues: dict[int, deque[WorkItem]] = {}
        self.deficits: dict[int, int] = {}
        self.buckets: dict[int, TokenBucket] = {}
        self.condition = threading.Condition()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._idle_event = threading.Event()
        self._idle_event.set()
        self._completed = 0
        self._inflight = 0

        self._sync_buckets()

    def _sync_buckets(self) -> None:
        for tid, ctx in self.policy.tenants.items():
            if tid in self.buckets:
                continue
            budget = self.policy.budgets.get(tid)
            bps = ctx.bandwidth_max_bps if ctx.bandwidth_max_bps > 0 \
                  else (budget.bytes_per_second if budget else 0)
            burst = (budget.burst_bytes if budget else 0) or max(bps // 10, 1 << 20)
            self.buckets[tid] = TokenBucket(refill_bps=bps, burst_bytes=burst)
            self.queues.setdefault(tid, deque())
            self.deficits.setdefault(tid, 0)

    # -------- public API --------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="MtcclScheduler",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self.condition:
            self.condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def submit(self, item: WorkItem) -> None:
        # Single-tenant bypass: zero-overhead direct call.
        if (self.policy.num_active() <= 1
                and self.mode == PolicyMode.SINGLE_PASSTHROUGH
                and item.tenant_id in self.policy.tenants):
            item.callback()
            return

        self._sync_buckets()
        with self.condition:
            self.queues.setdefault(item.tenant_id, deque()).append(item)
            self.deficits.setdefault(item.tenant_id, 0)
            self._inflight += 1
            self._idle_event.clear()
            self.condition.notify_all()

    def wait_idle(self, timeout: float | None = None) -> bool:
        return self._idle_event.wait(timeout=timeout)

    def stats(self) -> dict:
        return {
            "completed": self._completed,
            "inflight": self._inflight,
            "queue_depths": {tid: len(q) for tid, q in self.queues.items()},
            "mode": str(self.mode),
        }

    # -------- internal scheduler loop --------

    def _loop(self) -> None:
        while not self._stop.is_set():
            item = self._pick_next()
            if item is None:
                with self.condition:
                    if all(len(q) == 0 for q in self.queues.values()):
                        self._idle_event.set()
                        self.condition.wait(timeout=0.05)
                continue

            if self.enforce_rate_limit:
                bucket = self.buckets.get(item.tenant_id)
                if bucket is not None:
                    granted, wait_ns = bucket.consume(item.nbytes)
                    if not granted:
                        # Re-queue at head and sleep briefly.
                        with self.condition:
                            self.queues[item.tenant_id].appendleft(item)
                        time.sleep(min(wait_ns / 1e9, 0.005))
                        continue

            try:
                item.callback()
            except Exception as e:  # noqa: BLE001
                print(f"[MtcclScheduler] callback failed tenant={item.tenant_id}: {e}",
                      flush=True)
            finally:
                with self.condition:
                    self._completed += 1
                    self._inflight = max(0, self._inflight - 1)
                    if self._inflight == 0 and all(len(q) == 0 for q in self.queues.values()):
                        self._idle_event.set()

    def _pick_next(self) -> Optional[WorkItem]:
        with self.condition:
            if self.mode == PolicyMode.STRICT_PRIORITY:
                return self._pick_strict_priority()
            if self.mode == PolicyMode.HYBRID:
                return self._pick_hybrid()
            return self._pick_drr()  # FAIR / SINGLE_PASSTHROUGH multi fallback

    # Deficit Round Robin (WFQ approximation)
    def _pick_drr(self) -> Optional[WorkItem]:
        candidates = [tid for tid, q in self.queues.items() if q]
        if not candidates:
            return None
        for tid in sorted(candidates):
            ctx = self.policy.tenants.get(tid)
            weight = ctx.weight if ctx else 1
            self.deficits[tid] += max(weight, 1) * self.quantum
            head = self.queues[tid][0]
            if head.nbytes <= self.deficits[tid]:
                self.deficits[tid] -= head.nbytes
                self.queues[tid].popleft()
                return head
        # Nothing ready this round; fall back to the smallest head.
        tid = min(candidates, key=lambda t: self.queues[t][0].nbytes)
        return self.queues[tid].popleft()

    def _pick_strict_priority(self) -> Optional[WorkItem]:
        now = time.perf_counter_ns()
        # Aging: bump effective priority for items waited > aging_ns.
        for q in self.queues.values():
            for it in q:
                if (now - it.enqueue_ts_ns) > self.aging_ns:
                    it.priority_boost = 1
        candidates = [(tid, q[0]) for tid, q in self.queues.items() if q]
        if not candidates:
            return None
        def key(pair):
            tid, head = pair
            ctx = self.policy.tenants.get(tid)
            base = int(ctx.qos_class) if ctx else 0
            return -(base + head.priority_boost)
        tid, head = min(candidates, key=key)
        self.queues[tid].popleft()
        return head

    def _pick_hybrid(self) -> Optional[WorkItem]:
        # Realtime/Premium go via strict priority; the rest via DRR.
        premium_ids = [tid for tid, ctx in self.policy.tenants.items()
                       if ctx.qos_class >= QoSClass.PREMIUM and self.queues.get(tid)]
        if premium_ids:
            tid = max(premium_ids, key=lambda t: int(self.policy.tenants[t].qos_class))
            return self.queues[tid].popleft()
        return self._pick_drr()


# ---------------------------------------------------------------------------
# Policy table persistence (control-plane → data-plane, design.md §6.2)
# ---------------------------------------------------------------------------

def publish_policy(table: PolicyTable, path: str = DEFAULT_POLICY_SHM) -> None:
    """Atomically publish a PolicyTable as JSON (tmpfile + rename)."""
    table.version += 1
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(table.to_json())
    os.replace(tmp, path)


def load_policy(path: str = DEFAULT_POLICY_SHM) -> PolicyTable:
    if not os.path.exists(path):
        return PolicyTable()
    with open(path) as f:
        return PolicyTable.from_json(f.read())


# ---------------------------------------------------------------------------
# Convenience: build a PolicyTable from a YAML/dict config (design.md §6.1)
# ---------------------------------------------------------------------------

def build_policy(mode: PolicyMode, tenants: list[dict]) -> PolicyTable:
    table = PolicyTable(mode=mode)
    for spec in tenants:
        ctx = TenantContext(
            tenant_id=int(spec["tenant_id"]),
            qos_class=QoSClass(int(spec.get("qos_class", QoSClass.BEST_EFFORT))),
            weight=int(spec.get("weight", 1)),
            sla_latency_ns=int(spec.get("sla_latency_ns", 0)),
            bandwidth_min_bps=int(spec.get("bandwidth_min_bps", 0)),
            bandwidth_max_bps=int(spec.get("bandwidth_max_bps", 0)),
            create_ts_ns=time.time_ns(),
        )
        table.tenants[ctx.tenant_id] = ctx
        bps = ctx.bandwidth_max_bps or 0
        burst = int(spec.get("burst_bytes", 0)) or max(bps // 10, 1 << 20)
        table.budgets[ctx.tenant_id] = BandwidthBudget(
            bytes_per_second=bps,
            burst_bytes=burst,
            last_refill_ts_ns=time.time_ns(),
        )
    return table


__all__ = [
    "MAX_TENANTS", "DEFAULT_TENANT",
    "QoSClass", "PolicyMode",
    "TenantContext", "BandwidthBudget", "PolicyTable",
    "TokenBucket", "TenantScheduler", "WorkItem",
    "publish_policy", "load_policy", "build_policy",
]
