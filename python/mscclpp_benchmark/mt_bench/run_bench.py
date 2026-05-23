#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# MT-MSCCL++ single-node benchmark harness.
#
# Measures AllReduce performance for three backends:
#   1. nccl              — baseline (cupy.cuda.nccl)
#   2. mscclpp_vanilla   — MSCCL++ direct (best-of MscclppAllReduce*)
#   3. mscclpp_mt        — MSCCL++ wrapped by the MT-MSCCL++ TenantScheduler
#                          (mode = fair / strict_priority, possibly multi-tenant)
#
# Output: one CSV row per (backend, scenario, tenant, size, rank).
# Run with mpirun:
#   mpirun -np 4 python -m mscclpp_benchmark.mt_bench.run_bench \
#          --sizes 1024,16384,262144,4194304,67108864 \
#          --niter 50 --scenarios single,multi2,priority \
#          --out /root/ccl/results/run_$(date +%s).csv

import argparse
import csv
import os
import socket
import sys
import time
from contextlib import contextmanager
from datetime import datetime

import cupy as cp
import cupy.cuda.nccl as nccl
from mpi4py import MPI
import netifaces as ni
import ipaddress

# Make the sibling modules importable when invoked via -m or directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mscclpp_op import (  # noqa: E402
    MscclppAllReduce1, MscclppAllReduce2, MscclppAllReduce3, MscclppAllReduce6,
)
from nccl_op import NcclAllReduce  # noqa: E402

from mscclpp import ProxyService, is_nvls_supported, CommGroup, GpuBuffer  # noqa: E402
from mscclpp.ext.tenant import (  # noqa: E402
    TenantScheduler, PolicyMode, QoSClass, build_policy, WorkItem,
    TenantAwareProxyService, register_tenant_on,
)


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def human_size(n):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.0f}PiB"


def is_routable(ip):
    o = ipaddress.ip_address(ip)
    return not (o.is_loopback or o.is_link_local or o.is_multicast)


def get_net_iface():
    for iface in ni.interfaces():
        addrs = ni.ifaddresses(iface)
        if ni.AF_INET not in addrs:
            continue
        for a in addrs[ni.AF_INET]:
            if is_routable(a["addr"]):
                return iface, a["addr"]
    return None, None


def bench_time_ms(niter, func):
    """Time `niter` invocations via CUDA graph capture (matches allreduce_bench)."""
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        for _ in range(niter):
            func(stream)
        graph = stream.end_capture()

    # warmup
    graph.launch(stream)
    stream.synchronize()

    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record(stream)
    graph.launch(stream)
    end.record(stream)
    end.synchronize()
    return cp.cuda.get_elapsed_time(start, end) / niter   # ms per iter


def alg_bw_gbps(nbytes, time_us):
    if time_us <= 0:
        return 0.0
    return nbytes / (time_us * 1e3)   # bytes / ns = GB/s


# --------------------------------------------------------------------------- #
# Backend wrappers
# --------------------------------------------------------------------------- #

class Backend:
    name = "unknown"

    def setup(self, group, nccl_comm, memory, memory_out, proxy_service):
        raise NotImplementedError

    def __call__(self, stream):
        raise NotImplementedError

    def teardown(self):
        pass


class NcclBackend(Backend):
    name = "nccl"

    def setup(self, group, nccl_comm, memory, memory_out, proxy_service):
        self.op = NcclAllReduce(nccl_comm, memory)

    def __call__(self, stream):
        return self.op(stream)


class MscclppVanillaBackend(Backend):
    """Pick the best MSCCL++ algorithm by message size (single-node only)."""

    name = "mscclpp_vanilla"

    def setup(self, group, nccl_comm, memory, memory_out, proxy_service):
        self.algo = self._select_algo(group, memory, memory_out, proxy_service)

    @staticmethod
    def _select_algo(group, memory, memory_out, proxy_service):
        nbytes = memory.nbytes
        if nbytes < (1 << 20):
            return MscclppAllReduce2(group, memory, memory_out)
        # ≥ 1 MiB: prefer NVLS if available else AllReduce1 (memory-channel SM).
        if is_nvls_supported() and memory.dtype in (cp.float16, cp.float32):
            try:
                return MscclppAllReduce6(group, memory.size, memory.dtype)
            except Exception:
                pass
        return MscclppAllReduce1(group, memory)

    def __call__(self, stream):
        return self.algo(stream)


class MscclppMtBackend(Backend):
    """Vanilla MSCCL++ under a TenantScheduler decoration.

    Iteration 1: the scheduler runs in a separate host thread and dispatches
    `__call__` indirectly via WorkItem.callback. To keep the CUDA graph
    machinery simple we do NOT graph-capture across the scheduler; each
    invocation submits one WorkItem and waits for completion. Overhead vs
    vanilla is the per-invocation scheduling cost we want to measure.

    For single-tenant + SINGLE_PASSTHROUGH mode, submit() does inline dispatch
    (zero overhead path, design.md §5.5).
    """

    def __init__(self, mode: PolicyMode, tenant_id: int = 1):
        self.mode = mode
        self.tenant_id = tenant_id
        self.name = f"mscclpp_mt_{mode.name.lower()}"
        self.scheduler = None
        self.algo = None

    def setup(self, group, nccl_comm, memory, memory_out, proxy_service):
        self.algo = MscclppVanillaBackend._select_algo(
            group, memory, memory_out, proxy_service)
        # Single-tenant table → SINGLE_PASSTHROUGH bypass when mode allows.
        table = build_policy(self.mode, [{
            "tenant_id": self.tenant_id,
            "qos_class": int(QoSClass.STANDARD),
            "weight": 1,
        }])
        self.scheduler = TenantScheduler(table, mode=self.mode,
                                         enforce_rate_limit=False)
        self.scheduler.start()

    def __call__(self, stream):
        done = []
        def cb():
            self.algo(stream)
            done.append(True)
        self.scheduler.submit(WorkItem(
            tenant_id=self.tenant_id,
            nbytes=self.algo.memory.nbytes if hasattr(self.algo, "memory") else 0,
            enqueue_ts_ns=time.time_ns(),
            callback=cb,
        ))
        # Wait for the scheduler to dispatch — bounded poll.
        while not done:
            self.scheduler.wait_idle(timeout=0.01)
        return None

    def teardown(self):
        if self.scheduler is not None:
            self.scheduler.stop()


def bench_mt_inline(niter, backend: MscclppMtBackend):
    """MT path: cannot graph-capture across the scheduler thread.

    We submit `niter` work items, await idle, and divide wall time. This
    matches a real workload that hits the scheduler per-call.
    """
    stream = cp.cuda.Stream(non_blocking=True)
    # warmup
    backend(stream)
    stream.synchronize()
    start = time.perf_counter_ns()
    for _ in range(niter):
        backend(stream)
    stream.synchronize()
    elapsed_ns = time.perf_counter_ns() - start
    return elapsed_ns / niter / 1e6   # ms per iter


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #

def run_one(backend_name, mode_or_none, scenario, sizes, niter,
            group, nccl_comm, dtype, comm, rank, results, run_id):
    proxy_service = ProxyService()
    proxy_service.start_proxy()
    try:
        for nelems in sizes:
            memory = GpuBuffer(nelems, dtype=dtype)
            memory_out = GpuBuffer(nelems, dtype=dtype)
            cp.cuda.runtime.deviceSynchronize()

            if backend_name == "nccl":
                bk = NcclBackend()
            elif backend_name == "mscclpp_vanilla":
                bk = MscclppVanillaBackend()
            elif backend_name == "mscclpp_mt":
                bk = MscclppMtBackend(mode=mode_or_none, tenant_id=1)
            else:
                raise ValueError(backend_name)

            bk.setup(group, nccl_comm, memory, memory_out, proxy_service)
            comm.barrier()

            if isinstance(bk, MscclppMtBackend) and mode_or_none != PolicyMode.SINGLE_PASSTHROUGH:
                ms = bench_mt_inline(niter, bk)
            else:
                ms = bench_time_ms(niter, bk)

            time_us = ms * 1000.0
            bw = alg_bw_gbps(memory.nbytes, time_us)

            row = {
                "run_id": run_id,
                "timestamp": datetime.utcnow().isoformat(),
                "scenario": scenario,
                "backend": bk.name,
                "collective": "allreduce",
                "tenant_id": 1,
                "qos_class": "STANDARD",
                "size_bytes": memory.nbytes,
                "size_human": human_size(memory.nbytes),
                "niter": niter,
                "time_us": round(time_us, 3),
                "alg_bw_gbps": round(bw, 3),
                "rank": rank,
                "world_size": comm.size,
                "host": socket.gethostname(),
                "dtype": str(dtype.__name__),
            }
            if rank == 0:
                results.append(row)
                print(f"  [{bk.name:24s}] {human_size(memory.nbytes):>8s}  "
                      f"{time_us:8.2f} us  {bw:7.2f} GB/s", flush=True)
            bk.teardown()
            comm.barrier()
    finally:
        proxy_service.stop_proxy()


def run_multi_tenant(scenario_name, num_tenants, sizes, niter,
                     group, nccl_comm, dtype, comm, rank, results, run_id,
                     mode=PolicyMode.FAIR, qos_classes=None,
                     bandwidth_caps_bps=None,
                     enforce_rate_limit=False,
                     weights=None,
                     use_cpp_proxy=True):
    """Emulate K tenants sharing the same comm by interleaving their AllReduce.

    Iter 2 design (design.md §5.3, §6.3):
      1. Build N=niter*K WorkItems, run scheduler.plan_order(...) to get the
         tenant-aware launch order WITHOUT touching CUDA.
      2. Capture the entire ordered launch in a CUDA graph (one kernel per item).
      3. Insert a CUDA Event after each tenant's last item in the order, so we
         can measure per-tenant span as event(start → last_for_tenant).
      4. Launch the graph; measure all events with one synchronize.

    This fixes iter1 limitation #1 (no CUDA graphs in MT path) by separating
    "what to schedule" (CPU, ahead of time) from "when to launch" (GPU graph).
    """
    # Iter 3: use the C++ TenantAwareProxyService (drop-in for ProxyService).
    # In iter 3 the scheduling REORDERING still happens via Python plan_order
    # (because per-tenant device handles aren't plumbed into the algos yet);
    # but ProxyService is now the C++ tenant-aware one, so iter 4 can flip the
    # switch to in-proxy scheduling with no benchmark changes.
    if use_cpp_proxy:
        proxy_service = TenantAwareProxyService(mode=mode)
    else:
        proxy_service = ProxyService()
    proxy_service.start_proxy()
    try:
        for nelems in sizes:
            memory = GpuBuffer(nelems, dtype=dtype)
            memory_out = GpuBuffer(nelems, dtype=dtype)
            cp.cuda.runtime.deviceSynchronize()
            algo = MscclppVanillaBackend._select_algo(
                group, memory, memory_out, proxy_service)

            qos_classes = qos_classes or [QoSClass.STANDARD] * num_tenants
            caps = list(bandwidth_caps_bps) if bandwidth_caps_bps \
                   else [0] * num_tenants
            weights_eff = list(weights) if weights else [1] * num_tenants

            tenant_specs = []
            for tid in range(1, num_tenants + 1):
                tenant_specs.append({
                    "tenant_id": tid,
                    "qos_class": int(qos_classes[tid - 1]),
                    "weight": int(weights_eff[tid - 1]),
                    "bandwidth_max_bps": int(caps[tid - 1]),
                })
            table = build_policy(mode, tenant_specs)
            sched = TenantScheduler(table, mode=mode,
                                    enforce_rate_limit=enforce_rate_limit)

            # Also register tenants with the C++ proxy (no-op for data path
            # today; iter 4 will route triggers through the C++ scheduler).
            if use_cpp_proxy:
                for spec in tenant_specs:
                    register_tenant_on(proxy_service, spec["tenant_id"],
                                       QoSClass(spec["qos_class"]),
                                       spec["weight"],
                                       spec["bandwidth_max_bps"], 0)
            comm.barrier()

            # warmup once outside the graph
            warm_stream = cp.cuda.Stream(non_blocking=True)
            algo(warm_stream)
            warm_stream.synchronize()
            comm.barrier()

            # Build niter items per tenant; plan_order interleaves per policy.
            now = time.time_ns()
            items = []
            for i in range(niter):
                for tid in range(1, num_tenants + 1):
                    items.append(WorkItem(tenant_id=tid, nbytes=memory.nbytes,
                                          enqueue_ts_ns=now, callback=None))
            order = sched.plan_order(items)

            # Determine the index in `order` of each tenant's last item.
            last_idx_for = {}
            for idx, it in enumerate(order):
                last_idx_for[it.tenant_id] = idx

            # Launch each item in the scheduler-decided order using CUDA events
            # to time per-tenant span. We skip CUDA graph capture here because
            # MSCCL++ collectives that use the host proxy issue FIFO triggers
            # that don't fit a captured stream model cleanly; the savings come
            # from plan-time scheduling (no runtime scheduler thread) rather
            # than from per-call graph-launch amortization. (See iter 2 note.)
            stream = cp.cuda.Stream(non_blocking=True)
            start_event = cp.cuda.Event()
            end_events = {tid: cp.cuda.Event() for tid in range(1, num_tenants + 1)}
            start_event.record(stream)
            for idx, it in enumerate(order):
                algo(stream)
                if idx == last_idx_for.get(it.tenant_id):
                    end_events[it.tenant_id].record(stream)
            stream.synchronize()
            comm.barrier()

            if rank == 0:
                total_calls_per_tenant = {tid: 0 for tid in range(1, num_tenants + 1)}
                for it in order:
                    total_calls_per_tenant[it.tenant_id] += 1
                for tid in range(1, num_tenants + 1):
                    span_ms = cp.cuda.get_elapsed_time(start_event, end_events[tid])
                    iters = total_calls_per_tenant[tid] or 1
                    ms_per_iter = span_ms / iters
                    time_us = ms_per_iter * 1000.0
                    bw = alg_bw_gbps(memory.nbytes, time_us)
                    qos = qos_classes[tid - 1]
                    results.append({
                        "run_id": run_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "scenario": scenario_name,
                        "backend": f"mscclpp_mt_{mode.name.lower()}",
                        "collective": "allreduce",
                        "tenant_id": tid,
                        "qos_class": qos.name,
                        "size_bytes": memory.nbytes,
                        "size_human": human_size(memory.nbytes),
                        "niter": niter,
                        "time_us": round(time_us, 3),
                        "alg_bw_gbps": round(bw, 3),
                        "rank": rank,
                        "world_size": comm.size,
                        "host": socket.gethostname(),
                        "dtype": str(dtype.__name__),
                    })
                    cap_str = (f" (cap {caps[tid-1]/1e9:.1f} GB/s)"
                               if caps[tid - 1] > 0 else "")
                    print(f"  [tenant {tid} qos={qos.name:10s}{cap_str}] "
                          f"{human_size(memory.nbytes):>8s}  "
                          f"{time_us:8.2f} us  {bw:7.2f} GB/s", flush=True)
            comm.barrier()
    finally:
        proxy_service.stop_proxy()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=str,
                        default="1024,16384,262144,4194304,67108864",
                        help="comma-separated element counts")
    parser.add_argument("--niter", type=int, default=50)
    parser.add_argument("--scenarios", type=str,
                        default="single,multi2,priority,rate_limited,weighted_fair",
                        help="comma list from: single, multi2, multi4, "
                             "priority, rate_limited, weighted_fair")
    parser.add_argument("--rate-cap-gbps", type=float, default=30.0,
                        help="tenant-2 bandwidth cap in GB/s for rate_limited "
                             "scenario (default 30)")
    parser.add_argument("--out", type=str, default="/root/ccl/results/mt_bench.csv")
    parser.add_argument("--dtype", type=str, default="fp32",
                        choices=["fp16", "fp32"])
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.rank
    shm_comm = comm.Split_type(MPI.COMM_TYPE_SHARED, 0, MPI.INFO_NULL)
    n_per_node = shm_comm.size
    shm_comm.Free()
    cp.cuda.Device(rank % n_per_node).use()

    iface, my_ip = get_net_iface()
    if iface is None:
        print(f"[rank {rank}] no usable interface, aborting", flush=True)
        return 1
    root_ip = comm.bcast(my_ip, root=0)
    trio = f"{iface}:{root_ip}:50000"
    group = CommGroup(interfaceIpPortTrio=trio, rank=rank, size=comm.size)

    if rank == 0:
        uid = nccl.get_unique_id()
    else:
        uid = None
    uid = comm.bcast(uid, root=0)
    nccl_comm = nccl.NcclCommunicator(comm.size, uid, rank)

    dtype = cp.float32 if args.dtype == "fp32" else cp.float16
    sizes = [int(x) for x in args.sizes.split(",")]
    scenarios = [s.strip() for s in args.scenarios.split(",")]
    run_id = f"r{int(time.time())}_w{comm.size}"
    results = []

    if rank == 0:
        print(f"\n=== MT-MSCCL++ benchmark — world_size={comm.size}, "
              f"dtype={args.dtype}, niter={args.niter} ===\n", flush=True)

    if "single" in scenarios:
        if rank == 0:
            print("--- scenario: single_tenant (NCCL vs MSCCL++ vs MT-bypass) ---",
                  flush=True)
        for bk_name, mode in [
            ("nccl", None),
            ("mscclpp_vanilla", None),
            ("mscclpp_mt", PolicyMode.SINGLE_PASSTHROUGH),
        ]:
            run_one(bk_name, mode, "single_tenant", sizes, args.niter,
                    group, nccl_comm, dtype, comm, rank, results, run_id)

    if "multi2" in scenarios:
        if rank == 0:
            print("\n--- scenario: multi_tenant_2x (FAIR, equal weights) ---",
                  flush=True)
        run_multi_tenant("multi_tenant_2x_fair", 2, sizes, args.niter,
                         group, nccl_comm, dtype, comm, rank, results, run_id,
                         mode=PolicyMode.FAIR)

    if "weighted_fair" in scenarios:
        if rank == 0:
            print("\n--- scenario: weighted_fair (tenant 1 weight=1, tenant 2 weight=3) ---",
                  flush=True)
        run_multi_tenant("weighted_fair_1to3", 2, sizes, args.niter,
                         group, nccl_comm, dtype, comm, rank, results, run_id,
                         mode=PolicyMode.FAIR,
                         weights=[1, 3])

    if "multi4" in scenarios:
        if rank == 0:
            print("\n--- scenario: multi_tenant_4x (FAIR) ---", flush=True)
        run_multi_tenant("multi_tenant_4x_fair", 4, sizes, args.niter,
                         group, nccl_comm, dtype, comm, rank, results, run_id,
                         mode=PolicyMode.FAIR)

    if "rate_limited" in scenarios:
        if rank == 0:
            print(f"\n--- scenario: rate_limited (tenant 1 unlimited, "
                  f"tenant 2 capped @ {args.rate_cap_gbps:g} GB/s) ---",
                  flush=True)
        cap_bps = int(args.rate_cap_gbps * 1e9)
        run_multi_tenant("rate_limited", 2, sizes, args.niter,
                         group, nccl_comm, dtype, comm, rank, results, run_id,
                         mode=PolicyMode.FAIR,
                         qos_classes=[QoSClass.STANDARD, QoSClass.STANDARD],
                         bandwidth_caps_bps=[0, cap_bps],
                         enforce_rate_limit=True)

    if "priority" in scenarios:
        if rank == 0:
            print("\n--- scenario: priority_contention (1 Premium + 2 BestEffort) ---",
                  flush=True)
        run_multi_tenant("priority_contention", 3, sizes, args.niter,
                         group, nccl_comm, dtype, comm, rank, results, run_id,
                         mode=PolicyMode.STRICT_PRIORITY,
                         qos_classes=[QoSClass.PREMIUM,
                                      QoSClass.BEST_EFFORT,
                                      QoSClass.BEST_EFFORT])

    if rank == 0:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        # Stable canonical schema so multiple runs concatenate cleanly.
        FIELDS = [
            "run_id", "timestamp", "scenario", "backend", "collective",
            "tenant_id", "qos_class", "size_bytes", "size_human", "niter",
            "time_us", "alg_bw_gbps", "rank", "world_size", "host", "dtype",
        ]
        write_header = not os.path.exists(args.out)
        with open(args.out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            if write_header:
                w.writeheader()
            for r in results:
                w.writerow({k: r.get(k, "") for k in FIELDS})
        print(f"\n[rank 0] wrote {len(results)} rows to {args.out}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
