#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# C++ tenant-aware proxy AllReduce benchmark.
#
# Typical 2-node x 2-GPU run:
#
#   MTCCL_K_STREAMS=1 mpirun --allow-run-as-root \
#     -np 4 -H suit2:2,suit1:2 \
#     -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH \
#     -x MSCCLPP_HOME -x CUPY_CACHE_DIR -x MPLCONFIGDIR \
#     -x MTCCL_K_STREAMS -x MSCCLPP_HCA_DEVICES \
#     -x MSCCLPP_SOCKET_IFNAME -x NCCL_SOCKET_IFNAME -x NCCL_IB_HCA \
#     python -m mtccl.mt_cpp_bench \
#       --dtype fp16 \
#       --sizes-bytes 12MiB \
#       --niter 50 \
#       --out /mnt/nfs/ccl/results/1mt/mtccl_bench.csv

from __future__ import annotations

import argparse
import csv
import ipaddress
import os
import re
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUPY_CACHE_DIR", "/tmp/cupy-cache")

import cupy as cp
import netifaces as ni

try:
    from mscclpp_benchmark.mscclpp_op import (
        MscclppAllReduce3,
        MscclppAllReduce4,
        MscclppAllReduce5,
    )
except ImportError:
    # Support direct execution from mscclpp/python/mtccl.
    bench_dir = Path(__file__).resolve().parents[1] / "mscclpp_benchmark"
    sys.path.insert(0, str(bench_dir))
    from mscclpp_op import (  # type: ignore  # noqa: E402
        MscclppAllReduce3,
        MscclppAllReduce4,
        MscclppAllReduce5,
    )

from mscclpp import CommGroup, GpuBuffer  # noqa: E402
from mscclpp.ext.tenant import (  # noqa: E402
    PolicyMode,
    QoSClass,
    TenantAwareProxyService,
    register_tenant_on,
)


CSV_FIELDS = [
    "run_id",
    "timestamp",
    "scenario",
    "backend",
    "collective",
    "tenant_id",
    "tenant_name",
    "qos_class",
    "policy",
    "configured_weight",
    "size_bytes",
    "size_human",
    "niter",
    "time_us",
    "alg_bw_gbps",
    "wallclock_gbps",
    "wall_s",
    "bytes_sent",
    "rank",
    "world_size",
    "host",
    "dtype",
    "sched_dispatched_triggers",
    "sched_dispatched_bytes",
    "token_bucket_waits",
    "drr_picks",
    "strict_priority_picks",
]


@dataclass(frozen=True)
class TenantSpec:
    tenant_id: int
    name: str
    qos: QoSClass
    weight: int
    bandwidth_cap_bps: int = 0


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    policy: str
    mode: PolicyMode
    tenants: tuple[TenantSpec, ...]


def human_size(nbytes: int) -> str:
    value = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            if unit == "B":
                return f"{int(value)}B"
            if value.is_integer():
                return f"{int(value)}{unit}"
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{int(value)}TiB"


def parse_size_bytes(token: str) -> int:
    raw = token.strip()
    if not raw:
        raise ValueError("empty size token")

    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([A-Za-z]*)", raw)
    if not match:
        raise ValueError(f"invalid size {raw!r}")

    value = float(match.group(1))
    unit = match.group(2).lower()
    multipliers = {
        "": 1,
        "b": 1,
        "k": 1024,
        "kb": 1024,
        "kib": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "mib": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
        "gib": 1024**3,
        "t": 1024**4,
        "tb": 1024**4,
        "tib": 1024**4,
    }
    if unit not in multipliers:
        raise ValueError(f"unsupported size unit in {raw!r}")
    size = int(value * multipliers[unit])
    if size <= 0:
        raise ValueError(f"size must be positive: {raw!r}")
    return size


def parse_size_list_bytes(spec: str) -> list[int]:
    return [parse_size_bytes(tok) for tok in spec.split(",") if tok.strip()]


def dtype_from_arg(name: str):
    if name == "fp16":
        return cp.float16
    if name == "fp32":
        return cp.float32
    if name == "int32":
        return cp.int32
    raise ValueError(f"unsupported dtype {name}")


def dtype_name(dtype) -> str:
    return cp.dtype(dtype).name


def bytes_to_nelems(size_bytes: int, dtype) -> int:
    itemsize = cp.dtype(dtype).itemsize
    if size_bytes % itemsize != 0:
        raise ValueError(
            f"{human_size(size_bytes)} is not divisible by dtype itemsize {itemsize}"
        )
    return size_bytes // itemsize


def is_routable(ip: str) -> bool:
    addr = ipaddress.ip_address(ip)
    return not (addr.is_loopback or addr.is_link_local or addr.is_multicast)


def get_net_iface() -> tuple[str, str]:
    requested = os.environ.get("MSCCLPP_SOCKET_IFNAME", "").strip()
    if requested:
        if requested not in ni.interfaces():
            raise RuntimeError(f"MSCCLPP_SOCKET_IFNAME={requested} was not found")
        addrs = ni.ifaddresses(requested).get(ni.AF_INET, [])
        for addr in addrs:
            ip = addr.get("addr")
            if ip and is_routable(ip):
                return requested, ip
        raise RuntimeError(f"MSCCLPP_SOCKET_IFNAME={requested} has no routable IPv4")

    for iface in ni.interfaces():
        for addr in ni.ifaddresses(iface).get(ni.AF_INET, []):
            ip = addr.get("addr")
            if ip and is_routable(ip):
                return iface, ip
    raise RuntimeError("no routable IPv4 interface found")


def alg_bw_gbps(nbytes: int, time_us: float) -> float:
    if time_us <= 0:
        return 0.0
    return nbytes / (time_us * 1e3)


def assign_tenant_to_allreduce3(algo, tenant_id: int) -> None:
    for rank in range(algo.group.nranks):
        if rank == algo.group.my_rank:
            continue
        algo.fst_round_port_chans[rank].set_tenant_id(tenant_id)
        algo.snd_round_port_chans[rank].set_tenant_id(tenant_id)

    algo.fst_device_handles = [
        algo.fst_round_port_chans[rank].device_handle().raw
        for rank in range(algo.group.nranks)
        if rank != algo.group.my_rank
    ]
    algo.snd_device_handles = [
        algo.snd_round_port_chans[rank].device_handle().raw
        for rank in range(algo.group.nranks)
        if rank != algo.group.my_rank
    ]
    algo.fst_device_handles_cp = cp.asarray(
        memoryview(b"".join(algo.fst_device_handles)), dtype=cp.uint8
    )
    algo.snd_device_handles_cp = cp.asarray(
        memoryview(b"".join(algo.snd_device_handles)), dtype=cp.uint8
    )
    algo.set_params(algo.nblocks, algo.block_size)


def assign_tenant_to_allreduce4(algo, tenant_id: int) -> None:
    for rank in range(algo.group.nranks):
        if rank == algo.group.my_rank:
            continue
        algo.reduce_scatter_port_channels[rank].set_tenant_id(tenant_id)
        algo.all_gather_port_channels[rank].set_tenant_id(tenant_id)

    algo.reduce_sactter_proxy_device_handles = [
        algo.reduce_scatter_port_channels[rank].device_handle().raw
        for rank in range(algo.group.nranks)
        if rank != algo.group.my_rank
    ]
    algo.all_gather_proxy_device_handles = [
        algo.all_gather_port_channels[rank].device_handle().raw
        for rank in range(algo.group.nranks)
        if rank != algo.group.my_rank
    ]
    algo.reduce_sactter_proxy_device_handles_cp = cp.asarray(
        memoryview(b"".join(algo.reduce_sactter_proxy_device_handles)),
        dtype=cp.uint8,
    )
    algo.all_gather_proxy_device_handles_cp = cp.asarray(
        memoryview(b"".join(algo.all_gather_proxy_device_handles)), dtype=cp.uint8
    )
    algo.set_params(algo.nblocks, algo.block_size, algo.pipeline_depth)


def assign_tenant_to_allreduce5(algo, tenant_id: int) -> None:
    def in_same_node(rank: int) -> bool:
        return rank // algo.nranks_per_node == algo.group.my_rank // algo.nranks_per_node

    for rank in range(algo.group.nranks):
        if rank != algo.group.my_rank and not in_same_node(rank):
            algo.port_channels[rank].set_tenant_id(tenant_id)

    algo.proxy_device_handles = [
        algo.port_channels[rank].device_handle().raw
        for rank in range(algo.group.nranks)
        if rank != algo.group.my_rank and not in_same_node(rank)
    ]
    algo.proxy_device_handles_cp = cp.asarray(
        memoryview(b"".join(algo.proxy_device_handles)), dtype=cp.uint8
    )
    algo.set_params(algo.nblocks, algo.block_size)


def validate_xnode_ar4_size(nelems: int, dtype, world_size: int, nranks_per_node: int) -> None:
    dtype_obj = cp.dtype(dtype)
    elems_per_int = max(1, 4 // dtype_obj.itemsize)
    if nelems % elems_per_int != 0:
        raise ValueError(
            f"AR4 requires nelems divisible by {elems_per_int} for {dtype_obj.name}; "
            f"got {nelems}"
        )
    int_elems = nelems // elems_per_int
    if int_elems % (world_size * 3) != 0:
        raise ValueError(
            "cross-node AR4 requires sizes compatible with world_size * "
            f"pipeline_depth = {world_size * 3}; got nelems={nelems}. "
            "For 2 nodes x 2 GPUs fp16, use sizes like 12MiB, 48MiB, 192MiB."
        )


def make_cpp_mt_algo(group, memory, memory_out, proxy_service, nranks_per_node: int):
    if group.nranks == nranks_per_node:
        return MscclppAllReduce3(group, memory, proxy_service), assign_tenant_to_allreduce3

    if memory.nbytes < (1 << 22):
        return (
            MscclppAllReduce5(group, memory, memory_out, nranks_per_node, proxy_service),
            assign_tenant_to_allreduce5,
        )

    validate_xnode_ar4_size(memory.size, memory.dtype, group.nranks, nranks_per_node)
    return (
        MscclppAllReduce4(group, memory, nranks_per_node, proxy_service),
        assign_tenant_to_allreduce4,
    )


def make_scenario_specs(rate_cap_gbps: float) -> dict[str, ScenarioSpec]:
    cap_bps = int(rate_cap_gbps * 1e9)
    tenant_a = "tenantA"
    tenant_b = "tenantB"
    tenant_c = "tenantC"
    return {
        "fair": ScenarioSpec(
            name="mtccl_fair_3tenant",
            policy="fair",
            mode=PolicyMode.FAIR,
            tenants=(
                TenantSpec(1, tenant_a, QoSClass.STANDARD, 50),
                TenantSpec(2, tenant_b, QoSClass.STANDARD, 50),
                TenantSpec(3, tenant_c, QoSClass.STANDARD, 50),
            ),
        ),
        "priority": ScenarioSpec(
            name="mtccl_priority_3tenant",
            policy="strict_priority",
            mode=PolicyMode.STRICT_PRIORITY,
            tenants=(
                TenantSpec(1, tenant_a, QoSClass.PREMIUM, 50),
                TenantSpec(2, tenant_b, QoSClass.STANDARD, 30),
                TenantSpec(3, tenant_c, QoSClass.BEST_EFFORT, 20),
            ),
        ),
        "weighted": ScenarioSpec(
            name="mtccl_weighted_3tenant",
            policy="fair",
            mode=PolicyMode.FAIR,
            tenants=(
                TenantSpec(1, tenant_a, QoSClass.STANDARD, 20),
                TenantSpec(2, tenant_b, QoSClass.STANDARD, 30),
                TenantSpec(3, tenant_c, QoSClass.STANDARD, 50),
            ),
        ),
        "rate_limited": ScenarioSpec(
            name="mtccl_rate_limited_3tenant",
            policy="fair",
            mode=PolicyMode.FAIR,
            tenants=(
                TenantSpec(1, tenant_a, QoSClass.STANDARD, 50),
                TenantSpec(2, tenant_b, QoSClass.STANDARD, 50),
                TenantSpec(3, tenant_c, QoSClass.STANDARD, 50, cap_bps),
            ),
        ),
    }


def normalize_scenarios(raw: str, specs: dict[str, ScenarioSpec]) -> list[ScenarioSpec]:
    aliases = {
        "mtccl_fair_3tenant": "fair",
        "mtccl_priority_3tenant": "priority",
        "mtccl_weighted_3tenant": "weighted",
        "mtccl_rate_limited_3tenant": "rate_limited",
        "cpp_mt_fair": "fair",
        "cpp_mt_priority": "priority",
        "cpp_mt_weighted": "weighted",
        "cpp_mt_rate_limited": "rate_limited",
    }
    requested = [part.strip() for part in raw.split(",") if part.strip()]
    if not requested:
        raise ValueError("--scenarios cannot be empty")
    if "all" in requested:
        requested = ["fair", "priority", "weighted", "rate_limited"]

    out = []
    seen = set()
    for name in requested:
        key = aliases.get(name, name)
        if key not in specs:
            valid = ",".join(["fair", "priority", "weighted", "rate_limited", "all"])
            raise ValueError(f"unknown scenario {name!r}; valid values: {valid}")
        if key not in seen:
            out.append(specs[key])
            seen.add(key)
    return out


def parse_ops(raw: str | None, tenants: tuple[TenantSpec, ...], default_ops: int) -> dict[int, int]:
    if raw is None or not raw.strip():
        values = [default_ops] * len(tenants)
    else:
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
        if len(values) == 1:
            values = values * len(tenants)
        elif len(values) != len(tenants):
            raise ValueError(
                f"--ops expects one value or {len(tenants)} comma-separated values "
                f"for {[tenant.name for tenant in tenants]}, got {raw!r}"
            )

    if any(value <= 0 for value in values):
        raise ValueError("--ops values must be positive")
    return {tenant.tenant_id: value for tenant, value in zip(tenants, values)}


def kstream_enabled(single_stream: bool) -> bool:
    if single_stream:
        os.environ["MTCCL_K_STREAMS"] = "0"
        return False
    os.environ["MTCCL_K_STREAMS"] = "1"
    return True


def backend_label(mode: PolicyMode, use_k_streams: bool) -> str:
    suffix = "kstream" if use_k_streams else "single_stream_smoke"
    if mode == PolicyMode.STRICT_PRIORITY:
        return f"mscclpp_cpp_mt_strict_priority_{suffix}"
    if mode == PolicyMode.HYBRID:
        return f"mscclpp_cpp_mt_hybrid_{suffix}"
    return f"mscclpp_cpp_mt_fair_{suffix}"


def run_cpp_scenario(
    scenario: ScenarioSpec,
    size_bytes_list: list[int],
    niter: int,
    ops_arg: str | None,
    dtype,
    group: CommGroup,
    comm,
    rank: int,
    nranks_per_node: int,
    run_id: str,
    use_cuda_graph: bool,
    use_k_streams: bool,
    sched_window_size: int,
    proxy_debug: bool,
) -> list[dict]:
    rows = []
    ops_by_tenant = parse_ops(ops_arg, scenario.tenants, niter)
    max_ops = max(ops_by_tenant.values())

    for size_bytes in size_bytes_list:
        nelems = bytes_to_nelems(size_bytes, dtype)
        proxy_service = TenantAwareProxyService(
            mode=scenario.mode,
            scheduling_window_size=sched_window_size,
            debug=proxy_debug,
        )
        for tenant in scenario.tenants:
            register_tenant_on(
                proxy_service,
                tenant.tenant_id,
                tenant.qos,
                tenant.weight,
                int(tenant.bandwidth_cap_bps),
                0,
            )

        proxy_service.start_proxy()
        try:
            algos = {}
            for tenant in scenario.tenants:
                memory = GpuBuffer(nelems, dtype=dtype)
                memory_out = GpuBuffer(nelems, dtype=dtype)
                cp.cuda.runtime.deviceSynchronize()
                algo, assign_tenant = make_cpp_mt_algo(
                    group,
                    memory,
                    memory_out,
                    proxy_service,
                    nranks_per_node,
                )
                assign_tenant(algo, tenant.tenant_id)
                algos[tenant.tenant_id] = (algo, memory)
            comm.barrier()

            warm_stream = cp.cuda.Stream(non_blocking=True)
            for tenant in scenario.tenants:
                algos[tenant.tenant_id][0](warm_stream)
            warm_stream.synchronize()
            comm.barrier()

            if use_k_streams:
                streams = {
                    tenant.tenant_id: cp.cuda.Stream(non_blocking=True)
                    for tenant in scenario.tenants
                }
            else:
                streams = {scenario.tenants[0].tenant_id: cp.cuda.Stream(non_blocking=True)}

            primary = next(iter(streams.values()))
            graph = None
            graphs = {}
            if use_cuda_graph:
                if use_k_streams:
                    for tenant in scenario.tenants:
                        tid = tenant.tenant_id
                        stream = streams[tid]
                        with stream:
                            stream.begin_capture()
                            for _ in range(ops_by_tenant[tid]):
                                algos[tid][0](stream)
                            graphs[tid] = stream.end_capture()
                    for tenant in scenario.tenants:
                        tid = tenant.tenant_id
                        graphs[tid].launch(streams[tid])
                    for stream in streams.values():
                        stream.synchronize()
                else:
                    stream = primary
                    with stream:
                        stream.begin_capture()
                        for op_idx in range(max_ops):
                            for tenant in scenario.tenants:
                                tid = tenant.tenant_id
                                if op_idx < ops_by_tenant[tid]:
                                    algos[tid][0](stream)
                        graph = stream.end_capture()
                    graph.launch(stream)
                    stream.synchronize()
                comm.barrier()

            start_event = cp.cuda.Event()
            end_events = {tenant.tenant_id: cp.cuda.Event() for tenant in scenario.tenants}
            wall_start = time.time()
            start_event.record(primary)
            for stream in streams.values():
                if stream is not primary:
                    stream.wait_event(start_event)

            if use_cuda_graph:
                if use_k_streams:
                    for tenant in scenario.tenants:
                        tid = tenant.tenant_id
                        graphs[tid].launch(streams[tid])
                        end_events[tid].record(streams[tid])
                else:
                    graph.launch(primary)
                    for tenant in scenario.tenants:
                        end_events[tenant.tenant_id].record(primary)
            else:
                if use_k_streams:
                    for tenant in scenario.tenants:
                        tid = tenant.tenant_id
                        stream = streams[tid]
                        for _ in range(ops_by_tenant[tid]):
                            algos[tid][0](stream)
                        end_events[tid].record(stream)
                else:
                    stream = primary
                    for op_idx in range(max_ops):
                        for tenant in scenario.tenants:
                            tid = tenant.tenant_id
                            if op_idx < ops_by_tenant[tid]:
                                algos[tid][0](stream)
                                if op_idx + 1 == ops_by_tenant[tid]:
                                    end_events[tid].record(stream)

            for stream in streams.values():
                stream.synchronize()
            wall_s = max(time.time() - wall_start, 1e-9)
            comm.barrier()

            if rank == 0:
                counters = (
                    proxy_service.scheduler_debug_counters()
                    if hasattr(proxy_service, "scheduler_debug_counters")
                    else {}
                )
                backend = backend_label(scenario.mode, use_k_streams)
                actual_size_bytes = algos[scenario.tenants[0].tenant_id][1].nbytes
                for tenant in scenario.tenants:
                    tid = tenant.tenant_id
                    sched = counters.get(tid, {})
                    span_ms = cp.cuda.get_elapsed_time(start_event, end_events[tid])
                    ops = ops_by_tenant[tid]
                    time_us = span_ms * 1000.0 / ops
                    bytes_sent = actual_size_bytes * ops
                    row = {
                        "run_id": run_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "scenario": scenario.name,
                        "backend": backend,
                        "collective": "allreduce",
                        "tenant_id": tid,
                        "tenant_name": tenant.name,
                        "qos_class": tenant.qos.name,
                        "policy": scenario.policy,
                        "configured_weight": tenant.weight,
                        "size_bytes": actual_size_bytes,
                        "size_human": human_size(actual_size_bytes),
                        "niter": ops,
                        "time_us": round(time_us, 3),
                        "alg_bw_gbps": round(alg_bw_gbps(actual_size_bytes, time_us), 3),
                        "wallclock_gbps": round(bytes_sent / wall_s / 1e9, 3),
                        "wall_s": round(wall_s, 6),
                        "bytes_sent": bytes_sent,
                        "rank": rank,
                        "world_size": comm.size,
                        "host": socket.gethostname(),
                        "dtype": dtype_name(dtype),
                        "sched_dispatched_triggers": int(
                            sched.get("sched_dispatched_triggers", 0)
                        ),
                        "sched_dispatched_bytes": int(
                            sched.get("sched_dispatched_bytes", 0)
                        ),
                        "token_bucket_waits": int(sched.get("token_bucket_waits", 0)),
                        "drr_picks": int(sched.get("drr_picks", 0)),
                        "strict_priority_picks": int(
                            sched.get("strict_priority_picks", 0)
                        ),
                    }
                    rows.append(row)
                    cap = (
                        f" cap={tenant.bandwidth_cap_bps / 1e9:g}GB/s"
                        if tenant.bandwidth_cap_bps
                        else ""
                    )
                    print(
                        f"  [{scenario.name} {tenant.name} {tenant.qos.name}"
                        f" w={tenant.weight} ops={ops}{cap}] "
                        f"{row['size_human']:>8s} {time_us:9.2f} us "
                        f"{row['alg_bw_gbps']:7.3f} GB/s "
                        f"wall={row['wallclock_gbps']:7.3f} GB/s "
                        f"disp={row['sched_dispatched_triggers']} "
                        f"drr={row['drr_picks']} sp={row['strict_priority_picks']} "
                        f"wait={row['token_bucket_waits']}",
                        flush=True,
                    )
            comm.barrier()
        finally:
            proxy_service.stop_proxy()

        if rank == 0:
            print("", flush=True)

    return rows


def write_csv(path: str, rows: list[dict]) -> None:
    out = Path(path)
    if out.parent:
        out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    with out.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MT-MSCCL++ C++ tenant-aware proxy AllReduce benchmark."
    )
    parser.add_argument("--dtype", choices=["fp16", "fp32", "int32"], default="fp16")
    parser.add_argument(
        "--sizes-bytes",
        default=None,
        help="Comma-separated message sizes in bytes or human form, e.g. 12MiB,48MiB.",
    )
    parser.add_argument(
        "--sizes",
        default=None,
        help="Optional comma-separated element counts. Ignored when --sizes-bytes is set.",
    )
    parser.add_argument("--niter", type=int, default=50)
    parser.add_argument(
        "--ops",
        default=None,
        help=(
            "AllReduce ops per tenant. Use one integer for all tenants, or "
            "comma-separated values in tenant order, e.g. 100,50,20."
        ),
    )
    parser.add_argument(
        "--scenarios",
        default="fair,priority",
        help="Comma list: fair,priority,weighted,rate_limited,all.",
    )
    parser.add_argument(
        "--rate-cap-gbps",
        type=float,
        default=10.0,
        help="Rate cap used by the rate_limited scenario, in GB/s.",
    )
    parser.add_argument(
        "--sched-window-size",
        type=int,
        default=5,
        help="TenantAwareProxyService scheduling window size.",
    )
    parser.add_argument(
        "--no-cuda-graph",
        action="store_true",
        help="Launch kernels directly instead of capturing per-tenant CUDA graphs.",
    )
    parser.add_argument(
        "--single-stream",
        action="store_true",
        help="Use one shared CUDA stream instead of one stream per tenant.",
    )
    parser.add_argument(
        "--proxy-debug",
        action="store_true",
        help="Print C++ tenant-aware proxy trigger dispatch order and trigger fields.",
    )
    parser.add_argument(
        "--out",
        default="/mnt/nfs/ccl/results/1mt/mtccl_bench.csv",
        help="CSV output path.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.niter <= 0:
        raise ValueError("--niter must be positive")
    args.sched_window_size = max(1, int(args.sched_window_size))

    dtype = dtype_from_arg(args.dtype)
    if args.sizes_bytes is not None:
        size_bytes_list = parse_size_list_bytes(args.sizes_bytes)
    elif args.sizes is not None:
        itemsize = cp.dtype(dtype).itemsize
        size_bytes_list = [int(nelems) * itemsize for nelems in args.sizes.split(",")]
    else:
        size_bytes_list = parse_size_list_bytes("12MiB")

    scenario_specs = make_scenario_specs(args.rate_cap_gbps)
    scenarios = normalize_scenarios(args.scenarios, scenario_specs)
    use_k_streams = kstream_enabled(args.single_stream)
    use_cuda_graph = not args.no_cuda_graph

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.rank
    shm_comm = comm.Split_type(MPI.COMM_TYPE_SHARED, 0, MPI.INFO_NULL)
    local_rank = shm_comm.rank
    nranks_per_node = shm_comm.size
    shm_comm.Free()
    cp.cuda.Device(local_rank).use()

    iface, my_ip = get_net_iface()
    root_ip = comm.bcast(my_ip, root=0)
    group = CommGroup(
        interfaceIpPortTrio=f"{iface}:{root_ip}:50000",
        rank=rank,
        size=comm.size,
    )

    run_id = f"mtccl_{int(time.time())}_w{comm.size}"
    rows: list[dict] = []

    if rank == 0:
        print(
            "\n=== MT-MSCCL++ C++ proxy benchmark "
            f"world={comm.size} nranks_per_node={nranks_per_node} "
            f"dtype={args.dtype} niter={args.niter} "
            f"ops={args.ops or args.niter} "
            f"k_streams={int(use_k_streams)} cuda_graph={int(use_cuda_graph)} ===\n",
            flush=True,
        )
        print(f"Selected interface: {iface} ({my_ip}), root_ip={root_ip}", flush=True)
        print(
            "Sizes: " + ", ".join(human_size(size) for size in size_bytes_list),
            flush=True,
        )
        print(
            "Scenarios: " + ", ".join(scenario.name for scenario in scenarios),
            flush=True,
        )
        print("", flush=True)

    for scenario in scenarios:
        if rank == 0:
            print(f"--- scenario: {scenario.name} ({scenario.policy}) ---", flush=True)
        rows.extend(
            run_cpp_scenario(
                scenario=scenario,
                size_bytes_list=size_bytes_list,
                niter=args.niter,
                ops_arg=args.ops,
                dtype=dtype,
                group=group,
                comm=comm,
                rank=rank,
                nranks_per_node=nranks_per_node,
                run_id=run_id,
                use_cuda_graph=use_cuda_graph,
                use_k_streams=use_k_streams,
                sched_window_size=args.sched_window_size,
                proxy_debug=args.proxy_debug,
            )
        )

    if rank == 0:
        write_csv(args.out, rows)
        print(f"[rank 0] wrote {len(rows)} rows to {args.out}", flush=True)

    group = None
    return 0


if __name__ == "__main__":
    sys.exit(main())
