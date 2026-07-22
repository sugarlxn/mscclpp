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

from mscclpp import CommGroup, GpuBuffer, ProxyService  # noqa: E402
from mscclpp.ext.tenant import (  # noqa: E402
    PolicyMode,
    QoSClass,
    TenantAwareProxyService,
    register_tenant_on,
)


CSV_FIELDS = [
    "run_id",
    "trial",
    "timestamp",
    "scenario",
    "backend",
    "collective",
    "tenant_id",
    "tenant_name",
    "qos_class",
    "policy",
    "scheduler_variant",
    "small_threshold_bytes",
    "size_class",
    "configured_weight",
    "size_bytes",
    "size_human",
    "niter",
    "time_us",
    "job_time_us",
    "job_time_ms",
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
    "size_aware_bypass_triggers",
    "size_aware_bypass_bytes",
    "token_bucket_waits",
    "drr_picks",
    "strict_priority_picks",
    "scheduler_wait_samples",
    "scheduler_wait_avg_ns",
    "scheduler_wait_p50_ns",
    "scheduler_wait_p99_ns",
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


def make_cpp_mt_algo(group, memory, memory_out, proxy_service, nranks_per_node: int, algorithm: str = "auto"):
    if group.nranks == nranks_per_node:
        return MscclppAllReduce3(group, memory, proxy_service), assign_tenant_to_allreduce3

    if algorithm == "ar5" or (algorithm == "auto" and memory.nbytes <= (1 << 22)):
        return (
            MscclppAllReduce5(group, memory, memory_out, nranks_per_node, proxy_service),
            assign_tenant_to_allreduce5,
        )

    if algorithm not in {"auto", "ar4"}:
        raise ValueError(f"unsupported cross-node algorithm {algorithm!r}")

    validate_xnode_ar4_size(memory.size, memory.dtype, group.nranks, nranks_per_node)
    return (
        MscclppAllReduce4(group, memory, nranks_per_node, proxy_service),
        assign_tenant_to_allreduce4,
    )


def make_scenario_specs(rate_cap_gbps: float) -> dict[str, ScenarioSpec]:
    cap_bps = int(rate_cap_gbps * 1e9)
    inference = "inference"
    training_a = "trainingA"
    training_b = "trainingB"
    tenant_a = "tenantA"
    tenant_b = "tenantB"
    tenant_c = "tenantC"
    return {
        "single": ScenarioSpec(
            name="single_tenant",
            policy="single_tenant",
            mode=PolicyMode.STRICT_PRIORITY,
            tenants=(TenantSpec(1, "tenantA", QoSClass.STANDARD, 1),),
        ),
        "infer_priority": ScenarioSpec(
            name="mtccl_infer_priority_3tenant",
            policy="strict_priority",
            mode=PolicyMode.STRICT_PRIORITY,
            tenants=(
                TenantSpec(1, training_a, QoSClass.BEST_EFFORT, 1),
                TenantSpec(2, training_b, QoSClass.BEST_EFFORT, 1),
                TenantSpec(3, inference, QoSClass.REALTIME, 1),
            ),
        ),
        "equal_priority": ScenarioSpec(
            name="mtccl_equal_priority_3tenant",
            policy="strict_priority_equal_qos",
            mode=PolicyMode.STRICT_PRIORITY,
            tenants=(
                TenantSpec(1, training_a, QoSClass.STANDARD, 1),
                TenantSpec(2, training_b, QoSClass.STANDARD, 1),
                TenantSpec(3, inference, QoSClass.STANDARD, 1),
            ),
        ),
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
        "mtccl_infer_priority_3tenant": "infer_priority",
        "mtccl_equal_priority_3tenant": "equal_priority",
        "inference_priority": "infer_priority",
        "same_priority": "equal_priority",
        "mtccl_fair_3tenant": "fair",
        "mtccl_priority_3tenant": "priority",
        "mtccl_weighted_3tenant": "weighted",
        "mtccl_rate_limited_3tenant": "rate_limited",
        "cpp_mt_fair": "fair",
        "cpp_mt_priority": "priority",
        "cpp_mt_weighted": "weighted",
        "cpp_mt_rate_limited": "rate_limited",
        "single_tenant": "single",
    }
    requested = [part.strip() for part in raw.split(",") if part.strip()]
    if not requested:
        raise ValueError("--scenarios cannot be empty")
    if "all" in requested:
        requested = ["single", "infer_priority", "equal_priority", "fair", "priority", "weighted", "rate_limited"]

    out = []
    seen = set()
    for name in requested:
        key = aliases.get(name, name)
        if key not in specs:
            valid = ",".join([
                "infer_priority",
                "equal_priority",
                "fair",
                "priority",
                "weighted",
                "rate_limited",
                "all",
                "single",
            ])
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


def make_size_profiles(
    sizes: list[int],
    tenant_sizes: list[int] | None,
    tenants: tuple[TenantSpec, ...],
) -> list[dict[int, int]]:
    if tenant_sizes is not None:
        if len(tenant_sizes) == 1:
            tenant_sizes = tenant_sizes * len(tenants)
        elif len(tenant_sizes) != len(tenants):
            raise ValueError(
                f"--tenant-sizes-bytes expects one value or {len(tenants)} "
                f"comma-separated values for {[tenant.name for tenant in tenants]}"
            )
        return [{tenant.tenant_id: size for tenant, size in zip(tenants, tenant_sizes)}]

    return [
        {tenant.tenant_id: size for tenant in tenants}
        for size in sizes
    ]


def describe_size_profile(profile: dict[int, int], tenants: tuple[TenantSpec, ...]) -> str:
    return ", ".join(
        f"{tenant.name}={human_size(profile[tenant.tenant_id])}"
        for tenant in tenants
    )


def kstream_enabled(single_stream: bool) -> bool:
    if single_stream:
        os.environ["MTCCL_K_STREAMS"] = "0"
        return False
    os.environ["MTCCL_K_STREAMS"] = "1"
    return True


def backend_label(variant: str, mode: PolicyMode, use_k_streams: bool) -> str:
    suffix = "kstream" if use_k_streams else "single_stream_smoke"
    if variant == "native":
        return f"mscclpp_native_{suffix}"
    if variant == "fifo":
        return f"mscclpp_size_aware_fifo_{suffix}"
    if mode == PolicyMode.STRICT_PRIORITY:
        return f"tapcs_size_aware_priority_{suffix}"
    if mode == PolicyMode.HYBRID:
        return f"mscclpp_cpp_mt_hybrid_{suffix}"
    return f"mscclpp_cpp_mt_fair_{suffix}"


def run_cpp_scenario(
    scenario: ScenarioSpec,
    size_profiles: list[dict[int, int]],
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
    inference_delay_ms: float,
    variant: str,
    small_threshold_bytes: int,
    aging_ns: int,
    algorithm: str,
    trial: int,
) -> list[dict]:
    rows = []
    ops_by_tenant = parse_ops(ops_arg, scenario.tenants, niter)
    max_ops = max(ops_by_tenant.values())
    inference_delay_s = max(0.0, inference_delay_ms) / 1000.0
    launch_tenants = tuple(
        tenant for tenant in scenario.tenants if tenant.name != "inference"
    ) + tuple(tenant for tenant in scenario.tenants if tenant.name == "inference")

    for size_profile in size_profiles:
        if variant == "native":
            proxy_service = ProxyService()
        else:
            mode = PolicyMode.FIFO if variant == "fifo" else scenario.mode
            proxy_service = TenantAwareProxyService(
                mode=mode,
                scheduling_window_size=sched_window_size,
                debug=proxy_debug,
                small_collective_threshold_bytes=small_threshold_bytes,
                aging_ns=aging_ns,
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
                proxy_service.set_tenant_collective_bytes(
                    tenant.tenant_id, size_profile[tenant.tenant_id]
                )

        proxy_service.start_proxy()
        try:
            algos = {}
            for tenant in scenario.tenants:
                size_bytes = size_profile[tenant.tenant_id]
                nelems = bytes_to_nelems(size_bytes, dtype)
                memory = GpuBuffer(nelems, dtype=dtype)
                memory_out = GpuBuffer(nelems, dtype=dtype)
                cp.cuda.runtime.deviceSynchronize()
                algo, assign_tenant = make_cpp_mt_algo(
                    group,
                    memory,
                    memory_out,
                    proxy_service,
                    nranks_per_node,
                    algorithm,
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
                    for tenant in launch_tenants:
                        tid = tenant.tenant_id
                        if tenant.name == "inference" and inference_delay_s > 0:
                            time.sleep(inference_delay_s)
                        graphs[tid].launch(streams[tid])
                        end_events[tid].record(streams[tid])
                else:
                    graph.launch(primary)
                    for tenant in scenario.tenants:
                        end_events[tenant.tenant_id].record(primary)
            else:
                if use_k_streams:
                    for tenant in launch_tenants:
                        tid = tenant.tenant_id
                        if tenant.name == "inference" and inference_delay_s > 0:
                            time.sleep(inference_delay_s)
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
                backend = backend_label(variant, scenario.mode, use_k_streams)
                for tenant in scenario.tenants:
                    tid = tenant.tenant_id
                    sched = counters.get(tid, {})
                    span_ms = cp.cuda.get_elapsed_time(start_event, end_events[tid])
                    ops = ops_by_tenant[tid]
                    job_time_us = span_ms * 1000.0
                    time_us = job_time_us / ops
                    actual_size_bytes = algos[tid][1].nbytes
                    bytes_sent = actual_size_bytes * ops
                    row = {
                        "run_id": run_id,
                        "trial": trial,
                        "timestamp": datetime.utcnow().isoformat(),
                        "scenario": scenario.name,
                        "backend": backend,
                        "collective": "allreduce",
                        "tenant_id": tid,
                        "tenant_name": tenant.name,
                        "qos_class": tenant.qos.name,
                        "policy": scenario.policy,
                        "scheduler_variant": variant,
                        "small_threshold_bytes": small_threshold_bytes,
                        "size_class": (
                            "small"
                            if actual_size_bytes <= small_threshold_bytes
                            else "large"
                        ),
                        "configured_weight": tenant.weight,
                        "size_bytes": actual_size_bytes,
                        "size_human": human_size(actual_size_bytes),
                        "niter": ops,
                        "time_us": round(time_us, 3),
                        "job_time_us": round(job_time_us, 3),
                        "job_time_ms": round(span_ms, 3),
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
                        "size_aware_bypass_triggers": int(
                            sched.get("size_aware_bypass_triggers", 0)
                        ),
                        "size_aware_bypass_bytes": int(
                            sched.get("size_aware_bypass_bytes", 0)
                        ),
                        "token_bucket_waits": int(sched.get("token_bucket_waits", 0)),
                        "drr_picks": int(sched.get("drr_picks", 0)),
                        "strict_priority_picks": int(
                            sched.get("strict_priority_picks", 0)
                        ),
                        "scheduler_wait_samples": int(
                            sched.get("scheduler_wait_samples", 0)
                        ),
                        "scheduler_wait_avg_ns": int(
                            sched.get("scheduler_wait_avg_ns", 0)
                        ),
                        "scheduler_wait_p50_ns": int(
                            sched.get("scheduler_wait_p50_ns", 0)
                        ),
                        "scheduler_wait_p99_ns": int(
                            sched.get("scheduler_wait_p99_ns", 0)
                        ),
                    }
                    rows.append(row)
                    cap = (
                        f" cap={tenant.bandwidth_cap_bps / 1e9:g}GB/s"
                        if tenant.bandwidth_cap_bps
                        else ""
                    )
                    print(
                        f"  [{variant} {scenario.name} {tenant.name} {tenant.qos.name}"
                        f" w={tenant.weight} ops={ops}{cap}] "
                        f"{row['size_human']:>8s} {time_us:9.2f} us/op "
                        f"job={job_time_us / 1000.0:9.3f} ms "
                        f"{row['alg_bw_gbps']:7.3f} GB/s "
                        f"wall={row['wallclock_gbps']:7.3f} GB/s "
                        f"disp={row['sched_dispatched_triggers']} "
                        f"drr={row['drr_picks']} sp={row['strict_priority_picks']} "
                        f"wait={row['token_bucket_waits']} "
                        f"qavg={row['scheduler_wait_avg_ns'] / 1000.0:.1f}us "
                        f"q50={row['scheduler_wait_p50_ns'] / 1000.0:.1f}us "
                        f"q99={row['scheduler_wait_p99_ns'] / 1000.0:.1f}us",
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
    existing_rows = []
    rewrite_existing = False
    if out.exists():
        with out.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and reader.fieldnames != CSV_FIELDS:
                rewrite_existing = True
                existing_rows = list(reader)

    if rewrite_existing:
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in existing_rows:
                writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})

    write_header = not out.exists() or out.stat().st_size == 0
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
    parser.add_argument(
        "--tenant-sizes-bytes",
        default=None,
        help=(
            "Optional per-tenant message sizes in scenario tenant order. "
            "For infer/equal scenarios the order is trainingA,trainingB,inference; "
            "example: 12MiB,12MiB,256KiB."
        ),
    )
    parser.add_argument("--niter", type=int, default=50)
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Independent contention trials; use >=10 for P50/P95/P99 analysis.",
    )
    parser.add_argument(
        "--algorithm",
        choices=["auto", "ar4", "ar5"],
        default="auto",
        help="Cross-node AllReduce implementation. Use ar5 for the exact threshold sweep.",
    )
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
        default="infer_priority,equal_priority",
        help=(
            "Comma list: single,infer_priority,equal_priority,fair,priority,"
            "weighted,rate_limited,all."
        ),
    )
    parser.add_argument(
        "--variants",
        default="native,fifo,tapcs",
        help="Comparison backends: native,size-aware FIFO baseline,TAPCS.",
    )
    parser.add_argument(
        "--small-threshold-bytes",
        default="4MiB",
        help="Fixed TAPCS small/large boundary (default: 4MiB).",
    )
    parser.add_argument(
        "--aging-ms",
        type=float,
        default=100.0,
        help="Priority aging interval in milliseconds.",
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
        "--inference-delay-ms",
        type=float,
        default=0.0,
        help=(
            "Delay inference tenant launch by this many milliseconds after "
            "training tenants launch in k-stream mode."
        ),
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
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    args.sched_window_size = max(1, int(args.sched_window_size))

    dtype = dtype_from_arg(args.dtype)
    if args.sizes_bytes is not None:
        size_bytes_list = parse_size_list_bytes(args.sizes_bytes)
    elif args.sizes is not None:
        itemsize = cp.dtype(dtype).itemsize
        size_bytes_list = [int(nelems) * itemsize for nelems in args.sizes.split(",")]
    else:
        size_bytes_list = parse_size_list_bytes("12MiB")
    tenant_size_bytes = (
        parse_size_list_bytes(args.tenant_sizes_bytes)
        if args.tenant_sizes_bytes is not None
        else None
    )

    scenario_specs = make_scenario_specs(args.rate_cap_gbps)
    scenarios = normalize_scenarios(args.scenarios, scenario_specs)
    variants = [value.strip().lower() for value in args.variants.split(",") if value.strip()]
    aliases = {"priority": "tapcs", "proposed": "tapcs", "size_fifo": "fifo"}
    variants = [aliases.get(value, value) for value in variants]
    invalid_variants = sorted(set(variants) - {"native", "fifo", "tapcs"})
    if invalid_variants:
        raise ValueError(f"unknown --variants values: {invalid_variants}")
    small_threshold_bytes = parse_size_bytes(args.small_threshold_bytes)
    aging_ns = max(1, int(args.aging_ms * 1_000_000))
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
            f"k_streams={int(use_k_streams)} cuda_graph={int(use_cuda_graph)} "
            f"inference_delay_ms={args.inference_delay_ms:g} ===\n",
            flush=True,
        )
        print(f"Selected interface: {iface} ({my_ip}), root_ip={root_ip}", flush=True)
        print(
            "Sizes: " + (
                ", ".join(human_size(size) for size in size_bytes_list)
                if tenant_size_bytes is None
                else "per-tenant " + ", ".join(human_size(size) for size in tenant_size_bytes)
            ),
            flush=True,
        )
        print(
            "Scenarios: " + ", ".join(scenario.name for scenario in scenarios),
            flush=True,
        )
        print(
            f"Variants: {','.join(variants)}; small/large boundary="
            f"{human_size(small_threshold_bytes)}; aging={args.aging_ms:g}ms",
            flush=True,
        )
        print("", flush=True)

    for scenario in scenarios:
        size_profiles = make_size_profiles(
            size_bytes_list,
            tenant_size_bytes,
            scenario.tenants,
        )
        if rank == 0:
            print(f"--- scenario: {scenario.name} ({scenario.policy}) ---", flush=True)
            for profile in size_profiles:
                print(
                    "    sizes: " + describe_size_profile(profile, scenario.tenants),
                    flush=True,
                )
        for variant in variants:
            for trial in range(args.trials):
                rows.extend(
                    run_cpp_scenario(
                    scenario=scenario,
                    size_profiles=size_profiles,
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
                    inference_delay_ms=args.inference_delay_ms,
                    variant=variant,
                    small_threshold_bytes=small_threshold_bytes,
                    aging_ns=aging_ns,
                    algorithm=args.algorithm,
                    trial=trial,
                    )
                )

    if rank == 0:
        write_csv(args.out, rows)
        print(f"[rank 0] wrote {len(rows)} rows to {args.out}", flush=True)

    group = None
    return 0


if __name__ == "__main__":
    sys.exit(main())
