#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# NCCL allreduce micro-benchmark for the 2-node x 2-rank case.
# Intended usage:
#   mpirun -n 4 python test_2node_allreduce.py --sizes 1024,2048
#
# To study contention on one NIC, launch two copies at the same time. By
# default the script constrains NCCL to mlx5_0 via NCCL_IB_HCA.

import argparse
import os
import socket
import sys
from datetime import datetime

cp = None
nccl = None
MPI = None
NcclAllReduce = None


def load_runtime_modules():
    global cp, nccl, MPI, NcclAllReduce
    import cupy as _cp
    import cupy.cuda.nccl as _nccl
    from mpi4py import MPI as _MPI

    # Make mscclpp_benchmark/nccl_op.py importable when this file is run directly.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from nccl_op import NcclAllReduce as _NcclAllReduce

    cp = _cp
    nccl = _nccl
    MPI = _MPI
    NcclAllReduce = _NcclAllReduce


def human_size(nbytes):
    value = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024.0
    return f"{value:.1f}PiB"


def parse_sizes(sizes):
    out = []
    for item in sizes.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    if not out:
        raise ValueError("--sizes must contain at least one integer")
    return out


def dtype_from_arg(name):
    if name == "fp16":
        return cp.float16
    if name == "fp32":
        return cp.float32
    if name == "int32":
        return cp.int32
    raise ValueError(name)


def alg_bw_gbps(nbytes, time_us):
    if time_us <= 0:
        return 0.0
    return nbytes / (time_us * 1e3)


def bench_time_us(niter, op, use_graph=True):
    stream = cp.cuda.Stream(non_blocking=True)

    if use_graph:
        with stream:
            stream.begin_capture()
            for _ in range(niter):
                op(stream)
            graph = stream.end_capture()

        graph.launch(stream)
        stream.synchronize()

        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record(stream)
        graph.launch(stream)
        end.record(stream)
        end.synchronize()
        return cp.cuda.get_elapsed_time(start, end) * 1000.0 / niter

    op(stream)
    stream.synchronize()
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record(stream)
    for _ in range(niter):
        op(stream)
    end.record(stream)
    end.synchronize()
    return cp.cuda.get_elapsed_time(start, end) * 1000.0 / niter


def check_once(buf, op, comm):
    rank = comm.rank
    world = comm.size
    buf.fill(rank + 1)
    cp.cuda.runtime.deviceSynchronize()
    stream = cp.cuda.Stream(non_blocking=True)
    op(stream)
    stream.synchronize()
    expected = world * (world + 1) // 2
    ok = bool(cp.allclose(buf, expected, rtol=1e-2, atol=1e-2))
    return comm.allreduce(ok, op=MPI.LAND)


def main():
    parser = argparse.ArgumentParser(
        description="2-node x 2-rank NCCL allreduce bandwidth test")
    parser.add_argument("--sizes", type=str, default="1024,2048",
                        help="comma-separated element counts, e.g. 1024,2048")
    parser.add_argument("--niter", type=int, default=100,
                        help="timed allreduce iterations per size")
    parser.add_argument("--dtype", type=str, default="fp32",
                        choices=["fp16", "fp32", "int32"])
    parser.add_argument("--hca", type=str, default="mlx5_0",
                        help="NCCL_IB_HCA value; use empty string to leave unset")
    parser.add_argument("--no-graph", action="store_true",
                        help="disable CUDA graph timing")
    parser.add_argument("--check", action="store_true",
                        help="run one correctness check per size before timing")
    parser.add_argument("--allow-non-2x2", action="store_true",
                        help="do not require world_size=4 and local ranks=2")
    args = parser.parse_args()

    if args.hca:
        os.environ["NCCL_IB_HCA"] = args.hca

    load_runtime_modules()

    comm = MPI.COMM_WORLD
    rank = comm.rank
    world = comm.size

    shm_comm = comm.Split_type(MPI.COMM_TYPE_SHARED, 0, MPI.INFO_NULL)
    local_rank = shm_comm.rank
    local_size = shm_comm.size
    shm_comm.Free()

    if not args.allow_non_2x2 and (world != 4 or local_size != 2):
        if rank == 0:
            print("Expected 2 nodes x 2 ranks: world_size=4 and local_size=2. "
                  "Pass --allow-non-2x2 to override.", flush=True)
        return 2

    cp.cuda.Device(local_rank).use()

    if rank == 0:
        uid = nccl.get_unique_id()
    else:
        uid = None
    uid = comm.bcast(uid, root=0)
    nccl_comm = nccl.NcclCommunicator(world, uid, rank)

    dtype = dtype_from_arg(args.dtype)
    sizes = parse_sizes(args.sizes)
    use_graph = not args.no_graph

    if rank == 0:
        hca = os.environ.get("NCCL_IB_HCA", "<unset>")
        print(f"# test_2node_allreduce {datetime.utcnow().isoformat()}Z", flush=True)
        print(f"# world_size={world} local_size={local_size} dtype={args.dtype} "
              f"niter={args.niter} graph={use_graph} NCCL_IB_HCA={hca}",
              flush=True)
        print("size_elems,size_bytes,size_human,time_us,alg_bw_gbps",
              flush=True)

    for nelems in sizes:
        buf = cp.empty(nelems, dtype=dtype)
        buf.fill(rank + 1)
        cp.cuda.runtime.deviceSynchronize()
        op = NcclAllReduce(nccl_comm, buf)

        if args.check:
            ok = check_once(buf, op, comm)
            if rank == 0 and not ok:
                print(f"correctness check failed for nelems={nelems}",
                      file=sys.stderr, flush=True)
                return 3
            buf.fill(rank + 1)
            cp.cuda.runtime.deviceSynchronize()

        comm.barrier()
        time_us = bench_time_us(args.niter, op, use_graph=use_graph)
        comm.barrier()

        nbytes = buf.nbytes
        bw = alg_bw_gbps(nbytes, time_us)
        if rank == 0:
            print(f"{nelems},{nbytes},{human_size(nbytes)},"
                  f"{time_us:.3f},{bw:.3f}", flush=True)

    comm.barrier()
    if rank == 0:
        print(f"# done host={socket.gethostname()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
