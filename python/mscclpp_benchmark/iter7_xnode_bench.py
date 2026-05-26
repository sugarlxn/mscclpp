"""iter7_xnode_bench: 2-node MSCCL++ AR4/AR5 vs NCCL baseline.

Skips allreduce_bench's autotune (which crashes with
`cudaErrorMisalignedAddress` on some AR5 cross-node configs); uses each
algo's default block / pipeline params.
"""
import os, sys, time, ctypes
import cupy as cp
import netifaces as ni
from mpi4py import MPI
from prettytable import PrettyTable
import cupy.cuda.nccl as nccl

import ipaddress
from mscclpp import CommGroup, ProxyService, GpuBuffer
from mscclpp_op import MscclppAllReduce4, MscclppAllReduce5
from nccl_op import NcclAllReduce


def routable(ip):
    o = ipaddress.ip_address(ip)
    return not (o.is_loopback or o.is_link_local or o.is_multicast)


def pick_iface():
    for iface in ni.interfaces():
        addrs = ni.ifaddresses(iface).get(ni.AF_INET, [])
        for a in addrs:
            if routable(a["addr"]):
                return iface, a["addr"]
    return None, None


def bench_us(niter, func):
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        for _ in range(niter):
            func(stream)
        graph = stream.end_capture()
    graph.launch(stream)
    stream.synchronize()
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record(stream)
    graph.launch(stream)
    end.record(stream)
    end.synchronize()
    return cp.cuda.get_elapsed_time(start, end) * 1000.0 / niter  # us


def main():
    shm_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED, 0, MPI.INFO_NULL)
    n_per_node = shm_comm.size
    shm_comm.Free()
    rank = MPI.COMM_WORLD.rank
    world = MPI.COMM_WORLD.size
    cp.cuda.Device(rank % n_per_node).use()

    iface, my_ip = pick_iface()
    root_ip = MPI.COMM_WORLD.bcast(my_ip, root=0)
    group = CommGroup(interfaceIpPortTrio=f"{iface}:{root_ip}:50000",
                      rank=rank, size=world)

    uid = nccl.get_unique_id() if rank == 0 else None
    uid = MPI.COMM_WORLD.bcast(uid, root=0)
    nccl_comm = nccl.NcclCommunicator(world, uid, rank)

    if rank == 0:
        print(f"=== iter7_xnode (world={world} n_per_node={n_per_node}) ===")
        tb = PrettyTable()
        tb.field_names = ["Size", "Algo", "MSCCL++ µs", "MSCCL++ GB/s",
                          "NCCL µs", "NCCL GB/s", "speed-up"]

    # AR4-only.  Use `3 * 2^N` element counts to match allreduce_bench's
    # original sweep (AR4 kernel chunks require nelems divisible by
    # nranks * pipeline_depth; the 3-factor keeps things clean).  Bytes shown
    # at FP16 (×2).
    nelems_list = [
        (3 * 2**21, "AR4_12MiB"),    # 12 MiB
        (3 * 2**23, "AR4_48MiB"),    # 48 MiB
        (3 * 2**25, "AR4_192MiB"),   # 192 MiB
        (3 * 2**26, "AR4_384MiB"),   # 384 MiB
    ]
    niter = 20
    dtype = cp.float16

    for nelems, tag in nelems_list:
        nbytes = nelems * dtype().itemsize
        memory = GpuBuffer(nelems, dtype=dtype)
        memory_out = GpuBuffer(nelems, dtype=dtype)
        cp.cuda.runtime.deviceSynchronize()
        proxy = ProxyService()

        algo = MscclppAllReduce4(group, memory, n_per_node, proxy)

        proxy.start_proxy()
        MPI.COMM_WORLD.barrier()
        ms_us = bench_us(niter, algo)
        ms_bw = nbytes / ms_us / 1e3   # GB/s

        nccl_call = NcclAllReduce(nccl_comm, memory)
        nc_us = bench_us(niter, nccl_call)
        nc_bw = nbytes / nc_us / 1e3

        MPI.COMM_WORLD.barrier()
        proxy.stop_proxy()
        if rank == 0:
            tb.add_row([
                f"{nbytes//1024} KiB" if nbytes < 1<<20 else f"{nbytes//(1<<20)} MiB",
                tag,
                f"{ms_us:.2f}", f"{ms_bw:.2f}",
                f"{nc_us:.2f}", f"{nc_bw:.2f}",
                f"{nc_us/ms_us:.2f}x",
            ])

    if rank == 0:
        print(tb)
        outdir = "/root/ccl/results"
        os.makedirs(outdir, exist_ok=True)
        with open(f"{outdir}/iter7_xnode_2rank.txt", "w") as f:
            f.write(str(tb) + "\n")


if __name__ == "__main__":
    main()
