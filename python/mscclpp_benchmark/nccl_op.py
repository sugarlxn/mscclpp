# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import cupy.cuda.nccl as nccl
from mpi4py import MPI
import cupy as cp


def _get_nccl_dtype(dtype):
    dtype = cp.dtype(dtype)
    if dtype == cp.dtype(cp.float32):
        return nccl.NCCL_FLOAT32
    if dtype == cp.dtype(cp.float16):
        return nccl.NCCL_FLOAT16
    if dtype == cp.dtype(cp.int32):
        return nccl.NCCL_INT32
    raise RuntimeError(f"Make sure that the data type {dtype} is mapped to the correct NCCL data type")


class NcclAllReduce:
    def __init__(self, nccl_comm: nccl.NcclCommunicator, memory: cp.ndarray):
        self.nccl_comm = nccl_comm
        self.memory = memory
        self.nccl_dtype = _get_nccl_dtype(memory.dtype)

    def __call__(self, stream):
        stream_ptr = stream.ptr if stream else 0
        self.nccl_comm.allReduce(
            self.memory.data.ptr, self.memory.data.ptr, self.memory.size, self.nccl_dtype, nccl.NCCL_SUM, stream_ptr
        )
        return self.memory


class NcclAllGather:
    def __init__(
        self,
        nccl_comm: nccl.NcclCommunicator,
        memory: cp.ndarray,
        memory_out: cp.ndarray = None,
        nranks: int = None,
    ):
        self.nccl_comm = nccl_comm
        self.memory = memory
        self.nccl_dtype = _get_nccl_dtype(memory.dtype)
        self.nranks = MPI.COMM_WORLD.size if nranks is None else nranks
        if self.nranks <= 0:
            raise RuntimeError("NCCL allgather nranks must be positive")

        output_size = self.memory.size * self.nranks
        if memory_out is None:
            memory_out = cp.empty(output_size, dtype=self.memory.dtype)
        if memory_out.dtype != self.memory.dtype:
            raise RuntimeError("NCCL allgather output buffer dtype must match input buffer dtype")
        if memory_out.size < output_size:
            raise RuntimeError(
                f"NCCL allgather output buffer must have at least {output_size} elements, got {memory_out.size}"
            )
        self.memory_out = memory_out

    def __call__(self, stream):
        stream_ptr = stream.ptr if stream else 0
        self.nccl_comm.allGather(
            self.memory.data.ptr, self.memory_out.data.ptr, self.memory.size, self.nccl_dtype, stream_ptr
        )
        return self.memory_out
