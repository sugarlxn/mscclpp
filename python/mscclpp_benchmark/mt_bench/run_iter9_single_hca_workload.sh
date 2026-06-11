#!/usr/bin/env bash
set -euo pipefail

export MTCCL_K_STREAMS=1

mpirun --allow-run-as-root \
  -np 4 -H 172.16.8.36:2,172.16.8.38:2 \
  --mca plm_rsh_args "-o StrictHostKeyChecking=no" \
  --mca pml ob1 --mca btl tcp,self \
  --mca btl_tcp_if_include 172.16.8.0/22 \
  --mca oob_tcp_if_include 172.16.8.0/22 \
  -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x MTCCL_K_STREAMS \
  -x NCCL_IB_DISABLE=1 \
  -x MSCCLPP_HCA_DEVICES=mlx5_0,mlx5_0 \
  python -m mscclpp_benchmark.mt_bench.run_bench \
    --dtype fp16 \
    --sizes 6144,98304,1572864,25165824,100663296,201326592,402653184 \
    --niter 200 \
    --rate-cap-gbps 10 \
    --scenarios cpp_mt_fair,cpp_mt_weighted,cpp_mt_priority,cpp_mt_rate_limited \
    --out /root/ccl/results/mt_bench_2node_iter9_single_hca_mlx5_0_fp16_TO314M.csv
