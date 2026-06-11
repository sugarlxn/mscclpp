#!/usr/bin/env bash
set -euo pipefail

SIZES="${SIZES:-6144,98304,1572864,25165824,100663296,201326592}"
SCRIPT="${SCRIPT:-mscclpp/python/mscclpp_benchmark/mt_bench/test_2node_allreduce.py}"
LOG_DIR="${LOG_DIR:-/root/ccl/results}"

mkdir -p "${LOG_DIR}"

run_workload() {
  local workload_id="$1"
  local log_file="${LOG_DIR}/nccl_allreduce_workload_${workload_id}_mlx5_0.log"

  echo "[workload ${workload_id}] writing ${log_file}"
  mpirun --allow-run-as-root \
    -np 4 -H 172.16.8.36:2,172.16.8.38:2 \
    --mca plm_rsh_args "-o StrictHostKeyChecking=no" \
    --mca pml ob1 --mca btl tcp,self \
    --mca btl_tcp_if_include 172.16.8.0/22 \
    --mca oob_tcp_if_include 172.16.8.0/22 \
    -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH \
    -x NCCL_IB_DISABLE=0 \
    -x NCCL_IB_HCA=mlx5_0 \
    -x NCCL_DEBUG=INFO \
    -x NCCL_DEBUG_SUBSYS=INIT,NET \
    "${SCRIPT}" --sizes "${SIZES}" \
    >"${log_file}" 2>&1
}

run_workload 1 &
pid1="$!"

run_workload 2 &
pid2="$!"

wait "${pid1}"
status1="$?"

wait "${pid2}"
status2="$?"

echo "[workload 1] exit ${status1}"
echo "[workload 2] exit ${status2}"

if [[ "${status1}" -ne 0 || "${status2}" -ne 0 ]]; then
  exit 1
fi
