#!/usr/bin/env bash
#SBATCH --job-name=qex-quda-build
#SBATCH --partition=skx-dev
#SBATCH --account=TG-MCA93S002
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=build-sycl/quda-build-%j.log
# Compile the library on a CPU node; no NVIDIA allocation is needed.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source ./load-mpi-env.sh
export CUDA_PATH=${CUDA_PATH:-/home1/apps/nvidia/Linux_x86_64/24.5/cuda}
export QUDA_CUDA_MATH_ROOT=${QUDA_CUDA_MATH_ROOT:-/home1/apps/nvidia/Linux_x86_64/24.5/math_libs}
export QUDA_EIGEN_INCLUDE=${QUDA_EIGEN_INCLUDE:-/home1/apps/nvidia25/eigen/3.4.0/include/eigen3}
export SYCL_TARGET=${SYCL_TARGET:-h100} BUILD_JOBS=${BUILD_JOBS:-32}
LOGS="$PWD/deps-quda/build-records/${SLURM_JOB_ID}"
mkdir -p "$LOGS"
finish() {
  local status=$?
  trap - EXIT
  printf '%s\n' "$status" > "$LOGS/exit-status.txt"
  date -Is > "$LOGS/finished.txt"
  exit "$status"
}
trap finish EXIT
hostname > "$LOGS/host.txt"
date -Is > "$LOGS/started.txt"
/usr/bin/time -v -o "$LOGS/resource-usage.txt" ./scripts/build-quda.sh
