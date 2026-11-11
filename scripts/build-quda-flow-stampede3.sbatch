#!/usr/bin/env bash
#SBATCH --job-name=qex-quda-flow-build
#SBATCH --partition=skx-dev
#SBATCH --account=TG-MCA93S002
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=build-sycl/quda-flow-build-%j.log
# Submit after the matching QUDA library build has succeeded.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export CUDA_PATH=${CUDA_PATH:-/home1/apps/nvidia/Linux_x86_64/25.3/cuda}
export QEX_BUILD_ROOT=${QEX_BUILD_ROOT:-$PWD/build-sycl/llvm-v7.2.0-cuda12.8}
export BUILD_JOBS=${BUILD_JOBS:-16}
export QEX_QUDA_BUILD_RECORD=${QEX_QUDA_BUILD_RECORD:-$QEX_BUILD_ROOT/quda-build-records/${SLURM_JOB_ID}}
exec ./scripts/build-quda-flow.sh --run-cpu-tests
