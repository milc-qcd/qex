#!/usr/bin/env bash
#SBATCH --job-name=qex-h100-check
#SBATCH --partition=h100
#SBATCH --account=TG-MCA93S002
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --output=build-sycl/h100-validation-%j.log

# Submit from the qex-gpu-sycl repository root. Numeric binaries must be
# prebuilt; the validation script compiles only its small device preflight.
set -euo pipefail
ROOT=${QEX_SYCL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
if [[ ! -f "$ROOT/scripts/check-hisq-sycl.sh" ]]; then
  echo "Submit from the qex-gpu-sycl root, or set QEX_SYCL_ROOT to it." >&2
  exit 2
fi
cd "$ROOT"
source ./load-mpi-env.sh
# Use the same compiler/runtime and rebuilt MPI dependencies as the numeric
# binaries. A site override must point at the matching toolchain environment.
source "${H100_SYCL_ENV:-$ROOT/scripts/load-nvidia-sycl-env.sh}"
export SYCL_TARGET=h100 SKIP_BUILD=1
export OMP_NUM_THREADS=2 I_MPI_FABRICS=shm I_MPI_HYDRA_BOOTSTRAP=fork
export QEX_VALIDATION_LOGS="$QEX_BUILD_ROOT/validation-h100/job_${SLURM_JOB_ID}"
mkdir -p "$QEX_VALIDATION_LOGS"
finish() {
  local status=$?
  trap - EXIT
  printf '%s\n' "$status" > "$QEX_VALIDATION_LOGS/exit-status.txt"
  date -Is > "$QEX_VALIDATION_LOGS/finished.txt"
  exit "$status"
}
trap finish EXIT
hostname > "$QEX_VALIDATION_LOGS/host.txt"
"$SYCL_CXX" --version
sha256sum "$QEX_BUILD_ROOT/SYCL-h100/"{check_sycl_smoke,check_hisq_gpu,check_gaugeflow_gpu,check_device_halo,check_gpu_mem,hisqflow} > "$QEX_VALIDATION_LOGS/binaries.sha256"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee "$QEX_VALIDATION_LOGS/nvidia-smi.txt"
ONEAPI_DEVICE_SELECTOR=cuda:gpu sycl-ls | tee "$QEX_VALIDATION_LOGS/devices.txt"
./scripts/check-hisq-sycl.sh
