#!/usr/bin/env bash
#SBATCH --job-name=qex-h100-check
#SBATCH --partition=h100
#SBATCH --account=TG-MCA93S002
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --output=qex-h100-check-%j.log

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
# Optional site setup for a CUDA-enabled runtime matching the compiled
# oneAPI 2024 binaries. Keep its MPI ABI compatible with deps-mpi.
if [[ -n "${H100_SYCL_ENV:-}" ]]; then
  source "$H100_SYCL_ENV"
fi
export SYCL_TARGET=h100 SKIP_BUILD=1
export OMP_NUM_THREADS=2 I_MPI_FABRICS=shm I_MPI_HYDRA_BOOTSTRAP=fork
hostname
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
./scripts/check-hisq-sycl.sh
