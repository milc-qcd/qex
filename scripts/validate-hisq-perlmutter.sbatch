#!/usr/bin/env bash
#SBATCH --job-name=qex-a100-check
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=qex-a100-check-%j.log

# Submit from the repository root with sbatch -A <NERSC_GPU_ACCOUNT> ...
# Build on Perlmutter first, and export the same compiler/runtime/module
# environment. PERLMUTTER_SYCL_ENV optionally names an environment setup file.
set -euo pipefail
ROOT=${QEX_SYCL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$ROOT"
if [[ -n "${PERLMUTTER_SYCL_ENV:-}" ]]; then
  source "$PERLMUTTER_SYCL_ENV"
fi
source ./scripts/load-perlmutter-env.sh
BUILD="$QEX_BUILD_ROOT/SYCL-a100"
LOGS="$QEX_BUILD_ROOT/validation-${SLURM_JOB_ID:-manual}"
mkdir -p "$LOGS"
export ONEAPI_DEVICE_SELECTOR=cuda:gpu OMP_NUM_THREADS=2
export OMP_PROC_BIND=close OMP_PLACES=cores
for program in "$QEX_BUILD_ROOT/CPU/check_hisq_gpu" \
    "$BUILD/check_cuda_device" "$BUILD/check_sycl_smoke" "$BUILD/check_hisq_gpu"; do
  if [[ ! -x "$program" ]]; then
    echo "Missing $program; build on Perlmutter before submitting." >&2
    exit 1
  fi
done
trap 'echo "Validation failed; logs: $LOGS" >&2' ERR
run_gpu() {
  local ranks=$1
  shift
  # Slurm gives each rank one visible GPU; QEX selects that GPU.
  srun --nodes=1 --ntasks="$ranks" --cpus-per-task=4 \
    --gpus-per-task=1 --gpu-bind=single:1 --cpu-bind=cores "$@"
}
run_gpu 4 "$BUILD/check_cuda_device" A100 > "$LOGS/devices.log" 2>&1
run_gpu 1 "$QEX_BUILD_ROOT/CPU/check_hisq_gpu" > "$LOGS/cpu.log" 2>&1
run_gpu 4 "$BUILD/check_sycl_smoke" > "$LOGS/smoke.log" 2>&1
run_gpu 1 "$BUILD/check_hisq_gpu" > "$LOGS/sycl.log" 2>&1
# Avoid short outer local extents in the CPU length-three reference shifts.
run_gpu 4 "$BUILD/check_hisq_gpu" -lat:8,8,8,16 > "$LOGS/mpi4.log" 2>&1
run_gpu 2 "$BUILD/check_hisq_gpu" -lat:2,4,4,8 -haloOnly:1 > "$LOGS/narrow.log" 2>&1
grep -E 'device:|PASS:' "$LOGS"/*.log
echo "Perlmutter validation logs: $LOGS"
