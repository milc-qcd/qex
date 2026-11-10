#!/usr/bin/env bash
#SBATCH --job-name=qex-quda-flow-build
#SBATCH --partition=skx-dev
#SBATCH --account=TG-MCA93S002
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=build-sycl/quda-flow-build-%j.log
# Submit after the matching QUDA library build has succeeded.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source ./load-mpi-env.sh
source ./scripts/load-nvidia-sycl-env.sh
export SYCL_TARGET=h100 BUILD_JOBS=${BUILD_JOBS:-16}
export OMP_NUM_THREADS=2 I_MPI_FABRICS=shm I_MPI_HYDRA_BOOTSTRAP=fork
LOGS="$QEX_BUILD_ROOT/quda-build-records/${SLURM_JOB_ID}"
mkdir -p "$LOGS"
finish() {
  local status=$?
  trap - EXIT
  printf '%s\n' "$status" > "$LOGS/exit-status.txt"
  date -Is > "$LOGS/finished.txt"
  exit "$status"
}
trap finish EXIT
# Explicitly omit QUDA for the CPU-only regression and rejection checks.
export QUDA_DIR=""
for target in hisqflow check_hisq_gpu; do
  ./scripts/build-hisq-sycl.sh CPU "$target" > "$LOGS/build-cpu-$target.log" 2>&1
done
mpiexec -n 2 "$QEX_BUILD_ROOT/CPU/check_hisq_gpu" -lat:8,8,8,8 > "$LOGS/cpu-mpi2.log" 2>&1
if "$QEX_BUILD_ROOT/CPU/hisqflow" --solver=quda > "$LOGS/no-quda-rejection.log" 2>&1; then
  echo 'FAIL: non-QUDA binary accepted --solver=quda' >&2; exit 1
fi
grep -F 'requires rebuilding with QUDA_DIR' "$LOGS/no-quda-rejection.log"
if "$QEX_BUILD_ROOT/CPU/hisqflow" --solver=invalid > "$LOGS/invalid-solver-rejection.log" 2>&1; then
  echo 'FAIL: invalid solver accepted' >&2; exit 1
fi
grep -F 'Unknown HISQ solver' "$LOGS/invalid-solver-rejection.log"
source ./scripts/load-quda-env.sh
[[ -f "$QUDA_DIR/lib/libquda.so" ]]
./scripts/build-hisq-sycl.sh SYCL check_cuda_device > "$LOGS/build-device.log" 2>&1
for target in check_quda_hisq hisqflow; do
  ./scripts/build-hisq-sycl.sh SYCL "$target" > "$LOGS/build-$target.log" 2>&1
done
sha256sum "$QEX_BUILD_ROOT/SYCL-h100-quda/"{hisqflow,check_quda_hisq} \
  "$QEX_BUILD_ROOT/SYCL-h100/check_cuda_device" "$QUDA_DIR/lib/libquda.so" > "$LOGS/binaries.sha256"
ldd "$QEX_BUILD_ROOT/SYCL-h100-quda/hisqflow" > "$LOGS/application-libraries.txt"
echo 'PASS: CPU regression/rejection checks and QUDA/SYCL application/test builds'
