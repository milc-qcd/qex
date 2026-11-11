#!/usr/bin/env bash
# Compile the CPU reference and QUDA/SYCL application without a GPU allocation.
# Numerical CPU tests are opt-in and must run inside an allocation.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
RUN_CPU_TESTS=0
case "${1:-}" in
  "") ;;
  --run-cpu-tests) RUN_CPU_TESTS=1 ;;
  *) echo "Usage: $0 [--run-cpu-tests]" >&2; exit 2 ;;
esac
if [[ "$RUN_CPU_TESTS" == 1 && -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Run numerical CPU tests in a Slurm allocation." >&2
  exit 2
fi
source ./load-mpi-env.sh
source ./scripts/load-nvidia-sycl-env.sh
export SYCL_TARGET=h100 BUILD_JOBS=${BUILD_JOBS:-2}
export OMP_NUM_THREADS=2 I_MPI_FABRICS=shm I_MPI_HYDRA_BOOTSTRAP=fork
LOGS=${QEX_QUDA_BUILD_RECORD:-$QEX_BUILD_ROOT/quda-build-records/local-$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$LOGS"
LOGS=$(cd -- "$LOGS" && pwd)
hostname > "$LOGS/host.txt"
date -Is > "$LOGS/started.txt"
git rev-parse HEAD > "$LOGS/qex-commit.txt"
git diff --binary HEAD > "$LOGS/qex-changes.patch"
finish() {
  local status=$?
  trap - EXIT
  printf '%s\n' "$status" > "$LOGS/exit-status.txt"
  date -Is > "$LOGS/finished.txt"
  exit "$status"
}
trap finish EXIT
printf 'building\n' > "$LOGS/exit-status.txt"
QUDA_INSTALL=${QUDA_DIR:-}
# Explicitly omit QUDA for the CPU reference and command-line rejection checks.
export QUDA_DIR=""
for target in hisqflow check_hisq_gpu; do
  ./scripts/build-hisq-sycl.sh CPU "$target" > "$LOGS/build-cpu-$target.log" 2>&1
done
if "$QEX_BUILD_ROOT/CPU/hisqflow" --solver=quda > "$LOGS/no-quda-rejection.log" 2>&1; then
  echo 'FAIL: non-QUDA binary accepted --solver=quda' >&2; exit 1
fi
grep -F 'requires rebuilding with QUDA_DIR' "$LOGS/no-quda-rejection.log"
if "$QEX_BUILD_ROOT/CPU/hisqflow" --solver=invalid > "$LOGS/invalid-solver-rejection.log" 2>&1; then
  echo 'FAIL: invalid solver accepted' >&2; exit 1
fi
grep -F 'Unknown HISQ solver' "$LOGS/invalid-solver-rejection.log"
if [[ "$RUN_CPU_TESTS" == 1 ]]; then
  ibrun -n 2 -o 0 "$QEX_BUILD_ROOT/CPU/check_hisq_gpu" -lat:8,8,8,8 > "$LOGS/cpu-mpi2.log" 2>&1
fi
export QUDA_DIR="$QUDA_INSTALL"
source ./scripts/load-quda-env.sh
[[ -f "$QUDA_DIR/lib/libquda.so" ]]
./scripts/build-hisq-sycl.sh SYCL check_cuda_device > "$LOGS/build-device.log" 2>&1
for target in check_quda_hisq hisqflow; do
  ./scripts/build-hisq-sycl.sh SYCL "$target" > "$LOGS/build-$target.log" 2>&1
done
sha256sum "$QEX_BUILD_ROOT/SYCL-h100-quda/"{hisqflow,check_quda_hisq} \
  "$QEX_BUILD_ROOT/SYCL-h100/check_cuda_device" "$QEX_BUILD_ROOT/CPU/"{hisqflow,check_hisq_gpu} \
  "$QUDA_DIR/lib/libquda.so" > "$LOGS/binaries.sha256"
ldd "$QEX_BUILD_ROOT/SYCL-h100-quda/hisqflow" > "$LOGS/application-libraries.txt"
"$CUDA_PATH/bin/nvcc" --version > "$LOGS/cuda-version.txt"
"$SYCL_CXX" --version > "$LOGS/sycl-compiler-version.txt"
echo "PASS: CPU/QUDA application and test builds; record: $LOGS"
