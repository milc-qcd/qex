#!/usr/bin/env bash
# Small, single-host validation; override the selector to test a real GPU.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SYCL_TARGET=${SYCL_TARGET:-generic}
SKIP_BUILD=${SKIP_BUILD:-0}
case "$SKIP_BUILD" in 0|1) ;; *) echo "SKIP_BUILD must be 0 or 1" >&2; exit 2 ;; esac
BUILD_ROOT=${QEX_BUILD_ROOT:-$ROOT/build-sycl}
SYCL_BUILD="$BUILD_ROOT/SYCL"
LOGS="$BUILD_ROOT/validation"
case "$SYCL_TARGET" in
  generic) export ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR:-opencl:cpu} ;;
  h100|a100)
    SYCL_BUILD="$BUILD_ROOT/SYCL-$SYCL_TARGET"
    LOGS="$BUILD_ROOT/validation-$SYCL_TARGET"
    case "$SYCL_TARGET" in h100) GPU_MODEL=H100 ;; a100) GPU_MODEL=A100 ;; esac
    export ONEAPI_DEVICE_SELECTOR=cuda:gpu
    ;;
  *) echo "SYCL_TARGET must be generic, h100 or a100" >&2; exit 2 ;;
esac
export SYCL_TARGET
mkdir -p "$LOGS"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export CL_CONFIG_CPU_MAX_COMPUTE_UNITS=${CL_CONFIG_CPU_MAX_COMPUTE_UNITS:-2}
export I_MPI_FABRICS=${I_MPI_FABRICS:-shm}
export I_MPI_HYDRA_BOOTSTRAP=${I_MPI_HYDRA_BOOTSTRAP:-fork}
if [[ "$SYCL_TARGET" != generic ]]; then
  mkdir -p "$SYCL_BUILD"
  "${SYCL_CXX:-icpx}" -std=c++17 -fsycl \
    "$ROOT/tests/backend/check_cuda_device.cpp" -o "$SYCL_BUILD/check_cuda_device" \
    > "$LOGS/build-device.log" 2>&1
  if ! "$SYCL_BUILD/check_cuda_device" "$GPU_MODEL" > "$LOGS/device.log" 2>&1; then
    cat "$LOGS/device.log" >&2
    echo "$GPU_MODEL validation stopped before running the physics checks. Logs: $LOGS" >&2
    exit 1
  fi
fi
if [[ "$SKIP_BUILD" == 0 ]]; then
  "$ROOT/scripts/build-hisq-sycl.sh" CPU check_hisq_gpu > "$LOGS/build-cpu.log" 2>&1
  "$ROOT/scripts/build-hisq-sycl.sh" SYCL check_sycl_smoke > "$LOGS/build-smoke.log" 2>&1
  "$ROOT/scripts/build-hisq-sycl.sh" SYCL check_hisq_gpu > "$LOGS/build-sycl.log" 2>&1
  for target in check_gpu_mem check_device_halo check_gaugeflow_gpu; do
    "$ROOT/scripts/build-hisq-sycl.sh" SYCL "$target" > "$LOGS/build-$target.log" 2>&1
  done
fi
for program in "$BUILD_ROOT/CPU/check_hisq_gpu" \
    "$SYCL_BUILD/check_sycl_smoke" "$SYCL_BUILD/check_hisq_gpu" \
    "$SYCL_BUILD/check_gpu_mem" "$SYCL_BUILD/check_device_halo" \
    "$SYCL_BUILD/check_gaugeflow_gpu"; do
  if [[ ! -x "$program" ]]; then
    echo "Missing validation binary: $program; build it first or use SKIP_BUILD=0." >&2
    exit 1
  fi
done
"$BUILD_ROOT/CPU/check_hisq_gpu" > "$LOGS/cpu.log" 2>&1
"$SYCL_BUILD/check_sycl_smoke" > "$LOGS/smoke.log" 2>&1
"$SYCL_BUILD/check_hisq_gpu" > "$LOGS/sycl.log" 2>&1
"$SYCL_BUILD/check_hisq_gpu" -rectangle:1 > "$LOGS/rectangle.log" 2>&1
for target in check_gpu_mem check_device_halo check_gaugeflow_gpu; do
  "$SYCL_BUILD/$target" > "$LOGS/$target.log" 2>&1
done
"${MPIEXEC:-mpiexec}" -n 2 "$SYCL_BUILD/check_device_halo" \
  -lat:2,4,4,8 > "$LOGS/device-halo-mpi2.log" 2>&1
"${MPIEXEC:-mpiexec}" -n 2 "$SYCL_BUILD/check_gaugeflow_gpu" \
  -lat:8,8,8,8 > "$LOGS/gauge-mpi2.log" 2>&1
"${MPIEXEC:-mpiexec}" -n 2 "$SYCL_BUILD/check_hisq_gpu" \
  -lat:8,8,8,8 > "$LOGS/mpi2.log" 2>&1
"${MPIEXEC:-mpiexec}" -n 2 "$SYCL_BUILD/check_hisq_gpu" \
  -lat:2,4,4,8 -haloOnly:1 > "$LOGS/narrow.log" 2>&1
grep -E 'SYCL device:|PASS:' "$LOGS/cpu.log" "$LOGS/smoke.log" \
  "$LOGS/sycl.log" "$LOGS/mpi2.log" "$LOGS/narrow.log" \
  "$LOGS/rectangle.log" "$LOGS/check_gpu_mem.log" "$LOGS/check_device_halo.log" \
  "$LOGS/check_gaugeflow_gpu.log" "$LOGS/device-halo-mpi2.log" "$LOGS/gauge-mpi2.log"
echo "Validation logs: $LOGS"
