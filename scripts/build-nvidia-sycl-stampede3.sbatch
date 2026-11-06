#!/usr/bin/env bash
#SBATCH --job-name=qex-cuda-build
#SBATCH --partition=pvc
#SBATCH --account=TG-MCA93S002
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=build-sycl/cuda-build-%j.log
# Compilation and small CPU-device reference checks; no NVIDIA GPU is needed.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source ./load-mpi-env.sh
source ./scripts/load-nvidia-sycl-env.sh
export BUILD_JOBS=${BUILD_JOBS:-16}
export OMP_NUM_THREADS=2 CL_CONFIG_CPU_MAX_COMPUTE_UNITS=2
export I_MPI_FABRICS=shm I_MPI_HYDRA_BOOTSTRAP=fork
LOGS="$QEX_BUILD_ROOT/build-logs"
mkdir -p "$LOGS"
finish() {
  local status=$?
  trap - EXIT
  printf '%s\n' "$status" > "$LOGS/build-exit-status.txt"
  date -Is > "$LOGS/build-finished.txt"
  exit "$status"
}
trap finish EXIT
"$SYCL_CXX" --version
if [[ ! -f "$QMP_DIR/lib/libqmp.a" || ! -f "$QIO_DIR/lib/libqio.a" ]]; then
  ./scripts/build-flow-mpi-deps.sh > "$LOGS/mpi-dependencies.log" 2>&1
fi
build() {
  local backend=$1 target=$2
  echo "$(date -Is): build $SYCL_TARGET $backend $target"
  ./scripts/build-hisq-sycl.sh "$backend" "$target" > "$LOGS/$SYCL_TARGET-$backend-$target.log" 2>&1
}
export SYCL_TARGET=generic
build CPU check_hisq_gpu
"$QEX_BUILD_ROOT/CPU/check_hisq_gpu" > "$LOGS/new-compiler-cpu-check.log" 2>&1
build SYCL check_sycl_smoke
build SYCL check_hisq_gpu
ONEAPI_DEVICE_SELECTOR=opencl:cpu "$QEX_BUILD_ROOT/SYCL/check_sycl_smoke" > "$LOGS/new-compiler-sycl-cpu-smoke.log" 2>&1
ONEAPI_DEVICE_SELECTOR=opencl:cpu mpiexec -n 2 "$QEX_BUILD_ROOT/SYCL/check_hisq_gpu" -lat:8,8,8,8 > "$LOGS/new-compiler-sycl-cpu-mpi2.log" 2>&1
for SYCL_TARGET in h100 a100; do
  export SYCL_TARGET
  for target in check_cuda_device check_sycl_smoke check_hisq_gpu check_gaugeflow_gpu check_device_halo check_gpu_mem hisqflow; do
    build SYCL "$target"
  done
  sha256sum "$QEX_BUILD_ROOT/SYCL-$SYCL_TARGET/"{check_cuda_device,check_sycl_smoke,check_hisq_gpu,check_gaugeflow_gpu,check_device_halo,check_gpu_mem,hisqflow} > "$LOGS/$SYCL_TARGET-binaries.sha256"
done
echo "PASS: LLVM SYCL CPU checks and H100/A100 application/check builds"
