# Source this on Perlmutter after loading PrgEnv-gnu/cray-mpich/cudatoolkit
# and activating a CUDA-enabled SYCL compiler/runtime. No module names or
# Stampede3 installation paths are assumed here.
_QEX_PERLMUTTER_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -z "${SYCL_CXX:-}" ]] || ! command -v "$SYCL_CXX" >/dev/null 2>&1; then
  echo "Set SYCL_CXX to the installed CUDA-enabled icpx/clang++ executable first." >&2
  return 1
fi
if ! command -v "${CRAY_CXX_WRAPPER:-CC}" >/dev/null 2>&1; then
  echo "Load a Perlmutter programming environment with CrayPE and cray-mpich first." >&2
  return 1
fi
export CUDA_PATH=${CUDA_PATH:-${CUDA_HOME:-${CUDA_ROOT:-}}}
if [[ ! -f "$CUDA_PATH/include/cuda.h" ]]; then
  echo "Set CUDA_PATH to the loaded CUDA toolkit (containing include/cuda.h)." >&2
  return 1
fi
export SYCL_TARGET=a100
export QEX_BUILD_ROOT=${QEX_BUILD_ROOT:-$_QEX_PERLMUTTER_ROOT/build-sycl/perlmutter}
export QMP_DIR=${QMP_DIR:-$_QEX_PERLMUTTER_ROOT/deps-perlmutter/qmp}
export QIO_DIR=${QIO_DIR:-$_QEX_PERLMUTTER_ROOT/deps-perlmutter/qio}
export CXX="$_QEX_PERLMUTTER_ROOT/scripts/sycl-cray-cxx.py"
# The current flow exchanges host buffers; device-pointer MPI is not used.
export MPICH_GPU_SUPPORT_ENABLED=0
unset _QEX_PERLMUTTER_ROOT
