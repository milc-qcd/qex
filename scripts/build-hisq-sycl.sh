#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BACKEND=${1:-SYCL}
TARGET=${2:-check_hisq_gpu}
SYCL_TARGET=${SYCL_TARGET:-generic}
case "$BACKEND" in CPU|SYCL) ;; *) echo "Backend must be CPU or SYCL" >&2; exit 2;; esac
case "$SYCL_TARGET" in generic|h100|a100) ;; *) echo "SYCL_TARGET must be generic, h100 or a100" >&2; exit 2;; esac
case "$TARGET" in
  check_cuda_device) SOURCE="$ROOT/tests/backend/check_cuda_device.cpp" ;;
  hisqflow) SOURCE="$ROOT/src/alphas/hisqflow.nim" ;;
  *) SOURCE="$ROOT/tests/backend/$TARGET.nim" ;;
esac
BUILD_ROOT=${QEX_BUILD_ROOT:-$ROOT/build-sycl}
BUILD="$BUILD_ROOT/$BACKEND"
EXTRA_FLAGS=()
if [[ "$BACKEND" == SYCL && "$SYCL_TARGET" != generic ]]; then
  BUILD="$BUILD_ROOT/SYCL-$SYCL_TARGET"
  case "$SYCL_TARGET" in h100) CUDA_ARCH=sm_90 ;; a100) CUDA_ARCH=sm_80 ;; esac
  CUDA_TOOLKIT=${CUDA_PATH:-${CUDA_HOME:-}}
  if [[ -z "$CUDA_TOOLKIT" ]]; then
    echo "NVIDIA GPU builds require CUDA_PATH (or CUDA_HOME) and a SYCL compiler/runtime with the NVIDIA CUDA backend." >&2
    exit 2
  fi
  if [[ ! -f "$CUDA_TOOLKIT/include/cuda.h" || ! -d "$CUDA_TOOLKIT/nvvm/libdevice" ]]; then
    echo "CUDA toolkit headers or nvvm/libdevice are missing under: $CUDA_TOOLKIT" >&2
    exit 2
  fi
  # Nim passes these through to the C++ compiler at compile AND link time.
  # Current LLVM uses the architecture-qualified target; oneAPI 2024 uses
  # backend forwarding. Explicit architecture selection also works on login
  # nodes where no NVIDIA GPU is available for native-target detection.
  case "${SYCL_CUDA_ARCH_STYLE:-legacy}" in
    legacy) CUDA_FLAGS="-fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=$CUDA_ARCH" ;;
    target) CUDA_FLAGS="-fsycl-targets=nvidia_gpu_$CUDA_ARCH" ;;
    *) echo "SYCL_CUDA_ARCH_STYLE must be legacy or target" >&2; exit 2 ;;
  esac
  EXTRA_FLAGS+=("--passC:$CUDA_FLAGS" "--passL:$CUDA_FLAGS")
  # Preserve shell quoting in the compiler command Nim constructs.
  printf -v CUDA_OPTION '%q' "--cuda-path=$CUDA_TOOLKIT"
  EXTRA_FLAGS+=("--passC:$CUDA_OPTION" "--passL:$CUDA_OPTION")
fi
# A QUDA-enabled executable gets its own cache/output directory. Existing
# flow-only binaries stay available for the CPU-solver comparison.
if [[ -n "${QUDA_DIR:-}" && "$TARGET" != check_cuda_device ]]; then
  if [[ "$BACKEND" == SYCL && "$SYCL_TARGET" == generic ]]; then
    echo "This QUDA profile requires a CUDA SYCL target (h100 or a100)." >&2
    exit 2
  fi
  [[ -f "$QUDA_DIR/include/quda.h" && -f "$QUDA_DIR/include/quda_milc_interface.h" ]] || {
    echo "Missing QUDA installation headers under $QUDA_DIR" >&2; exit 2;
  }
  QUDA_CUDA_LIB=${CUDA_LIB_DIR:-${CUDA_PATH:?Set CUDA_PATH for QUDA}/lib64}
  [[ -f "$QUDA_CUDA_LIB/libcudart.so" ]] || {
    echo "Set CUDA_LIB_DIR to the CUDA runtime library directory." >&2; exit 2;
  }
  BUILD="$BUILD-quda"
  EXTRA_FLAGS+=("-d:qudaDir:$QUDA_DIR" "-d:cudaLibDir:$QUDA_CUDA_LIB" -d:qexQudaCuda)
  if [[ -n "${QUDA_GCC_TOOLCHAIN:-}" ]]; then
    printf -v QUDA_GCC_OPTION '%q' "--gcc-toolchain=$QUDA_GCC_TOOLCHAIN"
    EXTRA_FLAGS+=("--passC:$QUDA_GCC_OPTION" "--passL:$QUDA_GCC_OPTION")
    printf -v QUDA_GCC_RPATH '%q' "-Wl,-rpath,$QUDA_GCC_TOOLCHAIN/lib64"
    EXTRA_FLAGS+=("--passL:$QUDA_GCC_RPATH")
  fi
  # QUDA also uses NVML. Link its toolkit stub on login/CPU nodes; the
  # actual driver library is resolved at runtime on the GPU node.
  if [[ -f "$QUDA_CUDA_LIB/stubs/libnvidia-ml.so" ]]; then
    printf -v QUDA_NVML_OPTION '%q' "-L$QUDA_CUDA_LIB/stubs"
    EXTRA_FLAGS+=("--passL:$QUDA_NVML_OPTION -lnvidia-ml")
  fi
  if [[ -n "${CUDA_MATH_LIB_DIR:-}" ]]; then
    EXTRA_FLAGS+=("-d:cudaMathLibDir:$CUDA_MATH_LIB_DIR")
  fi
  if [[ "$BACKEND" == SYCL ]]; then EXTRA_FLAGS+=(-d:qexQudaSycl); fi
  printf -v QUDA_CUDA_INCLUDE '%q' "-I${CUDA_PATH:?}/include"
  EXTRA_FLAGS+=("--passC:$QUDA_CUDA_INCLUDE")
fi
mkdir -p "$BUILD"
if [[ "$TARGET" == check_cuda_device ]]; then
  if [[ "$BACKEND" != SYCL ]]; then
    echo "check_cuda_device requires Backend=SYCL" >&2; exit 2
  fi
  # Enumeration only: the smoke/physics binaries exercise the target kernels.
  exec "${SYCL_CXX:-icpx}" -std=c++17 -fsycl "$SOURCE" -o "$BUILD/$TARGET"
fi
export VLEN=${VLEN:-8}
# Select the same SYCL compiler through Intel MPI, Open MPI or MPICH wrappers.
export I_MPI_CXX=${I_MPI_CXX:-${SYCL_CXX:-icpx}}
export OMPI_CXX=${OMPI_CXX:-${SYCL_CXX:-icpx}}
export MPICH_CXX=${MPICH_CXX:-${SYCL_CXX:-icpx}}
exec "${NIM:-nim}" cpp --cc:clang \
  --clang.cpp.exe:"${CXX:-mpicxx}" --clang.cpp.linkerexe:"${CXX:-mpicxx}" \
  --parallelBuild:"${BUILD_JOBS:-2}" --mm:refc --threads:on --tlsEmulation:off \
  -d:release --assertions:on \
  --path:"$ROOT/src" --path:"$ROOT/src/alphas" \
  -d:qmpDir:"${QMP_DIR:-$ROOT/deps-mpi/qmp}" \
  -d:qioDir:"${QIO_DIR:-$ROOT/deps-mpi/qio}" \
  -d:Backend="$BACKEND" -d:hisqFlowGpu \
  --nimcache:"$BUILD/nimcache-$TARGET" --out:"$BUILD/$TARGET" \
  "${EXTRA_FLAGS[@]}" "$SOURCE"
