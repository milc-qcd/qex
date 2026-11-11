#!/usr/bin/env bash
# Build the pinned NVIDIA QUDA library against the MPI used by QEX.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
QUDA_REVISION=af7e15080e247795ef076b6d9284478755a9a683
DEP_ROOT=${QEX_QUDA_ROOT:-$ROOT/deps-quda}
SOURCE="$DEP_ROOT/source"
case "${SYCL_TARGET:-h100}" in
  h100) ARCH=sm_90 ;;
  a100) ARCH=sm_80 ;;
  *) echo "QUDA build requires SYCL_TARGET=h100 or a100" >&2; exit 2 ;;
esac
mkdir -p "$DEP_ROOT"
# A bounded compiler probe and the batch build may overlap; serialize them.
exec 9>"$DEP_ROOT/build-$ARCH.lock"
flock 9
BUILD=${QUDA_BUILD_DIR:-$DEP_ROOT/build-$ARCH}
PREFIX=${QUDA_DIR:-$DEP_ROOT/$ARCH}
CUDA_TOOLKIT=${CUDA_PATH:?Set CUDA_PATH to the CUDA toolkit}
# Resolve the HPC SDK's bin/nvcc symlink: its parent CUDA wrapper directory
# does not contain targets/, which CMake needs when linking compiler probes.
CUDA_TOOLKIT=$(dirname -- "$(dirname -- "$(readlink -f "$CUDA_TOOLKIT/bin/nvcc")")")
QUDA_CUDA_VERSION=$(awk '$1 == "#define" && $2 == "CUDA_VERSION" {print $3; exit}' "$CUDA_TOOLKIT/include/cuda.h")
if [[ "$ARCH" == sm_90 && "$QUDA_CUDA_VERSION" -lt 12060 ]]; then
  echo "This QUDA revision needs CUDA >= 12.6 for H100; select CUDA 12.8 on Stampede3." >&2
  exit 2
fi
EXTRA_CMAKE=()
if [[ -n "${QUDA_EIGEN_INCLUDE:-}" ]]; then
  EXTRA_CMAKE+=(-DQUDA_DOWNLOAD_EIGEN=OFF "-DEIGEN_INCLUDE_DIR=$QUDA_EIGEN_INCLUDE")
fi
if [[ -n "${QUDA_CUDA_MATH_ROOT:-}" ]]; then
  EXTRA_CMAKE+=("-DCMAKE_LIBRARY_PATH=$QUDA_CUDA_MATH_ROOT/lib64" "-DCMAKE_INCLUDE_PATH=$QUDA_CUDA_MATH_ROOT/include")
fi
HOST_CC=${QUDA_HOST_CC:-gcc}
HOST_CXX=${QUDA_HOST_CXX:-g++}
[[ -x "$CUDA_TOOLKIT/bin/nvcc" ]]
if [[ ! -d "$SOURCE/.git" ]]; then
  mkdir -p "$DEP_ROOT"
  git init "$SOURCE"
  git -C "$SOURCE" remote add origin https://github.com/lattice/quda.git
  git -C "$SOURCE" fetch --depth 1 origin "$QUDA_REVISION"
  git -C "$SOURCE" checkout --detach FETCH_HEAD
fi
[[ "$(git -C "$SOURCE" rev-parse HEAD)" == "$QUDA_REVISION" ]]
[[ -z "$(git -C "$SOURCE" status --porcelain)" ]]
I_MPI_CC="$HOST_CC" I_MPI_CXX="$HOST_CXX" cmake -S "$SOURCE" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_INSTALL_LIBDIR=lib \
  -DCMAKE_C_COMPILER="$(command -v "$HOST_CC")" -DCMAKE_CXX_COMPILER="$(command -v "$HOST_CXX")" \
  -DCMAKE_CUDA_COMPILER="$CUDA_TOOLKIT/bin/nvcc" -DCUDAToolkit_ROOT="$CUDA_TOOLKIT" \
  -DMPI_C_COMPILER="$(command -v mpicc)" -DMPI_CXX_COMPILER="$(command -v mpicxx)" \
  -DQUDA_TARGET_TYPE=CUDA -DQUDA_GPU_ARCH="$ARCH" \
  -DQUDA_DIRAC_DEFAULT_OFF=ON -DQUDA_DIRAC_STAGGERED=ON \
  -DQUDA_INTERFACE_MILC=ON -DQUDA_INTERFACE_QDP=OFF \
  -DQUDA_MPI=ON -DQUDA_QMP=OFF -DQUDA_QIO=OFF -DQUDA_OPENMP=OFF \
  -DQUDA_MULTIGRID=OFF -DQUDA_BUILD_ALL_TESTS=OFF -DQUDA_INSTALL_ALL_TESTS=OFF \
  -DQUDA_BUILD_SHAREDLIB=ON "${EXTRA_CMAKE[@]}"
if [[ "${1:-}" == --configure-only ]]; then exit 0; fi
cmake --build "$BUILD" --target quda --parallel "${BUILD_JOBS:-8}"
# QUDA's top-level install also requests libquda_test even when tests are off.
# The lib subdirectory installs the production library, headers, and CMake files.
cmake --install "$BUILD/lib"
printf 'QUDA revision: %s\nQUDA_DIR=%s\n' "$QUDA_REVISION" "$PREFIX"
