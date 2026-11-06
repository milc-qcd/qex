#!/usr/bin/env bash
# Source after the MPI environment. Keep compiler, headers and runtime together.
_qex_nvidia_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export QEX_NVIDIA_SYCL_PREFIX=${QEX_NVIDIA_SYCL_PREFIX:-$_qex_nvidia_root/../opt/llvm-sycl-v7.2.0-cuda}
if [[ ! -x "$QEX_NVIDIA_SYCL_PREFIX/bin/clang++" || ! -f "$QEX_NVIDIA_SYCL_PREFIX/lib/libur_adapter_cuda.so" ]]; then
  echo "Missing CUDA-enabled SYCL installation: $QEX_NVIDIA_SYCL_PREFIX" >&2
  return 1
fi
# Intel compiler modules add old SYCL headers to these search paths. Preserve
# other dependencies, including MPI headers, while removing compiler headers.
_qex_clean_compiler_path() {
  local name=$1 entry result=""
  local -a entries
  IFS=: read -r -a entries <<< "${!name-}"
  for entry in "${entries[@]}"; do
    [[ -z "$entry" || "$entry" == */oneapi/compiler/* ]] && continue
    result=${result:+$result:}$entry
  done
  if [[ -n "$result" ]]; then export "$name=$result"; else unset "$name"; fi
}
for _qex_path in CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH; do
  _qex_clean_compiler_path "$_qex_path"
done
export PATH="$QEX_NVIDIA_SYCL_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$QEX_NVIDIA_SYCL_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="$QEX_NVIDIA_SYCL_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export SYCL_CXX="$QEX_NVIDIA_SYCL_PREFIX/bin/clang++"
export I_MPI_CXX="$SYCL_CXX" OMPI_CXX="$SYCL_CXX" MPICH_CXX="$SYCL_CXX"
export I_MPI_CC="$QEX_NVIDIA_SYCL_PREFIX/bin/clang"
export MPI_CC=${MPI_CC:-mpicc}
export QEX_DEPS_ROOT=${QEX_DEPS_ROOT:-$_qex_nvidia_root/deps-sycl/llvm-v7.2.0}
export QMP_DIR=${QMP_DIR:-$QEX_DEPS_ROOT/qmp}
export QIO_DIR=${QIO_DIR:-$QEX_DEPS_ROOT/qio}
export QEX_QMP="$QMP_DIR" QEX_QIO="$QIO_DIR"
export CUDA_PATH=${CUDA_PATH:-/home1/apps/nvidia/Linux_x86_64/24.5/cuda}
export SYCL_TARGET=${SYCL_TARGET:-h100}
export SYCL_CUDA_ARCH_STYLE=target
# QEX uses OpenMP on the host. A plain -fopenmp also requests GPU OpenMP
# offload in this compiler when combined with SYCL NVIDIA targets.
export OMPFLAG=${QEX_HOST_OMP_FLAGS:-"-Xarch_host -fopenmp=libiomp5 -idirafter /opt/intel/oneapi/compiler/2024.0/opt/compiler/include"}
export QEX_BUILD_ROOT=${QEX_BUILD_ROOT:-$_qex_nvidia_root/build-sycl/llvm-v7.2.0}
unset -f _qex_clean_compiler_path
unset _qex_nvidia_root _qex_path
