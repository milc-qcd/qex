#!/usr/bin/env bash
# Source after load-mpi-env.sh and load-nvidia-sycl-env.sh on Stampede3.
_qex_quda_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
case "${SYCL_TARGET:-h100}" in
  h100) _qex_quda_arch=sm_90 ;;
  a100) _qex_quda_arch=sm_80 ;;
  *) echo "Select SYCL_TARGET=h100 or a100 for the CUDA QUDA library" >&2; return 1 ;;
esac
export QUDA_DIR=${QUDA_DIR:-$_qex_quda_root/deps-quda/$_qex_quda_arch}
export CUDA_LIB_DIR=${CUDA_LIB_DIR:-${CUDA_PATH:?}/lib64}
# The HPC SDK keeps its math libraries beside cuda/VERSION; a standalone
# toolkit normally puts them in the CUDA runtime library directory instead.
_qex_quda_cuda_root=$(dirname -- "$(dirname -- "$(readlink -f "$CUDA_PATH/bin/nvcc")")")
_qex_quda_math="$_qex_quda_cuda_root/../../math_libs/lib64"
if [[ ! -d "$_qex_quda_math" ]]; then _qex_quda_math=$CUDA_LIB_DIR; fi
export CUDA_MATH_LIB_DIR=${CUDA_MATH_LIB_DIR:-$_qex_quda_math}
# QUDA is built with Stampede3's GCC 13.2; clang otherwise selects system GCC 11.
export QUDA_GCC_TOOLCHAIN=${QUDA_GCC_TOOLCHAIN:-/opt/apps/gcc/13.2.0}
export LD_LIBRARY_PATH="$QUDA_DIR/lib:$CUDA_LIB_DIR:$CUDA_MATH_LIB_DIR:$QUDA_GCC_TOOLCHAIN/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Use host-buffer MPI until CUDA-aware MPI is separately validated.
export QUDA_ENABLE_GDR=0 QUDA_ENABLE_P2P=0
# endQuda must leave the CUDA primary context alive for SYCL.
export QUDA_DEVICE_RESET=0
unset _qex_quda_root _qex_quda_arch _qex_quda_cuda_root _qex_quda_math
