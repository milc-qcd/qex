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
export CUDA_MATH_LIB_DIR=${CUDA_MATH_LIB_DIR:-/home1/apps/nvidia/Linux_x86_64/24.5/math_libs/lib64}
export LD_LIBRARY_PATH="$QUDA_DIR/lib:$CUDA_LIB_DIR:$CUDA_MATH_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Use host-buffer MPI until CUDA-aware MPI is separately validated.
export QUDA_ENABLE_GDR=0 QUDA_ENABLE_P2P=0
# endQuda must leave the CUDA primary context alive for SYCL.
export QUDA_DEVICE_RESET=0
unset _qex_quda_root _qex_quda_arch
