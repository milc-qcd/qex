# Source this before building or running the local MPI-enabled QEX build.
module purge
module load intel/24.0
module load impi/21.11

export PATH=/work2/11599/mdai07/stampede3/opt/bin:$PATH
export QEX_BUILD=/work2/11599/mdai07/stampede3/qex-gpu-sycl/build-mpi
export QEX_QMP=/work2/11599/mdai07/stampede3/qex-gpu-sycl/deps-mpi/qmp
export QEX_QIO=/work2/11599/mdai07/stampede3/qex-gpu-sycl/deps-mpi/qio
