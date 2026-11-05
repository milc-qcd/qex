#!/usr/bin/env python3
"""Run a CUDA-enabled SYCL compiler with the loaded CrayPE MPI flags."""
import os
import shlex
import subprocess
import sys

compiler = os.environ.get("SYCL_CXX")
if not compiler:
    sys.exit("Set SYCL_CXX to a CUDA-enabled icpx or clang++ executable.")
wrapper = os.environ.get("CRAY_CXX_WRAPPER", "CC")
args = sys.argv[1:]
compile_only = any(arg in args for arg in ("-c", "-E", "-S", "-fsyntax-only"))
try:
    def cray_flags(kind):
        return shlex.split(subprocess.check_output(
            [wrapper, "-dynamic", f"--cray-print-opts={kind}"], text=True))

    includes = cray_flags("cflags")
    libraries = [] if compile_only else cray_flags("libs")
    # Libraries follow object files and QEX's QMP/QIO static archives.
    os.execvp(compiler, [compiler, *includes, *args, *libraries])
except (OSError, subprocess.CalledProcessError, ValueError) as error:
    sys.exit(f"Cannot invoke the SYCL compiler with CrayPE MPI flags: {error}")
