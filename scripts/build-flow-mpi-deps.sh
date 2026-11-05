#!/usr/bin/env bash
# Build clean copies of the same QMP/QIO sources against the target MPI.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
DEP_ROOT=${QEX_DEPS_ROOT:-$ROOT/deps-perlmutter}
QMP_PREFIX=${QMP_DIR:-$DEP_ROOT/qmp}
QIO_PREFIX=${QIO_DIR:-$DEP_ROOT/qio}
QMP_ARCHIVE=${QMP_TARBALL:-$ROOT/deps-mpi/qmp-2.5.4.tar.gz}
QIO_ARCHIVE=${QIO_TARBALL:-$ROOT/deps-mpi/qio-3.0.0.tar.gz}
# cc is the Cray MPI wrapper; use MPI_CC=mpicc on another system.
MPI_CC=${MPI_CC:-cc}
for archive in "$QMP_ARCHIVE" "$QIO_ARCHIVE"; do
  if [[ ! -f "$archive" ]]; then
    echo "Missing source archive: $archive. Set QMP_TARBALL and QIO_TARBALL to the copied release tarballs." >&2
    exit 2
  fi
done
command -v "$MPI_CC" >/dev/null
mkdir -p "$DEP_ROOT"
# Use fresh source directories; never reconfigure the original Stampede3 sources.
WORK=$(mktemp -d "$DEP_ROOT/build-XXXXXXXX")
mkdir "$WORK/qmp" "$WORK/qio"
tar -xzf "$QMP_ARCHIVE" --strip-components=1 -C "$WORK/qmp"
tar -xzf "$QIO_ARCHIVE" --strip-components=1 -C "$WORK/qio"
(
  cd "$WORK/qmp"
  ./configure --prefix="$QMP_PREFIX" --with-qmp-comms-type=mpi \
    CC="$MPI_CC" CFLAGS="${CFLAGS:--O3 -fPIC -std=gnu99}"
  make -j "${BUILD_JOBS:-2}"
  make install
)
(
  cd "$WORK/qio"
  ./configure --prefix="$QIO_PREFIX" --with-qmp="$QMP_PREFIX" \
    CC="$MPI_CC" CFLAGS="${CFLAGS:--O3 -fPIC -std=gnu99}"
  make -j "${BUILD_JOBS:-2}"
  make install
)
printf 'QMP_DIR=%s\nQIO_DIR=%s\nBuild sources/logs: %s\n' "$QMP_PREFIX" "$QIO_PREFIX" "$WORK"
