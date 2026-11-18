# Frozen CPU reference data

See the [collaborator instructions](../../../../docs/COLLABORATOR_GPU_TESTS.md)
for build, run, and comparison commands.

- `small/fixtures`: exact unit and nontrivial `8^4` test gauges.
- `small/references`: original CPU wall-driver correlators, per-hit results,
  gauge observables, source options, and gauge hashes.
- `production/inputs`: full `64^3 x 96`, configuration-4800 flow/rank inputs.
- `production/references`: the successful CPU job's correlators, gauge
  observables, phase timings, source metadata, and validation results.
- `production/timing-report.md` and `timings.tsv`: completed CPU timing report.
- `production/manifest.json`: exact sources, gauge/source payload hashes,
  CPU build provenance, and comparison tolerances.
- `cpu-source`: four byte-for-byte source files from the timed CPU build.
  These historical sources are data for reproduction, not replacements for
  the current GPU implementation.

Numerical outputs and timing JSON are copied without recomputation.
Absolute paths in reference source metadata were reduced to basenames.
The CPU build-settings file retains its original command as provenance;
its machine-specific paths must be changed when rebuilding elsewhere.
`SHA256SUMS` verifies the distributed files. The CPU executable, production
gauge, and large source QIO file are not included.
