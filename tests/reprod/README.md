# Reproducibility tests

The [collaborator GPU test guide](../../docs/COLLABORATOR_GPU_TESTS.md)
provides ready-to-run checks against the saved CPU results:

- `check_wall_sources.py`: exact corner/Z4 source fields, multiple source
  times, MPI/thread replay, per-hit and averaged correlators, and gauge flow.
- `check_flow_timings.py --results=...`: exclusive phase accounting and
  complete stage/solve coverage for the small source suite.
- `check_randomwall_flow.py`: independent single-wall FFT/RK3 reference.
- `check_cpu_benchmark.py`: verify reference checksums, prepare/run the full
  configuration-4800 replay, and compare numerical results and timings.
- `test_cpu_benchmark_comparison.py`: offline fault-injection tests of the
  production comparator.

Frozen inputs, CPU source snapshots, numerical outputs, and timings are in
[`reference-data/cpu-flow-20261118`](reference-data/cpu-flow-20261118).
The small gauges are included. The production gauge must be supplied
separately and is identified by its exact SHA-256.
