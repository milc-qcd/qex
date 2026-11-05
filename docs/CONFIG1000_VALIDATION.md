# Configuration 1000: full GPU comparison

On November 5, 2026, the full resident gauge and fermion flow passed against
the saved 0.15 fm CPU result for configuration **1000**, using actual Intel
Max 1550 GPUs on Stampede3. All **10,280** saved numeric values passed;
no physics-code changes or tolerance relaxation were needed for this run.

## Run and reference

- GPU job: **3685973**, completed with exit status 0, node `c551-003`, partition `pvc`.
- Backend: generic SYCL binary with `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`.
- Device inventory: eight visible Level Zero GPU devices, reported as Intel Data Center GPU Max 1550; QEX maps MPI rank modulo the number of visible devices.
- Placement: 32 MPI ranks, two OpenMP threads per rank, three allocated host cores per rank, one node; four ranks per visible device.
- Physics implementation: commit `3c408c82f560e0392f63e451e70b13205abc389b`.
- Binary SHA-256: `a4f02155ef2a6e8348fa0cb458919fa6e217593dd6fd7fb07fc6b3b9ecf8e61d`.
- Reference CPU job: `3650117_1`, completed October 28, 2026 on one ICX node, with 32 MPI ranks and two threads per rank.
- Lattice: `32 x 32 x 32 x 48`; rank geometry `[2,2,4,2]`; SIMD geometry `[2,2,1,2]` (VLEN=8), FP64.
- Wilson gauge flow, step `0.02`, final time `1.50`; HISQ spectrum mass `0.06730`, directional fermion-flow mass `0.0`.
- Gauge input: `/scratch/11599/mdai07/lat015phys/l3248f211b580m002426m06730m8447a.1000.ildg`.
- Input SHA-256: `31aea78504bb359dd5c1733d4d64147a3ea3dabb2fd21a2d0b11604b90531ce0`.

The saved `hisqflow.nim` and `conservedA0.nim` CPU sources are byte-for-byte
identical to the corresponding sources used by this binary. The initial
plaquette is `0.5353887077731858`; each color solve takes 376 iterations.
Gauge force, smearing, gauge updates, directional fermion flow, RK combinations
and halo packing run on the GPU. Inversion, measurements and I/O remain on CPU.

## Numerical results

The original output validator passed: 76 gauge times, 16 correlator times,
48 timeslices, all finite, final flow time 1.50. The comparator requires exact
JSON keys and array lengths and checks every numeric value with the tolerance
chosen before submission:

```text
abs(new - reference) <= 1e-10 + 1e-8 * abs(reference)
```

| Output | Values | Failed | Maximum absolute difference | Relative L2 difference |
| --- | ---: | ---: | ---: | ---: |
| gaugeobs | 1064 | 0 | 6.919465e-14 | 6.153061e-15 |
| correlator | 9216 | 0 | 2.832223e-11 | 8.543958e-15 |

Relative L2 means `norm(new-reference)/norm(reference)` over the stated output.
The correlator aggregate includes large pion/vector channels. The channel
breakdown below also exposes differences in the smaller current channels.

| Correlator | Maximum absolute difference | Relative L2 difference |
| --- | ---: | ---: |
| `pi_05_ss` | 1.719513e-11 | 5.489214e-15 |
| `pi_5_ss` | 1.648459e-11 | 4.226268e-15 |
| `rho_0i_ss` | 2.684519e-11 | 8.605505e-15 |
| `rho_i_ss` | 2.832223e-11 | 7.750227e-15 |
| `A0P_thin_re_ss` | 1.155342e-11 | 2.493849e-14 |
| `A0P_thin_im_ss` | 1.506528e-11 | 7.001196e-13 |
| `cA0P_re_ss` | 2.034994e-11 | 3.675578e-14 |
| `cA0P_im_ss` | 1.108669e-11 | 5.187064e-13 |
| `A0P_hisq_re_ss` | 1.901412e-11 | 3.556012e-14 |
| `A0P_hisq_im_ss` | 9.102941e-12 | 4.936628e-13 |
| `A0P_re_ss` | 1.901412e-11 | 3.556012e-14 |
| `A0P_im_ss` | 9.102941e-12 | 4.936628e-13 |

The worst difference divided by its allowed tolerance is `0.0048254`, in
`A0P_thin_im_ss` at flow time 0.00, timeslice 5. That is an absolute difference
of `1.76048e-12` against reference `0.02648363254007258`, before flow begins.
The largest gauge difference is `6.91947e-14` in topological charge at flow
time 0.64. No time or channel was excluded from comparison.

## Timing and scope

The application-reported initialization-to-finalization time was **313.362 s**
(5 min 13 s), versus **690.874 s** (11 min 31 s) in the saved CPU run: an observed
**2.20x** reduction for these two executions. The CPU models and resources differ,
so this is not a controlled accelerator speedup benchmark. Slurm elapsed time
for the GPU job was 5 min 18 s. GPU memory capacity was sufficient for this
32-rank decomposition across the eight visible devices; peak device memory
was not measured, and memory fit on a different decomposition is untested.

Before the full run, single-GPU gauge force/smearing/RK checks and single-
and two-rank coupled gauge/fermion checks passed on Intel Max 1550 devices.
The largest coupled fermion error in the two-rank test was `2.09221e-15`.
The small gauge test covered Wilson and rectangle actions; this full
configuration used Wilson only. H100/A100 execution and Perlmutter MPI/runtime
integration still require validation.

A companion SYCL CPU-device job, `3685967`, was stopped after the complete
GPU comparison passed, to release its allocation. Its partial output is not
counted as a completed full-configuration validation.

## Artifacts and reproduction

The original `015phys` outputs were preserved and their hashes checked again
after the GPU run. The isolated run root is:

```text
/work2/11599/mdai07/stampede3/test/hisqflow_directional_20261026/gpu-sycl-validation/015phys-cfg1000-20261105-pvc-gpu
```

It contains the exact binary, input files, reference copies, SHA-256 manifest,
submission record, and source-equivalence record. `attempts/job_3685973/` holds
`run.log`, device inventory, both full output files, the completeness check,
`comparison.json`, `comparison.csv`, resource usage, and completion status.
The complete JSON report includes per-flow-time and per-channel errors.
The comparison report records the original node-local candidate path; the
preserved candidate outputs are in the attempt directory above.

Git tracks the [summary](validation/config1000-pvc-summary.json) and
[per-channel CSV](validation/config1000-pvc-comparison.csv), along with the
[comparator](../scripts/compare-flow-output.py) and
[batch script](../scripts/compare-015phys-config1000.sbatch). To repeat the
comparison using the preserved outputs:

```bash
cd /work2/11599/mdai07/stampede3/qex-gpu-sycl
python3 scripts/compare-flow-output.py \
  --reference /work2/11599/mdai07/stampede3/test/hisqflow_directional_20261026/gpu-sycl-validation/015phys-cfg1000-20261105-pvc-gpu/reference \
  --candidate /work2/11599/mdai07/stampede3/test/hisqflow_directional_20261026/gpu-sycl-validation/015phys-cfg1000-20261105-pvc-gpu/attempts/job_3685973 \
  --configuration 1000 --rtol 1e-8 --atol 1e-10 \
  --report /tmp/config1000-comparison.json --csv /tmp/config1000-comparison.csv
```

To repeat the compute run with the preserved binary and inputs, submit from
the isolated run root; the batch script makes a new job-specific attempt:

```bash
cd /work2/11599/mdai07/stampede3/test/hisqflow_directional_20261026/gpu-sycl-validation/015phys-cfg1000-20261105-pvc-gpu
sbatch --export=ALL,QEX_DEVICE_SELECTOR=level_zero:gpu run.sbatch
```
