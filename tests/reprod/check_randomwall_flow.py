"""End-to-end Z4-wall checks against an independent free-HISQ FFT/RK3 reference.

Run inside a compute allocation with NumPy installed. Supply an 8^4 unit-gauge
file named with the strange mass m2 (0.2) and configuration suffix .1000.ildg.
The CPU and SYCL/QUDA executables accept the same test command.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import struct
import subprocess

import numpy as np

from randomwall_reference import reference, wall


def execute(command, work, *, threads=1, expect_error=None):
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_DYNAMIC="FALSE",
               I_MPI_FABRICS="shm", I_MPI_HYDRA_BOOTSTRAP="fork", I_MPI_PIN="0")
    with (work / "run.log").open("w") as log:
        process = subprocess.Popen(command, cwd=work, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            status = process.wait(timeout=240)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    text = (work / "run.log").read_text()
    if expect_error is None:
        assert status == 0, (command, status, text[-4000:])
        assert "Total time (Init - Finalize)" in text
    else:
        assert status != 0 and expect_error in text, (command, status, text[-4000:])
    return text


def read_source(path):
    records = {}
    with path.open("rb") as stream:
        while header := stream.read(144):
            magic, version, flags, length, kind = struct.unpack(">IHHQ128s", header)
            assert magic == 0x456789AB and version == 1
            kind = kind.split(b"\0")[0].decode()
            records[kind] = stream.read(length)
            stream.read((-length) % 8)
    return (json.loads(records["scidac-record-xml"].rstrip(b"\0")),
            np.frombuffer(records["scidac-binary-data"], dtype=">c16").reshape(8, 8, 8, 8, 3))


def compare(left, right):
    assert left.keys() == right.keys()
    worst = 0.0
    for tau in left:
        assert left[tau].keys() == right[tau].keys()
        for channel in left[tau]:
            a, b = np.asarray(left[tau][channel]), np.asarray(right[tau][channel])
            np.testing.assert_allclose(a, b, rtol=2e-8, atol=2e-10,
                                       err_msg=f"{tau}/{channel}")
            worst = max(worst, float(np.max(abs(a-b))))
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--unit-gauge", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--solver", choices=["qex", "quda"], default="qex")
    parser.add_argument("--corner-reference-executable", type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir()
    suffix = ".1000.ildg"
    assert str(args.unit_gauge).endswith(suffix)
    base = str(args.unit_gauge.resolve())[:-len(suffix)]
    geometry = [8, 8, 8, 8]
    report = {}

    def command(work, seed=2, ranks=1, source_time=None, source="random-wall",
                save=True, executable=None, solver=None):
        (work / "lattice.json").write_text(json.dumps({
            "lattice-geometry": geometry, "rank-geometry": [1, 1, 1, ranks],
            "simd-geometry": [2, 2, 2, 1],
        }))
        (work / "flow.json").write_text(json.dumps({
            "C0p0": {"action": "Wilson", "time-increments": [0.02],
                     "maximum-flow-times": [0.12], "loops": 1},
        }))
        values = ["mpirun", "-n", str(ranks), str(executable or args.executable),
                  f"--solver={solver or args.solver}", f"--source={source}",
                  f"--base-filename={base}", "--configuration=1000",
                  f"--lattice-json={work}/lattice.json", f"--flow-json={work}/flow.json",
                  f"--output={work}"]
        if seed is not None:
            values.append(f"--seed={seed}")
        if source_time is not None:
            values.append(f"--source-time={source_time}")
        if save:
            values.append(f"--save-source={work}/source.qio")
        return values

    def run(name, seed=2, ranks=1, threads=1, save=True, source_time=None):
        work = args.output / name
        work.mkdir()
        text = execute(command(work, seed, ranks, source_time, save=save), work, threads=threads)
        metadata = json.loads((work / "correlator_C0p0_1000.source.json").read_text())
        data = json.loads((work / "correlator_C0p0_1000").read_text())
        gauge = json.loads((work / "gaugeobs_C0p0_1000").read_text())
        actual_seed = metadata["seed"]
        assert seed is None or actual_seed == seed
        assert metadata["source-kind"] == "random-wall"
        assert metadata["solver"] == args.solver
        assert metadata["inversions-per-flavor"] == 1
        assert metadata["source-norm2"] == 1536
        assert "source-corner" not in metadata
        t0, expected_source = wall(actual_seed, (8, 8, 8, 8), source_time=source_time)
        assert metadata["source-time"] == t0
        assert metadata["source-time-fixed"] == (source_time is not None)
        if save:
            saved_metadata, saved = read_source(work / "source.qio")
            assert metadata["source-field-verified"]
            assert saved_metadata["seed"] == actual_seed and saved_metadata["source-time"] == t0
            np.testing.assert_array_equal(saved, expected_source)
        else:
            assert "source-field" not in metadata
        if args.solver == "quda":
            residuals = re.findall(r"QUDA true relative residual squared: (\S+)", text)
            assert len(residuals) == 1
            assert math.isfinite(float(residuals[0])) and 0 <= float(residuals[0]) <= 1.01e-20
            assert "SYCL device: NVIDIA" in text
        else:
            solves = re.findall(r"^stagSolve:\s+(\d+)\s+\S+s\s+\S+Gf/s\s+(\S+)\s*$", text, re.M)
            assert len(solves) == 1, solves
            assert 0 <= float(solves[0][1]) <= 1.001e-20
        assert set(data) == {"0.00", "0.10"}
        assert set(gauge) == {f"{i*.02:.2f}" for i in range(7)}
        worst = 0.0
        for tau, channels in data.items():
            _, pp, conserved, thin = reference(
                actual_seed, geometry, 0.2, [0.02]*round(float(tau)/0.02),
                source_time=source_time)
            expected = {
                "PP_ss": pp, "P-P_ss": -pp,
                "A0P_thin_re_ss": thin.real, "A0P_thin_im_ss": thin.imag,
                "cA0P_re_ss": conserved.real, "cA0P_im_ss": conserved.imag,
            }
            worst = max(worst, compare({tau: channels}, {tau: expected}))
        for observation in gauge.values():
            np.testing.assert_allclose(observation["plaquette"], 3, atol=1e-12)
        report[name] = {"passed": True, "seed": actual_seed, "source_time": t0,
                        "source_field_verified": save, "inversions": 1,
                        "maximum_absolute_error": worst, "mpi_ranks": ranks}
        print(f"PASS {name}: t0={t0}, max FFT-reference error={worst:.3g}", flush=True)
        return data, metadata

    serial, _ = run("odd-serial", seed=2)
    distributed, _ = run("odd-mpi2", seed=2, ranks=2, threads=2)
    report["MPI_agreement"] = {"passed": True, "maximum_absolute_error": compare(serial, distributed)}
    run("even-no-save", seed=5, save=False)
    run("fixed-time", seed=2, source_time=0, ranks=2, threads=2)
    clock, metadata = run("clock-mpi2", seed=None, ranks=2, threads=2)
    replay, _ = run("clock-replay", seed=metadata["seed"])
    report["clock_seed_replay"] = {"passed": True, "maximum_absolute_error": compare(clock, replay)}

    # All ranks must terminate promptly when rank 0 discovers an existing record.
    record = args.output / "odd-mpi2/correlator_C0p0_1000.source.json"
    before = record.read_bytes()
    work = args.output / "duplicate-record"
    work.mkdir()
    values = command(work, ranks=2)
    values.append(f"--output={record.parent}")
    execute(values, work, threads=2, expect_error="source record already exists")
    assert record.read_bytes() == before
    report["record_preservation_MPI"] = {"passed": True}
    # A saved source must also survive a new run requesting the same QIO path.
    source = args.output / "odd-mpi2/source.qio"
    before = source.read_bytes()
    work = args.output / "duplicate-field"
    work.mkdir()
    values = command(work, ranks=2)
    values.append(f"--save-source={source}")
    execute(values, work, threads=2, expect_error="source field already exists")
    assert source.read_bytes() == before
    report["field_preservation_MPI"] = {"passed": True}
    for name, option, message in [
        ("corner-option", "--source-corner=0", "applies only to corner-wall"),
        ("invalid-source", "--source=unknown", "--source must be"),
        ("invalid-time", "--source-time=8", "entries must be in"),
        ("invalid-seed", "--seed=-1", "--seed must be"),
    ]:
        work = args.output / name
        work.mkdir()
        values = command(work, ranks=2)
        values.append(option)
        execute(values, work, expect_error=message)
        assert not (work / "correlator_C0p0_1000.source.json").exists()
        report[name] = {"passed": True}
    if args.corner_reference_executable:
        results = []
        for name, executable, solver in [
            ("corner-reference", args.corner_reference_executable, "qex"),
            ("corner-current", args.executable, args.solver),
        ]:
            work = args.output / name
            work.mkdir()
            values = command(work, seed=12345, ranks=2, source_time=7,
                             source="corner-wall", save=False,
                             executable=executable, solver=solver)
            values.append("--source-corner=5")
            execute(values, work, threads=2)
            results.append(json.loads((work / "correlator_C0p0_1000").read_text()))
        # The new corner path retains every legacy channel and adds PP aliases.
        legacy, current = results
        current = {tau: {key: current[tau][key] for key in legacy[tau]} for tau in legacy}
        report["corner_regression"] = {"passed": True, "maximum_absolute_error": compare(legacy, current)}
    report["status"] = "passed"
    (args.output / "checks.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS: all random-wall flow checks", flush=True)


if __name__ == "__main__":
    main()
