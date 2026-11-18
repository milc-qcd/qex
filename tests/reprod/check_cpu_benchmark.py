"""Prepare/run GPU replay of the frozen 0.09 fm CPU benchmark and compare results."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import struct
import subprocess
import time

from check_flow_timings import CPU_PHASES, GPU_PHASES

REFERENCE = Path(__file__).resolve().parent / "reference-data/cpu-flow-20261118"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_reference():
    entries = (REFERENCE / "SHA256SUMS").read_text().splitlines()
    for line in entries:
        expected, name = line.split("  ", 1)
        path = (REFERENCE / name).resolve()
        require(REFERENCE.resolve() in path.parents, "invalid reference path")
        require(digest(path) == expected, f"reference checksum mismatch: {name}")
    return len(entries)


def payload_digest(path):
    with path.open("rb") as stream:
        while header := stream.read(144):
            require(len(header) == 144, "truncated LIME header")
            magic, version, _, size = struct.unpack_from(">IHHQ", header)
            require(magic == 0x456789AB and version == 1, "invalid LIME header")
            kind = header[16:144].split(b"\0")[0]
            if kind == b"scidac-binary-data":
                remaining, h = size, hashlib.sha256()
                while remaining:
                    block = stream.read(min(8 * 1024 * 1024, remaining))
                    require(bool(block), "truncated source payload")
                    remaining -= len(block)
                    h.update(block)
                return {"bytes": size, "sha256": h.hexdigest()}
            stream.seek(size + (-size) % 8, 1)
    raise ValueError("no source binary payload")


def compare_observables(candidate, reference, atol, rtol, *, extra_channels=False):
    require(candidate.keys() == reference.keys(), "flow-time grids differ")
    channels = set(next(iter(reference.values())))
    for row in reference.values():
        require(set(row) == channels, "inconsistent reference channels")
    extra = set(next(iter(candidate.values()))) - channels
    for row in candidate.values():
        require(set(row) == channels | extra, "inconsistent candidate channels")
        require(extra_channels or not extra, "unexpected candidate channels")
        # Also reject nonfinite values in channels outside the shared CPU set.
        for value in row.values():
            values = value if isinstance(value, list) else [value]
            require(all(isinstance(x, (int, float)) and math.isfinite(x) for x in values),
                    "nonfinite or nonnumeric observable")
    tests = {}
    for channel in sorted(channels):
        left, right = [], []
        for tau in sorted(reference, key=float):
            a, b = candidate[tau][channel], reference[tau][channel]
            if isinstance(b, list):
                require(isinstance(a, list) and len(a) == len(b), f"{channel}: array length")
                left.extend(a)
                right.extend(b)
            else:
                require(isinstance(a, (int, float)), f"{channel}: expected scalar")
                left.append(a)
                right.append(b)
        require(all(math.isfinite(x) for x in right), "nonfinite CPU reference")
        peak = max(map(abs, right))
        difference = max(abs(a - b) for a, b in zip(left, right))
        tolerance = atol + rtol * peak
        tests[channel] = {"passed": difference <= tolerance, "values": len(right),
                          "maximum_absolute_difference": difference,
                          "reference_peak": peak, "allowed_difference": tolerance}
    return {"passed": all(v["passed"] for v in tests.values()),
            "channels": tests, "uncompared_extra_channels": sorted(extra),
            "absolute_tolerance": atol, "relative_channel_peak_tolerance": rtol}


def timing_counts(manifest, spec, solver):
    n, measurements = manifest["flow_steps"], manifest["correlator_measurements"]
    corner = spec["source_mode"] == "corner-wall"
    fields = spec["inversions"]
    return {
        "gauge_flow": 6*n+1, "fermion_flow": 3*n, "hisq_links_flow": 3*n,
        "hisq_links_solve": 1,
        "hisq_links_contractions": measurements * (2 if corner else 1),
        "contractions": measurements * (3 if corner else 2),
        "solve": fields, "solve_verification": fields if solver == "quda" else 0,
        "solver_cleanup": int(solver == "quda"),
        "flow_setup": 2*manifest["flow_intervals"], "flow_download": n,
        "gauge_measurements": n+2, "source_generation": fields+int(not corner),
        "source_io": fields if spec["save_source"] else 0,
        "gauge_input": 1, "output_io": 3, "flow_diagnostics": 1, "other": 1,
    }


def validate_timing(timing, manifest, spec, plan):
    require(timing["schema_version"] == 1 and timing["units"] == "seconds", "timing schema")
    require(timing["clock"] == "monotonic wall clock", "timing clock")
    require(timing["backend"] == plan["backend"], "wrong timing backend")
    require(timing["solver"] == plan["solver"], "wrong solver")
    require(timing["flow_implementation"] == "coupled", "expected coupled flow")
    require(timing["mpi_ranks"] == math.prod(plan["lattice"]["rank-geometry"]), "MPI rank count")
    for key, value in {"source_mode": spec["source_mode"], "source_count": 1,
                       "solution_fields": spec["inversions"], "seed": spec["seed"],
                       "configuration": manifest["configuration"]}.items():
        require(timing[key] == value, f"timing {key}")
    phases = timing["phases"]
    require(set(phases) == CPU_PHASES | GPU_PHASES | {"other"}, "phase names")
    counts = timing_counts(manifest, spec, plan["solver"])
    counts["total"] = 1
    for name, entry in {**phases, "total": timing["total"]}.items():
        a, b, c = [entry[f"rank_{stat}_seconds"] for stat in ["min", "mean", "max"]]
        require(all(math.isfinite(x) and x >= 0 for x in [a, b, c]), f"{name}: seconds")
        require(a-1e-9 <= b <= c+1e-9, f"{name}: rank statistics")
        require(c <= timing["total"]["rank_max_seconds"]+1e-9, f"{name}: exceeds total")
        require(entry["rank_min_segments"] == entry["rank_max_segments"] == counts[name],
                f"{name}: segment count, expected {counts[name]}")
        require(c > 0 if counts[name] else c == 0, f"{name}: missing or spurious work")
    discrepancy = abs(sum(p["rank_mean_seconds"] for p in phases.values())
                      - timing["total"]["rank_mean_seconds"])
    require(discrepancy <= max(1e-8, timing["total"]["rank_mean_seconds"]*1e-12),
            "exclusive timing phases do not sum to total")
    return {"passed": True, "accounting_error_seconds": discrepancy, "segments": counts}


def compare_case(results, name, manifest, plan):
    spec = manifest["cases"][name]
    case = results / name
    reference = REFERENCE / "production/references" / name
    cfg = manifest["configuration"]
    meta = read(case / f"correlator_C0p0_{cfg}.source.json")
    ref_meta = read(reference / f"metadata_C0p0_{cfg}.json")
    for key in ["source-kind", "seed", "source-count", "source-times", "rng-scheme",
                "lattice-geometry", "configuration", "mass", "inversions-per-flavor"]:
        require(meta[key] == ref_meta[key], f"source metadata mismatch: {key}")
    require(meta["solver"] == plan["solver"], "source metadata solver")
    require(len(meta["sources"]) == spec["inversions"], "source field count")
    for candidate, expected in zip(meta["sources"], ref_meta["sources"]):
        for key in ["source-index", "source-time", "source-norm2"]:
            require(candidate[key] == expected[key], f"source field {key}")
        if name == "corner":
            for key in ["source-color", "source-corner"]:
                require(candidate[key] == expected[key], f"source field {key}")
    if name == "corner":
        require(meta["source-corners"] == [spec["source_corner"]], "source corner")
    source_payload = None
    if spec["save_source"]:
        require(meta["sources"][0]["source-field-verified"] is True, "source readback")
        source = case / Path(meta["sources"][0]["source-field"]).name
        source_payload = payload_digest(source)
        require(source_payload == manifest["random_source_payload"],
                "saved random wall differs from the CPU binary payload")
    log = (case / "run.log").read_text(errors="replace")
    require(len(re.findall(r"Total time \(Init - Finalize\):", log)) == 1, "clean finalization")
    if plan["backend"] == "SYCL":
        require("SYCL device:" in log, "SYCL device not reported")
    if plan["solver"] == "quda":
        residuals = re.findall(r"QUDA true relative residual squared: (\S+)", log)
    else:
        residuals = re.findall(r"^stagSolve:.*?([0-9.eE+-]+)\s*$", log, re.M)
    residuals = list(map(float, residuals))
    require(len(residuals) == spec["inversions"], "solve count")
    require(all(math.isfinite(x) and 0 <= x <= 1.01e-20 for x in residuals), "solve residual")
    atol = manifest["absolute_tolerance"]
    rtol = manifest["relative_channel_peak_tolerance"]
    correlators = compare_observables(read(case / f"correlator_C0p0_{cfg}"),
                                     read(reference / f"correlator_C0p0_{cfg}"),
                                     atol, rtol, extra_channels=(name == "corner"))
    gauge = compare_observables(read(case / f"gaugeobs_C0p0_{cfg}"),
                                read(reference / f"gaugeobs_C0p0_{cfg}"), atol, rtol)
    timing = read(case / f"timing_C0p0_{cfg}.json")
    accounting = validate_timing(timing, manifest, spec, plan)
    return {"status": "passed" if correlators["passed"] and gauge["passed"] else "failed",
            "correlators": correlators, "gauge_observables": gauge,
            "solver_residuals_squared": residuals, "source_payload": source_payload,
            "timing_accounting": accounting, "candidate_timing": timing,
            "cpu_timing": read(reference / f"timing_C0p0_{cfg}.json")}


def compare_results(results):
    manifest = read(REFERENCE / "production/manifest.json")
    plan = read(results / "run-plan.json")
    require(plan["gauge"] == manifest["gauge"], "plan gauge identity")
    require(plan["lattice"]["lattice-geometry"] == manifest["geometry"], "plan geometry")
    require(plan["reference_sha256"] == digest(REFERENCE / "SHA256SUMS"), "reference version changed")
    execution = read(results / "run-status.json") if (results / "run-status.json").exists() else None
    report = {"status": "running", "cases": {}, "run_plan": plan,
              "scope": "Six shared PP/A0P channels and all gauge observables; "
                       "GPU corner mode has extra contraction work. Timing ratios "
                       "are elapsed-time ratios, not equal-work speedups."}
    rows = []
    for name in plan["cases"]:
        try:
            if execution is not None:
                require(execution["status"] == "finished"
                        and execution["cases"][name]["exit_code"] == 0,
                        "application did not exit successfully")
            result = compare_case(results, name, manifest, plan)
            for phase in list(result["candidate_timing"]["phases"]) + ["total"]:
                cpu = result["cpu_timing"]
                gpu = result["candidate_timing"]
                a = cpu["total"] if phase == "total" else cpu["phases"].get(phase)
                b = gpu["total"] if phase == "total" else gpu["phases"][phase]
                amean = a["rank_mean_seconds"] if a else None
                bmean = b["rank_mean_seconds"]
                rows.append([name, phase, amean, bmean,
                             a["rank_max_seconds"] if a else None,
                             b["rank_max_seconds"], amean/bmean if a and bmean else None])
        except (ValueError, KeyError, OSError, TypeError) as error:
            result = {"status": "failed", "error": str(error)}
        report["cases"][name] = result
    report["status"] = "passed" if report["cases"] and all(
        x["status"] == "passed" for x in report["cases"].values()) else "failed"
    save(results / "cpu-comparison.json", report)
    with (results / "timing-comparison.tsv").open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["case", "phase", "cpu_mean_seconds", "candidate_mean_seconds",
                         "cpu_max_seconds", "candidate_max_seconds", "cpu_mean_div_candidate_mean"])
        writer.writerows(rows)
    print(f"{report['status'].upper()}: {results / 'cpu-comparison.json'}", flush=True)
    return report["status"] == "passed"


def geometry(text):
    values = json.loads(text)
    require(isinstance(values, list) and len(values) == 4
            and all(type(x) is int and x > 0 for x in values), "geometry needs four positive integers")
    return values


def verify_gauge(path, manifest):
    require(path.name == manifest["gauge"]["filename"], "preserve the gauge filename (mass is parsed from it)")
    require(path.stat().st_size == manifest["gauge"]["size_bytes"], "gauge size")
    require(digest(path) == manifest["gauge"]["sha256"], "gauge SHA256 differs from CPU benchmark")


def prepare(args):
    manifest = read(REFERENCE / "production/manifest.json")
    gauge, executable, output = args.gauge.resolve(), args.executable.resolve(), args.output.resolve()
    require(executable.is_file() and os.access(executable, os.X_OK), "executable is not runnable")
    require(args.threads > 0, "threads must be positive")
    ranks, simd = geometry(args.rank_geometry), geometry(args.simd_geometry)
    require(math.prod(simd) == 8, "this test profile expects a VLEN=8 executable")
    require(all(g % r == 0 and (g//r) % s == 0 for g, r, s in zip(manifest["geometry"], ranks, simd)),
            "rank/SIMD geometry does not divide the lattice")
    verify_gauge(gauge, manifest)
    launcher = [word.format(ranks=math.prod(ranks)) for word in shlex.split(args.launcher)]
    require(bool(launcher), "MPI launcher must not be empty")
    output.mkdir(parents=True, exist_ok=False)
    cases = ["corner", "random_wall"] if args.case == "both" else [args.case]
    lattice = {"lattice-geometry": manifest["geometry"], "rank-geometry": ranks, "simd-geometry": simd}
    plan = {
        "schema_version": 1, "reference_sha256": digest(REFERENCE / "SHA256SUMS"),
        "gauge": manifest["gauge"], "gauge_path": str(gauge),
        "executable": str(executable), "executable_sha256": digest(executable),
        "backend": args.backend, "solver": args.solver, "threads_per_rank": args.threads,
        "lattice": lattice, "cases": cases, "commands": {},
    }
    for name in cases:
        directory = output / name
        directory.mkdir()
        spec = manifest["cases"][name]
        save(directory / "flow.json", read(REFERENCE / "production/inputs/flow-test.json"))
        save(directory / "lattice.json", lattice)
        command = launcher + [
            str(executable), f"--flow-json={directory}/flow.json",
            f"--lattice-json={directory}/lattice.json",
            f"--configuration={manifest['configuration']}",
            f"--base-filename={str(gauge).rsplit('.', 2)[0]}",
            f"--output={directory}", f"--source={spec['source_mode']}",
            "--sources=1", f"--seed={spec['seed']}",
            "--source-times=" + json.dumps(spec["source_times"], separators=(",", ":")),
            f"--solver={args.solver}",
        ]
        if name == "corner":
            command.append(f"--source-corner={spec['source_corner']}")
        if spec["save_source"]:
            command.append(f"--save-source={directory}/source_4800.qio")
        plan["commands"][name] = command
        save(directory / "command.json", command)
    save(output / "run-plan.json", plan)
    print(f"Prepared {output / 'run-plan.json'}; no simulation launched.", flush=True)


def run_plan(results):
    plan = read(results / "run-plan.json")
    manifest = read(REFERENCE / "production/manifest.json")
    require(plan["reference_sha256"] == digest(REFERENCE / "SHA256SUMS"), "reference version changed")
    require(digest(Path(plan["executable"])) == plan["executable_sha256"], "executable changed")
    verify_gauge(Path(plan["gauge_path"]), manifest)
    require(not any((results / name / "run.log").exists() for name in plan["cases"]),
            "run already attempted; prepare a fresh output directory")
    progress = {"status": "running", "cases": {}, "environment": {
        k: os.environ[k] for k in ["SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_JOB_PARTITION",
                                   "SLURM_NNODES", "SLURM_NTASKS", "ONEAPI_DEVICE_SELECTOR"]
        if k in os.environ}}
    for name in plan["cases"]:
        directory = results / name
        require(read(directory / "flow.json") == read(REFERENCE / "production/inputs/flow-test.json"),
                "flow input changed")
        require(read(directory / "lattice.json") == plan["lattice"], "lattice input changed")
        start = time.monotonic()
        print(f"Running {name}", flush=True)
        with (directory / "run.log").open("w") as log:
            completed = subprocess.run(plan["commands"][name], cwd=directory,
                                       env=dict(os.environ, OMP_NUM_THREADS=str(plan["threads_per_rank"])),
                                       stdout=log, stderr=subprocess.STDOUT)
        progress["cases"][name] = {"exit_code": completed.returncode,
                                   "launcher_seconds": time.monotonic()-start}
        save(results / "run-status.json", progress)
    progress["status"] = "finished"
    save(results / "run-status.json", progress)
    valid = compare_results(results)
    return valid and all(case["exit_code"] == 0 for case in progress["cases"].values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("verify-reference")
    p = sub.add_parser("prepare", help="Verify input and write exact commands; does not run simulations")
    p.add_argument("--executable", type=Path, required=True)
    p.add_argument("--gauge", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--launcher", default="mpirun -n {ranks}")
    p.add_argument("--rank-geometry", required=True,
                   help="Four factors matching the allocated MPI ranks, e.g. '[2,2,2,2]'")
    p.add_argument("--simd-geometry", default="[2,2,1,2]")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--case", choices=["both", "corner", "random_wall"], default="both")
    p.add_argument("--backend", choices=["SYCL", "CPU"], default="SYCL")
    p.add_argument("--solver", choices=["quda", "qex"], default="quda")
    for name in ["run", "compare"]:
        p = sub.add_parser(name)
        p.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    print(f"Verified {verify_reference()} reference files.", flush=True)
    if args.action == "prepare":
        prepare(args)
    elif args.action == "run":
        raise SystemExit(0 if run_plan(args.results.resolve()) else 1)
    elif args.action == "compare":
        raise SystemExit(0 if compare_results(args.results.resolve()) else 1)


if __name__ == "__main__":
    main()
