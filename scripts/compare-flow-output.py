#!/usr/bin/env python3
"""Compare complete gauge/correlator JSON output without hiding small channels."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def leaves(reference, candidate, path=()):
    if isinstance(reference, dict):
        if not isinstance(candidate, dict) or reference.keys() != candidate.keys():
            raise ValueError(f"keys differ at {path}")
        for key in reference:
            yield from leaves(reference[key], candidate[key], path + (key,))
    elif isinstance(reference, list):
        if not isinstance(candidate, list) or len(reference) != len(candidate):
            raise ValueError(f"array length differs at {path}")
        for index, (old, new) in enumerate(zip(reference, candidate)):
            yield from leaves(old, new, path + (index,))
    else:
        for value in (reference, candidate):
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"nonfinite/nonnumeric value at {path}: {value!r}")
        yield path, reference, candidate


def summarize(rows):
    worst = max(rows, key=lambda row: row["scaled_error"])
    difference_norm = math.sqrt(math.fsum(row["absolute_error"] ** 2 for row in rows))
    reference_norm = math.sqrt(math.fsum(row["reference"] ** 2 for row in rows))
    return {
        "count": len(rows),
        "failed_values": sum(row["scaled_error"] > 1 for row in rows),
        "max_absolute_error": max(row["absolute_error"] for row in rows),
        "max_scaled_error": worst["scaled_error"],
        "difference_l2": difference_norm,
        "reference_l2": reference_norm,
        "relative_l2": difference_norm / reference_norm if reference_norm else None,
        "worst": worst,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--label", default="C0p0")
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--legacy-thin-boundary", action="store_true",
                        help="correct the last thin-A0P reference bin from the pre-random-source fixed-origin format")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.rtol) or not math.isfinite(args.atol) or args.rtol < 0 or args.atol <= 0:
        parser.error("rtol must be nonnegative and atol strictly positive, both finite")
    report = {
        "reference": str(args.reference.resolve()),
        "candidate": str(args.candidate.resolve()),
        "configuration": args.configuration,
        "rtol": args.rtol, "atol": args.atol,
        "criterion": "abs(candidate-reference) <= atol + rtol*abs(reference), for every value",
        "files": {}, "passed": True,
    }
    if args.legacy_thin_boundary:
        source_name = f"correlator_{args.label}_{args.configuration}.source.json"
        if (args.reference / source_name).exists():
            parser.error("legacy correction requires a reference predating source metadata")
        metadata = json.loads((args.candidate / source_name).read_text())
        if metadata["source-time"] != 0 or metadata["source-corner"] != 0:
            parser.error("legacy correction requires the candidate source at time 0, corner 0")
        report["reference_transformation"] = "negate last bin of A0P_thin_re_ss and A0P_thin_im_ss; original reference file remains unchanged"
    summaries = []
    for kind in ("gaugeobs", "correlator"):
        name = f"{kind}_{args.label}_{args.configuration}"
        old_path, new_path = args.reference / name, args.candidate / name
        try:
            old, new = json.loads(old_path.read_text()), json.loads(new_path.read_text())
            if kind == "correlator" and args.legacy_thin_boundary:
                for channels_at_time in old.values():
                    for channel in ("A0P_thin_re_ss", "A0P_thin_im_ss"):
                        channels_at_time[channel][-1] *= -1
            rows, channels, times = [], {}, {}
            for path, reference, candidate in leaves(old, new):
                difference = abs(candidate - reference)
                row = {"path": list(path), "reference": reference, "candidate": candidate,
                       "absolute_error": difference,
                       "scaled_error": difference / (args.atol + args.rtol * abs(reference))}
                rows.append(row)
                channels.setdefault(str(path[1]), []).append(row)
                times.setdefault(str(path[0]), []).append(row)
            if not rows:
                raise ValueError("empty output")
            summary = summarize(rows)
            summary.update({"reference_sha256": checksum(old_path), "candidate_sha256": checksum(new_path),
                            "by_channel": {key: summarize(values) for key, values in channels.items()},
                            "by_flow_time": {key: summarize(values) for key, values in times.items()},
                            "largest_scaled_errors": sorted(rows, key=lambda row: row["scaled_error"], reverse=True)[:10]})
            report["files"][name] = summary
            report["passed"] &= summary["failed_values"] == 0
            for channel, values in channels.items():
                item = summarize(values)
                summaries.append({"file": kind, "channel": channel,
                                  **{key: value for key, value in item.items() if key != "worst"},
                                  "worst_path": json.dumps(item["worst"]["path"])})
            relative_l2 = summary["relative_l2"]
            relative_text = f"{relative_l2:.6g}" if relative_l2 is not None else "undefined (zero reference)"
            print(f"{kind}: {len(rows)} values; {summary['failed_values']} failures; "
                  f"relative L2={relative_text}; max absolute={summary['max_absolute_error']:.6g}")
        except (ValueError, OSError, TypeError, IndexError) as error:
            report["files"][name] = {"error": str(error)}
            report["passed"] = False
            print(f"{kind}: FAIL: {error}")
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with args.csv.open("w", newline="") as stream:
        if summaries:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(summaries)
    print("PASS: all values agree within tolerance" if report["passed"] else "FAIL: outputs differ")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
