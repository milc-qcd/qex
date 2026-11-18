"""Offline fault-injection tests for the collaborator comparison (no GPU/MPI)."""
import copy
import hashlib
from pathlib import Path
import struct
import tempfile
import unittest

import check_cpu_benchmark as check


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = check.read(check.REFERENCE / "production/manifest.json")
        cls.reference = check.REFERENCE / "production/references/corner"
        cls.corr = check.read(cls.reference / "correlator_C0p0_4800")

    def compare(self, candidate, *, extra=True):
        return check.compare_observables(candidate, self.corr, 1e-11, 1e-8,
                                         extra_channels=extra)

    def test_frozen_reference_integrity(self):
        self.assertGreater(check.verify_reference(), 40)

    def test_identical_data_and_extra_corner_channels(self):
        self.assertTrue(self.compare(self.corr)["passed"])
        candidate = copy.deepcopy(self.corr)
        for row in candidate.values():
            row["extra_legacy_channel"] = [0.] * 96
        result = self.compare(candidate)
        self.assertTrue(result["passed"])
        self.assertEqual(result["uncompared_extra_channels"], ["extra_legacy_channel"])
        with self.assertRaisesRegex(ValueError, "unexpected candidate channels"):
            self.compare(candidate, extra=False)

    def test_numerical_perturbation_fails(self):
        candidate = copy.deepcopy(self.corr)
        candidate["0.00"]["PP_ss"][0] += 1.
        result = self.compare(candidate)
        self.assertFalse(result["passed"])
        self.assertFalse(result["channels"]["PP_ss"]["passed"])

    def test_missing_measurement_channel_or_timeslice_fails(self):
        for mutation in ["flow", "channel", "time"]:
            candidate = copy.deepcopy(self.corr)
            if mutation == "flow":
                del candidate["0.10"]
            elif mutation == "channel":
                del candidate["0.00"]["PP_ss"]
            else:
                candidate["0.00"]["PP_ss"].pop()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.compare(candidate)

    def test_nonfinite_shared_or_extra_channel_fails(self):
        for channel in ["PP_ss", "extra_channel"]:
            candidate = copy.deepcopy(self.corr)
            candidate["0.00"][channel] = [float("nan")] * 96
            with self.subTest(channel=channel), self.assertRaises(ValueError):
                self.compare(candidate)

    def synthetic_corner(self, root):
        """Use real numerical CPU data and explicitly synthetic timer/log fixtures."""
        case = root / "corner"
        case.mkdir()
        for name in ["correlator_C0p0_4800", "gaugeobs_C0p0_4800"]:
            (case / name).write_bytes((self.reference / name).read_bytes())
        meta = check.read(self.reference / "metadata_C0p0_4800.json")
        meta["solver"] = "qex"
        check.save(case / "correlator_C0p0_4800.source.json", meta)
        plan = {"backend": "CPU", "solver": "qex", "lattice": {"rank-geometry": [1, 1, 2, 2]}}
        spec = self.manifest["cases"]["corner"]
        counts = check.timing_counts(self.manifest, spec, "qex")
        phases = {name: {"rank_min_seconds": float(n > 0),
                         "rank_mean_seconds": float(n > 0), "rank_max_seconds": float(n > 0),
                         "rank_min_segments": n, "rank_max_segments": n}
                  for name, n in counts.items()}
        total = sum(value["rank_mean_seconds"] for value in phases.values())
        timing = {"schema_version": 1, "units": "seconds", "clock": "monotonic wall clock",
                  "backend": "CPU", "solver": "qex", "flow_implementation": "coupled",
                  "mpi_ranks": 4, "source_mode": "corner-wall", "source_count": 1,
                  "solution_fields": 3, "seed": spec["seed"], "configuration": "4800",
                  "phases": phases, "total": {"rank_min_seconds": total,
                  "rank_mean_seconds": total, "rank_max_seconds": total,
                  "rank_min_segments": 1, "rank_max_segments": 1}}
        check.save(case / "timing_C0p0_4800.json", timing)
        (case / "run.log").write_text(
            "Synthetic fixture; not a GPU validation.\n"
            + "stagSolve: 669 1s 2Gf/s 9e-21\n" * 3
            + "Total time (Init - Finalize): 20 seconds.\n")
        return case, plan

    def test_complete_case_and_wrong_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case, plan = self.synthetic_corner(root)
            self.assertEqual(check.compare_case(root, "corner", self.manifest, plan)["status"], "passed")
            path = case / "correlator_C0p0_4800.source.json"
            meta = check.read(path)
            meta["seed"] += 1
            check.save(path, meta)
            with self.assertRaisesRegex(ValueError, "seed"):
                check.compare_case(root, "corner", self.manifest, plan)

    def test_missing_rk_stage_and_bad_accounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case, plan = self.synthetic_corner(root)
            path = case / "timing_C0p0_4800.json"
            good = check.read(path)
            for mutation in ["missing_stage", "accounting"]:
                timing = copy.deepcopy(good)
                if mutation == "missing_stage":
                    timing["phases"]["hisq_links_flow"]["rank_max_segments"] -= 1
                else:
                    for name in ["min", "mean", "max"]:
                        timing["phases"]["other"][f"rank_{name}_seconds"] += .25
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    check.validate_timing(timing, self.manifest, self.manifest["cases"]["corner"], plan)

    def test_saved_comparison_and_failed_application(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, plan = self.synthetic_corner(root)
            plan["gauge"] = self.manifest["gauge"]
            plan["lattice"]["lattice-geometry"] = self.manifest["geometry"]
            plan["cases"] = ["corner"]
            plan["reference_sha256"] = check.digest(check.REFERENCE / "SHA256SUMS")
            check.save(root / "run-plan.json", plan)
            self.assertTrue(check.compare_results(root))
            self.assertEqual(check.read(root / "cpu-comparison.json")["status"], "passed")
            self.assertIn("cpu_mean_seconds", (root / "timing-comparison.tsv").read_text())
            check.save(root / "run-status.json",
                       {"status": "finished", "cases": {"corner": {"exit_code": 1}}})
            self.assertFalse(check.compare_results(root))
            self.assertIn("exit successfully", check.read(root / "cpu-comparison.json")["cases"]["corner"]["error"])

    def test_source_payload_hash_and_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wall.qio"
            data = b"synthetic test payload"
            header = struct.pack(">IHHQ", 0x456789AB, 1, 0, len(data))
            header += b"scidac-binary-data".ljust(128, b"\0")
            path.write_bytes(header + data)
            self.assertEqual(check.payload_digest(path),
                             {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
            path.write_bytes(header + data[:-1])
            with self.assertRaisesRegex(ValueError, "truncated"):
                check.payload_digest(path)


if __name__ == "__main__":
    unittest.main()
