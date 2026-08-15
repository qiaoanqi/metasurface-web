import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import run_reference_resolution_escalation as escalation


def selected_cases():
    return [
        {
            "L": 120.0 + index,
            "W": 100.0 + index,
            "H": 300.0 + index,
            "P": 400.0 + index,
            "r": 1.2,
            "fill": 0.1,
            "sharpness_5nm": 0.1 * index,
            "selection": "test",
        }
        for index in range(8)
    ]


class ReferenceResolutionPlanTests(unittest.TestCase):
    def test_plan_has_frozen_full_matrix(self):
        tasks = escalation.build_tasks(selected_cases())
        self.assertEqual(len(tasks), 80)
        self.assertEqual({task["step_nm"] for task in tasks}, {0.5, 1.0})
        self.assertEqual(
            {(task["requested_nG"], task["Nxy"]) for task in tasks if task["step_nm"] == 1.0},
            set(escalation.CONFIGS_1NM),
        )
        self.assertEqual(
            len([task for task in tasks if task["step_nm"] == 0.5]),
            16,
        )

    def test_white_is_neutral_on_both_fine_grids(self):
        for wavelength in (escalation.WL_1NM, escalation.WL_HALF_NM):
            lab = escalation.labels_on_grid(wavelength, np.ones_like(wavelength))
            np.testing.assert_allclose(lab, [100.0, 0.0, 0.0], atol=1e-10)

    def test_thresholds_are_unchanged_from_failed_joint_gate(self):
        self.assertEqual(escalation.MEAN_DE_LIMIT, 1.15)
        self.assertEqual(escalation.PER_GEOMETRY_DE_LIMIT, 2.3)
        self.assertEqual(escalation.CONSERVATION_LIMIT, 1e-6)

    def test_atomic_checkpoint_retries_windows_sharing_violation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pkl"
            real_replace = __import__("os").replace
            calls = {"count": 0}

            def transient_replace(source, target):
                calls["count"] += 1
                if calls["count"] < 3:
                    raise PermissionError("synthetic sharing violation")
                return real_replace(source, target)

            with patch(
                "scripts.run_reference_resolution_escalation.os.replace",
                side_effect=transient_replace,
            ), patch(
                "scripts.run_reference_resolution_escalation.time.sleep", return_value=None
            ):
                escalation.atomic_pickle(path, {"ok": True})
            self.assertEqual(calls["count"], 3)
            self.assertTrue(path.is_file())

    def test_plan_declares_diagnostic_scope(self):
        meta = {
            "pool_sha256": "ABC",
            "selection_source": {"path": "failed.json", "sha256": "DEF"},
            "baseline_checkpoint": {"path": "baseline.pkl", "sha256": "123"},
            "selected_geometries": selected_cases(),
            "configs_1nm": [list(config) for config in escalation.CONFIGS_1NM],
            "fine_config": list(escalation.FINE_CONFIG),
            "thresholds": {
                "mean_joint_dE00_lt": 1.15,
                "all_joint_dE00_lt": 2.3,
                "pointwise_conservation_lte": 1e-6,
            },
            "runtime_hashes": {"runner": "HASH"},
        }
        plan = escalation.build_plan(meta)
        self.assertTrue(plan["plan_valid"])
        self.assertIn("cannot pass", plan["decision_rule"])
        self.assertEqual(plan["expected_tasks"], 80)
        self.assertIn("spectral_failure", plan["conditional_escalation"])
        self.assertEqual(plan["estimated_wall_hours_16_workers"], {"low": 8, "high": 12})

    def test_failed_task_produces_evidence_instead_of_crashing(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pkl"
            checkpoint.write_bytes(b"checkpoint")
            meta = {
                "pool_sha256": "ABC",
                "selected_geometries": selected_cases(),
                "thresholds": {},
                "selection_source": {"path": "failed.json", "sha256": "DEF"},
                "baseline_checkpoint": {"path": "baseline.pkl", "sha256": "123"},
                "runtime_hashes": {},
            }
            result = escalation.summarize(
                meta,
                {"failed": {"status": "failed", "error": "synthetic"}},
                {"results": {}},
                checkpoint,
            )
        self.assertFalse(result["passed"])
        self.assertEqual(result["classification"], "execution_or_spectrum_failure")


if __name__ == "__main__":
    unittest.main()
