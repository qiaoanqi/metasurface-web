import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.run_cross_solver_validation import (
    BASE_HARMONICS,
    epsilon_grid,
    evaluate_results,
    joint_gate_ready,
    polarization_vector,
    retained_order,
    select_cross_solver_geometries,
)


class CrossSolverValidationTests(unittest.TestCase):
    def geometry(self, index):
        return {
            "L": 90.0 + index * 3.0,
            "W": 80.0 + index,
            "H": 110.0 + index * 9.0,
            "P": 300.0 + index * 5.0,
            "r": (90.0 + index * 3.0) / (80.0 + index),
            "fill": 0.04 + index * 0.002,
            "sharpness_5nm": index / 20.0,
        }

    def base_result(self, index, pol, offset=0.0, stress=False):
        base_R = np.linspace(0.2, 0.6, 81)
        base_T = 1.0 - base_R
        third_R = base_R + offset
        third_T = 1.0 - third_R
        tag = "global_sharpness_1" if index == 0 else "global_sharpness_2"
        geometry = {**self.geometry(index), "stress": stress, "selection": tag}
        return {
            "id": f"base-{index}-{pol}", "mode": "base",
            "geometry_index": index, "geometry": geometry, "pol": pol,
            "status": "ok", "base_grcwa_R": base_R, "base_grcwa_T": base_T,
            "base_thirdparty_R": third_R, "base_thirdparty_T": third_T,
        }

    def stress_result(self, index, pol, third_offset=0.0):
        high_R = np.linspace(0.2, 0.6, 81)
        high_T = 1.0 - high_R
        high_third_R = high_R + third_offset
        high_third_T = 1.0 - high_third_R
        return {
            "id": f"stress-{index}-{pol}", "mode": "stress",
            "geometry_index": index, "geometry": {**self.geometry(index), "stress": True},
            "pol": pol, "status": "ok", "high_grcwa_R": high_R,
            "high_grcwa_T": high_T, "high_thirdparty_R": high_third_R,
            "high_thirdparty_T": high_third_T,
        }

    def complete_results(self, offset=0.0):
        results = {}
        for index in range(12):
            stress = index < 4
            for pol in ("p", "s"):
                base = self.base_result(index, pol, offset=offset, stress=stress)
                results[base["id"]] = base
                if stress:
                    high = self.stress_result(index, pol, third_offset=offset)
                    results[high["id"]] = high
        return results

    def test_selection_has_three_per_stratum_and_forced_cases(self):
        geometries = [self.geometry(index) for index in range(80)]
        for geometry in geometries:
            geometry["r"] = 1.5
            geometry["fill"] = 0.2
        geometries[10]["r"] = 1.0001
        geometries[30]["r"] = 3.0
        geometries[50]["fill"] = 0.69
        selected = select_cross_solver_geometries(geometries)
        self.assertEqual(len(selected), 12)
        self.assertEqual(
            [sum(item["sharpness_stratum"] == value for item in selected) for value in range(4)],
            [3, 3, 3, 3],
        )
        tags = {item["selection"] for item in selected}
        self.assertTrue(
            {"global_sharpness_1", "global_sharpness_2", "near_circle", "high_aspect", "high_fill"}
            <= tags
        )
        self.assertEqual(sum(item["stress"] for item in selected), 4)

    def test_mask_rotation_and_polarization_mapping(self):
        first = {"L": 240.0, "W": 120.0, "H": 300.0, "P": 450.0}
        second = {"L": 120.0, "W": 240.0, "H": 300.0, "P": 450.0}
        eps_first, _ = epsilon_grid(first, 550.0, 64)
        eps_second, _ = epsilon_grid(second, 550.0, 64)
        np.testing.assert_array_equal(eps_first, eps_second.T)
        np.testing.assert_array_equal(polarization_vector("s"), [1.0, 0.0])
        np.testing.assert_array_equal(polarization_vector("p"), [0.0, 1.0])
        self.assertEqual(BASE_HARMONICS, (11, 11))
        self.assertEqual(retained_order(131, 400.0), 121)

    def test_identical_converged_spectra_pass(self):
        results = self.complete_results()
        evaluation = evaluate_results(results, len(results))
        self.assertTrue(evaluation["passed"])
        self.assertEqual(evaluation["classification"], "passed")

    def test_converged_cross_solver_mismatch_blocks(self):
        results = self.complete_results(offset=0.2)
        evaluation = evaluate_results(results, len(results))
        self.assertFalse(evaluation["passed"])
        self.assertEqual(evaluation["classification"], "converged_cross_solver_divergence")

    def test_nonconverged_solver_is_classified_uncertain(self):
        results = self.complete_results()
        for result in results.values():
            if result["mode"] == "stress":
                result["high_grcwa_R"] = result["high_grcwa_R"] + 0.2
                result["high_grcwa_T"] = 1.0 - result["high_grcwa_R"]
        evaluation = evaluate_results(results, len(results))
        self.assertFalse(evaluation["passed"])
        self.assertEqual(evaluation["classification"], "uncertain_solver_not_converged")

    def test_failed_task_blocks(self):
        results = self.complete_results()
        key = next(iter(results))
        results[key]["status"] = "failed"
        results[key]["error"] = "synthetic failure"
        evaluation = evaluate_results(results, len(results))
        self.assertFalse(evaluation["passed"])
        self.assertEqual(evaluation["classification"], "thirdparty_or_runtime_unavailable")

    def test_joint_gate_requires_configured_version_and_pool(self):
        import scripts.run_cross_solver_validation as runner

        original_root = runner.ROOT
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / ".state").mkdir()
                (root / "pipeline_policy.json").write_text(
                    json.dumps({
                        "workflow": {"actions": [{
                            "gate": "joint_numerical_convergence",
                            "evidence_version": "paper2-joint-convergence-v2",
                        }]}
                    }), encoding="utf-8"
                )
                evidence = root / "evidence.json"
                evidence.write_text(json.dumps({
                    "evidence_version": "paper2-joint-convergence-v2",
                    "pool_sha256": "POOL-A",
                }), encoding="utf-8")
                (root / ".state" / "audit_result.json").write_text(json.dumps({
                    "training_gates": {"joint_numerical_convergence": True},
                    "gate_evidence": {"joint_numerical_convergence": {
                        "verified": True,
                        "evidence": [{"path": "evidence.json"}],
                    }},
                }), encoding="utf-8")
                runner.ROOT = root
                self.assertTrue(joint_gate_ready("POOL-A"))
                self.assertFalse(joint_gate_ready("POOL-B"))
                evidence.write_text(json.dumps({
                    "evidence_version": "paper2-joint-convergence-v1.1",
                    "pool_sha256": "POOL-A",
                }), encoding="utf-8")
                self.assertFalse(joint_gate_ready("POOL-A"))
        finally:
            runner.ROOT = original_root


if __name__ == "__main__":
    unittest.main()
