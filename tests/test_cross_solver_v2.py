import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import pipeline_supervisor as supervisor
from scripts import run_cross_solver_validation_v2 as cross
from scripts import run_replacement_pool as replacement


def selected_geometries():
    return [
        {
            "L": 140.0 + index,
            "W": 90.0 + index * 0.25,
            "H": 250.0 + index,
            "P": 420.0 + index,
            "r": (140.0 + index) / (90.0 + index * 0.25),
            "fill": 0.1 + index * 0.01,
            "sharpness_5nm": index * 0.02,
            "stress": index < 4,
        }
        for index in range(12)
    ]


class CrossSolverV2Tests(unittest.TestCase):
    def test_dynamic_stress_factor_design(self):
        low = cross.stress_configs(290, 384)
        self.assertEqual(
            {(item["nG_requested"], item["Nxy"]) for item in low},
            {(365, 384), (290, 512), (365, 512)},
        )
        high = cross.stress_configs(365, 512)
        self.assertEqual(
            {(item["nG_requested"], item["Nxy"]) for item in high},
            {(450, 512), (365, 768), (450, 768)},
        )

    def test_task_matrix_uses_all_base_and_only_four_stress_geometries(self):
        protocol = {"nG_requested": 290, "Nxy": 384}
        tasks, stress = cross.build_tasks(selected_geometries(), protocol)
        self.assertEqual(len(stress), 3)
        self.assertEqual(len(tasks), 12 * 2 + 4 * 2 * 3)
        self.assertEqual(len({task["id"] for task in tasks}), len(tasks))

    def build_identical_results(self):
        tasks, stress = cross.build_tasks(
            selected_geometries(), {"nG_requested": 290, "Nxy": 384}
        )
        R = np.linspace(0.1, 0.9, cross.WAVELENGTHS_NM.size)
        T = 1.0 - R
        results = {}
        for task in tasks:
            results[task["id"]] = {
                **task,
                "status": "ok",
                "grcwa_R": R,
                "grcwa_T": T,
                "thirdparty_R": R,
                "thirdparty_T": T,
            }
        return results, stress

    def test_identical_solvers_and_configs_pass(self):
        results, stress = self.build_identical_results()
        evaluation = cross.evaluate_results(results, stress)
        self.assertTrue(evaluation["passed"])
        self.assertEqual(evaluation["classification"], "passed")

    def test_solver_divergence_is_classified_after_self_convergence(self):
        results, stress = self.build_identical_results()
        for item in results.values():
            item["thirdparty_R"] = np.full(cross.WAVELENGTHS_NM.size, 0.9)
            item["thirdparty_T"] = np.full(cross.WAVELENGTHS_NM.size, 0.1)
        evaluation = cross.evaluate_results(results, stress)
        self.assertFalse(evaluation["passed"])
        self.assertEqual(evaluation["classification"], "converged_cross_solver_divergence")

    def test_joint_gate_rejects_old_version_and_wrong_pool_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate_path = root / "gate.json"
            evidence_path = root / "joint.json"
            evidence = {
                "evidence_version": "paper2-joint-convergence-v1.1",
                "passed": True,
                "pool_sha256": "NEW",
            }
            supervisor.atomic_json(evidence_path, evidence)
            gate = {
                "gates": {
                    "joint_numerical_convergence": {
                        "passed": True,
                        "evidence": [{
                            "path": "joint.json",
                            "sha256": supervisor.file_digest(evidence_path),
                        }],
                    }
                }
            }
            supervisor.atomic_json(gate_path, gate)
            with patch.object(supervisor, "GATE_STATE", gate_path), patch.object(
                replacement, "canonical_workspace_path", return_value=evidence_path
            ):
                self.assertFalse(cross.joint_v2_ready("NEW"))
                evidence["evidence_version"] = "paper2-joint-convergence-v2"
                supervisor.atomic_json(evidence_path, evidence)
                gate["gates"]["joint_numerical_convergence"]["evidence"][0][
                    "sha256"
                ] = supervisor.file_digest(evidence_path)
                supervisor.atomic_json(gate_path, gate)
                self.assertFalse(cross.joint_v2_ready("OLD"))
                self.assertTrue(cross.joint_v2_ready("NEW"))


if __name__ == "__main__":
    unittest.main()
