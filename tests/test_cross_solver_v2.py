import copy
import unittest
from unittest.mock import patch

import numpy as np

import pipeline_supervisor as supervisor
from scripts import run_cross_solver_validation_v2 as cross


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
                "wavelength_nm": cross.WAVELENGTHS_NM.copy(),
                "grcwa_R": R,
                "grcwa_T": T,
                "thirdparty_R": R,
                "thirdparty_T": T,
            }
        return results, stress

    def test_checkpoint_validation_rejects_unknown_mismatched_and_corrupt_results(self):
        tasks, _ = cross.build_tasks(
            selected_geometries(), {"nG_requested": 290, "Nxy": 384}
        )
        results, _ = self.build_identical_results()
        checkpoint = {"results": results}
        cross.validate_checkpoint_results(checkpoint, tasks, require_complete=True)

        corruptions = []
        unknown = copy.deepcopy(checkpoint)
        unknown["results"]["unknown-task"] = copy.deepcopy(next(iter(results.values())))
        corruptions.append(unknown)

        mismatched = copy.deepcopy(checkpoint)
        first_key = next(iter(mismatched["results"]))
        mismatched["results"][first_key]["id"] = "other-id"
        corruptions.append(mismatched)

        wrong_grid = copy.deepcopy(checkpoint)
        wrong_grid["results"][first_key]["wavelength_nm"] = cross.WAVELENGTHS_NM[:-1]
        corruptions.append(wrong_grid)

        nonfinite = copy.deepcopy(checkpoint)
        nonfinite["results"][first_key]["grcwa_R"][0] = np.nan
        corruptions.append(nonfinite)

        incomplete = copy.deepcopy(checkpoint)
        incomplete["results"].pop(first_key)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            cross.validate_checkpoint_results(incomplete, tasks, require_complete=True)

        for index, broken in enumerate(corruptions):
            with self.subTest(index=index):
                with self.assertRaises(ValueError):
                    cross.validate_checkpoint_results(broken, tasks)

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

    def test_joint_gate_uses_supervisor_semantic_verification(self):
        context = {"policy": {"workflow": {}}, "pool_audit": {"sha256": "NEW"}}
        with patch.object(
            supervisor,
            "verify_gate_evidence",
            return_value=({"joint_numerical_convergence": False}, {}),
        ) as verify:
            self.assertFalse(cross.joint_v2_ready(context))
        verify.assert_called_once_with(context["policy"], context["pool_audit"])

        with patch.object(
            supervisor,
            "verify_gate_evidence",
            return_value=({"joint_numerical_convergence": True}, {}),
        ):
            self.assertTrue(cross.joint_v2_ready(context))


if __name__ == "__main__":
    unittest.main()
