import unittest

import numpy as np

from scripts.audit_reference_resolution_result import (
    classify,
    physics_controls_pass,
    validate_result,
)


class ReferenceResolutionAuditTests(unittest.TestCase):
    def result_pair(self):
        wavelength = np.arange(380.0, 781.0, 1.0)
        expected = {
            "id": "task",
            "geometry_index": 0,
            "geometry": {"L": 200.0, "W": 150.0, "H": 300.0, "P": 400.0},
            "pol": "p",
            "requested_nG": 131,
            "retained_nG": 121,
            "Nxy": 256,
            "step_nm": 1,
            "wavelength_nm": wavelength,
        }
        result = {
            **expected,
            "status": "ok",
            "R": np.linspace(0.1, 0.9, wavelength.size),
        }
        result["T"] = 1.0 - result["R"]
        return result, expected

    def test_exact_result_validation(self):
        result, expected = self.result_pair()
        validate_result(result, expected)
        result["Nxy"] = 384
        with self.assertRaisesRegex(ValueError, "Nxy"):
            validate_result(result, expected)

    def test_physics_controls_are_all_required(self):
        payload = {
            "solver_verdict": "pass",
            "independent_checks": {
                "empty_layer_fresnel": {"R": 0.04, "analytic_R": 0.04, "rt": 1.0},
                "rotation_max_dR": 1e-12,
                "rotation_max_dT": 1e-12,
                "lw_circle_bitwise": True,
            },
        }
        self.assertTrue(physics_controls_pass(payload))
        payload["independent_checks"]["lw_circle_bitwise"] = False
        self.assertFalse(physics_controls_pass(payload))

    def test_classification_keeps_reference_and_production_separate(self):
        self.assertEqual(classify(False, True, False), "reference_requires_followup")
        self.assertEqual(classify(True, False, False), "implementation_control_failure")
        self.assertEqual(classify(True, True, False), "historical_production_budget_rejected")
        self.assertEqual(classify(True, True, True), "historical_5nm_sampling_rejected")


if __name__ == "__main__":
    unittest.main()
