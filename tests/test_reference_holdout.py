import unittest

import numpy as np

from scripts import freeze_reference_holdout_plan as freezer
from scripts import run_reference_resolution_escalation as base
from scripts import run_reference_resolution_holdout as holdout
from scripts.reference_protocol_selection import CONFIGS, evaluate_protocols


def geometry(index: int, sharpness: float) -> dict:
    L = 90.0 + index * 2.0
    W = 80.0 + index
    H = 120.0 + index * 5.0
    P = 300.0 + index * 3.0
    return {
        "L": L,
        "W": W,
        "H": H,
        "P": P,
        "r": L / W,
        "fill": np.pi * (L / 2.0) * (W / 2.0) / (P * P),
        "sharpness_5nm": sharpness,
    }


class ReferenceHoldoutTests(unittest.TestCase):
    def test_selection_is_deterministic_and_balanced(self):
        candidates = [geometry(index, index / 100.0) for index in range(40)]
        existing = [geometry(100 + index, 0.1 * index) for index in range(8)]
        first, boundaries = freezer.choose_holdout(candidates, existing)
        second, second_boundaries = freezer.choose_holdout(candidates, existing)
        self.assertEqual(first, second)
        self.assertEqual(boundaries, second_boundaries)
        self.assertEqual(len(first), 24)
        self.assertEqual(len({freezer.geometry_key(item) for item in first}), 24)
        counts = {
            label: sum(item["selection"] == label for item in first)
            for label in {item["selection"] for item in first}
        }
        self.assertEqual(sorted(counts.values()), [6, 6, 6, 6])

    def test_holdout_task_matrix_has_exact_240_tasks(self):
        plan = {"new_cases": [geometry(index, index / 24.0) for index in range(24)]}
        tasks = holdout.build_new_tasks(plan)
        self.assertEqual(len(tasks), 240)
        self.assertEqual(len({task["id"] for task in tasks}), 240)
        self.assertEqual({task["step_nm"] for task in tasks}, {0.5, 1.0})

    def test_direct_protocol_selection_uses_lowest_measured_cost_pass(self):
        selected = [geometry(0, 0.1), geometry(1, 0.2)]
        results = {}

        def identifier(index, pol, config, step):
            return f"{index}-{pol}-{config[0]}-{config[1]}-{step}"

        for index, _item in enumerate(selected):
            for pol in base.POLS:
                for name, config in CONFIGS.items():
                    wavelength = base.WL_1NM
                    R = np.linspace(0.1, 0.9, wavelength.size)
                    results[identifier(index, pol, config, 1.0)] = {
                        "wavelength_nm": wavelength,
                        "R": R,
                        "step_nm": 1.0,
                        "time_s": {"A": 10.0, "B": 20.0, "C": 30.0, "D": 40.0}[name],
                    }
                wavelength = base.WL_HALF_NM
                results[identifier(index, pol, base.FINE_CONFIG, 0.5)] = {
                    "wavelength_nm": wavelength,
                    "R": np.linspace(0.1, 0.9, wavelength.size),
                    "step_nm": 0.5,
                    "time_s": 80.0,
                }
        evaluation = evaluate_protocols(selected, results, identifier)
        self.assertTrue(evaluation["any_protocol_passed"])
        chosen = evaluation["lowest_cost_passing_protocol"]
        self.assertEqual(chosen["config_name"], "A")
        self.assertEqual(chosen["wavelength_step_nm"], 5.0)
        self.assertEqual(chosen["requested_nG"], 290)


if __name__ == "__main__":
    unittest.main()
