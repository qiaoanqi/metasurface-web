import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from scripts.run_joint_convergence import (
    atomic_pickle,
    build_tasks,
    retained_order,
    select_stratified,
)


class JointConvergencePlanTests(unittest.TestCase):
    def test_atomic_checkpoint_retries_transient_windows_lock(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pkl"
            real_replace = __import__("os").replace
            calls = {"count": 0}

            def transient_replace(source, target):
                calls["count"] += 1
                if calls["count"] < 3:
                    raise PermissionError("synthetic sharing violation")
                return real_replace(source, target)

            with patch("scripts.run_joint_convergence.os.replace", side_effect=transient_replace), patch(
                "scripts.run_joint_convergence.time.sleep", return_value=None
            ):
                atomic_pickle(path, {"ok": True})
            self.assertEqual(calls["count"], 3)
            self.assertTrue(path.is_file())

    def test_requested_orders_record_actual_retained_counts(self):
        self.assertEqual(retained_order(131, 400.0), 121)
        self.assertEqual(retained_order(201, 400.0), 169)
        self.assertEqual(retained_order(251, 400.0), 225)

    def test_plan_has_full_matrix_and_one_nm_sharp_checks(self):
        geometries = []
        for index in range(64):
            L = 90.0 + index * 2.0
            W = 85.0 + index
            P = 350.0 + index
            geometries.append({
                "L": L,
                "W": W,
                "H": 120.0 + index * 5.0,
                "P": P,
                "r": L / W,
                "fill": np.pi * (L / 2.0) * (W / 2.0) / P**2,
                "sharpness_5nm": float(index),
            })
        selected = select_stratified(geometries)
        tasks = build_tasks(selected)
        self.assertEqual(len(selected), 32)
        self.assertEqual(len([task for task in tasks if task["step_nm"] == 5]), 384)
        self.assertEqual(len([task for task in tasks if task["step_nm"] == 1]), 16)
        self.assertEqual({task["retained_nG"] for task in tasks}, {121, 169, 225})
        self.assertEqual({g["sharpness_stratum"] for g in selected}, {0, 1, 2, 3})


if __name__ == "__main__":
    unittest.main()
