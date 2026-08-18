import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import audit_reference_resolution_budget_v4 as audit
from scripts import run_reference_resolution_budget_v4 as runner


class ReferenceResolutionBudgetV4Tests(unittest.TestCase):
    def test_protocol_freezes_formal_task_set_and_safety(self):
        protocol = runner.load_protocol()
        tasks = runner.build_tasks(protocol)
        self.assertEqual(len(tasks), 48)
        self.assertEqual(len({task["id"] for task in tasks}), 48)
        self.assertEqual({task["geometry_index"] for task in tasks}, set(range(8)))
        self.assertEqual({task["pol"] for task in tasks}, {"p", "s"})
        self.assertEqual(
            {(task["requested_nG"], task["Nxy"], task["step_nm"]) for task in tasks},
            {(750, 1024, 0.5), (850, 1024, 1.0), (850, 1024, 0.5)},
        )
        self.assertFalse(protocol["training_allowed"])
        self.assertFalse(protocol["holdout_allowed"])
        self.assertFalse(protocol["old_pool_activation_allowed"])
        self.assertEqual(protocol["thresholds"]["mean_joint_dE00_lt"], 1.15)
        self.assertEqual(protocol["thresholds"]["all_joint_dE00_lt"], 2.3)

    def test_chunks_are_bounded_unique_and_complete(self):
        tasks = runner.build_tasks(runner.load_protocol())
        blocks = runner.build_blocks(tasks)
        self.assertEqual(len(blocks), 848)
        self.assertEqual(len({block["id"] for block in blocks}), 848)
        self.assertTrue(all(1 <= len(block["wavelength_nm"]) <= 40 for block in blocks))
        for task in tasks:
            pieces = sorted(
                (block for block in blocks if block["full_task_id"] == task["id"]),
                key=lambda block: block["block_index"],
            )
            joined = np.concatenate([block["wavelength_nm"] for block in pieces])
            self.assertTrue(np.array_equal(joined, task["wavelength_nm"]))

    def test_probe_seed_is_exactly_six_g02_spectra(self):
        tasks = runner.build_tasks(runner.load_protocol())
        results = {}
        runner.seed_probe(results, tasks, runner.load_protocol())
        self.assertEqual(len(results), 6)
        self.assertEqual({value["geometry_index"] for value in results.values()}, {2})
        expected = {task["id"] for task in tasks if task["geometry_index"] == 2}
        self.assertEqual(set(results), expected)

    def test_auditor_rejects_nonfinite_and_conservation_failure(self):
        protocol = audit.load_protocol()
        task = audit.build_tasks(protocol)[0]
        good = dict(task)
        good.update(status="ok", R=np.full(len(task["wavelength_nm"]), 0.4),
                    T=np.full(len(task["wavelength_nm"]), 0.6), time_s=1.0)
        audit.validate_result(good, task, 1e-6)
        bad = dict(good)
        bad["R"] = np.asarray(good["R"]).copy()
        bad["R"][0] = np.nan
        with self.assertRaises(ValueError):
            audit.validate_result(bad, task, 1e-6)
        bad = dict(good)
        bad["T"] = np.full(len(task["wavelength_nm"]), 0.7)
        with self.assertRaises(ValueError):
            audit.validate_result(bad, task, 1e-6)

    def test_atomic_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pkl"
            payload = {"version": runner.VERSION, "results": {"x": 1}, "blocks": {}}
            runner.atomic_pickle(path, payload)
            with path.open("rb") as handle:
                self.assertEqual(pickle.load(handle), payload)


if __name__ == "__main__":
    unittest.main()
