import json
import unittest

import numpy as np

from scripts import recover_reference_budget_v4_memory_failure as recovery


class ReferenceBudgetV4MemoryRecoveryTests(unittest.TestCase):
    @staticmethod
    def protocol():
        return json.loads(recovery.PROTOCOL_PATH.read_text(encoding="utf-8"))

    def test_protocol_freezes_exact_memory_failures(self):
        protocol = self.protocol()
        self.assertEqual(len(protocol["failed_task_ids"]), 2)
        self.assertEqual(protocol["required_failure"]["error_prefix"], "MemoryError:")
        self.assertEqual(protocol["chunking"]["chunk_size_wavelengths"], 40)
        self.assertEqual(protocol["chunking"]["expected_total_chunks"], 42)
        self.assertFalse(protocol["threshold_change_allowed"])
        self.assertFalse(protocol["training_allowed"])

    def test_blocks_cover_each_wavelength_exactly_once(self):
        protocol = self.protocol()
        tasks, full_tasks = recovery.build_block_tasks(protocol)
        self.assertEqual(len(tasks), 42)
        self.assertEqual(len({task["id"] for task in tasks}), 42)
        for pol in ("p", "s"):
            blocks = sorted((task for task in tasks if task["pol"] == pol), key=lambda task: task["block_index"])
            joined = np.concatenate([task["wavelength_nm"] for task in blocks])
            full = full_tasks[f"probe-v4-g02-{pol}-ng850-nxy1024-step0p5"]
            self.assertTrue(np.array_equal(joined, full["wavelength_nm"]))
            self.assertEqual(blocks[0]["start_index"], 0)
            self.assertEqual(blocks[-1]["stop_index"], 801)

    def test_failed_checkpoint_is_exact_and_successes_validate(self):
        protocol = self.protocol()
        sealed = dict(protocol)
        sealed["source_failed_checkpoint"] = dict(protocol["source_failed_checkpoint"])
        sealed["source_failed_checkpoint"]["path"] = protocol["raw_failure_seal"]
        state, payload = recovery.validate_source(sealed)
        self.assertGreater(len(payload), 0)
        failed = {key for key, item in state["results"].items() if item["status"] != "ok"}
        self.assertEqual(failed, set(protocol["failed_task_ids"]))


if __name__ == "__main__":
    unittest.main()
