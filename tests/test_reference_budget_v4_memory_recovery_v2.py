import json
import unittest

from scripts import recover_reference_budget_v4_memory_failure_v2 as recovery


class ReferenceBudgetV4MemoryRecoveryV2Tests(unittest.TestCase):
    def test_metadata_repair_is_deterministic(self):
        result = {"id": "block", "status": "ok", "R": [], "T": []}
        expected = {
            "source_task_id": "probe-v4-g02-p-ng850-nxy1024-step0p5",
            "block_index": 0,
            "start_index": 0,
            "stop_index": 40,
            "geometry_index": 2,
            "pol": "p",
            "requested_nG": 850,
            "retained_nG": 841,
            "Nxy": 1024,
            "step_nm": 0.5,
        }
        repaired = recovery.attach_block_metadata(result, expected)
        for field, value in expected.items():
            self.assertEqual(repaired[field], value)

    def test_v2_protocol_freezes_v1_checkpoint(self):
        protocol = json.loads(recovery.PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(protocol["source_recovery_checkpoint_v1"]["path"], ".state/reference_resolution_budget_v4_memory_recovery_checkpoint.pkl")
        self.assertEqual(protocol["chunking"]["expected_total_chunks"], 42)
        self.assertFalse(protocol["training_allowed"])


if __name__ == "__main__":
    unittest.main()
