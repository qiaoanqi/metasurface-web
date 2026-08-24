import tempfile
import unittest
from pathlib import Path

from pipeline_supervisor import atomic_json
from scripts import audit_reference_resolution_budget_v4_recovery_v1 as recovery


class ReferenceBudgetV4AuditRecoveryTests(unittest.TestCase):
    def active_dispatch(self, path: Path) -> None:
        atomic_json(
            path,
            {
                "schema_version": 1,
                "request_id": "v4-audit-recovery-test",
                "attempt": 1,
                "action": "reference_resolution_budget_v4",
                "status": "in_progress",
                "strategy_based_on": "7f41f20a7462402a8715",
            },
        )

    def test_frozen_producer_bytes_match_checkpoint_runtime_hashes(self):
        protocol = recovery.load_recovery_protocol()
        for path, expected in protocol["producer_runtime_hashes"].items():
            source = recovery.frozen_bytes(protocol["producer_commit"], path, expected)
            self.assertTrue(source)
        colorimetry = recovery.frozen_colorimetry(protocol)
        self.assertEqual(len(colorimetry[0]), 81)
        self.assertEqual(len(colorimetry[3]), 81)

    def test_completed_checkpoint_is_independently_auditable(self):
        with tempfile.TemporaryDirectory() as temporary:
            dispatch = Path(temporary) / "dispatch.json"
            self.active_dispatch(dispatch)
            result = recovery.audit(
                recovery.ROOT / ".state/reference_resolution_budget_v4_checkpoint.pkl",
                recovery.ROOT / ".state/reference_resolution_budget_v4.json",
                dispatch,
            )
        self.assertEqual(result["authorization_request"]["request_id"], "v4-audit-recovery-test")
        self.assertEqual(result["checkpoint"]["tasks"], 48)
        self.assertTrue(result["checks"]["runtime_hashes_verified"])
        self.assertTrue(result["checks"]["checkpoint_bytes_unchanged"])
        self.assertFalse(result["training_allowed"])
        self.assertIn(result["classification"], {
            "reference_resolution_budget_v4_passed",
            "reference_resolution_budget_v4_failed",
        })

    def test_checkpoint_byte_drift_is_rejected_before_scientific_classification(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            dispatch = directory / "dispatch.json"
            checkpoint = directory / "checkpoint.pkl"
            self.active_dispatch(dispatch)
            checkpoint.write_bytes(
                (recovery.ROOT / ".state/reference_resolution_budget_v4_checkpoint.pkl").read_bytes()
                + b"drift"
            )
            with self.assertRaisesRegex(ValueError, "checkpoint bytes changed"):
                recovery.audit(
                    checkpoint,
                    recovery.ROOT / ".state/reference_resolution_budget_v4.json",
                    dispatch,
                )


if __name__ == "__main__":
    unittest.main()
