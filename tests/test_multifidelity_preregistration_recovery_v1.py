import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import audit_multifidelity_preregistration_recovery_v1 as recovery


class MultifidelityPreregistrationRecoveryTests(unittest.TestCase):
    def test_recovery_rebases_ui_hashes_without_opening_gate(self):
        evidence = recovery.build_evidence()
        self.assertTrue(evidence["passed"])
        self.assertEqual(
            evidence["classification"],
            "multifidelity_preregistration_recovery_ready",
        )
        self.assertEqual(evidence["scientific_classification"], "multifidelity_preregistration_passed")
        self.assertFalse(evidence["gate_registration_allowed"])
        self.assertFalse(evidence["ack_write_allowed"])
        self.assertFalse(evidence["training_allowed"])
        self.assertEqual(evidence["request"]["attempt"], 3)
        self.assertEqual(
            evidence["ui_authorization"]["observed"]["ml_module.py"]["md5"],
            "4759BA473120FFFCEF0503DE0B3B1A28",
        )

    def test_worker_evidence_presence_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "worker.json"
            path.write_text(json.dumps({"forged": True}), encoding="ascii")
            with patch.object(recovery, "WORKER_EVIDENCE_PATH", path):
                with self.assertRaisesRegex(ValueError, "worker evidence exists"):
                    recovery.build_evidence()

    def test_matching_ack_presence_is_fail_closed(self):
        dispatch = recovery.load(recovery.DISPATCH_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ack.json"
            path.write_text(
                json.dumps(
                    {
                        "request_id": dispatch["request_id"],
                        "attempt": dispatch["attempt"],
                        "status": "completed",
                    }
                ),
                encoding="ascii",
            )
            with patch.object(recovery, "ACK_PATH", path):
                with self.assertRaisesRegex(ValueError, "matching executor ack exists"):
                    recovery.build_evidence()


if __name__ == "__main__":
    unittest.main()
