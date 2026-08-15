import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline_supervisor import file_digest
from scripts import launch_reference_resolution_holdout as launcher


class ReferenceHoldoutLauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.plan = self.root / "plan.json"
        self.evidence = self.root / "candidate.json"
        self.checkpoint = self.root / "candidate.pkl"
        self.audit = self.root / "audit.json"
        self.protocol = self.root / "protocol.json"
        self.candidate_plan = self.root / ".state" / "reference_resolution_v1_plan.json"
        self.candidate_plan.parent.mkdir()
        self.candidate_plan.write_text("{}\n", encoding="ascii")
        self.base_plan = {"pool": {"sha256": "POOL"}}
        self.plan.write_text("{}\n", encoding="ascii")
        self.checkpoint.write_bytes(b"checkpoint")
        self.evidence.write_text(
            json.dumps({
                "evidence_version": launcher.candidate_runner.VERSION,
                "passed": True,
                "pool_sha256": "POOL",
            }),
            encoding="ascii",
        )
        self.audit.write_text(
            json.dumps({
                "evidence_version": launcher.CANDIDATE_AUDIT_VERSION,
                "passed": True,
                "replacement_pool_required": True,
                "pool_sha256": "POOL",
                "checks": {
                    "reference_checkpoint_exact_80": True,
                    "reference_evidence_passed": True,
                    "physics_controls_passed": True,
                    "production_1nm_comparison_complete": True,
                },
                "inputs": {
                    "reference_evidence": {"path": "candidate.json", "sha256": file_digest(self.evidence)},
                    "reference_checkpoint": {"path": "candidate.pkl", "sha256": file_digest(self.checkpoint)},
                    "plan": {
                        "path": ".state/reference_resolution_v1_plan.json",
                        "sha256": file_digest(self.candidate_plan),
                    },
                },
            }),
            encoding="ascii",
        )
        self.protocol.write_text(
            json.dumps({
                "evidence_version": launcher.CANDIDATE_PROTOCOL_VERSION,
                "passed": True,
                "approved": False,
                "pool_sha256": "POOL",
                "source_evidence": {"path": "candidate.json", "sha256": file_digest(self.evidence)},
                "source_checkpoint": {"path": "candidate.pkl", "sha256": file_digest(self.checkpoint)},
            }),
            encoding="ascii",
        )

    def tearDown(self):
        self.tmp.cleanup()

    def validate(self):
        with (
            patch.object(launcher, "ROOT", self.root),
            patch.object(launcher, "EXPECTED_PLAN_SHA256", file_digest(self.plan)),
            patch.object(launcher.holdout, "load_plan", return_value=self.base_plan),
            patch.object(launcher.holdout, "load_candidate", return_value={"results": {}}),
        ):
            return launcher.validate_candidate_chain(
                self.plan, self.evidence, self.checkpoint, self.audit, self.protocol
            )

    def test_independent_audit_is_required_and_hash_linked(self):
        result = self.validate()
        self.assertEqual(result["candidate_audit"]["sha256"], file_digest(self.audit))
        payload = json.loads(self.audit.read_text(encoding="ascii"))
        payload["passed"] = False
        self.audit.write_text(json.dumps(payload), encoding="ascii")
        with self.assertRaisesRegex(ValueError, "independent candidate audit"):
            self.validate()

    def test_candidate_protocol_screening_cannot_self_approve(self):
        payload = json.loads(self.protocol.read_text(encoding="ascii"))
        payload["approved"] = True
        self.protocol.write_text(json.dumps(payload), encoding="ascii")
        with self.assertRaisesRegex(ValueError, "candidate protocol screening"):
            self.validate()

    def test_frozen_plan_hash_matches_registered_plan(self):
        self.assertEqual(
            file_digest(launcher.ROOT / ".state/reference_resolution_holdout_v1_plan.json"),
            launcher.EXPECTED_PLAN_SHA256,
        )

    def test_stale_lock_is_recoverable_but_live_lock_is_not(self):
        lock = self.root / "holdout.lock"
        lock.write_text('{"pid": 123}\n', encoding="ascii")
        with patch.object(launcher.supervisor, "pid_alive", return_value=False):
            self.assertTrue(launcher.lock_available(lock))
        with patch.object(launcher.supervisor, "pid_alive", return_value=True):
            self.assertFalse(launcher.lock_available(lock))


if __name__ == "__main__":
    unittest.main()
