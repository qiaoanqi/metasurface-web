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
        self.plan_path = self.root / "holdout-plan.json"
        self.v2_plan = self.root / "v2-plan.json"
        self.v2_checkpoint = self.root / "v2-checkpoint.pkl"
        self.v2_evidence = self.root / "v2-evidence.json"
        self.v2_audit = self.root / "v2-audit.json"
        self.archive = self.root / "archive.json"
        self.base_checkpoint = self.root / "base.pkl"
        for path, content in (
            (self.v2_plan, "{}"),
            (self.v2_evidence, "{}"),
            (self.v2_audit, '{"passed": true}'),
            (self.archive, "{}"),
        ):
            path.write_text(content, encoding="ascii")
        self.v2_checkpoint.write_bytes(b"v2")
        self.base_checkpoint.write_bytes(b"base")
        self.plan = {
            "pool": {"sha256": "P"},
            "source_v2_plan": self.binding(self.v2_plan),
            "source_v2_checkpoint": self.binding(self.v2_checkpoint),
            "source_v2_worker_evidence": self.binding(self.v2_evidence),
            "source_v2_independent_audit": self.binding(self.v2_audit),
            "archived_v1_holdout_plan": self.binding(self.archive),
            "source_base_checkpoint": self.binding(self.base_checkpoint),
        }
        self.plan_path.write_text(json.dumps(self.plan), encoding="ascii")

    def tearDown(self):
        self.tmp.cleanup()

    def binding(self, path):
        return {"path": path.name, "sha256": file_digest(path)}

    def validate(self, expected_sha=None):
        expected_sha = expected_sha or file_digest(self.plan_path)
        with (
            patch.object(launcher, "ROOT", self.root),
            patch.object(launcher.holdout, "ROOT", self.root),
            patch.object(launcher.holdout, "load_plan", return_value=self.plan),
            patch.object(launcher.holdout, "load_source_results", return_value={}),
        ):
            return launcher.validate_source_chain(
                self.plan_path,
                self.v2_plan,
                self.v2_checkpoint,
                self.v2_evidence,
                self.v2_audit,
                expected_sha,
            )

    def test_full_v2_source_chain_is_hash_linked(self):
        result = self.validate()
        self.assertEqual(result["source_v2_independent_audit"]["sha256"], file_digest(self.v2_audit))
        self.assertEqual(result["archived_v1_holdout_plan"]["sha256"], file_digest(self.archive))

    def test_dispatch_plan_hash_is_mandatory(self):
        with self.assertRaisesRegex(ValueError, "differs from dispatch"):
            self.validate("0" * 64)

    def test_any_v2_source_hash_change_is_rejected(self):
        self.v2_checkpoint.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "source_v2_checkpoint"):
            self.validate()

    def test_failed_v2_audit_is_rejected(self):
        self.v2_audit.write_text('{"passed": false}', encoding="ascii")
        self.plan["source_v2_independent_audit"] = self.binding(self.v2_audit)
        self.plan_path.write_text(json.dumps(self.plan), encoding="ascii")
        with self.assertRaisesRegex(ValueError, "not passed"):
            self.validate()

    def test_registered_plan_sha_requires_one_exact_binding(self):
        dispatch = {
            "strategy_evidence": [
                {"path": "holdout-plan.json", "sha256": file_digest(self.plan_path)}
            ]
        }
        with patch.object(launcher, "ROOT", self.root):
            self.assertEqual(
                launcher.registered_plan_sha(dispatch, self.plan_path), file_digest(self.plan_path)
            )
            dispatch["strategy_evidence"].append(dispatch["strategy_evidence"][0])
            with self.assertRaisesRegex(ValueError, "exactly one"):
                launcher.registered_plan_sha(dispatch, self.plan_path)

    def test_stale_lock_is_recoverable_but_live_lock_is_not(self):
        lock = self.root / "holdout.lock"
        lock.write_text('{"pid": 123}\n', encoding="ascii")
        with patch.object(launcher.supervisor, "pid_alive", return_value=False):
            self.assertTrue(launcher.lock_available(lock))
        with patch.object(launcher.supervisor, "pid_alive", return_value=True):
            self.assertFalse(launcher.lock_available(lock))


if __name__ == "__main__":
    unittest.main()
