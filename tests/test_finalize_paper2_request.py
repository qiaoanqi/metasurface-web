import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import finalize_paper2_request as finalizer
import pipeline_supervisor as supervisor


class Paper2FinalizerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state = self.root / ".state"
        self.state.mkdir()
        self.pool = self.root / "pool.pkl"
        self.pool.write_bytes(b"immutable test pool")
        self.pool_sha = supervisor.file_digest(self.pool)
        self.dispatch_path = self.state / "dispatch_request.json"
        self.ack_path = self.state / "executor_ack.json"
        self.gate_path = self.state / "gate_state.json"
        self.gate_path.write_text('{"schema_version": 1, "gates": {}}\n', encoding="ascii")
        self.protected = []
        for name in ("paper_oe.tex", "paper_oe.pdf", "论文.pdf"):
            path = self.root / name
            path.write_bytes(name.encode("utf-8"))
            self.protected.append({"path": name, "md5": supervisor.file_digest(path, "md5")})
        self.policy = {
            "executor_thread_id": "executor-thread",
            "protected_files": self.protected,
            "immutable_assets": [],
        }
        self.old_finalizer_root = finalizer.ROOT
        self.old_finalizer_file = finalizer.__file__
        self.old_supervisor_root = supervisor.ROOT
        self.old_gate_state = supervisor.GATE_STATE
        finalizer.ROOT = self.root
        fake_finalizer = self.root / "scripts" / "finalize_paper2_request.py"
        fake_finalizer.parent.mkdir()
        fake_finalizer.write_text("# test finalizer\n", encoding="ascii")
        finalizer.__file__ = str(fake_finalizer)
        supervisor.ROOT = self.root
        supervisor.GATE_STATE = self.gate_path
        self.patches = [
            patch.object(supervisor, "load_policy", return_value=self.policy),
            patch.object(
                supervisor,
                "resolve_active_pool",
                return_value=({"path": "pool.pkl"}, {"passed": True, "sha256": self.pool_sha}),
            ),
            patch.object(supervisor, "pid_alive", return_value=False),
        ]
        for item in self.patches:
            item.start()
        self.addCleanup(self._restore)

    def _restore(self):
        for item in reversed(self.patches):
            item.stop()
        finalizer.ROOT = self.old_finalizer_root
        finalizer.__file__ = self.old_finalizer_file
        supervisor.ROOT = self.old_supervisor_root
        supervisor.GATE_STATE = self.old_gate_state
        self.tmp.cleanup()

    def dispatch(self, action="joint_numerical_convergence", request_id="request-1"):
        payload = {
            "request_id": request_id,
            "attempt": 1,
            "action": action,
            "status": "in_progress",
            "payload": {"pool_sha256": self.pool_sha},
        }
        supervisor.atomic_json(self.dispatch_path, payload)
        return payload

    def active_ack(self, request_id="request-1", worker_pid=None):
        payload = {
            "request_id": request_id,
            "attempt": 1,
            "thread_id": "executor-thread",
            "status": "running",
            "worker_pid": worker_pid,
            "checkpoint_path": ".state/checkpoint.pkl",
            "checks": {"pool_sha256": self.pool_sha},
        }
        supervisor.atomic_json(self.ack_path, payload)
        return payload

    def audit_inputs(self, holdout=False):
        if holdout:
            evidence = self.state / "reference_resolution_holdout_v2.json"
            audit_path = self.state / "reference_resolution_holdout_v1_audit.json"
        else:
            evidence = self.state / "reference_resolution_budget_v2.json"
            audit_path = self.state / "reference_resolution_budget_v2_audit.json"
            plan = self.state / "reference_resolution_budget_v2_plan.json"
            plan.write_text("{}\n", encoding="ascii")
        evidence.write_text("{}\n", encoding="ascii")
        audit_path.write_text("{}\n", encoding="ascii")
        checkpoint = self.state / "checkpoint.pkl"
        checkpoint.write_bytes(b"checkpoint")
        return evidence, audit_path, checkpoint

    def test_write_terminal_ack_transitions_once_and_rejects_collisions(self):
        self.active_ack()
        payload = {"request_id": "request-1", "attempt": 1, "status": "failed", "value": 1}
        self.assertEqual(finalizer.write_terminal_ack(self.ack_path, payload), payload)
        self.assertEqual(finalizer.write_terminal_ack(self.ack_path, payload), payload)
        with self.assertRaisesRegex(ValueError, "terminal executor ack differs"):
            finalizer.write_terminal_ack(
                self.ack_path, {"request_id": "request-1", "attempt": 1, "status": "failed", "value": 2}
            )
        with self.assertRaisesRegex(ValueError, "identity collision"):
            finalizer.write_terminal_ack(
                self.ack_path, {"request_id": "other", "attempt": 1, "status": "failed", "value": 1}
            )

    def test_live_worker_blocks_finalization(self):
        self.dispatch()
        self.active_ack(worker_pid=os.getpid())
        with patch.object(supervisor, "pid_alive", return_value=True):
            with self.assertRaisesRegex(ValueError, "worker is alive"):
                finalizer.finalize(self.dispatch_path, self.ack_path)

    def test_joint_pass_and_negative_are_scientific_failures(self):
        for passed, classification in ((True, "budget_v2_converged"), (False, "budget_v2_still_insufficient")):
            with self.subTest(passed=passed):
                self.dispatch(request_id=f"joint-{passed}")
                self.active_ack(request_id=f"joint-{passed}")
                _evidence, audit_path, _checkpoint = self.audit_inputs()
                audit = {"passed": passed, "classification": classification, "checks": {"independent": True}}
                with patch.object(finalizer, "run_auditor", return_value=audit):
                    result = finalizer.finalize(self.dispatch_path, self.ack_path)
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["failure_class"], "scientific")
                self.assertFalse((self.state / "gate_state.json").read_text().find("joint_numerical") >= 0)

    def test_holdout_negative_and_execution_failure_do_not_register_gate(self):
        for audit in (
            {"passed": False, "classification": "reference_resolution_negative", "independent_reproduction": True},
            {"passed": False, "classification": "execution_integrity_failure", "independent_reproduction": False},
        ):
            with self.subTest(classification=audit["classification"]):
                self.dispatch(action="reference_resolution", request_id=audit["classification"])
                self.active_ack(request_id=audit["classification"])
                self.audit_inputs(holdout=True)
                with patch.object(finalizer, "run_auditor", return_value=audit):
                    result = finalizer.finalize(self.dispatch_path, self.ack_path)
                self.assertEqual(result["status"], "failed")
                expected = "permanent" if audit["classification"] == "execution_integrity_failure" else "scientific"
                self.assertEqual(result["failure_class"], expected)
                self.assertNotIn("reference_resolution", supervisor.load_json(self.gate_path, {}).get("gates", {}))

    def test_holdout_pass_registers_gate_and_completed_ack_self_validates(self):
        self.dispatch(action="reference_resolution", request_id="holdout-pass")
        self.active_ack(request_id="holdout-pass")
        self.audit_inputs(holdout=True)
        audit = {"passed": True, "classification": "reference_resolution_converged", "independent_reproduction": True}
        with patch.object(finalizer, "run_auditor", return_value=audit), patch.object(
            supervisor, "verify_gate_payload", return_value=(True, None)
        ):
            result = finalizer.finalize(self.dispatch_path, self.ack_path)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["checks"]["reference_gate_registered"])
        self.assertIn("reference_resolution", supervisor.load_json(self.gate_path)["gates"])
        valid, error = supervisor.validate_completed_ack(
            supervisor.load_json(self.ack_path), self.pool_sha, self.policy
        )
        self.assertTrue(valid, error)
        replay = finalizer.finalize(self.dispatch_path, self.ack_path)
        self.assertEqual(replay, result)

    def test_missing_auditor_output_becomes_permanent_diagnostic(self):
        self.dispatch(action="reference_resolution", request_id="missing-audit")
        self.active_ack(request_id="missing-audit")
        with patch.object(finalizer, "run_auditor", side_effect=ValueError("no canonical audit")):
            result = finalizer.finalize(self.dispatch_path, self.ack_path)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_class"], "permanent")
        self.assertEqual(len(result["evidence"]), 1)
        self.assertTrue((self.root / result["evidence"][0]["path"]).is_file())


if __name__ == "__main__":
    unittest.main()
