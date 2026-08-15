import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pipeline_supervisor as supervisor
from scripts import finalize_audited_gate as audited
from scripts import finalize_paper2_request as common


class AuditedGateFinalizerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state = self.root / ".state"
        self.state.mkdir()
        self.old = {
            "supervisor_root": supervisor.ROOT,
            "supervisor_state": supervisor.STATE,
            "gate": supervisor.GATE_STATE,
            "dispatch": supervisor.DISPATCH_REQUEST,
            "ack": supervisor.EXECUTOR_ACK,
            "audited_root": audited.ROOT,
            "common_root": common.ROOT,
        }
        supervisor.ROOT = self.root
        supervisor.STATE = self.state
        supervisor.GATE_STATE = self.state / "gate_state.json"
        supervisor.DISPATCH_REQUEST = self.state / "dispatch_request.json"
        supervisor.EXECUTOR_ACK = self.state / "executor_ack.json"
        audited.ROOT = self.root
        common.ROOT = self.root

    def tearDown(self):
        supervisor.ROOT = self.old["supervisor_root"]
        supervisor.STATE = self.old["supervisor_state"]
        supervisor.GATE_STATE = self.old["gate"]
        supervisor.DISPATCH_REQUEST = self.old["dispatch"]
        supervisor.EXECUTOR_ACK = self.old["ack"]
        audited.ROOT = self.old["audited_root"]
        common.ROOT = self.old["common_root"]
        self.tmp.cleanup()

    def test_pass_registers_provisional_gate_then_completed_ack_commits_it(self):
        pool = self.root / "pool.pkl"
        pool.write_bytes(b"pool")
        pool_sha = supervisor.file_digest(pool)
        paper = self.root / "paper.tex"
        paper.write_text("locked\n", encoding="ascii")
        worker = self.state / "worker.json"
        worker.write_text("{}\n", encoding="ascii")
        checkpoint = self.state / "checkpoint.pkl"
        checkpoint.write_bytes(b"checkpoint")
        audit_path = self.state / "audit.json"
        request = {"request_id": "request-1", "attempt": 1}
        dispatch = {
            **request,
            "action": "test_action",
            "status": "in_progress",
            "payload": {"pool_sha256": pool_sha},
        }
        ack = {
            **request,
            "thread_id": "executor-thread",
            "status": "running",
            "worker_pid": 999999,
            "checkpoint_path": ".state/checkpoint.pkl",
        }
        supervisor.atomic_json(supervisor.DISPATCH_REQUEST, dispatch)
        supervisor.atomic_json(supervisor.EXECUTOR_ACK, ack)
        supervisor.atomic_json(supervisor.GATE_STATE, {"schema_version": 1, "gates": {}})
        policy = {
            "executor_thread_id": "executor-thread",
            "protected_files": [
                {"path": "paper.tex", "md5": supervisor.file_digest(paper, "md5")}
            ],
            "immutable_assets": [],
            "workflow": {
                "actions": [
                    {
                        "action": "test_action",
                        "gate": "test_gate",
                        "evidence_version": "test-audit-v1",
                        "auditor_script": "scripts/test_auditor.py",
                        "worker_evidence": ".state/worker.json",
                        "audit_evidence": ".state/audit.json",
                        "checkpoint": ".state/checkpoint.pkl",
                        "finalizer": "scripts/finalize_audited_gate.py",
                        "scientific_failure_classifications": ["negative"],
                        "permanent_failure_classifications": [],
                    }
                ]
            },
        }
        audit = {
            "schema_version": 1,
            "evidence_version": "test-audit-v1",
            "request": request,
            "passed": True,
            "classification": "passed",
            "training_allowed": False,
            "pool_sha256": pool_sha,
            "independent_reproduction": True,
            "worker_evidence": {
                "path": ".state/worker.json",
                "sha256": supervisor.file_digest(worker),
            },
        }
        supervisor.atomic_json(audit_path, audit)
        with patch.object(supervisor, "load_policy", return_value=policy), patch.object(
            supervisor,
            "resolve_active_pool",
            return_value=({"path": "pool.pkl"}, {"passed": True, "sha256": pool_sha}),
        ), patch.object(supervisor, "pid_alive", return_value=False), patch.object(
            common, "verify_dispatch_strategy_evidence", return_value=None
        ), patch.object(audited, "run_auditor", return_value=audit), patch.object(
            supervisor, "verify_gate_payload", return_value=(True, None)
        ):
            result = audited.finalize(supervisor.DISPATCH_REQUEST, supervisor.EXECUTOR_ACK)
        self.assertEqual(result["status"], "completed")
        stored_ack = supervisor.load_json(supervisor.EXECUTOR_ACK)
        self.assertEqual(stored_ack["status"], "completed")
        gate = supervisor.load_json(supervisor.GATE_STATE)["gates"]["test_gate"]
        self.assertEqual(gate["phase"], "provisional_until_completed_ack")
        self.assertTrue(supervisor.completed_request_authorized(request, "test_action"))


if __name__ == "__main__":
    unittest.main()
