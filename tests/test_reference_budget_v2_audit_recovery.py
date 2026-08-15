import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

import pipeline_supervisor as supervisor
from scripts import arm_reference_budget_v2_audit_recovery as arm
from scripts import prepare_reference_budget_v2_retry as retry
from scripts import run_reference_resolution_budget_v2 as budget
from scripts.reference_budget_v2_lineage import binding, file_digest, validate_lineage


def cases():
    return [
        {"L": 120.0 + i, "W": 100.0 + i, "H": 300.0 + i, "P": 400.0 + i}
        for i in range(8)
    ]


def valid_result(task):
    samples = len(task["wavelength_nm"])
    return {
        **task,
        "status": "ok",
        "R": np.full(samples, 0.4),
        "T": np.full(samples, 0.6),
        "time_s": 1.0,
    }


class ReferenceBudgetV2AuditRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.state = self.root / ".state"
        self.state.mkdir()
        self.source = {"request_id": "source-request", "attempt": 1}
        self.active = {"request_id": "target-request", "attempt": 1}
        self.tasks = budget.build_tasks(cases())
        runner_path = self.root / "runner.py"
        runner_path.write_text("# frozen runner\n", encoding="ascii")
        runtime_hashes = {"runner.py": file_digest(runner_path)}
        self.plan = self.state / "reference_resolution_budget_v2_plan.json"
        supervisor.atomic_json(self.plan, {"plan": "frozen"})
        self.checkpoint = self.state / "reference_resolution_budget_v2_checkpoint.pkl"
        self.meta = {
            "version": budget.VERSION,
            "request": self.source,
            "plan_sha256": file_digest(self.plan),
            "pool_sha256": "A" * 64,
            "selected_geometries": cases(),
            "expected_tasks": len(self.tasks),
            "tasks": [
                {
                    key: task[key]
                    for key in (
                        "id",
                        "geometry_index",
                        "pol",
                        "requested_nG",
                        "Nxy",
                        "step_nm",
                    )
                }
                for task in self.tasks
            ],
            "runtime_hashes": runtime_hashes,
        }
        payload = {
            "meta": self.meta,
            "results": {task["id"]: valid_result(task) for task in self.tasks},
        }
        self.checkpoint.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
        self.evidence = self.state / "reference_resolution_budget_v2.json"
        supervisor.atomic_json(
            self.evidence,
            {
                "evidence_version": budget.VERSION,
                "request": self.source,
                "checkpoint": binding(self.root, self.checkpoint) | {"tasks": 96},
                "plan": binding(self.root, self.plan),
                "pool_sha256": "A" * 64,
                "training_allowed": False,
            },
        )
        self.recovery = self.state / "reference_resolution_budget_v2_audit_recovery_v1.json"
        supervisor.atomic_json(
            self.recovery,
            {
                "evidence_version": "paper2-reference-budget-v2-audit-recovery-v1",
                "source_request": self.source,
                "observation_only": True,
                "checkpoint_reuse_authorized": False,
                "scientific_outcome_authorized": False,
                "training_allowed": False,
            },
        )
        self.diagnostic = self.state / "finalization_diagnostics" / "source-request-attempt1.json"
        self.diagnostic.parent.mkdir()
        supervisor.atomic_json(
            self.diagnostic,
            {
                "schema_version": 1,
                "evidence_version": "paper2-finalization-diagnostic-v1",
                "classification": "execution_integrity_failure",
                "request": self.source | {"action": "joint_numerical_convergence"},
                "training_allowed": False,
            },
        )
        self.source_ack = {
            "request_id": self.source["request_id"],
            "attempt": 1,
            "status": "failed",
            "failure_class": "permanent",
            "checks": {"finalization_classification": "execution_integrity_failure"},
            "evidence": [
                binding(self.root, self.diagnostic),
                binding(self.root, self.checkpoint),
                binding(self.root, self.evidence),
            ],
        }
        source_dispatch = {
            "request_id": self.source["request_id"],
            "attempt": 1,
            "action": "joint_numerical_convergence",
            "status": "failed",
            "terminal_failure": True,
            "failure_class": "permanent",
        }
        self.history = self.state / "dispatch_history" / "source-request-attempt1.json"
        self.history.parent.mkdir()
        supervisor.atomic_json(
            self.history,
            {"request": source_dispatch, "final_ack": self.source_ack},
        )
        self.seal = self.state / "reference_resolution_budget_v2_post_terminal_seal_v1.json"
        supervisor.atomic_json(
            self.seal,
            {
                "evidence_version": "paper2-reference-budget-v2-post-terminal-seal-v1",
                "source_request": self.source,
                "target_request": self.active | {"strategy_revision": 9},
                "source_dispatch_history": binding(self.root, self.history),
                "source_final_ack": self.source_ack,
                "live_recovery_observation": binding(self.root, self.recovery),
                "checkpoint": binding(self.root, self.checkpoint) | {"tasks": 96},
                "worker_evidence": binding(self.root, self.evidence),
                "plan": binding(self.root, self.plan),
                "pool_sha256": "A" * 64,
                "runner_runtime_hashes": runtime_hashes,
                "audit_only": True,
                "checkpoint_mutation_authorized": False,
                "training_allowed": False,
            },
        )
        self.dispatch = {
            "request_id": self.active["request_id"],
            "attempt": 1,
            "action": "joint_numerical_convergence",
            "status": "in_progress",
            "strategy_revision": 9,
            "strategy_based_on": self.source["request_id"],
            "strategy_evidence": [
                binding(self.root, self.seal),
                binding(self.root, self.history),
            ],
        }
        self.ack = {
            "request_id": self.active["request_id"],
            "attempt": 1,
            "status": "running",
            "worker_pid": None,
            "checks": {
                "audit_only_recovery": True,
                "finalization_ready": True,
                "recovery_seal": binding(self.root, self.seal),
                "completed_tasks": 96,
                "training_allowed": False,
            },
        }

    def repack_diagnostic_lineage(self, diagnostic):
        supervisor.atomic_json(self.diagnostic, diagnostic)
        self.source_ack["evidence"][0] = binding(self.root, self.diagnostic)
        history = supervisor.load_json(self.history)
        history["final_ack"] = self.source_ack
        supervisor.atomic_json(self.history, history)
        seal = supervisor.load_json(self.seal)
        seal["source_final_ack"] = self.source_ack
        seal["source_dispatch_history"] = binding(self.root, self.history)
        supervisor.atomic_json(self.seal, seal)
        self.dispatch["strategy_evidence"] = [
            binding(self.root, self.seal),
            binding(self.root, self.history),
        ]

    def use_temp_arm_root(self):
        old_arm_root = arm.ROOT
        old_supervisor_root = supervisor.ROOT
        arm.ROOT = self.root
        supervisor.ROOT = self.root
        self.addCleanup(setattr, arm, "ROOT", old_arm_root)
        self.addCleanup(setattr, supervisor, "ROOT", old_supervisor_root)

    def test_complete_sealed_lineage_passes(self):
        result = validate_lineage(
            self.root, self.dispatch, self.ack, self.checkpoint, self.evidence
        )
        self.assertEqual(result["producer_request"], self.source)
        self.assertEqual(result["active_request"], self.active)

    def test_terminal_source_accepts_complete_bound_evidence(self):
        self.use_temp_arm_root()
        arm.validate_terminal_source(
            supervisor.load_json(self.history)["request"],
            self.source_ack,
            supervisor.load_json(self.recovery),
            self.checkpoint,
            self.evidence,
        )

    def test_repacked_unrelated_diagnostic_is_rejected(self):
        diagnostic = supervisor.load_json(self.diagnostic)
        diagnostic["request"] = {
            "request_id": "unrelated-request",
            "attempt": 99,
            "action": "other",
        }
        self.repack_diagnostic_lineage(diagnostic)
        source_dispatch = supervisor.load_json(self.history)["request"]
        self.use_temp_arm_root()
        with self.assertRaisesRegex(ValueError, "diagnostic identity"):
            arm.validate_terminal_source(
                source_dispatch,
                self.source_ack,
                supervisor.load_json(self.recovery),
                self.checkpoint,
                self.evidence,
            )
        with self.assertRaisesRegex(ValueError, "diagnostic identity"):
            validate_lineage(
                self.root, self.dispatch, self.ack, self.checkpoint, self.evidence
            )

    def test_missing_history_or_partial_checkpoint_fails_closed(self):
        original = self.history.read_bytes()
        self.history.write_bytes(b"{}\n")
        with self.assertRaisesRegex(ValueError, "strategy evidence binding mismatch"):
            validate_lineage(
                self.root, self.dispatch, self.ack, self.checkpoint, self.evidence
            )
        self.history.write_bytes(original)
        with self.checkpoint.open("rb") as handle:
            checkpoint = pickle.load(handle)
        checkpoint["results"].pop(next(iter(checkpoint["results"])))
        self.checkpoint.write_bytes(pickle.dumps(checkpoint, protocol=pickle.HIGHEST_PROTOCOL))
        with self.assertRaisesRegex(ValueError, "sealed checkpoint binding mismatch"):
            validate_lineage(
                self.root, self.dispatch, self.ack, self.checkpoint, self.evidence
            )

    def test_cross_request_prepare_is_audit_only_and_byte_preserving(self):
        old_root = retry.ROOT
        retry.ROOT = self.root
        self.addCleanup(setattr, retry, "ROOT", old_root)
        expected = dict(self.meta)
        expected["request"] = self.active
        before = self.checkpoint.read_bytes()
        result = retry.prepare_retry(
            self.checkpoint,
            self.evidence,
            self.state,
            self.active,
            {**self.ack, "status": "claimed"},
            expected,
            self.tasks,
            self.source["request_id"],
        )
        self.assertEqual(result["status"], "audit_existing_evidence")
        self.assertTrue(result["cross_request"])
        self.assertFalse(result["checkpoint_mutated"])
        self.assertEqual(self.checkpoint.read_bytes(), before)

    def test_cross_request_partial_resume_is_forbidden(self):
        old_root = retry.ROOT
        retry.ROOT = self.root
        self.addCleanup(setattr, retry, "ROOT", old_root)
        with self.checkpoint.open("rb") as handle:
            checkpoint = pickle.load(handle)
        checkpoint["results"].pop(next(iter(checkpoint["results"])))
        self.checkpoint.write_bytes(pickle.dumps(checkpoint, protocol=pickle.HIGHEST_PROTOCOL))
        expected = dict(self.meta)
        expected["request"] = self.active
        with self.assertRaisesRegex(ValueError, "partial resume is forbidden"):
            retry.prepare_retry(
                self.checkpoint,
                self.evidence,
                self.state,
                self.active,
                {**self.ack, "status": "claimed"},
                expected,
                self.tasks,
                self.source["request_id"],
            )


if __name__ == "__main__":
    unittest.main()
