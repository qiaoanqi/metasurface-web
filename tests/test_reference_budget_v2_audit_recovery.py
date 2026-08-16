import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

import pipeline_supervisor as supervisor
from scripts import arm_reference_budget_v2_audit_recovery as arm
from scripts import prepare_reference_budget_v2_retry as retry
from scripts import run_reference_resolution_budget_v2 as budget
from scripts.reference_budget_v2_lineage import (
    binding,
    file_digest,
    validate_lineage,
    validate_source_diagnostic,
)


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
        self.source = {"request_id": "abcdef1234567890abcd", "attempt": 1}
        self.active = {"request_id": "1234567890abcdefabcd", "attempt": 1}
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
                "source_request": self.source | {"action": "joint_numerical_convergence"},
                "observation_only": True,
                "checkpoint_reuse_authorized": False,
                "scientific_outcome_authorized": False,
                "training_allowed": False,
            },
        )
        self.diagnostic = self.state / "finalization_diagnostics" / (
            f"{self.source['request_id']}-attempt1.json"
        )
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
        self.history = self.state / "dispatch_history" / (
            f"{self.source['request_id']}-attempt1.json"
        )
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
                "target_request": self.active
                | {"max_attempts": 3, "strategy_revision": 9},
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
            "max_attempts": 3,
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

    def test_same_target_attempts_two_and_three_preserve_sealed_bytes(self):
        checkpoint_before = self.checkpoint.read_bytes()
        evidence_before = self.evidence.read_bytes()
        for attempt in (2, 3):
            with self.subTest(attempt=attempt):
                dispatch = self.dispatch | {"attempt": attempt}
                ack = self.ack | {"attempt": attempt}
                result = validate_lineage(
                    self.root, dispatch, ack, self.checkpoint, self.evidence
                )
                self.assertEqual(
                    result["active_request"],
                    {"request_id": self.active["request_id"], "attempt": attempt},
                )
                self.assertEqual(self.checkpoint.read_bytes(), checkpoint_before)
                self.assertEqual(self.evidence.read_bytes(), evidence_before)

    def test_attempt_bounds_request_and_revision_fail_closed(self):
        cases = (
            (self.dispatch | {"attempt": 4}, self.ack | {"attempt": 4}),
            (
                self.dispatch | {"request_id": "different-target-request"},
                self.ack | {"request_id": "different-target-request"},
            ),
            (self.dispatch | {"strategy_revision": 10}, self.ack),
        )
        for dispatch, ack in cases:
            with self.subTest(dispatch=dispatch):
                with self.assertRaisesRegex(ValueError, "seal request identity"):
                    validate_lineage(
                        self.root, dispatch, ack, self.checkpoint, self.evidence
                    )

    def test_each_attempt_requires_exact_checkpoint_and_evidence_bytes(self):
        checkpoint_before = self.checkpoint.read_bytes()
        evidence_before = self.evidence.read_bytes()
        dispatch = self.dispatch | {"attempt": 2}
        ack = self.ack | {"attempt": 2}
        self.checkpoint.write_bytes(checkpoint_before + b"changed")
        with self.assertRaisesRegex(ValueError, "sealed checkpoint binding mismatch"):
            validate_lineage(self.root, dispatch, ack, self.checkpoint, self.evidence)
        self.checkpoint.write_bytes(checkpoint_before)
        self.evidence.write_bytes(evidence_before + b" ")
        with self.assertRaisesRegex(ValueError, "sealed worker evidence binding mismatch"):
            validate_lineage(self.root, dispatch, ack, self.checkpoint, self.evidence)

    def test_source_diagnostic_requires_exact_version_identity_and_training_lock(self):
        source = self.source | {"action": "joint_numerical_convergence"}
        valid = supervisor.load_json(self.diagnostic)
        validate_source_diagnostic(valid, source)
        invalid_payloads = []
        for key, value in (
            ("evidence_version", "wrong-version"),
            ("training_allowed", True),
        ):
            payload = dict(valid)
            payload[key] = value
            invalid_payloads.append(payload)
        for key, value in (
            ("request_id", "other-request"),
            ("attempt", 2),
            ("action", "other-action"),
        ):
            payload = dict(valid)
            payload["request"] = dict(valid["request"])
            payload["request"][key] = value
            invalid_payloads.append(payload)
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaisesRegex(ValueError, "diagnostic identity"):
                    validate_source_diagnostic(payload, source)

    def test_terminal_source_accepts_complete_bound_evidence(self):
        self.use_temp_arm_root()
        arm.validate_terminal_source(
            supervisor.load_json(self.history)["request"],
            self.source_ack,
            supervisor.load_json(self.recovery),
            self.checkpoint,
            self.evidence,
        )

    def test_terminal_source_accepts_earlier_same_request_recovery_observation(self):
        self.use_temp_arm_root()
        terminal_source = self.source | {"attempt": 2}
        with self.checkpoint.open("rb") as handle:
            checkpoint = pickle.load(handle)
        checkpoint["meta"]["request"] = terminal_source
        self.checkpoint.write_bytes(
            pickle.dumps(checkpoint, protocol=pickle.HIGHEST_PROTOCOL)
        )
        evidence = supervisor.load_json(self.evidence)
        evidence["request"] = terminal_source
        evidence["checkpoint"] = binding(self.root, self.checkpoint) | {"tasks": 96}
        supervisor.atomic_json(self.evidence, evidence)
        diagnostic = supervisor.load_json(self.diagnostic)
        diagnostic["request"] = terminal_source | {"action": "joint_numerical_convergence"}
        supervisor.atomic_json(self.diagnostic, diagnostic)
        source_ack = dict(self.source_ack)
        source_ack["attempt"] = 2
        source_ack["evidence"] = [
            binding(self.root, self.diagnostic),
            binding(self.root, self.checkpoint),
            binding(self.root, self.evidence),
        ]
        source_dispatch = {
            "request_id": terminal_source["request_id"],
            "attempt": 2,
            "action": "joint_numerical_convergence",
            "status": "failed",
            "terminal_failure": True,
            "failure_class": "permanent",
        }
        arm.validate_terminal_source(
            source_dispatch,
            source_ack,
            supervisor.load_json(self.recovery),
            self.checkpoint,
            self.evidence,
        )

    def test_post_terminal_arm_is_idempotent_and_seals_auditable_target(self):
        self.history.unlink()
        self.seal.unlink()
        supervisor_path = self.root / "pipeline_supervisor.py"
        supervisor_path.write_text("# frozen supervisor\n", encoding="ascii")
        for name in arm.STATIC_EVIDENCE_PATHS:
            path = self.root / name
            if path in {self.plan, self.recovery, self.seal}:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# frozen {name}\n", encoding="ascii")

        policy_path = self.root / "pipeline_policy.json"
        policy = {
            "dispatch": {"max_attempts": 3},
            "strategy_override": {
                "enabled": True,
                "decision": "retry_same_gate",
                "revision": 2,
                "action": "joint_numerical_convergence",
                "based_on_request_id": "feedfacefeedfacefeed",
                "instruction_append": "older strategy",
                "evidence": [{"path": "runner.py", "sha256": file_digest(self.root / "runner.py")}],
            }
        }
        supervisor.atomic_json(policy_path, policy)
        integrity_path = self.state / "pipeline_integrity.json"
        supervisor.atomic_json(
            integrity_path,
            {
                "schema_version": 1,
                "policy_sha256": file_digest(policy_path),
                "supervisor_sha256": file_digest(supervisor_path),
                "protected_assets_revision": 26,
            },
        )
        source_dispatch = {
            "request_id": self.source["request_id"],
            "attempt": 1,
            "action": "joint_numerical_convergence",
            "status": "failed",
            "terminal_failure": True,
            "failure_class": "permanent",
            "strategy_revision": 2,
            "payload": {"pool_sha256": "A" * 64},
        }
        dispatch_path = self.state / "dispatch_request.json"
        ack_path = self.state / "executor_ack.json"
        supervisor.atomic_json(dispatch_path, source_dispatch)
        supervisor.atomic_json(ack_path, self.source_ack)

        old_arm_root = arm.ROOT
        old_supervisor_root = supervisor.ROOT
        old_supervisor_state = supervisor.STATE
        arm.ROOT = self.root
        supervisor.ROOT = self.root
        supervisor.STATE = self.state
        self.addCleanup(setattr, arm, "ROOT", old_arm_root)
        self.addCleanup(setattr, supervisor, "ROOT", old_supervisor_root)
        self.addCleanup(setattr, supervisor, "STATE", old_supervisor_state)

        def crash_after_policy(stage):
            if stage == "after_policy_replace":
                raise RuntimeError("injected crash after policy replace")

        with self.assertRaisesRegex(RuntimeError, "injected crash"):
            arm.apply(
                policy_path,
                integrity_path,
                dispatch_path,
                ack_path,
                fault_injector=crash_after_policy,
            )
        self.assertNotEqual(
            supervisor.file_digest(policy_path),
            str(supervisor.load_json(integrity_path)["policy_sha256"]).upper(),
        )
        result = arm.apply(policy_path, integrity_path, dispatch_path, ack_path)
        self.assertEqual(result["status"], "already_armed")
        self.assertEqual(result["strategy_revision"], 3)
        self.assertEqual(result["integrity_revision"], 27)
        repeated = arm.apply(policy_path, integrity_path, dispatch_path, ack_path)
        self.assertEqual(repeated["status"], "already_armed")

        updated_policy = supervisor.load_json(policy_path)
        strategy = updated_policy["strategy_override"]
        seal = supervisor.load_json(self.seal)
        self.assertEqual(seal["source_request"], self.source)
        self.assertEqual(seal["target_request"]["strategy_revision"], 3)
        self.assertEqual(seal["target_request"]["max_attempts"], 3)
        self.assertEqual(strategy["revision"], 3)
        self.assertEqual(strategy["based_on_request_id"], self.source["request_id"])
        history = supervisor.load_json(self.history)
        self.assertEqual(history["request"], source_dispatch)
        self.assertEqual(history["final_ack"], self.source_ack)

        target = seal["target_request"]
        target_dispatch = {
            "request_id": target["request_id"],
            "attempt": target["attempt"],
            "action": "joint_numerical_convergence",
            "status": "in_progress",
            "strategy_revision": target["strategy_revision"],
            "max_attempts": target["max_attempts"],
            "strategy_based_on": self.source["request_id"],
            "strategy_evidence": strategy["evidence"],
        }
        target_ack = {
            "request_id": target["request_id"],
            "attempt": target["attempt"],
            "status": "claimed",
            "worker_pid": None,
        }
        lineage = validate_lineage(
            self.root,
            target_dispatch,
            target_ack,
            self.checkpoint,
            self.evidence,
            require_ready=False,
        )
        self.assertEqual(lineage["producer_request"], self.source)
        self.assertEqual(
            lineage["active_request"],
            {"request_id": target["request_id"], "attempt": target["attempt"]},
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
