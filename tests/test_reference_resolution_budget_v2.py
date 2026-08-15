import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import freeze_reference_budget_v2 as freeze
from scripts import run_reference_resolution_budget_v2 as budget
from scripts import audit_reference_resolution_budget_v2 as audit
from scripts import advance_reference_budget_v2_strategy as advance
from scripts import advance_reference_holdout_strategy as advance_holdout


def cases():
    return [
        {"L": 120.0 + i, "W": 100.0 + i, "H": 300.0 + i, "P": 400.0 + i}
        for i in range(8)
    ]


class BudgetV2PlanTests(unittest.TestCase):
    def test_plan_is_fixed_factor_not_diagonal_only(self):
        plan = {
            "pool_sha256": "POOL",
            "selection": cases(),
            "source_v1_evidence": {"path": "v1.json", "sha256": "E"},
            "source_v1_plan": {"path": "plan.json", "sha256": freeze.V1_PLAN_SHA256},
        }
        self.assertEqual(freeze.EXTRA_CONFIGS, ((450, 512), (365, 768), (450, 768)))
        self.assertEqual(freeze.EXPECTED_NEW_TASKS, 96)

    def test_task_identity_contains_both_spectral_steps(self):
        tasks = budget.build_tasks(cases())
        self.assertEqual(len(tasks), 96)
        self.assertEqual({task["step_nm"] for task in tasks}, {0.5, 1.0})
        self.assertEqual({(task["requested_nG"], task["Nxy"]) for task in tasks}, set(freeze.EXTRA_CONFIGS))
        self.assertEqual(len({task["id"] for task in tasks}), 96)

    def test_every_spatial_axis_is_compared_on_both_grids(self):
        runner_specs = budget.spatial_axis_specs()
        auditor_specs = audit.spatial_axis_specs()
        self.assertEqual(runner_specs, auditor_specs)
        self.assertEqual(len(runner_specs), 6)
        for axis in ("order", "grid", "corner"):
            self.assertEqual(
                {step for name, _config, step in runner_specs if name.startswith(axis)},
                {0.5, 1.0},
            )

    def test_only_registered_provisional_plan_can_be_migrated(self):
        provisional = {
            "evidence_version": freeze.VERSION,
            "source_v1_evidence": {"path": ".state/joint_convergence_v1_1.json"},
            "expected_new_tasks": 96,
        }
        self.assertTrue(
            freeze.is_known_provisional_plan(provisional, freeze.PROVISIONAL_PLAN_SHA256)
        )
        self.assertFalse(freeze.is_known_provisional_plan(provisional, "0" * 64))
        provisional["source_v1_audit"] = {"path": "unexpected"}
        self.assertFalse(
            freeze.is_known_provisional_plan(provisional, freeze.PROVISIONAL_PLAN_SHA256)
        )

    def test_invalid_spectrum_fails_closed(self):
        tasks = budget.build_tasks(cases())
        results = {
            task["id"]: {"status": "failed", "error": "synthetic"}
            for task in tasks
        }
        result = budget.validate_results(results, tasks)
        self.assertFalse(result["passed"])
        self.assertEqual(len(result["failures"]), 96)


def valid_result(task):
    samples = len(task["wavelength_nm"])
    return {
        **{key: task[key] for key in task},
        "status": "ok",
        "R": np.full(samples, 0.5),
        "T": np.full(samples, 0.5),
        "time_s": 1.0,
    }


class BudgetV2IntegrityTests(unittest.TestCase):
    def test_task_id_tamper_fails_in_runner_and_auditor(self):
        task = budget.build_tasks(cases())[0]
        result = valid_result(task)
        result["id"] = "wrong-id"
        with self.assertRaises(ValueError):
            budget.validate_result(result, task)
        with self.assertRaises(ValueError):
            audit.validate_result(result, task)

    def test_spectral_array_tamper_fails_closed(self):
        task = budget.build_tasks(cases())[0]
        result = valid_result(task)
        result["R"][3] = np.nan
        with self.assertRaises(ValueError):
            audit.validate_result(result, task)

    def test_runtime_tamper_fails_closed(self):
        task = budget.build_tasks(cases())[0]
        result = valid_result(task)
        result["time_s"] = -1
        with self.assertRaises(ValueError):
            audit.validate_result(result, task)

    def test_threshold_tamper_fails_plan_validation(self):
        plan = {
            "schema_version": 1,
            "evidence_version": freeze.VERSION,
            "plan_valid": True,
            "source_failed_action": "joint_numerical_convergence",
            "pool_sha256": "A" * 64,
            "selection": cases(),
            "extra_configs": [list(x) for x in freeze.EXTRA_CONFIGS],
            "base_config": [365, 512],
            "steps_nm": [1.0, 0.5],
            "expected_new_tasks": 96,
            "thresholds": {
                "mean_joint_dE00_lt": 1.16,
                "all_joint_dE00_lt": 2.3,
                "pointwise_conservation_lte": 1e-6,
            },
        }
        with self.assertRaises(ValueError):
            audit.validate_plan(plan, audit.ROOT / "pipeline_policy.json")

    def test_runtime_hash_tamper_fails_closed(self):
        expected = {path: audit.file_digest(audit.ROOT / path) for path in audit.RUNTIME_PATHS}
        self.assertTrue(audit.runtime_hashes_match(expected))
        expected[audit.RUNTIME_PATHS[0]] = "0" * 64
        self.assertFalse(audit.runtime_hashes_match(expected))

    def test_forged_passed_claim_fails_closed(self):
        self.assertFalse(audit.worker_claim_matches(True, False))
        self.assertFalse(audit.worker_claim_matches(False, True))
        self.assertTrue(audit.worker_claim_matches(False, False))

    def test_strategy_advances_exact_failed_request_once(self):
        policy = {"strategy_override": {"revision": 1}}
        dispatch = {
            "request_id": "failed-request",
            "action": "joint_numerical_convergence",
            "status": "failed",
            "failure_class": "scientific",
            "terminal_failure": True,
            "strategy_revision": 1,
        }
        strategy = advance.build_strategy(
            policy, dispatch, [{"path": "evidence.json", "sha256": "A" * 64}]
        )
        self.assertEqual(strategy["revision"], 2)
        self.assertEqual(strategy["based_on_request_id"], "failed-request")
        self.assertEqual(strategy["decision"], "retry_same_gate")
        self.assertEqual(strategy["action"], dispatch["action"])
        self.assertIn("failed with failure_class=scientific", strategy["instruction_append"])

    def test_strategy_rejects_nonterminal_failure(self):
        dispatch = {
            "request_id": "failed-request",
            "action": "joint_numerical_convergence",
            "status": "failed",
            "failure_class": "scientific",
            "terminal_failure": False,
        }
        with self.assertRaises(ValueError):
            advance.build_strategy({}, dispatch, [])

    def test_strategy_application_is_idempotent(self):
        original_root = advance.ROOT
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            advance.ROOT = root
            try:
                for name in advance.EVIDENCE_PATHS:
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("evidence\n", encoding="ascii")
                checks = {
                    key: True for key in (
                        "frozen_plan_sha256_and_content",
                        "checkpoint_meta_and_runtime_hashes",
                        "reference_checkpoint_exact_80",
                        "worker_claim_matches_independent_recomputation",
                        "physics_controls_passed",
                    )
                }
                v1_audit_path = root / advance.EVIDENCE_PATHS[0]
                advance.atomic_json(v1_audit_path, {
                    "evidence_version": advance.V1_AUDIT_VERSION,
                    "passed": False,
                    "classification": "reference_spatial_budget_insufficient_order_and_grid",
                    "checks": checks,
                })
                v2_plan_path = root / advance.EVIDENCE_PATHS[1]
                advance.atomic_json(v2_plan_path, {
                    "evidence_version": advance.V2_PLAN_VERSION,
                    "plan_valid": True,
                    "source_failed_action": advance.ACTION,
                    "source_v1_audit": advance.binding(v1_audit_path),
                    "expected_new_tasks": 96,
                    "thresholds": {
                        "mean_joint_dE00_lt": 1.15,
                        "all_joint_dE00_lt": 2.3,
                        "pointwise_conservation_lte": 1e-6,
                    },
                })
                supervisor = root / "pipeline_supervisor.py"
                supervisor.write_text("# frozen\n", encoding="ascii")
                policy_path = root / "pipeline_policy.json"
                advance.atomic_json(policy_path, {"strategy_override": {"revision": 1}})
                integrity_path = root / ".state/pipeline_integrity.json"
                advance.atomic_json(integrity_path, {
                    "schema_version": 1,
                    "policy_sha256": advance.file_digest(policy_path),
                    "supervisor_sha256": advance.file_digest(supervisor),
                    "protected_assets_revision": 14,
                    "note": "test",
                })
                dispatch_path = root / ".state/dispatch_request.json"
                advance.atomic_json(dispatch_path, {
                    "request_id": "failed-request",
                    "action": advance.ACTION,
                    "status": "failed",
                    "failure_class": "scientific",
                    "terminal_failure": True,
                    "strategy_revision": 1,
                })
                first = advance.apply_strategy(policy_path, integrity_path, dispatch_path)
                second = advance.apply_strategy(policy_path, integrity_path, dispatch_path)
                self.assertEqual(first["status"], "updated")
                self.assertEqual(second["status"], "already_applied")
                self.assertEqual(first["strategy_revision"], 2)
                self.assertEqual(second["strategy_revision"], 2)
                self.assertEqual(first["integrity_revision"], 15)
                self.assertEqual(second["integrity_revision"], 15)
            finally:
                advance.ROOT = original_root


class HoldoutTransitionTests(unittest.TestCase):
    def failed_v2_dispatch(self):
        return {
            "request_id": "failed-budget-v2-request",
            "action": "joint_numerical_convergence",
            "status": "failed",
            "failure_class": "scientific",
            "terminal_failure": True,
            "attempt": 1,
            "strategy_revision": 2,
            "payload": {"pool_sha256": "A" * 64},
        }

    def test_passed_v2_builds_strictly_new_holdout_transition(self):
        dispatch = self.failed_v2_dispatch()
        strategy = advance_holdout.build_strategy(
            {"strategy_override": {"revision": 4}},
            dispatch,
            [{"path": "v2-audit.json", "sha256": "B" * 64}],
        )
        self.assertEqual(strategy["decision"], "transition_after_failure")
        self.assertEqual(strategy["from_action"], dispatch["action"])
        self.assertEqual(strategy["action"], "reference_resolution")
        self.assertEqual(strategy["revision"], 5)
        self.assertEqual(strategy["based_on_request_id"], dispatch["request_id"])
        self.assertIn("launch_reference_resolution_holdout.py", strategy["instruction_append"])

    def test_holdout_transition_rejects_live_request(self):
        dispatch = self.failed_v2_dispatch()
        dispatch["status"] = "in_progress"
        with self.assertRaisesRegex(ValueError, "terminal failed joint"):
            advance_holdout.build_strategy({}, dispatch, [])

    def test_holdout_transition_requires_budget_v2_revision(self):
        dispatch = self.failed_v2_dispatch()
        dispatch["strategy_revision"] = 1
        with self.assertRaisesRegex(ValueError, "budget-v2 strategy revision"):
            advance_holdout.build_strategy({}, dispatch, [])

    def test_passed_v2_evidence_chain_applies_transition_once(self):
        original_root = advance_holdout.ROOT
        original_selection_hashes = (
            advance_holdout.EXISTING_SELECTION_SHA256,
            advance_holdout.NEW_SELECTION_SHA256,
            advance_holdout.COMBINED_SELECTION_SHA256,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            advance_holdout.ROOT = root
            try:
                for name in advance_holdout.EVIDENCE_PATHS:
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"frozen evidence\n")

                pool_sha = "A" * 64
                base_checkpoint = root / ".state/reference_resolution_v1_checkpoint.pkl"
                base_checkpoint.write_bytes(b"frozen base checkpoint")
                checkpoint = root / ".state/reference_resolution_budget_v2_checkpoint.pkl"
                checkpoint.write_bytes(b"frozen checkpoint")
                plan_path = root / ".state/reference_resolution_budget_v2_plan.json"
                advance_holdout.atomic_json(plan_path, {
                    "schema_version": 1,
                    "evidence_version": advance_holdout.V2_PLAN_VERSION,
                    "plan_valid": True,
                    "source_failed_action": advance_holdout.FROM_ACTION,
                    "pool_sha256": pool_sha,
                    "thresholds": advance_holdout.THRESHOLDS,
                    "expected_new_tasks": 96,
                    "source_v1_checkpoint": advance_holdout.binding(base_checkpoint),
                })
                evidence_path = root / ".state/reference_resolution_budget_v2.json"
                advance_holdout.atomic_json(evidence_path, {
                    "evidence_version": advance_holdout.V2_EVIDENCE_VERSION,
                    "request": {
                        "request_id": "failed-budget-v2-request",
                        "attempt": 1,
                    },
                    "passed": True,
                    "training_allowed": False,
                    "pool_sha256": pool_sha,
                    "thresholds": advance_holdout.THRESHOLDS,
                    "plan": advance_holdout.binding(plan_path),
                    "checkpoint": advance_holdout.binding(checkpoint),
                })
                audit_path = root / ".state/reference_resolution_budget_v2_audit.json"
                advance_holdout.atomic_json(audit_path, {
                    "evidence_version": advance_holdout.V2_AUDIT_VERSION,
                    "request": {
                        "request_id": "failed-budget-v2-request",
                        "attempt": 1,
                    },
                    "passed": True,
                    "classification": "budget_v2_converged",
                    "training_allowed": False,
                    "pool_sha256": pool_sha,
                    "thresholds": advance_holdout.THRESHOLDS,
                    "checks": {
                        name: True for name in advance_holdout.REQUIRED_AUDIT_CHECKS
                    },
                    "plan": advance_holdout.binding(plan_path),
                    "checkpoint": advance_holdout.binding(checkpoint) | {"tasks": 96},
                })
                archived_plan_path = root / ".state/reference_resolution_holdout_v1_plan.json"
                holdout_plan_path = root / ".state/reference_resolution_holdout_v2_plan.json"
                runtime_paths = (
                    "scripts/freeze_reference_holdout_plan.py",
                    "scripts/reference_protocol_selection.py",
                    "scripts/launch_reference_resolution_holdout.py",
                    "scripts/run_reference_resolution_holdout.py",
                    "scripts/audit_reference_resolution_holdout.py",
                )
                protocols = [
                    {
                        "requested_nG": nG,
                        "Nxy": nxy,
                        "step_nm": step,
                        "role": role,
                    }
                    for (nG, nxy, step), role in zip(
                        (
                            (365, 512, 0.5),
                            (365, 768, 0.5),
                            (450, 512, 0.5),
                            (450, 768, 1.0),
                            (450, 768, 0.5),
                        ),
                        ("base", "order", "grid", "spectral", "final_reference"),
                    )
                ]
                existing_cases = cases()
                new_cases = [
                    {"L": 150.0 + i, "W": 90.0 + i, "H": 250.0 + i, "P": 420.0 + i}
                    for i in range(24)
                ]
                combined_cases = existing_cases + new_cases
                selection_hashes = tuple(
                    advance_holdout.selection_sha256(items)
                    for items in (existing_cases, new_cases, combined_cases)
                )
                (
                    advance_holdout.EXISTING_SELECTION_SHA256,
                    advance_holdout.NEW_SELECTION_SHA256,
                    advance_holdout.COMBINED_SELECTION_SHA256,
                ) = selection_hashes
                advance_holdout.atomic_json(holdout_plan_path, {
                    "schema_version": 1,
                    "evidence_version": advance_holdout.HOLDOUT_PLAN_VERSION,
                    "plan_valid": True,
                    "created_before_holdout_results": True,
                    "candidate_frozen_on_initial_eight_only": True,
                    "holdout_cannot_reselect_candidate": True,
                    "source_gate": "independent_reference_resolution_budget_v2_passed",
                    "source_v2_plan": advance_holdout.binding(plan_path),
                    "source_v2_checkpoint": advance_holdout.binding(checkpoint),
                    "source_v2_worker_evidence": advance_holdout.binding(evidence_path),
                    "source_v2_independent_audit": advance_holdout.binding(audit_path),
                    "source_base_checkpoint": advance_holdout.binding(base_checkpoint),
                    "archived_v1_holdout_plan": advance_holdout.binding(archived_plan_path),
                    "existing_cases": existing_cases,
                    "new_cases": new_cases,
                    "combined_cases": combined_cases,
                    "existing_selection_sha256": selection_hashes[0],
                    "new_selection_sha256": selection_hashes[1],
                    "combined_selection_sha256": selection_hashes[2],
                    "existing_case_count": 8,
                    "new_case_count": 24,
                    "combined_case_count": 32,
                    "polarizations": ["p", "s"],
                    "primary_gate_population": "24_new_holdout_geometries_only",
                    "combined_32_population_scope": "supplemental_reporting_only",
                    "task_protocols": protocols,
                    "expected_source_tasks": 80,
                    "expected_new_tasks": 240,
                    "expected_combined_tasks": 320,
                    "pool": {"sha256": pool_sha},
                    "thresholds": advance_holdout.THRESHOLDS,
                    "frozen_candidate": {
                        **protocols[0],
                        "wavelength_step_nm": protocols[0]["step_nm"],
                        "passed": True,
                    },
                    "final_reference": {
                        "requested_nG": 450,
                        "Nxy": 768,
                        "wavelength_step_nm": 0.5,
                    },
                    "runtime_hashes": {
                        name: advance_holdout.file_digest(root / name)
                        for name in runtime_paths
                    },
                })

                supervisor_path = root / "pipeline_supervisor.py"
                supervisor_path.write_text("# frozen supervisor\n", encoding="ascii")
                policy_path = root / "pipeline_policy.json"
                advance_holdout.atomic_json(
                    policy_path, {"strategy_override": {"revision": 2}}
                )
                integrity_path = root / ".state/pipeline_integrity.json"
                advance_holdout.atomic_json(integrity_path, {
                    "schema_version": 1,
                    "policy_sha256": advance_holdout.file_digest(policy_path),
                    "supervisor_sha256": advance_holdout.file_digest(supervisor_path),
                    "protected_assets_revision": 14,
                    "note": "test",
                })
                dispatch_path = root / ".state/dispatch_request.json"
                advance_holdout.atomic_json(dispatch_path, self.failed_v2_dispatch())

                first = advance_holdout.apply_strategy(
                    policy_path, integrity_path, dispatch_path
                )
                second = advance_holdout.apply_strategy(
                    policy_path, integrity_path, dispatch_path
                )
                strategy = advance_holdout.load_json(policy_path)["strategy_override"]
                self.assertEqual(first["status"], "updated")
                self.assertEqual(second["status"], "already_applied")
                self.assertEqual(first["strategy_revision"], 3)
                self.assertEqual(second["strategy_revision"], 3)
                self.assertEqual(first["integrity_revision"], 15)
                self.assertEqual(second["integrity_revision"], 15)
                self.assertEqual(strategy["decision"], "transition_after_failure")
                self.assertEqual(strategy["action"], "reference_resolution")
                self.assertEqual(
                    strategy["based_on_request_id"], "failed-budget-v2-request"
                )

                tampered = advance_holdout.load_json(audit_path)
                tampered["request"]["attempt"] = 2
                advance_holdout.atomic_json(audit_path, tampered)
                current_policy = advance_holdout.load_json(policy_path)
                current_policy["strategy_override"] = {"revision": 2}
                advance_holdout.atomic_json(policy_path, current_policy)
                current_lock = advance_holdout.load_json(integrity_path)
                current_lock["policy_sha256"] = advance_holdout.file_digest(policy_path)
                advance_holdout.atomic_json(integrity_path, current_lock)
                with self.assertRaisesRegex(ValueError, "approved diagnostic pass"):
                    advance_holdout.apply_strategy(
                        policy_path, integrity_path, dispatch_path
                    )
            finally:
                advance_holdout.ROOT = original_root
                (
                    advance_holdout.EXISTING_SELECTION_SHA256,
                    advance_holdout.NEW_SELECTION_SHA256,
                    advance_holdout.COMBINED_SELECTION_SHA256,
                ) = original_selection_hashes


if __name__ == "__main__":
    unittest.main()
