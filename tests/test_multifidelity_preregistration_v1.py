import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pipeline_supervisor as supervisor
from scripts import audit_multifidelity_preregistration_v1 as auditor
from scripts import update_multifidelity_policy_v1 as updater


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "protocols" / "paper2_multifidelity_preregistration_v1.json"
COST = ROOT / "protocols" / "paper2_multifidelity_cost_basis_v1.json"


class MultifidelityPreregistrationTests(unittest.TestCase):
    def test_committed_contract_freezes_budget_and_holdout_isolation(self):
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
        budget = plan["high_fidelity_budget"]
        self.assertEqual(budget["seed_train_geometries"], 96)
        self.assertEqual(budget["active_batch_geometries"], 32)
        self.assertEqual(budget["maximum_train_geometries"], 192)
        self.assertEqual(budget["maximum_unique_geometries_with_passive_control"], 480)
        self.assertTrue(plan["created_before_holdout_results"])
        self.assertTrue(plan["holdout_isolation"]["exclude_from_active_acquisition"])
        self.assertTrue(
            plan["inverse_design"]["holdout_targets_unlocked_only_after_model_and_stopping_lock"]
        )
        self.assertTrue(plan["inverse_design"]["holdout_targets_final_evaluation_only"])
        self.assertFalse(plan["failure_and_fallback"]["full_pool_automatic_launch"])
        self.assertFalse(plan["training_allowed"])

    def test_independent_validator_rejects_budget_tampering(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            pool = root / "pool.pkl"
            holdout = root / "holdout.json"
            cost = root / "cost.json"
            plan_path = root / "plan.json"
            pool.write_bytes(b"frozen-low-fidelity-pool")
            holdout.write_text("{}", encoding="ascii")
            cost_payload = json.loads(COST.read_text(encoding="utf-8"))
            supervisor.atomic_json(cost, cost_payload)
            plan = json.loads(PLAN.read_text(encoding="utf-8"))
            plan["fidelity_roles"]["low"]["path"] = str(pool.relative_to(ROOT)).replace("\\", "/")
            plan["fidelity_roles"]["low"]["sha256"] = supervisor.file_digest(pool)
            plan["holdout_isolation"]["manifest"] = auditor.binding(holdout)
            plan["cost_basis"] = auditor.binding(cost)
            supervisor.atomic_json(plan_path, plan)
            auditor.validate_plan_payload(plan_path, cost, pool, holdout)
            plan["high_fidelity_budget"]["maximum_train_geometries"] = 193
            supervisor.atomic_json(plan_path, plan)
            with self.assertRaisesRegex(ValueError, "budget"):
                auditor.validate_plan_payload(plan_path, cost, pool, holdout)

    def test_independent_validator_rejects_early_holdout_reveal(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            pool = root / "pool.pkl"
            holdout = root / "holdout.json"
            cost = root / "cost.json"
            plan_path = root / "plan.json"
            pool.write_bytes(b"frozen-low-fidelity-pool")
            holdout.write_text("{}", encoding="ascii")
            cost_payload = json.loads(COST.read_text(encoding="utf-8"))
            supervisor.atomic_json(cost, cost_payload)
            plan = json.loads(PLAN.read_text(encoding="utf-8"))
            plan["fidelity_roles"]["low"]["path"] = str(pool.relative_to(ROOT)).replace("\\", "/")
            plan["fidelity_roles"]["low"]["sha256"] = supervisor.file_digest(pool)
            plan["holdout_isolation"]["manifest"] = auditor.binding(holdout)
            plan["cost_basis"] = auditor.binding(cost)
            plan["inverse_design"]["holdout_targets_unlocked_only_after_model_and_stopping_lock"] = False
            supervisor.atomic_json(plan_path, plan)
            with self.assertRaisesRegex(ValueError, "holdout reveal"):
                auditor.validate_plan_payload(plan_path, cost, pool, holdout)

    def test_policy_transform_replaces_automatic_full_pool_gate(self):
        before = supervisor.load_json(ROOT / "pipeline_policy.json", {})
        after = updater.build_after(before)
        supervisor.validate_workflow_contract(after)
        actions = after["workflow"]["actions"]
        names = [item["action"] for item in actions]
        self.assertLess(names.index("multifidelity_preregistration"), names.index("reference_resolution"))
        replacement = next(item for item in actions if item["action"] == "replacement_pool_generation")
        self.assertTrue(replacement["manual_only"])
        self.assertFalse(replacement["automatic_launch_authorized"])
        required = after["workflow"]["required_before_training"]
        self.assertIn("multifidelity_preregistered", required)
        self.assertIn("multifidelity_data_ready", required)
        self.assertNotIn("replacement_pool_ready", required)

    def test_manual_fallback_is_not_selected(self):
        policy = {
            "workflow": {
                "actions": [
                    {"action": "replacement_pool_generation", "gate": "replacement_pool_ready", "manual_only": True},
                    {"action": "next_scientific_gate", "gate": "next_gate"},
                ]
            }
        }
        self.assertEqual(supervisor.select_workflow_action(policy, {}), "next_scientific_gate")

    def test_preregistration_gate_fails_closed_on_training_unlock(self):
        payload = {
            "evidence_version": "paper2-multifidelity-preregistration-audit-v1",
            "passed": True,
            "classification": "multifidelity_preregistration_passed",
            "training_allowed": True,
            "independent_reproduction": True,
            "pool_sha256": "ABC",
        }
        passed, error = supervisor.verify_multifidelity_preregistration_gate(payload, {"sha256": "ABC"})
        self.assertFalse(passed)
        self.assertIn("training lock", error)

    def test_data_gate_requires_all_bound_manifests(self):
        payload = {
            "evidence_version": "paper2-multifidelity-data-audit-v1",
            "passed": True,
            "classification": "multifidelity_data_passed",
            "training_allowed": False,
            "low_fidelity_pool_sha256": "ABC",
            "checks": {
                "reference_protocol_independently_approved": True,
                "seed_validation_test_geometry_disjoint": True,
                "p_s_pairs_complete": True,
                "holdout_geometry_excluded": True,
                "pointwise_conservation_passed": True,
                "protected_files_unchanged": True,
            },
        }
        passed, error = supervisor.verify_multifidelity_data_ready_gate(payload, {"sha256": "ABC"})
        self.assertFalse(passed)
        self.assertIn("data binding", error)


if __name__ == "__main__":
    unittest.main()
