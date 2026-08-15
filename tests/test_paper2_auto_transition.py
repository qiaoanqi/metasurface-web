import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline_supervisor import atomic_json
from scripts import paper2_auto_transition as transition


class Paper2AutoTransitionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.state = self.root / ".state"
        self.state.mkdir()
        self.patches = (
            patch.object(transition, "ROOT", self.root),
            patch.object(transition, "STATE", self.state),
            patch.object(transition, "DISPATCH", self.state / "dispatch_request.json"),
            patch.object(transition, "V1_AUDIT", self.state / "reference_resolution_v1_audit.json"),
            patch.object(transition, "V2_AUDIT", self.state / "reference_resolution_budget_v2_audit.json"),
        )
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temporary.cleanup()

    def dispatch(self, revision=1):
        value = {
            "request_id": "bound-request",
            "action": "joint_numerical_convergence",
            "status": "failed",
            "failure_class": "scientific",
            "terminal_failure": True,
            "attempt": 2,
            "strategy_revision": revision,
        }
        atomic_json(transition.DISPATCH, value)
        return value

    def test_nonterminal_request_is_idle(self):
        value = self.dispatch()
        value["status"] = "in_progress"
        atomic_json(transition.DISPATCH, value)
        with patch.object(transition, "run_command") as run:
            result = transition.advance_once()
        self.assertEqual(result["status"], "idle")
        run.assert_not_called()

    def test_v1_failure_freezes_and_advances_budget_v2(self):
        self.dispatch()
        checks = {
            name: True
            for name in (
                "frozen_plan_sha256_and_content",
                "checkpoint_meta_and_runtime_hashes",
                "reference_checkpoint_exact_80",
                "worker_claim_matches_independent_recomputation",
                "physics_controls_passed",
            )
        }
        atomic_json(
            transition.V1_AUDIT,
            {
                "evidence_version": "paper2-reference-resolution-audit-v1",
                "passed": False,
                "classification": "reference_spatial_budget_insufficient_order_and_grid",
                "checks": checks,
            },
        )
        with patch.object(
            transition, "run_command", side_effect=lambda script: {"script": script}
        ) as run:
            result = transition.advance_once()
        self.assertEqual(result["transition"], "reference_budget_v2")
        self.assertEqual(run.call_count, 2)

    def test_converged_v1_reference_still_advances_historical_failed_gate(self):
        self.dispatch()
        checks = {
            name: True
            for name in (
                "frozen_plan_sha256_and_content",
                "checkpoint_meta_and_runtime_hashes",
                "reference_checkpoint_exact_80",
                "worker_claim_matches_independent_recomputation",
                "physics_controls_passed",
            )
        }
        atomic_json(
            transition.V1_AUDIT,
            {
                "evidence_version": "paper2-reference-resolution-audit-v1",
                "passed": True,
                "classification": "historical_production_budget_rejected",
                "checks": checks,
            },
        )
        with patch.object(
            transition, "run_command", side_effect=lambda script: {"script": script}
        ) as run:
            result = transition.advance_once()
        self.assertEqual(result["transition"], "reference_budget_v2")
        self.assertEqual(run.call_count, 2)

    def test_v2_pass_is_bound_before_holdout_transition(self):
        dispatch = self.dispatch(revision=2)
        atomic_json(
            transition.V2_AUDIT,
            {
                "evidence_version": "paper2-reference-resolution-budget-v2-audit",
                "request": transition.request_identity(dispatch),
                "passed": True,
                "classification": "budget_v2_converged",
                "training_allowed": False,
            },
        )
        with patch.object(
            transition, "run_command", side_effect=lambda script: {"script": script}
        ) as run:
            result = transition.advance_once()
        self.assertEqual(result["transition"], "reference_resolution_holdout")
        self.assertEqual(run.call_count, 2)

    def test_v2_request_mismatch_fails_closed(self):
        self.dispatch(revision=2)
        atomic_json(
            transition.V2_AUDIT,
            {
                "evidence_version": "paper2-reference-resolution-budget-v2-audit",
                "request": {"request_id": "other", "attempt": 2},
                "passed": True,
                "classification": "budget_v2_converged",
                "training_allowed": False,
            },
        )
        with self.assertRaisesRegex(ValueError, "terminal request"):
            transition.advance_once()

    def test_v2_scientific_negative_stops_without_reselection(self):
        dispatch = self.dispatch(revision=2)
        atomic_json(
            transition.V2_AUDIT,
            {
                "evidence_version": "paper2-reference-resolution-budget-v2-audit",
                "request": transition.request_identity(dispatch),
                "passed": False,
                "classification": "budget_v2_still_insufficient",
                "training_allowed": False,
            },
        )
        with patch.object(transition, "run_command") as run:
            result = transition.advance_once()
        self.assertEqual(result["reason"], "terminal_scientific_negative")
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
