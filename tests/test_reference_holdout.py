import argparse
import copy
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pipeline_supervisor import file_digest
from scripts import audit_reference_resolution_holdout as auditor
from scripts import freeze_reference_holdout_plan as freezer
from scripts import run_reference_resolution_budget_v2 as budget
from scripts import run_reference_resolution_holdout as holdout
from scripts.reference_protocol_selection import CONFIGS, STEPS, evaluate_protocols


def geometry(index: int) -> dict:
    return {
        "L": 120.0 + index,
        "W": 90.0 + index / 2,
        "H": 250.0 + index,
        "P": 400.0 + index,
    }


def candidate(nG=365, Nxy=512, step=0.5) -> dict:
    return {
        "config_name": "BASE",
        "requested_nG": nG,
        "Nxy": Nxy,
        "wavelength_step_nm": step,
        "mean_task_seconds_estimate": 1.0,
        "estimated_wall_hours_16_workers_6000": 0.1,
        "passed": True,
    }


def task_plan(frozen=None) -> dict:
    frozen = frozen or candidate()
    manifest = freezer.protocol_manifest(frozen)
    cases = [geometry(index) for index in range(32)]
    return {
        "combined_cases": cases,
        "new_cases": cases[8:],
        "task_protocols": manifest,
        "frozen_candidate": frozen,
        "expected_source_tasks": 8 * 2 * len(manifest),
        "expected_new_tasks": 24 * 2 * len(manifest),
        "expected_combined_tasks": 32 * 2 * len(manifest),
    }


def source_fixture():
    selected = [geometry(index) for index in range(8)]
    results = {}
    for index in range(8):
        for pol in budget.POLS:
            for config in CONFIGS.values():
                for step in STEPS:
                    wavelength = budget.v1.WL_HALF_NM if step == 0.5 else budget.v1.WL_1NM
                    runtime = 1.0
                    if config == CONFIGS["BASE"] and step == 1.0:
                        runtime = 0.1
                    results[freezer.source_result_id(index, pol, config, step)] = {
                        "wavelength_nm": wavelength,
                        "R": np.full(wavelength.size, 0.4),
                        "T": np.full(wavelength.size, 0.6),
                        "time_s": runtime,
                    }
    return selected, results


class ReferenceHoldoutTests(unittest.TestCase):
    def test_dispatch_identity_is_frozen_and_strict(self):
        self.assertEqual(
            holdout.request_identity("request-1", 2),
            {"request_id": "request-1", "attempt": 2},
        )
        self.assertEqual(
            auditor.dispatch_identity(
                {
                    "action": "reference_resolution",
                    "status": "in_progress",
                    "request_id": "request-1",
                    "attempt": 2,
                }
            ),
            {"request_id": "request-1", "attempt": 2},
        )
        with self.assertRaises(ValueError):
            holdout.request_identity("", 1)
        with self.assertRaises(ValueError):
            auditor.dispatch_identity(
                {
                    "action": "reference_resolution",
                    "status": "failed",
                    "request_id": "request-1",
                    "attempt": 2,
                }
            )

    def test_checkpoint_identity_survives_retry_but_run_identity_does_not(self):
        first = holdout.request_identity("request-1", 1)
        second = holdout.request_identity("request-1", 2)
        self.assertNotEqual(first, second)
        self.assertEqual(
            holdout.checkpoint_request_identity(first),
            holdout.checkpoint_request_identity(second),
        )
        self.assertEqual(
            auditor.checkpoint_request_identity(first),
            auditor.checkpoint_request_identity(second),
        )

    def test_worker_evidence_reuses_older_attempt_without_rebinding_provenance(self):
        stored = {
            "evidence_version": holdout.VERSION,
            "request": {"request_id": "request-1", "attempt": 1},
            "passed": True,
        }
        current = copy.deepcopy(stored)
        current["request"]["attempt"] = 2
        self.assertTrue(holdout.retry_equivalent_evidence(stored, current))
        changed = copy.deepcopy(current)
        changed["passed"] = False
        self.assertFalse(holdout.retry_equivalent_evidence(stored, changed))
        newer = copy.deepcopy(stored)
        newer["request"]["attempt"] = 3
        self.assertFalse(holdout.retry_equivalent_evidence(newer, current))
        other = copy.deepcopy(current)
        other["request"]["request_id"] = "request-2"
        self.assertFalse(holdout.retry_equivalent_evidence(stored, other))
        self.assertEqual(
            auditor.reusable_worker_request(stored, current["request"]),
            stored["request"],
        )
        with self.assertRaises(ValueError):
            auditor.reusable_worker_request(newer, current["request"])

    def test_audit_retry_only_replaces_prior_execution_failure(self):
        failed = {
            "request": {"request_id": "request-1", "attempt": 1},
            "passed": False,
            "classification": "execution_integrity_failure",
        }
        passed = {
            "request": {"request_id": "request-1", "attempt": 1},
            "passed": True,
            "classification": "reference_holdout_passed",
        }
        self.assertTrue(auditor.retry_replaces_execution_failure(failed, passed))
        scientific = copy.deepcopy(failed)
        scientific["classification"] = "production_candidate_holdout_negative"
        self.assertFalse(auditor.retry_replaces_execution_failure(scientific, passed))

    def test_transient_audit_exception_does_not_create_canonical_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "audit.json"
            with (
                patch.object(auditor, "ROOT", root),
                patch.object(auditor, "build_audit", side_effect=OSError("temporary read lock")),
                patch.object(auditor.sys, "argv", ["audit", "--output", "audit.json"]),
            ):
                self.assertEqual(auditor.main(), 2)
            self.assertFalse(output.exists())

    def test_scientific_negative_audit_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "audit.json"
            negative = {
                "request": {"request_id": "request-1", "attempt": 1},
                "passed": False,
                "classification": "production_candidate_holdout_negative",
            }
            auditor.write_retry_safe_audit(output, negative)
            replacement = copy.deepcopy(negative)
            replacement["passed"] = True
            replacement["classification"] = "reference_holdout_passed"
            with self.assertRaisesRegex(ValueError, "differs"):
                auditor.write_retry_safe_audit(output, replacement)

    def test_second_attempt_resumes_partial_checkpoint_without_recomputing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "plan.json").write_text("{}\n", encoding="ascii")
            plan = {
                "task_protocols": [],
                "runtime_hashes": {},
                **{
                    name: {"path": f"{name}.bin", "sha256": name.upper()}
                    for name in (
                        "source_v2_plan",
                        "source_v2_checkpoint",
                        "source_v2_worker_evidence",
                        "source_v2_independent_audit",
                        "source_base_checkpoint",
                    )
                },
            }
            tasks = [
                {
                    "id": identifier,
                    "geometry_index": index,
                    "pol": "p",
                    "requested_nG": 450,
                    "Nxy": 768,
                    "step_nm": 0.5,
                }
                for index, identifier in enumerate(("task-a", "task-b"))
            ]
            first_pending = []
            second_pending = []

            class InterruptingPool:
                def __init__(self, _jobs):
                    pass

                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return False

                def imap_unordered(self, _func, pending, chunksize=1):
                    first_pending.extend(task["id"] for task in pending)
                    yield {"id": "task-a"}
                    raise RuntimeError("simulated interruption")

            class CompletingPool(InterruptingPool):
                def imap_unordered(self, _func, pending, chunksize=1):
                    second_pending.extend(task["id"] for task in pending)
                    for task in pending:
                        yield {"id": task["id"]}

            def summary(_plan, _sources, _results, _plan_path, _checkpoint_path, request):
                return {"request": request, "passed": True}

            def args(attempt):
                return argparse.Namespace(
                    plan="plan.json",
                    checkpoint="checkpoint.pkl",
                    evidence="evidence.json",
                    lock="holdout.lock",
                    n_jobs=1,
                    request_id="stable-request",
                    attempt=attempt,
                )

            patches = (
                patch.object(holdout, "ROOT", root),
                patch.object(holdout, "load_plan", return_value=plan),
                patch.object(holdout, "load_source_results", return_value={}),
                patch.object(holdout, "build_new_tasks", return_value=tasks),
                patch.object(holdout, "validate_results", return_value={"passed": True}),
                patch.object(holdout, "summarize", side_effect=summary),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
                with patch.object(holdout, "Pool", InterruptingPool):
                    with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                        holdout.run(args(1))
                with patch.object(holdout, "Pool", CompletingPool):
                    result = holdout.run(args(2))

            with (root / "checkpoint.pkl").open("rb") as handle:
                checkpoint = pickle.load(handle)
            self.assertEqual(first_pending, ["task-a", "task-b"])
            self.assertEqual(second_pending, ["task-b"])
            self.assertEqual(checkpoint["meta"]["request"], {"request_id": "stable-request"})
            self.assertEqual(set(checkpoint["results"]), {"task-a", "task-b"})
            self.assertEqual(result["request"], {"request_id": "stable-request", "attempt": 2})

    def test_archived_selection_is_exact_and_disjoint(self):
        path = freezer.ROOT / ".state/reference_resolution_holdout_v1_plan.json"
        plan = freezer.load_archived_plan(path)
        self.assertEqual(file_digest(path), freezer.ARCHIVED_PLAN_SHA256)
        self.assertEqual(freezer.selection_sha256(plan["existing_cases"]), freezer.ARCHIVED_EXISTING_SHA256)
        self.assertEqual(freezer.selection_sha256(plan["new_cases"]), freezer.ARCHIVED_NEW_SHA256)
        existing = {tuple(item[key] for key in ("L", "W", "H", "P")) for item in plan["existing_cases"]}
        new = {tuple(item[key] for key in ("L", "W", "H", "P")) for item in plan["new_cases"]}
        self.assertFalse(existing & new)

    def test_minimum_holdout_matrix_has_exact_240_tasks(self):
        plan = task_plan(candidate(365, 512, 0.5))
        tasks = holdout.build_new_tasks(plan)
        self.assertEqual(len(tasks), 240)
        self.assertEqual(len({task["id"] for task in tasks}), 240)
        self.assertEqual({task["step_nm"] for task in tasks}, {0.5, 1.0})

    def test_candidate_outside_minimum_matrix_is_explicitly_added(self):
        plan = task_plan(candidate(365, 512, 1.0))
        self.assertEqual(len(plan["task_protocols"]), 6)
        self.assertEqual(plan["task_protocols"][-1]["role"], "frozen_production_candidate")
        self.assertEqual(len(holdout.build_new_tasks(plan)), 288)

    def test_candidate_is_selected_only_on_initial_eight(self):
        selected, results = source_fixture()
        evaluation = evaluate_protocols(selected, results, freezer.source_result_id)
        self.assertEqual(evaluation["selection_population"], "initial_eight_cases_only")
        self.assertFalse(evaluation["holdout_used_for_selection"])
        chosen = evaluation["lowest_cost_passing_protocol"]
        self.assertEqual((chosen["requested_nG"], chosen["Nxy"], chosen["wavelength_step_nm"]), (365, 512, 1.0))
        with self.assertRaisesRegex(ValueError, "initial eight"):
            evaluate_protocols(selected + [geometry(8)], results, freezer.source_result_id)

    def test_classification_taxonomy_is_fail_closed(self):
        passed = {name: {"passed": True} for name in holdout.COMPARISON_NAMES}
        self.assertEqual(holdout.classify(False, {}), "execution_integrity_failure")
        failed_budget = copy.deepcopy(passed)
        failed_budget[holdout.COMPARISON_NAMES[0]]["passed"] = False
        self.assertEqual(holdout.classify(True, failed_budget), "reference_budget_insufficient")
        failed_candidate = copy.deepcopy(passed)
        failed_candidate[holdout.COMPARISON_NAMES[4]]["passed"] = False
        self.assertEqual(
            holdout.classify(True, failed_candidate), "production_candidate_holdout_negative"
        )
        self.assertEqual(holdout.classify(True, passed), "reference_holdout_passed")

    def test_auditor_does_not_call_worker_summary_or_selector(self):
        source = Path(auditor.__file__).read_text(encoding="utf-8")
        self.assertNotIn("holdout.summarize", source)
        self.assertNotIn("evaluate_protocols(", source)

    def test_auditor_rejects_task_identity_grid_and_runtime_tampering(self):
        plan = task_plan()
        task = auditor.build_tasks(plan, range(8, 9))[0]
        samples = len(task["wavelength_nm"])
        result = {
            **task,
            "status": "ok",
            "R": np.full(samples, 0.4),
            "T": np.full(samples, 0.6),
            "time_s": 1.0,
        }
        auditor.validate_result(result, task)
        for name, value in (
            ("id", "tampered"),
            ("wavelength_nm", np.asarray(task["wavelength_nm"]) + 0.1),
            ("time_s", -1.0),
        ):
            changed = copy.deepcopy(result)
            changed[name] = value
            with self.assertRaises(ValueError):
                auditor.validate_result(changed, task)

    def test_auditor_rejects_threshold_tampering(self):
        archived = json.loads(
            (freezer.ROOT / ".state/reference_resolution_holdout_v1_plan.json").read_text(
                encoding="utf-8"
            )
        )
        frozen = candidate()
        manifest = freezer.protocol_manifest(frozen)
        plan = {
            "schema_version": 1,
            "evidence_version": freezer.VERSION,
            "plan_valid": True,
            "created_before_holdout_results": True,
            "candidate_frozen_on_initial_eight_only": True,
            "holdout_cannot_reselect_candidate": True,
            "existing_cases": archived["existing_cases"],
            "new_cases": archived["new_cases"],
            "combined_cases": archived["combined_cases"],
            "frozen_candidate": frozen,
            "task_protocols": manifest,
            "expected_source_tasks": 8 * 2 * len(manifest),
            "expected_new_tasks": 24 * 2 * len(manifest),
            "expected_combined_tasks": 32 * 2 * len(manifest),
            "thresholds": {
                "mean_joint_dE00_lt": 1.16,
                "all_joint_dE00_lt": 2.3,
                "pointwise_conservation_lte": 1e-6,
            },
            "runtime_hashes": {
                path: file_digest(freezer.ROOT / path) for path in freezer.RUNTIME_PATHS
            },
        }
        with self.assertRaisesRegex(ValueError, "thresholds changed"):
            auditor.validate_plan(plan, Path(auditor.__file__))


class V2SourceBindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.plan = self.root / "plan.json"
        self.checkpoint = self.root / "checkpoint.pkl"
        self.evidence = self.root / "evidence.json"
        self.audit = self.root / "audit.json"
        self.plan.write_text(json.dumps({"pool_sha256": "P", "thresholds": {}}), encoding="ascii")
        self.checkpoint.write_bytes(b"checkpoint")
        evidence = {
            "evidence_version": budget.VERSION,
            "request": {"request_id": "v2-request", "attempt": 1},
            "passed": True,
            "training_allowed": False,
            "pool_sha256": "P",
            "thresholds": {},
            "plan": {"path": "plan.json", "sha256": file_digest(self.plan)},
            "checkpoint": {"path": "checkpoint.pkl", "sha256": file_digest(self.checkpoint)},
        }
        self.evidence.write_text(json.dumps(evidence), encoding="ascii")
        checks = {name: True for name in freezer.REQUIRED_V2_AUDIT_CHECKS}
        audit = {
            "evidence_version": freezer.v2_audit.VERSION,
            "request": {"request_id": "v2-request", "attempt": 1},
            "passed": True,
            "classification": "budget_v2_converged",
            "training_allowed": False,
            "pool_sha256": "P",
            "thresholds": {},
            "checks": checks,
            "worker_claim": {"passed": True, "matches_independent_recomputation": True},
            "plan": {"path": "plan.json", "sha256": file_digest(self.plan)},
            "checkpoint": {
                "path": "checkpoint.pkl",
                "sha256": file_digest(self.checkpoint),
                "tasks": budget.EXPECTED_TASKS,
            },
        }
        self.audit.write_text(json.dumps(audit), encoding="ascii")

    def tearDown(self):
        self.tmp.cleanup()

    def validate(self):
        payload = json.loads(self.audit.read_text(encoding="ascii"))
        with (
            patch.object(freezer, "ROOT", self.root),
            patch.object(freezer.v2_audit, "validate_plan"),
            patch.object(freezer.v2_audit, "build_audit", return_value=payload),
            patch.object(
                freezer.v2_audit,
                "validate_v1_source",
                return_value=({}, {}, {"results": {}}),
            ),
            patch("pickle.load", return_value={"results": {}}),
        ):
            return freezer.validate_v2_source(
                self.plan, self.checkpoint, self.evidence, self.audit
            )

    def test_passed_independent_v2_audit_is_required(self):
        self.validate()
        payload = json.loads(self.audit.read_text(encoding="ascii"))
        payload["passed"] = False
        self.audit.write_text(json.dumps(payload), encoding="ascii")
        with self.assertRaisesRegex(ValueError, "independent v2 audit"):
            self.validate()

    def test_any_bound_source_hash_change_is_rejected(self):
        payload = json.loads(self.evidence.read_text(encoding="ascii"))
        payload["checkpoint"]["sha256"] = "0" * 64
        self.evidence.write_text(json.dumps(payload), encoding="ascii")
        with self.assertRaisesRegex(ValueError, "checkpoint SHA256"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
