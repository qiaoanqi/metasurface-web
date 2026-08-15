#!/usr/bin/env python3
"""Independently verify the pre-holdout multi-fidelity contract."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pipeline_supervisor as supervisor  # noqa: E402
from scripts import freeze_multifidelity_preregistration_v1 as freezer  # noqa: E402


VERSION = "paper2-multifidelity-preregistration-audit-v1"
POOL_PATH = freezer.POOL_PATH
PLAN_PATH = ROOT / "protocols" / "paper2_multifidelity_preregistration_v1.json"
COST_PATH = ROOT / "protocols" / "paper2_multifidelity_cost_basis_v1.json"


def load(path: Path) -> dict[str, Any]:
    payload = supervisor.load_json(path, {}) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def binding(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve().relative_to(ROOT)).replace("\\", "/"),
        "sha256": supervisor.file_digest(path),
    }


def validate_plan_payload(
    plan_path: Path = PLAN_PATH,
    cost_path: Path = COST_PATH,
    pool_path: Path = POOL_PATH,
    holdout_plan_path: Path = freezer.HOLDOUT_PLAN_PATH,
) -> dict[str, Any]:
    plan = load(plan_path)
    cost = load(cost_path)
    if plan.get("schema_version") != 1 or plan.get("evidence_version") != freezer.VERSION:
        raise ValueError("multifidelity preregistration version mismatch")
    if plan.get("preregistration_revision") != 1:
        raise ValueError("unsupported multifidelity preregistration revision")
    if plan.get("created_before_holdout_results") is not True:
        raise ValueError("the multifidelity plan was not frozen before holdout results")
    if plan.get("holdout_outcome_paths_absent_at_registration") is not True:
        raise ValueError("the plan does not attest to a clean pre-holdout registration")
    low = plan.get("fidelity_roles", {}).get("low", {})
    if (
        low.get("path") != str(pool_path.relative_to(ROOT)).replace("\\", "/")
        or str(low.get("sha256", "")).upper() != supervisor.file_digest(pool_path)
        or low.get("records") != 6000
        or low.get("geometries") != 3000
        or low.get("polarizations") != ["p", "s"]
        or low.get("production_truth") is not False
    ):
        raise ValueError("low-fidelity pool binding or role is invalid")
    if (
        plan.get("fidelity_roles", {}).get("high", {}).get("protocol_source")
        != "approved_v2_reference_holdout_candidate"
        or plan["fidelity_roles"]["high"].get("must_bind_to_independent_reference_audit") is not True
    ):
        raise ValueError("high-fidelity protocol is not bound to the independent reference audit")
    isolation = plan.get("holdout_isolation", {})
    expected_holdout = binding(holdout_plan_path)
    if (
        isolation.get("manifest") != expected_holdout
        or isolation.get("geometry_count") != 32
        or isolation.get("exclude_from_selection_training_validation_early_stopping") is not True
        or isolation.get("exclude_from_active_acquisition") is not True
        or isolation.get("exclude_from_hyperparameter_selection") is not True
    ):
        raise ValueError("holdout isolation contract is invalid")
    budget = plan.get("high_fidelity_budget", {})
    expected_budget = {
        "seed_train_geometries": 96,
        "seed_maximin_geometries": 64,
        "seed_fixed_random_geometries": 32,
        "active_batch_geometries": 32,
        "default_train_geometries": 160,
        "maximum_train_geometries": 192,
        "validation_geometries": 64,
        "test_geometries": 96,
        "passive_control_batch_geometries": 32,
        "passive_control_batches": 4,
        "minimum_active_batches": 2,
        "maximum_active_batches": 3,
        "polarization_records_per_geometry": 2,
    }
    if any(budget.get(key) != value for key, value in expected_budget.items()):
        raise ValueError("high-fidelity budget differs from the frozen contract")
    if budget.get("never_expand_after_maximum") is not True:
        raise ValueError("high-fidelity budget lacks a hard upper bound")
    split = plan.get("geometry_split", {})
    if (
        split.get("version") != "sha256-ranked-80-10-10-v1"
        or split.get("source_geometry_count") != 3000
        or split.get("source_record_count") != 6000
        or split.get("split_before_holdout_exclusion") is not True
        or split.get("p_s_pair_unit") is not True
    ):
        raise ValueError("geometry split contract is invalid")
    model = plan.get("multifidelity_model", {})
    if (
        model.get("primary") != "low_fidelity_plus_high_fidelity_residual"
        or model.get("residual_basis_components") != 16
        or model.get("ensemble_seeds") != [42, 123, 456, 789, 2026]
        or model.get("no_model_selection_on_holdout") is not True
    ):
        raise ValueError("multifidelity model contract is invalid")
    stopping = plan.get("stopping", {})
    if (
        stopping.get("minimum_batches") != 2
        or stopping.get("hard_stop") != "maximum 192 high-fidelity active-train geometries"
        or stopping.get("test_reveal_only_after_stop") is not True
        or stopping.get("thresholds_are_frozen") is not True
    ):
        raise ValueError("stopping contract is invalid")
    inverse = plan.get("inverse_design", {})
    if (
        inverse.get("target_source")
        != "24 primary frozen-reference geometries plus 64 final-test high-fidelity geometries"
        or inverse.get("target_count") != 88
        or inverse.get("holdout_targets_unlocked_only_after_model_and_stopping_lock") is not True
        or inverse.get("holdout_targets_final_evaluation_only") is not True
        or inverse.get("independent_high_fidelity_rcwa_verification") is not True
    ):
        raise ValueError("inverse-design holdout reveal contract is invalid")
    if plan.get("failure_and_fallback", {}).get("full_pool_automatic_launch") is not False:
        raise ValueError("full high-fidelity pool is still automatically launchable")
    if plan.get("training_allowed") is not False or plan.get("forbidden") is None:
        raise ValueError("training and safety locks are missing")
    cost_binding = plan.get("cost_basis", {})
    if cost_binding != binding(cost_path):
        raise ValueError("cost basis binding mismatch")
    if cost.get("evidence_version") != "paper2-multifidelity-cost-basis-v1" or cost.get("training_allowed") is not False:
        raise ValueError("cost basis evidence is invalid")
    if int(cost.get("workers", 0)) != 16 or cost.get("paired_polarizations") is not True:
        raise ValueError("cost basis does not describe paired 16-worker execution")
    return plan


def build_evidence(mode: str) -> dict[str, Any]:
    plan = validate_plan_payload()
    pool_sha = supervisor.file_digest(POOL_PATH)
    protected = supervisor.audit_protected_files(supervisor.load_policy())
    try:
        request = supervisor.current_request_identity("multifidelity_preregistration")
        pre_registration_only = False
    except ValueError:
        request = None
        pre_registration_only = True
    checks = {
        "preregistration_schema_and_revision": True,
        "low_fidelity_pool_hash_and_pairing_contract": True,
        "holdout_manifest_bound_and_excluded": True,
        "budget_and_stopping_rules_frozen": True,
        "model_and_acquisition_rules_frozen": True,
        "cost_basis_reproduced": True,
        "paper1_and_immutable_assets_unchanged": all(item.get("passed") for item in protected),
        "training_forbidden": True,
    }
    return {
        "schema_version": 1,
        "evidence_version": VERSION,
        "independent_reproduction": mode == "auditor",
        "request": request,
        "pre_registration_only": pre_registration_only,
        "passed": all(checks.values()),
        "classification": "multifidelity_preregistration_passed" if all(checks.values()) else "multifidelity_preregistration_invalid",
        "pool_sha256": pool_sha,
        "plan": binding(PLAN_PATH),
        "cost_basis": binding(COST_PATH),
        "holdout_manifest": binding(freezer.HOLDOUT_PLAN_PATH),
        "checks": checks,
        "protected_files": protected,
        "training_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("worker", "auditor"), default="auditor")
    parser.add_argument("--output", default=".state/multifidelity_preregistration_v1_audit.json")
    args = parser.parse_args()
    evidence = build_evidence(args.mode)
    output = ROOT / args.output
    if output.is_file():
        existing = load(output)
        if existing != evidence:
            raise SystemExit("existing multifidelity preregistration evidence differs")
    else:
        supervisor.atomic_json(output, evidence)
    print(json.dumps({"passed": evidence["passed"], "classification": evidence["classification"]}, sort_keys=True))
    return 0 if evidence["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
