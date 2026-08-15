#!/usr/bin/env python3
"""Advance the failed joint gate to the hash-bound budget-v2 diagnostic."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_supervisor import atomic_json, file_digest, load_json  # noqa: E402


ACTION = "joint_numerical_convergence"
V1_AUDIT_VERSION = "paper2-reference-resolution-audit-v1"
V2_PLAN_VERSION = "paper2-reference-budget-v2-plan"
EVIDENCE_PATHS = (
    ".state/reference_resolution_v1_audit.json",
    ".state/reference_resolution_budget_v2_plan.json",
    "scripts/run_reference_resolution_budget_v2.py",
    "scripts/audit_reference_resolution_budget_v2.py",
    "scripts/freeze_reference_budget_v2.py",
    "scripts/advance_reference_budget_v2_strategy.py",
    "tests/test_reference_resolution_budget_v2.py",
)


def binding(path: Path) -> dict:
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": file_digest(path),
    }


def validate_terminal_dispatch(dispatch: dict) -> None:
    if dispatch.get("action") != ACTION or dispatch.get("status") != "failed":
        raise ValueError("strategy advancement requires the failed joint gate")
    if str(dispatch.get("failure_class", "")).lower() != "scientific":
        raise ValueError("strategy advancement requires a scientific failure")
    if dispatch.get("terminal_failure") is not True:
        raise ValueError("failed joint request is not terminal")
    request_id = dispatch.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("failed joint request lacks request_id")


def validate_scientific_inputs(v1_audit: dict, v2_plan: dict) -> None:
    if (
        v1_audit.get("evidence_version") != V1_AUDIT_VERSION
        or v1_audit.get("passed") is not False
        or v1_audit.get("classification")
        in {"execution_integrity_failure", "worker_evidence_integrity_failure"}
    ):
        raise ValueError("v1 independent audit is not a scientific failure")
    for key in (
        "frozen_plan_sha256_and_content",
        "checkpoint_meta_and_runtime_hashes",
        "reference_checkpoint_exact_80",
        "worker_claim_matches_independent_recomputation",
        "physics_controls_passed",
    ):
        if v1_audit.get("checks", {}).get(key) is not True:
            raise ValueError(f"v1 independent audit prerequisite failed: {key}")
    if (
        v2_plan.get("evidence_version") != V2_PLAN_VERSION
        or v2_plan.get("plan_valid") is not True
        or v2_plan.get("source_v1_audit") != binding(ROOT / EVIDENCE_PATHS[0])
        or v2_plan.get("source_failed_action") != ACTION
        or v2_plan.get("expected_new_tasks") != 96
    ):
        raise ValueError("v2 frozen plan is invalid or not bound to the v1 audit")
    expected_thresholds = {
        "mean_joint_dE00_lt": 1.15,
        "all_joint_dE00_lt": 2.3,
        "pointwise_conservation_lte": 1e-6,
    }
    if v2_plan.get("thresholds") != expected_thresholds:
        raise ValueError("v2 strategy cannot change thresholds")


def build_strategy(policy: dict, dispatch: dict, evidence: list[dict]) -> dict:
    validate_terminal_dispatch(dispatch)
    current = policy.get("strategy_override", {})
    revision = max(
        int(current.get("revision", 0)),
        int(dispatch.get("strategy_revision", 0)),
    ) + 1
    return {
        "enabled": True,
        "decision": "retry_same_gate",
        "revision": revision,
        "action": ACTION,
        "based_on_request_id": dispatch["request_id"],
        "instruction_append": (
            "Run only the hash-bound numerical-budget v2 diagnostic: python "
            "scripts/run_reference_resolution_budget_v2.py --n-jobs 16. Preserve its independent "
            "checkpoint and raw 1 nm/0.5 nm R/T arrays. When complete, run python "
            "scripts/audit_reference_resolution_budget_v2.py and acknowledge this request with the "
            "independent audit evidence. This is diagnostic-only: do not register the historical nG131 "
            "pool, change thresholds, launch the 32-case holdout, activate a pool, or enable training."
        ),
        "evidence": evidence,
    }


def apply_strategy(policy_path: Path, integrity_path: Path, dispatch_path: Path) -> dict:
    policy = load_json(policy_path, {}) or {}
    integrity = load_json(integrity_path, {}) or {}
    dispatch = load_json(dispatch_path, {}) or {}
    if integrity.get("schema_version") != 1:
        raise ValueError("pipeline integrity lock is invalid")
    if file_digest(policy_path) != str(integrity.get("policy_sha256", "")).upper():
        raise ValueError("policy does not match the current integrity lock")
    supervisor_path = ROOT / "pipeline_supervisor.py"
    if file_digest(supervisor_path) != str(integrity.get("supervisor_sha256", "")).upper():
        raise ValueError("supervisor does not match the current integrity lock")
    v1_audit = load_json(ROOT / EVIDENCE_PATHS[0], {}) or {}
    v2_plan = load_json(ROOT / EVIDENCE_PATHS[1], {}) or {}
    validate_scientific_inputs(v1_audit, v2_plan)
    evidence = []
    for name in EVIDENCE_PATHS:
        path = ROOT / name
        if not path.is_file():
            raise ValueError(f"strategy evidence is missing: {name}")
        evidence.append(binding(path))
    current_strategy = policy.get("strategy_override", {})
    if current_strategy.get("based_on_request_id") == dispatch.get("request_id"):
        if (
            current_strategy.get("enabled") is True
            and current_strategy.get("decision") == "retry_same_gate"
            and current_strategy.get("action") == ACTION
            and current_strategy.get("evidence") == evidence
            and int(current_strategy.get("revision", 0))
            > int(dispatch.get("strategy_revision", 0))
        ):
            return {
                "status": "already_applied",
                "strategy_revision": int(current_strategy["revision"]),
                "based_on_request_id": dispatch["request_id"],
                "integrity_revision": int(integrity["protected_assets_revision"]),
                "policy_sha256": file_digest(policy_path),
                "supervisor_sha256": file_digest(supervisor_path),
                "evidence": evidence,
            }
        raise ValueError("existing strategy for this failed request differs")
    updated = copy.deepcopy(policy)
    updated["strategy_override"] = build_strategy(policy, dispatch, evidence)
    atomic_json(policy_path, updated)
    new_revision = int(integrity.get("protected_assets_revision", 0)) + 1
    new_lock = {
        "schema_version": 1,
        "policy_sha256": file_digest(policy_path),
        "supervisor_sha256": file_digest(supervisor_path),
        "protected_assets_revision": new_revision,
        "note": integrity.get(
            "note",
            "Update this lock atomically with an intentional policy or supervisor revision; a mismatch blocks dispatch.",
        ),
    }
    atomic_json(integrity_path, new_lock)
    return {
        "status": "updated",
        "strategy_revision": updated["strategy_override"]["revision"],
        "based_on_request_id": dispatch["request_id"],
        "integrity_revision": new_revision,
        "policy_sha256": new_lock["policy_sha256"],
        "supervisor_sha256": new_lock["supervisor_sha256"],
        "evidence": evidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="pipeline_policy.json")
    parser.add_argument("--integrity", default=".state/pipeline_integrity.json")
    parser.add_argument("--dispatch", default=".state/dispatch_request.json")
    args = parser.parse_args()
    result = apply_strategy(ROOT / args.policy, ROOT / args.integrity, ROOT / args.dispatch)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
