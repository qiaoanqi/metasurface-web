#!/usr/bin/env python3
"""Transition a passed budget-v2 diagnostic into the frozen reference holdout."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_supervisor import atomic_json, file_digest, load_json  # noqa: E402


FROM_ACTION = "joint_numerical_convergence"
TO_ACTION = "reference_resolution"
V2_AUDIT_VERSION = "paper2-reference-resolution-budget-v2-audit"
V2_EVIDENCE_VERSION = "paper2-reference-resolution-budget-v2"
V2_PLAN_VERSION = "paper2-reference-budget-v2-plan"
HOLDOUT_PLAN_VERSION = "paper2-reference-holdout-v2-plan"
THRESHOLDS = {
    "mean_joint_dE00_lt": 1.15,
    "all_joint_dE00_lt": 2.3,
    "pointwise_conservation_lte": 1e-6,
}
EXISTING_SELECTION_SHA256 = "887D23BC5650C2FA5D7B13FA040AF962EAA89B6FCB2BD10B39ABAC0F01EF83E7"
NEW_SELECTION_SHA256 = "DD18455E230FC73661D80B6C3C40779A09D5C9F567D4D652F8A62BB09BF66BA8"
COMBINED_SELECTION_SHA256 = "5A9CE8F9C831ADE87E1DD81FFE7EF8574A318B72D624CBB5927433F90A172D4F"
EVIDENCE_PATHS = (
    ".state/reference_resolution_budget_v2_audit.json",
    ".state/reference_resolution_budget_v2.json",
    ".state/reference_resolution_budget_v2_checkpoint.pkl",
    ".state/reference_resolution_budget_v2_plan.json",
    ".state/reference_resolution_holdout_v2_plan.json",
    ".state/reference_resolution_holdout_v1_plan.json",
    "scripts/prepare_reference_budget_v2_retry.py",
    "scripts/freeze_reference_holdout_plan.py",
    "scripts/reference_protocol_selection.py",
    "scripts/launch_reference_resolution_holdout.py",
    "scripts/run_reference_resolution_holdout.py",
    "scripts/audit_reference_resolution_holdout.py",
    "scripts/advance_reference_holdout_strategy.py",
    "scripts/paper2_auto_transition.py",
    "tests/test_reference_resolution_budget_v2.py",
    "tests/test_reference_budget_v2_retry.py",
    "tests/test_reference_holdout_launcher.py",
    "tests/test_reference_holdout.py",
)
REQUIRED_AUDIT_CHECKS = (
    "plan_and_source_hashes_verified",
    "v1_scientific_failure_verified",
    "exact_new_task_set",
    "new_spectra_valid",
    "runtime_hashes_verified",
    "order_converged",
    "grid_converged",
    "corner_converged",
    "spectral_450x512_converged",
    "spectral_365x768_converged",
    "spectral_450x768_converged",
)


def binding(path: Path) -> dict:
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": file_digest(path),
    }


def validate_terminal_dispatch(dispatch: dict) -> None:
    if dispatch.get("action") != FROM_ACTION or dispatch.get("status") != "failed":
        raise ValueError("holdout transition requires a terminal failed joint request")
    if str(dispatch.get("failure_class", "")).lower() != "scientific":
        raise ValueError("holdout transition requires a scientific failure")
    if dispatch.get("terminal_failure") is not True:
        raise ValueError("failed joint request is not terminal")
    if int(dispatch.get("strategy_revision", 0)) < 2:
        raise ValueError("holdout transition requires the completed budget-v2 strategy revision")
    request_id = dispatch.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("failed joint request lacks request_id")


def require_binding(item: object, path: Path, label: str) -> None:
    if item != binding(path):
        raise ValueError(f"{label} binding mismatch")


def reusable_request(previous: object, active: dict) -> bool:
    return bool(
        isinstance(previous, dict)
        and previous.get("request_id") == active.get("request_id")
        and 1 <= int(previous.get("attempt", 0)) <= int(active.get("attempt", 0))
    )


def selection_sha256(selection: list[dict]) -> str:
    payload = json.dumps(selection, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def resolve_bound_path(item: object, label: str) -> Path:
    if not isinstance(item, dict):
        raise ValueError(f"{label} binding is missing")
    relative = item.get("path")
    digest = str(item.get("sha256", "")).upper()
    if not isinstance(relative, str) or not relative or len(digest) != 64:
        raise ValueError(f"{label} binding is malformed")
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} path escapes the project root") from exc
    if not path.is_file() or binding(path) != item:
        raise ValueError(f"{label} binding mismatch")
    return path


def validate_v2_pass(
    dispatch: dict,
    audit: dict,
    evidence: dict,
    plan: dict,
    checkpoint_path: Path,
    holdout_plan: dict,
    holdout_plan_path: Path,
) -> None:
    validate_terminal_dispatch(dispatch)
    request = {
        "request_id": dispatch["request_id"],
        "attempt": int(dispatch.get("attempt", 0)),
    }
    pool_sha = str(dispatch.get("payload", {}).get("pool_sha256", "")).upper()
    if not pool_sha:
        raise ValueError("failed joint request lacks a pool SHA256 binding")
    if (
        audit.get("evidence_version") != V2_AUDIT_VERSION
        or audit.get("passed") is not True
        or audit.get("classification") != "budget_v2_converged"
        or audit.get("training_allowed") is not False
        or str(audit.get("pool_sha256", "")).upper() != pool_sha
        or audit.get("thresholds") != THRESHOLDS
        or not reusable_request(audit.get("request"), request)
    ):
        raise ValueError("independent budget-v2 audit is not an approved diagnostic pass")
    for name in REQUIRED_AUDIT_CHECKS:
        if audit.get("checks", {}).get(name) is not True:
            raise ValueError(f"budget-v2 independent audit check failed: {name}")

    plan_path = ROOT / ".state/reference_resolution_budget_v2_plan.json"
    evidence_path = ROOT / ".state/reference_resolution_budget_v2.json"
    if (
        plan.get("schema_version") != 1
        or plan.get("evidence_version") != V2_PLAN_VERSION
        or plan.get("plan_valid") is not True
        or plan.get("source_failed_action") != FROM_ACTION
        or str(plan.get("pool_sha256", "")).upper() != pool_sha
        or plan.get("thresholds") != THRESHOLDS
        or int(plan.get("expected_new_tasks", -1)) != 96
    ):
        raise ValueError("budget-v2 plan is invalid or changed")
    if (
        evidence.get("evidence_version") != V2_EVIDENCE_VERSION
        or evidence.get("passed") is not True
        or evidence.get("training_allowed") is not False
        or str(evidence.get("pool_sha256", "")).upper() != pool_sha
        or evidence.get("thresholds") != THRESHOLDS
        or evidence.get("request") != audit.get("request")
    ):
        raise ValueError("budget-v2 worker evidence is not a hash-bound pass")
    require_binding(evidence.get("plan"), plan_path, "budget-v2 evidence plan")
    require_binding(evidence.get("checkpoint"), checkpoint_path, "budget-v2 evidence checkpoint")
    require_binding(audit.get("plan"), plan_path, "budget-v2 audit plan")
    expected_audit_checkpoint = binding(checkpoint_path) | {"tasks": 96}
    if audit.get("checkpoint") != expected_audit_checkpoint:
        raise ValueError("budget-v2 audit checkpoint binding mismatch")

    archived_plan_path = ROOT / ".state/reference_resolution_holdout_v1_plan.json"
    expected_sources = {
        "source_v2_plan": plan_path,
        "source_v2_checkpoint": checkpoint_path,
        "source_v2_worker_evidence": evidence_path,
        "source_v2_independent_audit": ROOT / ".state/reference_resolution_budget_v2_audit.json",
        "archived_v1_holdout_plan": archived_plan_path,
    }
    for name, path in expected_sources.items():
        require_binding(holdout_plan.get(name), path, f"v2 holdout {name}")
    source_base_checkpoint = resolve_bound_path(
        holdout_plan.get("source_base_checkpoint"), "v2 holdout source_base_checkpoint"
    )
    if holdout_plan.get("source_base_checkpoint") != plan.get("source_v1_checkpoint"):
        raise ValueError("v2 holdout base checkpoint differs from the budget-v2 plan")
    if source_base_checkpoint == checkpoint_path:
        raise ValueError("v2 holdout base and extension checkpoints must be distinct")

    existing_cases = holdout_plan.get("existing_cases")
    new_cases = holdout_plan.get("new_cases")
    combined_cases = holdout_plan.get("combined_cases")
    if (
        not isinstance(existing_cases, list)
        or not isinstance(new_cases, list)
        or not isinstance(combined_cases, list)
        or len(existing_cases) != 8
        or len(new_cases) != 24
        or len(combined_cases) != 32
        or combined_cases != existing_cases + new_cases
        or selection_sha256(existing_cases) != EXISTING_SELECTION_SHA256
        or selection_sha256(new_cases) != NEW_SELECTION_SHA256
        or selection_sha256(combined_cases) != COMBINED_SELECTION_SHA256
        or holdout_plan.get("existing_selection_sha256") != EXISTING_SELECTION_SHA256
        or holdout_plan.get("new_selection_sha256") != NEW_SELECTION_SHA256
        or holdout_plan.get("combined_selection_sha256") != COMBINED_SELECTION_SHA256
    ):
        raise ValueError("v2 holdout geometry manifest is invalid or changed")
    if len({selection_sha256([item]) for item in combined_cases}) != 32:
        raise ValueError("v2 holdout geometry manifest contains duplicates")

    protocols = holdout_plan.get("task_protocols")
    if not isinstance(protocols, list) or len(protocols) < 5:
        raise ValueError("v2 holdout task protocol manifest is incomplete")
    try:
        protocol_keys = {
            (int(item["requested_nG"]), int(item["Nxy"]), float(item["step_nm"]))
            for item in protocols
        }
        candidate = holdout_plan["frozen_candidate"]
        candidate_key = (
            int(candidate["requested_nG"]),
            int(candidate["Nxy"]),
            float(candidate["wavelength_step_nm"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("v2 holdout protocol or candidate manifest is malformed") from exc
    if len(protocol_keys) != len(protocols) or candidate_key not in protocol_keys:
        raise ValueError("v2 holdout candidate is absent or duplicated in the task manifest")
    expected_new_tasks = 24 * 2 * len(protocols)
    if (
        holdout_plan.get("schema_version") != 1
        or holdout_plan.get("evidence_version") != HOLDOUT_PLAN_VERSION
        or holdout_plan.get("plan_valid") is not True
        or holdout_plan.get("created_before_holdout_results") is not True
        or holdout_plan.get("candidate_frozen_on_initial_eight_only") is not True
        or holdout_plan.get("holdout_cannot_reselect_candidate") is not True
        or holdout_plan.get("source_gate")
        != "independent_reference_resolution_budget_v2_passed"
        or int(holdout_plan.get("existing_case_count", -1)) != 8
        or int(holdout_plan.get("new_case_count", -1)) != 24
        or int(holdout_plan.get("combined_case_count", -1)) != 32
        or holdout_plan.get("polarizations") != ["p", "s"]
        or holdout_plan.get("primary_gate_population")
        != "24_new_holdout_geometries_only"
        or holdout_plan.get("combined_32_population_scope")
        != "supplemental_reporting_only"
        or int(holdout_plan.get("expected_source_tasks", -1))
        != 8 * 2 * len(protocols)
        or int(holdout_plan.get("expected_new_tasks", -1)) != expected_new_tasks
        or int(holdout_plan.get("expected_combined_tasks", -1))
        != 32 * 2 * len(protocols)
        or str(holdout_plan.get("pool", {}).get("sha256", "")).upper() != pool_sha
        or holdout_plan.get("thresholds") != THRESHOLDS
        or not isinstance(holdout_plan.get("frozen_candidate"), dict)
        or holdout_plan["frozen_candidate"].get("passed") is not True
        or holdout_plan.get("final_reference")
        != {"requested_nG": 450, "Nxy": 768, "wavelength_step_nm": 0.5}
    ):
        raise ValueError("v2 frozen holdout plan is invalid or changed")

    runtime_hashes = holdout_plan.get("runtime_hashes")
    if not isinstance(runtime_hashes, dict):
        raise ValueError("v2 holdout runtime hashes are missing")
    for name in (
        "scripts/freeze_reference_holdout_plan.py",
        "scripts/reference_protocol_selection.py",
        "scripts/launch_reference_resolution_holdout.py",
        "scripts/run_reference_resolution_holdout.py",
        "scripts/audit_reference_resolution_holdout.py",
    ):
        path = ROOT / name
        if str(runtime_hashes.get(name, "")).upper() != file_digest(path):
            raise ValueError(f"v2 holdout runtime hash mismatch: {name}")


def strategy_instruction() -> str:
    return (
        "Run only the fail-closed holdout launcher: python "
        "scripts/launch_reference_resolution_holdout.py --n-jobs 16. This request is the "
        "reference_resolution transition authorized by the passed independent budget-v2 audit and the "
        "v2 plan that froze one candidate on the initial eight cases. Use only the 24 untouched holdout "
        "geometries for the primary gate and never reselect a candidate from their results. Never run the "
        "holdout worker directly, register a gate before the independent v2 holdout audit passes, change "
        "thresholds, activate or overwrite a pool, modify paper 1, or enable training."
    )


def build_strategy(policy: dict, dispatch: dict, evidence: list[dict]) -> dict:
    validate_terminal_dispatch(dispatch)
    current = policy.get("strategy_override", {})
    revision = max(
        int(current.get("revision", 0)),
        int(dispatch.get("strategy_revision", 0)),
    ) + 1
    return {
        "enabled": True,
        "decision": "transition_after_failure",
        "revision": revision,
        "action": TO_ACTION,
        "from_action": FROM_ACTION,
        "based_on_request_id": dispatch["request_id"],
        "instruction_append": strategy_instruction(),
        "evidence": evidence,
    }


def apply_strategy(policy_path: Path, integrity_path: Path, dispatch_path: Path) -> dict:
    policy = load_json(policy_path, {}) or {}
    integrity = load_json(integrity_path, {}) or {}
    dispatch = load_json(dispatch_path, {}) or {}
    validate_terminal_dispatch(dispatch)
    if integrity.get("schema_version") != 1:
        raise ValueError("pipeline integrity lock is invalid")
    if file_digest(policy_path) != str(integrity.get("policy_sha256", "")).upper():
        raise ValueError("policy does not match the current integrity lock")
    supervisor_path = ROOT / "pipeline_supervisor.py"
    if file_digest(supervisor_path) != str(integrity.get("supervisor_sha256", "")).upper():
        raise ValueError("supervisor does not match the current integrity lock")

    paths = {name: ROOT / name for name in EVIDENCE_PATHS}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError(f"holdout transition evidence is missing: {missing}")
    audit = load_json(paths[EVIDENCE_PATHS[0]], {}) or {}
    worker_evidence = load_json(paths[EVIDENCE_PATHS[1]], {}) or {}
    plan = load_json(paths[EVIDENCE_PATHS[3]], {}) or {}
    holdout_plan = load_json(paths[EVIDENCE_PATHS[4]], {}) or {}
    validate_v2_pass(
        dispatch,
        audit,
        worker_evidence,
        plan,
        paths[EVIDENCE_PATHS[2]],
        holdout_plan,
        paths[EVIDENCE_PATHS[4]],
    )
    evidence = [binding(paths[name]) for name in EVIDENCE_PATHS]
    source_base_checkpoint = resolve_bound_path(
        holdout_plan.get("source_base_checkpoint"), "v2 holdout source_base_checkpoint"
    )
    source_binding = binding(source_base_checkpoint)
    if source_binding not in evidence:
        evidence.append(source_binding)
    current = policy.get("strategy_override", {})
    if current.get("based_on_request_id") == dispatch.get("request_id"):
        expected_fields = {
            "enabled": True,
            "decision": "transition_after_failure",
            "action": TO_ACTION,
            "from_action": FROM_ACTION,
            "based_on_request_id": dispatch["request_id"],
            "instruction_append": strategy_instruction(),
            "evidence": evidence,
        }
        if (
            all(current.get(key) == value for key, value in expected_fields.items())
            and int(current.get("revision", 0)) > int(dispatch.get("strategy_revision", 0))
        ):
            return {
                "status": "already_applied",
                "strategy_revision": int(current["revision"]),
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
