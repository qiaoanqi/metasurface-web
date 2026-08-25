#!/usr/bin/env python3
"""Audit-only recovery for a stale pre-holdout multifidelity audit.

This verifier rebases only control-plane/protected-file evidence. It never
recomputes the pool, creates worker evidence, changes the failed dispatch, or
registers a gate without a matching completed acknowledgement.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pipeline_supervisor as supervisor  # noqa: E402
from scripts import audit_multifidelity_preregistration_v1 as source_auditor  # noqa: E402


VERSION = "paper2-multifidelity-preregistration-audit-recovery-v1"
PROTOCOL_PATH = ROOT / "protocols/paper2_multifidelity_preregistration_audit_recovery_v1.json"
SOURCE_AUDIT_PATH = ROOT / ".state/multifidelity_preregistration_v1_audit.json"
WORKER_EVIDENCE_PATH = ROOT / ".state/multifidelity_preregistration_v1.json"
DISPATCH_PATH = ROOT / ".state/dispatch_request.json"
ACK_PATH = ROOT / ".state/executor_ack.json"
UI_AUTH_PATH = ROOT / "protocols/ui_mainline_authorization_v1.json"


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


def require_binding(item: Any, label: str) -> Path:
    if not isinstance(item, dict):
        raise ValueError(f"{label} binding is missing")
    path = supervisor.workspace_file(item.get("path"))
    if path is None or not path.is_file():
        raise ValueError(f"{label} path is missing or outside workspace")
    actual = supervisor.file_digest(path)
    if actual != str(item.get("sha256", "")).upper():
        raise ValueError(f"{label} SHA256 mismatch")
    return path


def validate_ui_authorization(protocol: dict[str, Any]) -> dict[str, Any]:
    item = protocol.get("ui_authorization")
    path = require_binding(item, "UI authorization")
    authorization = load(path)
    if (
        authorization.get("evidence_version") != "ui-mainline-authorization-v1"
        or authorization.get("authorized_by_user") is not True
        or authorization.get("training_allowed") is not False
        or authorization.get("holdout_allowed") is not False
        or authorization.get("paper1_modification_allowed") is not False
    ):
        raise ValueError("UI authorization safety envelope is invalid")
    observed: dict[str, Any] = {}
    for entry in authorization.get("authorized_files", []):
        if not isinstance(entry, dict):
            raise ValueError("UI authorization file entry is malformed")
        path_value = entry.get("path")
        file_path = supervisor.workspace_file(path_value)
        if file_path is None or not file_path.is_file():
            raise ValueError(f"authorized UI file is missing: {path_value}")
        if "current_md5" in entry:
            actual = supervisor.file_digest(file_path, "md5")
            if actual != str(entry["current_md5"]).upper():
                raise ValueError(f"authorized UI MD5 mismatch: {path_value}")
            observed[path_value] = {"md5": actual}
        elif "current_sha256" in entry:
            actual = supervisor.file_digest(file_path)
            if actual != str(entry["current_sha256"]).upper():
                raise ValueError(f"authorized UI SHA256 mismatch: {path_value}")
            observed[path_value] = {"sha256": actual}
        else:
            raise ValueError(f"authorized UI file has no current hash: {path_value}")
    return {"binding": binding(path), "authorization": authorization, "observed": observed}


def validate_source_audit(protocol: dict[str, Any], pool_sha: str) -> dict[str, Any]:
    source_spec = protocol.get("source_audit")
    source_path = require_binding(source_spec, "source audit")
    source = load(source_path)
    if (
        source.get("evidence_version") != source_spec.get("evidence_version")
        or source.get("passed") is not True
        or source.get("classification") != source_spec.get("required_classification")
        or source.get("independent_reproduction") is not True
        or source.get("request") is not None
        or str(source.get("pool_sha256", "")).upper() != pool_sha
    ):
        raise ValueError("source audit is not the frozen independent preregistration pass")
    source_audit_plan = source.get("plan")
    source_audit_cost = source.get("cost_basis")
    source_audit_holdout = source.get("holdout_manifest")
    for item, label in (
        (source_audit_plan, "source plan"),
        (source_audit_cost, "source cost basis"),
        (source_audit_holdout, "source holdout manifest"),
    ):
        require_binding(item, label)
    return {"binding": binding(source_path), "payload": source}


def validate_current_state(protocol: dict[str, Any], pool_sha: str) -> dict[str, Any]:
    dispatch = load(DISPATCH_PATH)
    if (
        dispatch.get("action") != protocol.get("action")
        or dispatch.get("status") != "failed"
        or int(dispatch.get("attempt", 0)) != int(protocol["request_contract"]["attempt"])
        or not isinstance(dispatch.get("request_id"), str)
        or not dispatch["request_id"]
    ):
        raise ValueError("dispatch is not the registered failed request attempt")
    if WORKER_EVIDENCE_PATH.exists():
        raise ValueError("multifidelity worker evidence exists; recovery refuses to synthesize or reuse it")
    ack = load(ACK_PATH)
    if (
        ack.get("request_id") == dispatch.get("request_id")
        and int(ack.get("attempt", 0)) == int(dispatch.get("attempt", 0))
    ):
        raise ValueError("matching executor ack exists; recovery refuses to overwrite it")
    policy = supervisor.load_policy()
    integrity = supervisor.verify_policy_integrity(policy)
    if integrity.get("passed") is not True:
        raise ValueError("policy/integrity lock does not pass")
    lock = load(ROOT / ".state/pipeline_integrity.json")
    if int(lock.get("protected_assets_revision", 0)) < int(
        protocol["control_plane"]["minimum_integrity_revision"]
    ):
        raise ValueError("pipeline integrity revision is below the recovery protocol minimum")
    actions = [
        item
        for item in policy.get("workflow", {}).get("actions", [])
        if isinstance(item, dict) and item.get("action") == protocol.get("action")
    ]
    if len(actions) != 1:
        raise ValueError("policy does not register exactly one multifidelity recovery action")
    action = actions[0]
    expected_registration = {
        "recovery_protocol": str(PROTOCOL_PATH.relative_to(ROOT)).replace("\\", "/"),
        "recovery_auditor": "scripts/audit_multifidelity_preregistration_recovery_v1.py",
        "recovery_audit_evidence": ".state/multifidelity_preregistration_v1_audit_recovery_v1.json",
        "audit_only_recovery": True,
        "worker_rerun_allowed": False,
        "gate_registration_requires_completed_ack": True,
    }
    if any(action.get(key) != value for key, value in expected_registration.items()):
        raise ValueError("policy recovery action registration is incomplete or drifted")
    if action.get("recovery_protocol_sha256") != supervisor.file_digest(PROTOCOL_PATH):
        raise ValueError("policy recovery protocol hash is stale")
    if action.get("recovery_auditor_sha256") != supervisor.file_digest(Path(__file__)):
        raise ValueError("policy recovery auditor hash is stale")
    pool_path = supervisor.workspace_file(policy["pool"]["path"])
    if pool_path is None or not pool_path.is_file() or supervisor.file_digest(pool_path) != pool_sha:
        raise ValueError("current pool SHA256 differs from the frozen preregistration")
    source_plan = protocol["scientific_bindings"]["plan"]
    source_cost = protocol["scientific_bindings"]["cost_basis"]
    source_holdout = protocol["scientific_bindings"]["holdout_manifest"]
    for item, label in (
        (source_plan, "current plan"),
        (source_cost, "current cost basis"),
        (source_holdout, "current holdout manifest"),
    ):
        require_binding(item, label)
    source_audit = load(SOURCE_AUDIT_PATH)
    for key, expected in (
        ("plan", source_plan),
        ("cost_basis", source_cost),
        ("holdout_manifest", source_holdout),
    ):
        if source_audit.get(key) != expected:
            raise ValueError(f"source audit {key} binding differs from current frozen bytes")
    return {
        "dispatch": dispatch,
        "policy_integrity": integrity,
        "policy_recovery_action": expected_registration,
        "integrity_lock": binding(ROOT / ".state/pipeline_integrity.json"),
        "policy": binding(ROOT / "pipeline_policy.json"),
        "pool": binding(pool_path),
    }


def build_evidence() -> dict[str, Any]:
    protocol = load(PROTOCOL_PATH)
    if protocol.get("evidence_version") != VERSION:
        raise ValueError("recovery protocol version mismatch")
    pool_sha = str(protocol["scientific_bindings"]["pool_sha256"]).upper()
    source = validate_source_audit(protocol, pool_sha)
    # Re-run only the static contract validator; no spectra, pool, or worker is recomputed.
    source_auditor.validate_plan_payload()
    state = validate_current_state(protocol, pool_sha)
    ui = validate_ui_authorization(protocol)
    current_protected = supervisor.audit_protected_files(supervisor.load_policy())
    if not all(item.get("passed") is True for item in current_protected):
        raise ValueError("current policy protected-file ledger does not pass")
    runtime = {
        "pipeline_supervisor.py": supervisor.file_digest(ROOT / "pipeline_supervisor.py"),
        "scripts/audit_multifidelity_preregistration_v1.py": supervisor.file_digest(
            ROOT / "scripts/audit_multifidelity_preregistration_v1.py"
        ),
        "scripts/freeze_multifidelity_preregistration_v1.py": supervisor.file_digest(
            ROOT / "scripts/freeze_multifidelity_preregistration_v1.py"
        ),
    }
    dispatch = state["dispatch"]
    checks = {
        "source_audit_scientific_passed": True,
        "source_audit_bytes_unchanged": True,
        "pool_bytes_unchanged": True,
        "plan_cost_and_holdout_bytes_unchanged": True,
        "ui_hash_authorization_matches": True,
        "policy_integrity_passed": True,
        "current_protected_ledger_passed": True,
        "stale_protected_hash_rebased": True,
        "no_worker_evidence": True,
        "no_matching_ack": True,
        "gate_registration_blocked_without_completed_ack": True,
        "training_forbidden": True,
        "holdout_forbidden": True,
    }
    return {
        "schema_version": 1,
        "evidence_version": VERSION,
        "action": protocol["action"],
        "passed": all(checks.values()),
        "classification": "multifidelity_preregistration_recovery_ready",
        "scientific_classification": "multifidelity_preregistration_passed",
        "gate_registration_allowed": False,
        "ack_write_allowed": False,
        "worker_rerun_allowed": False,
        "training_allowed": False,
        "holdout_allowed": False,
        "request": {
            "request_id": dispatch["request_id"],
            "attempt": int(dispatch["attempt"]),
            "status": dispatch["status"],
        },
        "source_audit": source,
        "recovery_protocol": binding(PROTOCOL_PATH),
        "pool": state["pool"],
        "plan": protocol["scientific_bindings"]["plan"],
        "cost_basis": protocol["scientific_bindings"]["cost_basis"],
        "holdout_manifest": protocol["scientific_bindings"]["holdout_manifest"],
        "policy": state["policy"],
        "integrity_lock": state["integrity_lock"],
        "policy_integrity": state["policy_integrity"],
        "ui_authorization": ui,
        "runtime_hashes": runtime,
        "protected_files": current_protected,
        "checks": checks,
        "blocker": {
            "classification": "execution_integrity_failure",
            "reason": "failed dispatch attempt 3 has no matching completed acknowledgement; gate and ack writes are forbidden",
            "requires": "a newly authorized, policy-registered audit-only request before any gate registration",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=".state/multifidelity_preregistration_v1_audit_recovery_v1.json")
    args = parser.parse_args()
    evidence = build_evidence()
    output = ROOT / args.output
    if output.is_file():
        existing = load(output)
        if existing != evidence:
            raise SystemExit("existing multifidelity recovery evidence differs")
    else:
        supervisor.atomic_json(output, evidence)
    print(json.dumps({"passed": evidence["passed"], "classification": evidence["classification"], "gate_registration_allowed": evidence["gate_registration_allowed"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
