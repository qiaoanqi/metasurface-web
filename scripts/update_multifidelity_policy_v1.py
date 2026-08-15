#!/usr/bin/env python3
"""Atomically install the v1 multi-fidelity contract in the pipeline policy."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pipeline_supervisor as supervisor  # noqa: E402
from scripts import policy_integrity_transaction as transaction  # noqa: E402


POLICY_PATH = ROOT / "pipeline_policy.json"
INTEGRITY_PATH = ROOT / ".state" / "pipeline_integrity.json"
PLAN_PATH = ROOT / "protocols" / "paper2_multifidelity_preregistration_v1.json"
COST_PATH = ROOT / "protocols" / "paper2_multifidelity_cost_basis_v1.json"
SUPERVISOR_TEST_PATH = ROOT / "tests" / "test_pipeline_supervisor.py"


def binding(path: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": supervisor.file_digest(path),
    }


def build_after(before: dict) -> dict:
    policy = copy.deepcopy(before)
    workflow = policy.setdefault("workflow", {})
    actions = workflow.get("actions", [])
    mf_action = {
        "action": "multifidelity_preregistration",
        "gate": "multifidelity_preregistered",
        "evidence_version": "paper2-multifidelity-preregistration-audit-v1",
        "runner": "python scripts/audit_multifidelity_preregistration_v1.py --mode worker",
        "auditor": "python scripts/audit_multifidelity_preregistration_v1.py --mode auditor",
        "auditor_script": "scripts/audit_multifidelity_preregistration_v1.py",
        "worker_evidence": ".state/multifidelity_preregistration_v1.json",
        "audit_evidence": ".state/multifidelity_preregistration_v1_audit.json",
        "finalizer": "scripts/finalize_audited_gate.py",
        "finalization_mode": "generic",
        "binding": "pool",
        "diagnostic_only": False,
    }
    data_action = {
        "action": "multifidelity_data_preparation",
        "gate": "multifidelity_data_ready",
        "evidence_version": "paper2-multifidelity-data-audit-v1",
        "implementation_state": "ready",
        "manual_only": True,
        "binding": "pool",
    }
    if not any(item.get("gate") == "multifidelity_preregistered" for item in actions):
        reference_index = next(
            (index for index, item in enumerate(actions) if item.get("action") == "reference_resolution"),
            len(actions),
        )
        actions.insert(reference_index, mf_action)
        actions.insert(reference_index + 2, data_action)
    for item in actions:
        if item.get("action") == "replacement_pool_generation":
            item["manual_only"] = True
            item["automatic_launch_authorized"] = False
            item["fallback_only"] = True
    workflow["actions"] = actions
    workflow["contract_revision"] = "paper2-workflow-contract-v2-multifidelity"
    required = list(workflow.get("required_before_training", []))
    required = [gate for gate in required if gate != "replacement_pool_ready"]
    for gate in ("multifidelity_preregistered", "multifidelity_data_ready"):
        if gate not in required:
            required.append(gate)
    workflow["required_before_training"] = required
    policy["multifidelity"] = {
        "protocol_revision": "paper2-multifidelity-preregistration-v1",
        "preregistration": binding(PLAN_PATH),
        "cost_basis": binding(COST_PATH),
        "low_fidelity_is_coverage_only": True,
        "high_fidelity_requires_reference_audit": True,
        "full_pool_fallback_requires_independent_necessity_audit": True,
        "automatic_full_pool_launch": False,
    }
    policy["fallbacks"] = {
        "full_high_fidelity_pool": {
            "action": "replacement_pool_generation",
            "automatic_launch_authorized": False,
            "requires": ["multifidelity_scientific_negative", "full_pool_necessity_audit_passed"],
            "must_use_new_versioned_output": True,
            "must_not_overwrite_low_fidelity_pool": True,
        }
    }
    for item in policy.get("strategy_override", {}).get("evidence", []):
        if item.get("path") == "tests/test_pipeline_supervisor.py":
            item["sha256"] = supervisor.file_digest(SUPERVISOR_TEST_PATH)
    return policy


def main() -> int:
    before_policy = supervisor.load_json(POLICY_PATH, {}) or {}
    before_integrity = supervisor.load_json(INTEGRITY_PATH, {}) or {}
    after_policy = build_after(before_policy)
    if after_policy == before_policy:
        print('{"status":"already_installed"}')
        return 0
    after_integrity = copy.deepcopy(before_integrity)
    after_integrity["protected_assets_revision"] = int(before_integrity.get("protected_assets_revision", 0)) + 1
    after_integrity["policy_sha256"] = transaction.json_file_sha256(after_policy)
    after_integrity["supervisor_sha256"] = supervisor.file_digest(ROOT / "pipeline_supervisor.py")
    after_integrity["note"] = "multifidelity preregistration contract installed; full pool is manual fallback"
    result = transaction.apply_policy_integrity_transaction(
        POLICY_PATH,
        INTEGRITY_PATH,
        before_policy,
        before_integrity,
        after_policy,
        after_integrity,
    )
    print('{"status":"committed","policy_sha256":"%s","integrity_sha256":"%s","supervisor_sha256":"%s","revision":%d}' % (
        result["policy_sha256"],
        result["integrity_sha256"],
        after_integrity["supervisor_sha256"],
        after_integrity["protected_assets_revision"],
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
