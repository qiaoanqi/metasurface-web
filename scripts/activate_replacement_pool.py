#!/usr/bin/env python3
"""Atomically activate a strictly audited, versioned paper 2 replacement pool."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pipeline_supervisor as supervisor  # noqa: E402


VERSION = "paper2-active-pool-v1"


def action_spec(policy: dict, gate: str) -> dict:
    return next(item for item in policy["workflow"]["actions"] if item["gate"] == gate)


def protocol_bound_gates(policy: dict) -> set[str]:
    return {
        item["gate"]
        for item in policy["workflow"]["actions"]
        if item.get("binding") == "solver_protocol"
    }


def replacement_spec(policy: dict, evidence: dict) -> dict:
    expected = action_spec(policy, "replacement_pool_ready")
    if evidence.get("schema_version") != 1 or evidence.get("passed") is not True:
        raise ValueError("replacement evidence must be schema v1 with passed=true")
    if evidence.get("evidence_version") != expected.get("evidence_version"):
        raise ValueError("replacement evidence version mismatch")
    override = evidence.get("pool_spec")
    if not isinstance(override, dict):
        raise ValueError("replacement evidence lacks pool_spec")
    spec = copy.deepcopy(policy["pool"])
    spec.update(override)
    old_path = str(policy["pool"]["path"]).replace("\\", "/").casefold()
    new_path = str(spec.get("path", "")).replace("\\", "/").casefold()
    if not new_path or new_path == old_path:
        raise ValueError("replacement pool must use a new versioned path")
    forbidden = {
        str(item.get("path", "")).replace("\\", "/").casefold()
        for item in (*policy.get("protected_files", []), *policy.get("immutable_assets", []))
    }
    if new_path in forbidden:
        raise ValueError("replacement pool path is immutable or protected")
    return spec


def build_gate_state(existing: dict, policy: dict, pool_manifest: Path, evidence_path: Path) -> dict:
    old_gates = existing.get("gates", {}) if isinstance(existing, dict) else {}
    keep = protocol_bound_gates(policy)
    gates = {name: value for name, value in old_gates.items() if name in keep}
    checked_at = supervisor.now_iso()
    gates["pool_manifest_frozen"] = {
        "passed": True,
        "checked_at": checked_at,
        "evidence": [{
            "path": str(pool_manifest.relative_to(ROOT)).replace("\\", "/"),
            "sha256": supervisor.file_digest(pool_manifest),
        }],
    }
    gates["replacement_pool_ready"] = {
        "passed": True,
        "checked_at": checked_at,
        "evidence": [{
            "path": str(evidence_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": supervisor.file_digest(evidence_path),
        }],
    }
    return {"schema_version": 1, "gates": gates}


def activate(evidence_path: Path, active_path: Path, pool_manifest_path: Path) -> dict:
    policy = supervisor.load_policy()
    integrity = supervisor.verify_policy_integrity(policy)
    if integrity.get("passed") is not True:
        raise ValueError("pipeline policy integrity is not verified")
    evidence = supervisor.load_json(evidence_path, {}) or {}
    spec = replacement_spec(policy, evidence)
    pool_path = supervisor.workspace_file(spec["path"])
    if pool_path is None or not pool_path.is_file():
        raise ValueError("replacement pool is missing or outside workspace")
    pool_sha256 = supervisor.file_digest(pool_path)
    if str(evidence.get("pool_sha256", "")).upper() != pool_sha256:
        raise ValueError("replacement evidence pool SHA256 mismatch")
    audit = supervisor.audit_pool(pool_path, spec)
    if audit.get("passed") is not True:
        raise ValueError(f"replacement pool strict audit failed: {audit.get('errors', [])[:5]}")
    protected = supervisor.audit_protected_files(policy)
    if not all(item.get("passed") for item in protected):
        raise ValueError("protected paper 1 asset changed; refuse activation")
    old_path = ROOT / policy["pool"]["path"]
    previous = {
        "path": str(old_path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": supervisor.file_digest(old_path),
        "md5": supervisor.file_digest(old_path, "md5"),
    }
    pool_manifest = {
        "schema_version": 1,
        "immutable": True,
        "strict_validation_passed": True,
        "pool_path": str(pool_path.relative_to(ROOT)).replace("\\", "/"),
        "pool_sha256": pool_sha256,
        "pool_md5": supervisor.file_digest(pool_path, "md5"),
        "records": audit["records"],
        "expected_records": audit["expected_records"],
        "geometries": audit["geometries"],
        "complete_pairs": audit["complete_pairs"],
        "duplicate_keys": audit["duplicate_keys"],
        "activation_evidence": {
            "path": str(evidence_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": supervisor.file_digest(evidence_path),
        },
        "previous_pool": previous,
        "training_allowed": False,
    }
    if pool_manifest_path.exists():
        current = supervisor.load_json(pool_manifest_path, {}) or {}
        if current != pool_manifest:
            raise ValueError("existing active pool manifest differs; use a new versioned path")
    else:
        supervisor.atomic_json(pool_manifest_path, pool_manifest)
    active = {
        "schema_version": 1,
        "evidence_version": VERSION,
        "active": True,
        "pool_spec": evidence["pool_spec"],
        "pool_sha256": pool_sha256,
        "previous_pool": previous,
        "activation_evidence": {
            "path": str(evidence_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": supervisor.file_digest(evidence_path),
        },
        "pool_manifest": {
            "path": str(pool_manifest_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": supervisor.file_digest(pool_manifest_path),
        },
        "training_allowed": False,
    }
    gate_state = supervisor.load_json(supervisor.GATE_STATE, {}) or {}
    next_gates = build_gate_state(gate_state, policy, pool_manifest_path, evidence_path)
    supervisor.atomic_json(supervisor.GATE_STATE, next_gates)
    if active_path.exists():
        current = supervisor.load_json(active_path, {}) or {}
        if current != active:
            raise ValueError("existing active pool selection differs; bump the activation version")
    else:
        supervisor.atomic_json(active_path, active)
    return active


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", default=".state/replacement_pool_v1.json")
    parser.add_argument("--active", default=".state/active_pool.json")
    parser.add_argument("--pool-manifest", default=".state/pool_manifest_active_v1.json")
    args = parser.parse_args()
    active = activate(ROOT / args.evidence, ROOT / args.active, ROOT / args.pool_manifest)
    print(json.dumps({"active": True, "pool_sha256": active["pool_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
