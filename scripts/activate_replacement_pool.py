#!/usr/bin/env python3
"""Atomically activate a strictly audited, versioned paper 2 replacement pool."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pipeline_supervisor as supervisor  # noqa: E402
from scripts import run_replacement_pool as replacement  # noqa: E402


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
    if evidence.get("schema_version") != 1 or evidence.get("passed") is not True:
        raise ValueError("replacement evidence must be schema v1 with passed=true")
    if evidence.get("evidence_version") != replacement.EVIDENCE_VERSION:
        raise ValueError("replacement evidence version mismatch")
    approved = evidence.get("approved_protocol")
    if not isinstance(approved, dict):
        raise ValueError("replacement evidence lacks an approved protocol")
    protocol_path = replacement.canonical_workspace_path(str(approved.get("path", "")))
    if not protocol_path.is_file():
        raise ValueError("approved replacement protocol is missing")
    protocol_sha256 = replacement.file_digest(protocol_path)
    if protocol_sha256 != str(approved.get("sha256", "")).upper():
        raise ValueError("approved replacement protocol hash mismatch")
    context = replacement.validate_protocol(protocol_path)
    protocol_spec = context["protocol"].get("pool_spec")
    if not isinstance(protocol_spec, dict):
        raise ValueError("approved replacement protocol lacks pool_spec")
    evidence_spec = evidence.get("pool_spec")
    if not isinstance(evidence_spec, dict):
        raise ValueError("replacement evidence lacks pool_spec")
    if evidence_spec != protocol_spec:
        raise ValueError("replacement evidence pool_spec differs from the approved protocol")
    pool_sha256 = str(evidence.get("pool_sha256", "")).upper()
    activation_id = hashlib.sha256(
        f"{protocol_sha256}|{pool_sha256}".encode("ascii")
    ).hexdigest()[:24]
    if evidence.get("activation_id") != activation_id:
        raise ValueError("replacement activation_id is not bound to protocol and pool hashes")
    return copy.deepcopy(protocol_spec)


def validate_audit(audit_path: Path, evidence_path: Path, evidence: dict) -> dict:
    audit = supervisor.load_json(audit_path, {}) or {}
    worker, error = supervisor.audited_worker_payload(
        audit,
        action="replacement_pool_generation",
        audit_version="paper2-replacement-pool-audit-v1",
        worker_version=replacement.EVIDENCE_VERSION,
        audit_fields={"independent_audit"},
        auditor_runtime_paths=supervisor.AUDITOR_RUNTIME_PATHS[
            "replacement_pool_generation"
        ],
    )
    if error or worker != evidence:
        raise ValueError(error or "replacement audit worker evidence differs")
    if (
        audit.get("passed") is not True
        or audit.get("training_allowed") is not False
        or audit.get("independent_audit") != evidence.get("audit")
    ):
        raise ValueError("replacement independent audit contract is invalid")
    expected_worker = {
        "path": str(evidence_path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": supervisor.file_digest(evidence_path),
    }
    if audit.get("worker_evidence") != expected_worker:
        raise ValueError("replacement audit is not bound to the worker evidence")
    return audit


def build_gate_state(
    existing: dict,
    policy: dict,
    pool_manifest: Path,
    evidence_path: Path,
    *,
    checked_at: str | None = None,
    activation_id: str | None = None,
    pool_manifest_sha256: str | None = None,
) -> dict:
    old_gates = existing.get("gates", {}) if isinstance(existing, dict) else {}
    keep = protocol_bound_gates(policy)
    gates = {name: value for name, value in old_gates.items() if name in keep}
    checked_at = checked_at or supervisor.now_iso()
    gates["pool_manifest_frozen"] = {
        "passed": True,
        "checked_at": checked_at,
        "evidence": [{
            "path": str(pool_manifest.relative_to(ROOT)).replace("\\", "/"),
            "sha256": pool_manifest_sha256 or supervisor.file_digest(pool_manifest),
        }],
    }
    gates["replacement_pool_ready"] = {
        "passed": True,
        "checked_at": checked_at,
        "activation_id": activation_id,
        "evidence": [{
            "path": str(evidence_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": supervisor.file_digest(evidence_path),
        }],
    }
    return {"schema_version": 1, "gates": gates}


def json_digest(payload: dict) -> str:
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def supervisor_json_digest(payload: dict) -> str:
    encoded = (
        json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest().upper()


def write_if_absent_or_equal(path: Path, payload: dict) -> None:
    if path.exists():
        current = supervisor.load_json(path, {}) or {}
        if current != payload:
            raise ValueError(f"activation target already differs: {path}")
        return
    supervisor.atomic_json(path, payload)


def transaction_payload(
    activation_id: str,
    evidence_path: Path,
    targets: dict[str, tuple[Path, dict]],
    previous_gate_state: dict,
    checked_at: str,
) -> dict:
    return {
        "schema_version": 1,
        "evidence_version": "paper2-active-pool-transaction-v1",
        "activation_id": activation_id,
        "status": "prepared",
        "evidence": {
            "path": str(evidence_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": supervisor.file_digest(evidence_path),
        },
        "checked_at": checked_at,
        "previous_gate_state": previous_gate_state,
        "previous_gate_state_sha256": json_digest(previous_gate_state),
        "targets": {
            name: {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "payload_sha256": json_digest(payload),
            }
            for name, (path, payload) in targets.items()
        },
        "updated_at": supervisor.now_iso(),
    }


def write_transaction(path: Path, transaction: dict, status: str) -> dict:
    updated = copy.deepcopy(transaction)
    updated["status"] = status
    updated["updated_at"] = supervisor.now_iso()
    supervisor.atomic_json(path, updated)
    return updated


def activate(
    evidence_path: Path,
    active_path: Path,
    pool_manifest_path: Path,
    transaction_path: Path | None = None,
    audit_path: Path | None = None,
) -> dict:
    policy = supervisor.load_policy()
    integrity = supervisor.verify_policy_integrity(policy)
    if integrity.get("passed") is not True:
        raise ValueError("pipeline policy integrity is not verified")
    evidence = supervisor.load_json(evidence_path, {}) or {}
    audit_path = audit_path or (ROOT / ".state" / "replacement_pool_v1_audit.json")
    validate_audit(audit_path, evidence_path, evidence)
    spec = replacement_spec(policy, evidence)
    pool_path = replacement.canonical_workspace_path(
        spec["path"], require_replacement_dir=True
    )
    if not pool_path.is_file():
        raise ValueError("replacement pool is missing or outside workspace")
    pool_sha256 = supervisor.file_digest(pool_path)
    if str(evidence.get("pool_sha256", "")).upper() != pool_sha256:
        raise ValueError("replacement evidence pool SHA256 mismatch")
    strict_audit = supervisor.audit_pool(pool_path, spec)
    if strict_audit.get("passed") is not True:
        raise ValueError(
            f"replacement pool strict audit failed: {strict_audit.get('errors', [])[:5]}"
        )
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
        "records": strict_audit["records"],
        "expected_records": strict_audit["expected_records"],
        "geometries": strict_audit["geometries"],
        "complete_pairs": strict_audit["complete_pairs"],
        "duplicate_keys": strict_audit["duplicate_keys"],
        "activation_evidence": {
            "path": str(audit_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": supervisor.file_digest(audit_path),
        },
        "activation_id": evidence["activation_id"],
        "approved_protocol": evidence["approved_protocol"],
        "pool_spec_sha256": json_digest(spec),
        "previous_pool": previous,
        "training_allowed": False,
    }
    active = {
        "schema_version": 1,
        "evidence_version": VERSION,
        "active": True,
        "pool_spec": evidence["pool_spec"],
        "pool_sha256": pool_sha256,
        "activation_id": evidence["activation_id"],
        "approved_protocol": evidence["approved_protocol"],
        "previous_pool": previous,
        "activation_evidence": {
            "path": str(audit_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": supervisor.file_digest(audit_path),
        },
        "pool_manifest": {
            "path": str(pool_manifest_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": supervisor_json_digest(pool_manifest),
        },
        "training_allowed": False,
    }
    transaction_path = transaction_path or (ROOT / ".state" / "active_pool_transaction_v1.json")
    current_gate_state = supervisor.load_json(supervisor.GATE_STATE, {}) or {}
    existing_transaction = (
        supervisor.load_json(transaction_path, {}) or {} if transaction_path.exists() else {}
    )
    if existing_transaction:
        if existing_transaction.get("activation_id") != evidence["activation_id"]:
            raise ValueError("another active-pool transaction already exists")
        checked_at = str(existing_transaction.get("checked_at", ""))
        gate_state = existing_transaction.get("previous_gate_state")
        if not checked_at or not isinstance(gate_state, dict):
            raise ValueError("existing active-pool transaction is incomplete")
        if json_digest(gate_state) != existing_transaction.get("previous_gate_state_sha256"):
            raise ValueError("existing active-pool transaction was tampered")
    else:
        checked_at = supervisor.now_iso()
        gate_state = current_gate_state
    next_gates = build_gate_state(
        gate_state,
        policy,
        pool_manifest_path,
        audit_path,
        checked_at=checked_at,
        activation_id=evidence["activation_id"],
        pool_manifest_sha256=supervisor_json_digest(pool_manifest),
    )
    targets = {
        "pool_manifest": (pool_manifest_path, pool_manifest),
        "active_pool": (active_path, active),
        "gate_state": (supervisor.GATE_STATE, next_gates),
    }
    transaction = transaction_payload(
        evidence["activation_id"], audit_path, targets, gate_state, checked_at
    )
    if existing_transaction:
        expected_targets = transaction["targets"]
        if existing_transaction.get("targets") != expected_targets:
            raise ValueError("existing active-pool transaction targets differ")
        if existing_transaction.get("evidence") != transaction["evidence"]:
            raise ValueError("existing active-pool transaction evidence differs")
        if existing_transaction.get("status") == "committed":
            for path, payload in targets.values():
                write_if_absent_or_equal(path, payload)
            return active
        transaction = existing_transaction
    else:
        supervisor.atomic_json(transaction_path, transaction)

    # Commit marker ordering is deliberate: until gate_state is last, the
    # workflow cannot leave replacement_pool_generation. Every step is
    # idempotent, so a watchdog can safely call activate again after a crash.
    write_if_absent_or_equal(pool_manifest_path, pool_manifest)
    transaction = write_transaction(transaction_path, transaction, "manifest_written")
    write_if_absent_or_equal(active_path, active)
    transaction = write_transaction(transaction_path, transaction, "active_written")
    current_gates = supervisor.load_json(supervisor.GATE_STATE, {}) or {}
    if current_gates not in (gate_state, next_gates):
        raise ValueError("gate state changed during active-pool transaction")
    if current_gates != next_gates:
        supervisor.atomic_json(supervisor.GATE_STATE, next_gates)
    write_transaction(transaction_path, transaction, "committed")
    return active


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", default=".state/replacement_pool_v1.json")
    parser.add_argument("--active", default=".state/active_pool.json")
    parser.add_argument("--pool-manifest", default=".state/pool_manifest_active_v1.json")
    parser.add_argument("--transaction", default=".state/active_pool_transaction_v1.json")
    parser.add_argument("--audit", default=".state/replacement_pool_v1_audit.json")
    args = parser.parse_args()
    active = activate(
        ROOT / args.evidence,
        ROOT / args.active,
        ROOT / args.pool_manifest,
        ROOT / args.transaction,
        ROOT / args.audit,
    )
    print(json.dumps({"active": True, "pool_sha256": active["pool_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
