#!/usr/bin/env python3
"""Recoverable transaction for the policy and its integrity lock."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from pipeline_supervisor import atomic_json, file_digest, load_json


VERSION = "paper2-policy-integrity-transaction-v1"
ACTIVE_STATUSES = {"prepared", "policy_replaced"}
FaultInjector = Callable[[str], None]


def json_file_sha256(payload: dict[str, Any]) -> str:
    """Hash the exact ASCII bytes written by pipeline_supervisor.atomic_json."""
    text = json.dumps(
        payload,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    return hashlib.sha256(text.encode("ascii")).hexdigest().upper()


def _transaction_dir(integrity_path: Path) -> Path:
    return integrity_path.parent / "policy_integrity_transactions"


def _normalized(path: Path) -> str:
    return str(path.resolve())


def _journal_path(
    policy_path: Path,
    integrity_path: Path,
    before_policy_sha256: str,
    after_policy_sha256: str,
) -> Path:
    raw = (
        f"{_normalized(policy_path)}|{_normalized(integrity_path)}|"
        f"{before_policy_sha256}|{after_policy_sha256}"
    ).encode("utf-8")
    transaction_id = hashlib.sha256(raw).hexdigest()[:20]
    return _transaction_dir(integrity_path) / f"policy-integrity-{transaction_id}.json"


def _snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "sha256": json_file_sha256(payload),
        "payload": payload,
    }


def _validate_snapshot(snapshot: object, label: str) -> dict[str, Any]:
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("payload"), dict):
        raise ValueError(f"transaction {label} snapshot is malformed")
    payload = snapshot["payload"]
    expected = str(snapshot.get("sha256", "")).upper()
    if not expected or json_file_sha256(payload) != expected:
        raise ValueError(f"transaction {label} snapshot hash mismatch")
    return payload


def _validate_journal(
    journal: dict[str, Any], policy_path: Path, integrity_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if (
        journal.get("schema_version") != 1
        or journal.get("evidence_version") != VERSION
        or journal.get("policy_path") != _normalized(policy_path)
        or journal.get("integrity_path") != _normalized(integrity_path)
    ):
        raise ValueError("policy/integrity transaction journal identity mismatch")
    before = journal.get("before")
    after = journal.get("after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError("policy/integrity transaction snapshots are missing")
    before_policy = _validate_snapshot(before.get("policy"), "before policy")
    before_integrity = _validate_snapshot(before.get("integrity"), "before integrity")
    after_policy = _validate_snapshot(after.get("policy"), "after policy")
    after_integrity = _validate_snapshot(after.get("integrity"), "after integrity")
    after_policy_sha = str(after["policy"]["sha256"]).upper()
    if str(after_integrity.get("policy_sha256", "")).upper() != after_policy_sha:
        raise ValueError("transaction integrity snapshot does not bind the after-policy hash")
    return before_policy, before_integrity, after_policy, after_integrity


def _current_sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"transaction target is missing: {path}")
    return file_digest(path)


def _set_status(path: Path, journal: dict[str, Any], status: str) -> dict[str, Any]:
    updated = dict(journal)
    updated["status"] = status
    atomic_json(path, updated)
    return updated


def _finish_journal(
    journal_path: Path,
    journal: dict[str, Any],
    policy_path: Path,
    integrity_path: Path,
) -> dict[str, Any]:
    before_policy, before_integrity, after_policy, after_integrity = _validate_journal(
        journal, policy_path, integrity_path
    )
    before_policy_sha = json_file_sha256(before_policy)
    before_integrity_sha = json_file_sha256(before_integrity)
    after_policy_sha = json_file_sha256(after_policy)
    after_integrity_sha = json_file_sha256(after_integrity)
    policy_sha = _current_sha256(policy_path)
    integrity_sha = _current_sha256(integrity_path)
    if policy_sha not in {before_policy_sha, after_policy_sha}:
        raise ValueError("policy changed outside the pending transaction")
    if integrity_sha not in {before_integrity_sha, after_integrity_sha}:
        raise ValueError("integrity lock changed outside the pending transaction")

    if policy_sha != after_policy_sha:
        atomic_json(policy_path, after_policy)
    if integrity_sha != after_integrity_sha:
        atomic_json(integrity_path, after_integrity)
    if (
        _current_sha256(policy_path) != after_policy_sha
        or _current_sha256(integrity_path) != after_integrity_sha
    ):
        raise ValueError("policy/integrity transaction recovery did not converge")
    committed = _set_status(journal_path, journal, "committed")
    return {
        "status": "recovered",
        "journal": str(journal_path),
        "policy_sha256": after_policy_sha,
        "integrity_sha256": after_integrity_sha,
        "transaction": committed,
    }


def recover_policy_integrity_transaction(
    policy_path: Path,
    integrity_path: Path,
) -> dict[str, Any] | None:
    directory = _transaction_dir(integrity_path)
    if not directory.is_dir():
        return None
    active: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(directory.glob("policy-integrity-*.json")):
        journal = load_json(path, {}) or {}
        if (
            journal.get("policy_path") == _normalized(policy_path)
            and journal.get("integrity_path") == _normalized(integrity_path)
            and journal.get("status") in ACTIVE_STATUSES
        ):
            active.append((path, journal))
    if len(active) > 1:
        raise ValueError("multiple unfinished policy/integrity transactions")
    if not active:
        return None
    return _finish_journal(active[0][0], active[0][1], policy_path, integrity_path)


def apply_policy_integrity_transaction(
    policy_path: Path,
    integrity_path: Path,
    before_policy: dict[str, Any],
    before_integrity: dict[str, Any],
    after_policy: dict[str, Any],
    after_integrity: dict[str, Any],
    *,
    fault_injector: FaultInjector | None = None,
) -> dict[str, Any]:
    recovered = recover_policy_integrity_transaction(policy_path, integrity_path)
    if recovered is not None:
        if (
            load_json(policy_path, {}) == after_policy
            and load_json(integrity_path, {}) == after_integrity
        ):
            return recovered
        raise ValueError("a different policy/integrity transaction was recovered")

    before_policy_snapshot = _snapshot(before_policy)
    before_integrity_snapshot = _snapshot(before_integrity)
    after_policy_snapshot = _snapshot(after_policy)
    after_integrity_snapshot = _snapshot(after_integrity)
    if _current_sha256(policy_path) != before_policy_snapshot["sha256"]:
        raise ValueError("policy changed before transaction preparation")
    if _current_sha256(integrity_path) != before_integrity_snapshot["sha256"]:
        raise ValueError("integrity lock changed before transaction preparation")
    if str(after_integrity.get("policy_sha256", "")).upper() != after_policy_snapshot["sha256"]:
        raise ValueError("new integrity lock does not bind the new policy")

    journal_path = _journal_path(
        policy_path,
        integrity_path,
        before_policy_snapshot["sha256"],
        after_policy_snapshot["sha256"],
    )
    journal = {
        "schema_version": 1,
        "evidence_version": VERSION,
        "status": "prepared",
        "policy_path": _normalized(policy_path),
        "integrity_path": _normalized(integrity_path),
        "before": {
            "policy": before_policy_snapshot,
            "integrity": before_integrity_snapshot,
        },
        "after": {
            "policy": after_policy_snapshot,
            "integrity": after_integrity_snapshot,
        },
    }
    if journal_path.exists():
        existing = load_json(journal_path, {}) or {}
        if existing != journal:
            if existing.get("status") == "committed":
                expected = dict(existing)
                expected["status"] = "prepared"
                if expected == journal:
                    policy_sha = _current_sha256(policy_path)
                    integrity_sha = _current_sha256(integrity_path)
                    if (
                        policy_sha == after_policy_snapshot["sha256"]
                        and integrity_sha == after_integrity_snapshot["sha256"]
                    ):
                        return {
                            "status": "already_committed",
                            "journal": str(journal_path),
                            "policy_sha256": after_policy_snapshot["sha256"],
                            "integrity_sha256": after_integrity_snapshot["sha256"],
                            "transaction": existing,
                        }
                    if (
                        policy_sha == before_policy_snapshot["sha256"]
                        and integrity_sha == before_integrity_snapshot["sha256"]
                    ):
                        atomic_json(journal_path, journal)
                    else:
                        raise ValueError(
                            "committed policy/integrity transaction targets changed unexpectedly"
                        )
                else:
                    raise ValueError("policy/integrity transaction journal collision")
            else:
                raise ValueError("policy/integrity transaction journal collision")
    else:
        atomic_json(journal_path, journal)

    atomic_json(policy_path, after_policy)
    journal = _set_status(journal_path, journal, "policy_replaced")
    if fault_injector is not None:
        fault_injector("after_policy_replace")
    atomic_json(integrity_path, after_integrity)
    if (
        _current_sha256(policy_path) != after_policy_snapshot["sha256"]
        or _current_sha256(integrity_path) != after_integrity_snapshot["sha256"]
    ):
        raise ValueError("policy/integrity transaction did not commit exact snapshots")
    journal = _set_status(journal_path, journal, "committed")
    return {
        "status": "committed",
        "journal": str(journal_path),
        "policy_sha256": after_policy_snapshot["sha256"],
        "integrity_sha256": after_integrity_snapshot["sha256"],
        "transaction": journal,
    }
