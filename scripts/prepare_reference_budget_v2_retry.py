#!/usr/bin/env python3
"""Safely authorize a budget-v2 checkpoint for a later retry attempt."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_supervisor import atomic_json, file_digest, load_json, pid_alive  # noqa: E402
from scripts import run_reference_resolution_budget_v2 as runner  # noqa: E402
from scripts.reference_budget_v2_lineage import validate_lineage  # noqa: E402


VERSION = "paper2-reference-resolution-budget-v2-retry-v1"
ACTIVE_ACK_STATUSES = {"accepted", "claimed", "running", "in_progress"}


def reusable_request(
    previous: object,
    active: dict[str, Any],
    lineage_request_id: str | None = None,
) -> bool:
    return bool(
        isinstance(previous, dict)
        and (
            (
                previous.get("request_id") == active.get("request_id")
                and 1 <= int(previous.get("attempt", 0)) <= int(active.get("attempt", 0))
            )
            or (
                isinstance(lineage_request_id, str)
                and lineage_request_id
                and previous.get("request_id") == lineage_request_id
                and int(previous.get("attempt", 0)) >= 1
            )
        )
    )


def expected_meta(plan: dict, plan_path: Path, request: dict[str, Any]) -> dict:
    tasks = runner.build_tasks(plan["selection"])
    return {
        "version": runner.VERSION,
        "request": request,
        "plan_sha256": file_digest(plan_path),
        "pool_sha256": plan["pool_sha256"],
        "selected_geometries": plan["selection"],
        "expected_tasks": len(tasks),
        "tasks": [
            {
                key: task[key]
                for key in (
                    "id",
                    "geometry_index",
                    "pol",
                    "requested_nG",
                    "Nxy",
                    "step_nm",
                )
            }
            for task in tasks
        ],
        "runtime_hashes": {
            path: file_digest(ROOT / path)
            for path in (
                "rcwa_batch.py",
                "paper2_colorimetry.py",
                "color_utils.py",
                "scripts/run_reference_resolution_budget_v2.py",
            )
        },
    }


def validate_retry_ack(ack: dict, request: dict[str, Any]) -> None:
    if (
        ack.get("request_id") != request["request_id"]
        or int(ack.get("attempt", 0)) != request["attempt"]
        or ack.get("status") not in ACTIVE_ACK_STATUSES
    ):
        raise ValueError("retry preparation requires the active attempt acknowledgement")
    worker_pid = ack.get("worker_pid")
    if worker_pid is not None and pid_alive(worker_pid):
        raise ValueError("cannot rebind a checkpoint while its worker is alive")


def validate_checkpoint(
    checkpoint: dict,
    expected: dict,
    tasks: list[dict],
    active_request: dict[str, Any],
    lineage_request_id: str | None = None,
) -> dict[str, Any]:
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("meta"), dict):
        raise ValueError("budget-v2 retry checkpoint is malformed")
    stored_request = checkpoint["meta"].get("request")
    if not reusable_request(stored_request, active_request, lineage_request_id):
        raise ValueError("budget-v2 checkpoint belongs to a different or newer request")
    stored_static = dict(checkpoint["meta"])
    expected_static = dict(expected)
    stored_static.pop("request", None)
    expected_static.pop("request", None)
    if stored_static != expected_static:
        raise ValueError("budget-v2 checkpoint protocol changed outside retry identity")
    results = checkpoint.get("results", {})
    runner.validate_checkpoint_results(results, tasks)
    return {"request_id": stored_request["request_id"], "attempt": int(stored_request["attempt"])}


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".retry.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def journal_path(directory: Path, previous: dict[str, Any], active: dict[str, Any]) -> Path:
    request_id = str(active["request_id"])
    if not request_id or any(not (char.isalnum() or char in "-_") for char in request_id):
        raise ValueError("request_id is unsafe for a retry journal path")
    if previous.get("request_id") == request_id:
        name = (
            f"reference_resolution_budget_v2_retry_{request_id}_"
            f"a{previous['attempt']}_to_a{active['attempt']}.json"
        )
    else:
        previous_id = str(previous.get("request_id", ""))
        if not previous_id or any(not (char.isalnum() or char in "-_") for char in previous_id):
            raise ValueError("source request_id is unsafe for a retry journal path")
        name = (
            f"reference_resolution_budget_v2_lineage_{previous_id}_a{previous['attempt']}_"
            f"to_{request_id}_a{active['attempt']}.json"
        )
    return directory / name


def prepare_retry(
    checkpoint_path: Path,
    evidence_path: Path,
    journal_dir: Path,
    active_request: dict[str, Any],
    ack: dict,
    expected: dict,
    tasks: list[dict],
    lineage_request_id: str | None = None,
) -> dict:
    validate_retry_ack(ack, active_request)
    if not checkpoint_path.is_file():
        if evidence_path.exists():
            raise ValueError("budget-v2 evidence exists without its checkpoint")
        return {
            "status": "start_fresh",
            "request": active_request,
            "results": 0,
            "training_allowed": False,
        }
    with checkpoint_path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    previous = validate_checkpoint(
        checkpoint, expected, tasks, active_request, lineage_request_id
    )
    checkpoint_before = file_digest(checkpoint_path)

    cross_request = previous["request_id"] != active_request["request_id"]
    if cross_request:
        if previous["request_id"] != lineage_request_id:
            raise ValueError("cross-request checkpoint lacks sealed strategy lineage")
        if len(checkpoint.get("results", {})) != len(tasks) or not evidence_path.is_file():
            raise ValueError(
                "cross-request recovery requires complete evidence; partial resume is forbidden"
            )
        evidence = load_json(evidence_path, {}) or {}
        checkpoint_binding = evidence.get("checkpoint", {})
        if (
            evidence.get("request") != previous
            or checkpoint_binding.get("path")
            != str(checkpoint_path.relative_to(ROOT)).replace("\\", "/")
            or str(checkpoint_binding.get("sha256", "")).upper() != checkpoint_before
            or int(checkpoint_binding.get("tasks", 0)) != len(tasks)
        ):
            raise ValueError("sealed cross-request worker evidence is invalid")
        return {
            "status": "audit_existing_evidence",
            "request": active_request,
            "produced_request": previous,
            "checkpoint_sha256": checkpoint_before,
            "results": len(checkpoint.get("results", {})),
            "cross_request": True,
            "checkpoint_mutated": False,
            "training_allowed": False,
        }

    if evidence_path.is_file():
        evidence = load_json(evidence_path, {}) or {}
        checkpoint_binding = evidence.get("checkpoint", {})
        if (
            reusable_request(
                evidence.get("request"), active_request, lineage_request_id
            )
            and checkpoint_binding.get("path")
            == str(checkpoint_path.relative_to(ROOT)).replace("\\", "/")
            and str(checkpoint_binding.get("sha256", "")).upper() == checkpoint_before
        ):
            return {
                "status": "audit_existing_evidence",
                "request": active_request,
                "produced_request": evidence["request"],
                "checkpoint_sha256": checkpoint_before,
                "results": len(checkpoint.get("results", {})),
                "training_allowed": False,
            }
        raise ValueError("existing budget-v2 evidence is not reusable by this retry")

    if previous == active_request:
        request_id = str(active_request["request_id"])
        pending = list(
            journal_dir.glob(
                f"reference_resolution_budget_v2_retry_{request_id}_a*_to_a{active_request['attempt']}.json"
            )
        )
        if len(pending) > 1:
            raise ValueError("multiple budget-v2 retry journals target the active attempt")
        if pending:
            journal_payload = load_json(pending[0], {}) or {}
            if (
                journal_payload.get("status") == "prepared"
                and journal_payload.get("request") == active_request
                and str(journal_payload.get("checkpoint_sha256_after", "")).upper()
                == checkpoint_before
            ):
                journal_payload["status"] = "completed"
                atomic_json(pending[0], journal_payload)
                return journal_payload
        return {
            "status": "ready",
            "request": active_request,
            "checkpoint_sha256": checkpoint_before,
            "results": len(checkpoint.get("results", {})),
            "training_allowed": False,
        }

    updated = dict(checkpoint)
    updated["meta"] = dict(checkpoint["meta"])
    updated["meta"]["request"] = active_request
    serialized = pickle.dumps(updated, protocol=pickle.HIGHEST_PROTOCOL)
    checkpoint_after = hashlib.sha256(serialized).hexdigest().upper()
    journal = journal_path(journal_dir, previous, active_request)
    common = {
        "schema_version": 1,
        "evidence_version": VERSION,
        "request": active_request,
        "previous_request": previous,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "results": len(checkpoint.get("results", {})),
        "expected_tasks": len(tasks),
        "checks": {
            "same_request_id": previous["request_id"] == active_request["request_id"],
            "authorized_strategy_lineage": bool(
                previous["request_id"] == active_request["request_id"]
                or previous["request_id"] == lineage_request_id
            ),
            "monotonic_attempt": True,
            "static_checkpoint_protocol_unchanged": True,
            "partial_results_valid": True,
            "worker_not_alive": True,
        },
        "training_allowed": False,
    }
    if journal.exists():
        existing = load_json(journal, {}) or {}
        comparable = dict(existing)
        comparable.pop("status", None)
        if comparable != common:
            raise ValueError("budget-v2 retry journal collision")
        actual = file_digest(checkpoint_path)
        if actual == checkpoint_after:
            completed = dict(common) | {"status": "completed"}
            atomic_json(journal, completed)
            return completed
        if actual != checkpoint_before:
            raise ValueError("checkpoint changed after retry journal preparation")

    atomic_json(journal, dict(common) | {"status": "prepared"})
    atomic_bytes(checkpoint_path, serialized)
    if file_digest(checkpoint_path) != checkpoint_after:
        raise ValueError("checkpoint retry rebind digest mismatch")
    completed = dict(common) | {"status": "completed"}
    atomic_json(journal, completed)
    return completed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dispatch", default=".state/dispatch_request.json")
    parser.add_argument("--ack", default=".state/executor_ack.json")
    parser.add_argument("--plan", default=".state/reference_resolution_budget_v2_plan.json")
    parser.add_argument("--v1-audit", default=".state/reference_resolution_v1_audit.json")
    parser.add_argument("--v1-evidence", default=".state/reference_resolution_v1.json")
    parser.add_argument("--v1-checkpoint", default=".state/reference_resolution_v1_checkpoint.pkl")
    parser.add_argument("--checkpoint", default=".state/reference_resolution_budget_v2_checkpoint.pkl")
    parser.add_argument("--evidence", default=".state/reference_resolution_budget_v2.json")
    parser.add_argument("--journal-dir", default=".state")
    args = parser.parse_args()

    dispatch_path = ROOT / args.dispatch
    ack_path = ROOT / args.ack
    checkpoint_path = ROOT / args.checkpoint
    evidence_path = ROOT / args.evidence
    dispatch = load_json(dispatch_path, {}) or {}
    active_request = runner.dispatch_identity(dispatch_path)
    ack = load_json(ack_path, {}) or {}
    lineage = None
    lineage_request_id = None
    if checkpoint_path.is_file():
        with checkpoint_path.open("rb") as handle:
            stored = pickle.load(handle)
        stored_request = stored.get("meta", {}).get("request") if isinstance(stored, dict) else None
        if (
            isinstance(stored_request, dict)
            and stored_request.get("request_id") != active_request["request_id"]
        ):
            lineage = validate_lineage(
                ROOT,
                dispatch,
                ack,
                checkpoint_path,
                evidence_path,
                require_ready=False,
            )
            lineage_request_id = lineage["producer_request"]["request_id"]
    plan_path = ROOT / args.plan
    plan, _evidence, _baseline = runner.load_inputs(
        plan_path,
        ROOT / args.v1_audit,
        ROOT / args.v1_evidence,
        ROOT / args.v1_checkpoint,
    )
    tasks = runner.build_tasks(plan["selection"])
    result = prepare_retry(
        checkpoint_path,
        evidence_path,
        ROOT / args.journal_dir,
        active_request,
        ack,
        expected_meta(plan, plan_path, active_request),
        tasks,
        lineage_request_id,
    )
    if result.get("cross_request") is True:
        if not isinstance(lineage, dict):
            raise ValueError("audit-only result lacks validated lineage")
        now = datetime.now().astimezone()
        updated_ack = dict(ack)
        updated_ack.update(
            {
                "status": "running",
                "worker_pid": None,
                "checkpoint_path": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"),
                "heartbeat_at": now.isoformat(timespec="seconds"),
                "lease_expires_at": (now + timedelta(hours=2)).isoformat(timespec="seconds"),
            }
        )
        checks = dict(updated_ack.get("checks", {}))
        checks.update(
            {
                "audit_only_recovery": True,
                "finalization_ready": True,
                "recovery_seal": lineage["seal"],
                "completed_tasks": 96,
                "training_allowed": False,
                "checkpoint_mutated": False,
            }
        )
        updated_ack["checks"] = checks
        atomic_json(ack_path, updated_ack)
        validate_lineage(ROOT, dispatch, updated_ack, checkpoint_path, evidence_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
