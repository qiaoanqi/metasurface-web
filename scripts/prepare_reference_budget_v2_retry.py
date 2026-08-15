#!/usr/bin/env python3
"""Safely authorize a budget-v2 checkpoint for a later retry attempt."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_supervisor import atomic_json, file_digest, load_json, pid_alive  # noqa: E402
from scripts import run_reference_resolution_budget_v2 as runner  # noqa: E402


VERSION = "paper2-reference-resolution-budget-v2-retry-v1"
ACTIVE_ACK_STATUSES = {"accepted", "claimed", "running", "in_progress"}


def reusable_request(previous: object, active: dict[str, Any]) -> bool:
    return bool(
        isinstance(previous, dict)
        and previous.get("request_id") == active.get("request_id")
        and 1 <= int(previous.get("attempt", 0)) <= int(active.get("attempt", 0))
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
) -> dict[str, Any]:
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("meta"), dict):
        raise ValueError("budget-v2 retry checkpoint is malformed")
    stored_request = checkpoint["meta"].get("request")
    if not reusable_request(stored_request, active_request):
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
    return directory / (
        f"reference_resolution_budget_v2_retry_{request_id}_"
        f"a{previous['attempt']}_to_a{active['attempt']}.json"
    )


def prepare_retry(
    checkpoint_path: Path,
    evidence_path: Path,
    journal_dir: Path,
    active_request: dict[str, Any],
    ack: dict,
    expected: dict,
    tasks: list[dict],
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
    previous = validate_checkpoint(checkpoint, expected, tasks, active_request)
    checkpoint_before = file_digest(checkpoint_path)

    if evidence_path.is_file():
        evidence = load_json(evidence_path, {}) or {}
        checkpoint_binding = evidence.get("checkpoint", {})
        if (
            reusable_request(evidence.get("request"), active_request)
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
            "same_request_id": True,
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

    active_request = runner.dispatch_identity(ROOT / args.dispatch)
    ack = load_json(ROOT / args.ack, {}) or {}
    plan_path = ROOT / args.plan
    plan, _evidence, _baseline = runner.load_inputs(
        plan_path,
        ROOT / args.v1_audit,
        ROOT / args.v1_evidence,
        ROOT / args.v1_checkpoint,
    )
    tasks = runner.build_tasks(plan["selection"])
    result = prepare_retry(
        ROOT / args.checkpoint,
        ROOT / args.evidence,
        ROOT / args.journal_dir,
        active_request,
        ack,
        expected_meta(plan, plan_path, active_request),
        tasks,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
