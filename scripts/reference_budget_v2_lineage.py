#!/usr/bin/env python3
"""Validate the sealed cross-request lineage for budget-v2 audit-only recovery."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


SEAL_VERSION = "paper2-reference-budget-v2-post-terminal-seal-v1"
RECOVERY_VERSION = "paper2-reference-budget-v2-audit-recovery-v1"
WORKER_VERSION = "paper2-reference-resolution-budget-v2"
DIAGNOSTIC_VERSION = "paper2-finalization-diagnostic-v1"
ACTION = "joint_numerical_convergence"
ACTIVE_ACK_STATUSES = {"accepted", "claimed", "running", "in_progress"}


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def resolve(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("lineage path is missing")
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("lineage path escapes workspace") from exc
    return path


def binding(root: Path, path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve().relative_to(root.resolve())).replace("\\", "/"),
        "sha256": file_digest(path),
    }


def require_binding(root: Path, item: object, label: str) -> Path:
    if not isinstance(item, dict):
        raise ValueError(f"{label} binding is missing")
    path = resolve(root, item.get("path"))
    expected = str(item.get("sha256", "")).upper()
    if not path.is_file() or not expected or file_digest(path) != expected:
        raise ValueError(f"{label} binding mismatch")
    return path


def exact_bound(evidence: object, expected: dict[str, str], label: str) -> None:
    if not isinstance(evidence, list):
        raise ValueError(f"{label} evidence list is missing")
    matches = [
        item
        for item in evidence
        if isinstance(item, dict)
        and item.get("path") == expected["path"]
        and str(item.get("sha256", "")).upper() == expected["sha256"]
    ]
    if len(matches) != 1:
        raise ValueError(f"{label} is not bound exactly once")


def validate_raw_results(checkpoint: dict[str, Any]) -> None:
    meta = checkpoint.get("meta")
    results = checkpoint.get("results")
    if (
        not isinstance(meta, dict)
        or int(meta.get("expected_tasks", 0)) != 96
        or not isinstance(meta.get("tasks"), list)
        or len(meta["tasks"]) != 96
        or not isinstance(results, dict)
        or len(results) != 96
    ):
        raise ValueError("audit-only lineage requires an exact 96-task checkpoint")
    task_ids = [task.get("id") for task in meta["tasks"] if isinstance(task, dict)]
    if len(task_ids) != 96 or len(set(task_ids)) != 96 or set(results) != set(task_ids):
        raise ValueError("audit-only checkpoint task identity mismatch")
    for task_id, result in results.items():
        if not isinstance(result, dict) or result.get("status") != "ok":
            raise ValueError(f"invalid source result status: {task_id}")
        wavelength = np.asarray(result.get("wavelength_nm"), dtype=float)
        reflectance = np.asarray(result.get("R"), dtype=float)
        transmittance = np.asarray(result.get("T"), dtype=float)
        if (
            wavelength.ndim != 1
            or wavelength.size == 0
            or reflectance.shape != wavelength.shape
            or transmittance.shape != wavelength.shape
            or not np.isfinite(wavelength).all()
            or not np.isfinite(reflectance).all()
            or not np.isfinite(transmittance).all()
            or float(np.max(np.abs(reflectance + transmittance - 1.0))) > 1e-6
        ):
            raise ValueError(f"invalid source spectrum: {task_id}")


def validate_source_diagnostic(payload: dict[str, Any], source: dict[str, Any]) -> None:
    if (
        not isinstance(source, dict)
        or set(source) != {"request_id", "attempt", "action"}
        or not isinstance(source.get("request_id"), str)
        or not source["request_id"]
        or not isinstance(source.get("attempt"), int)
        or isinstance(source.get("attempt"), bool)
        or source["attempt"] < 1
        or source.get("action") != ACTION
    ):
        raise ValueError("source diagnostic identity is invalid")
    if (
        payload.get("schema_version") != 1
        or payload.get("evidence_version") != DIAGNOSTIC_VERSION
        or payload.get("classification") != "execution_integrity_failure"
        or payload.get("request") != source
        or payload.get("training_allowed") is not False
    ):
        raise ValueError("source finalization diagnostic identity is invalid")


def find_seal(root: Path, dispatch: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    evidence = dispatch.get("strategy_evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("audit-only request has no frozen strategy evidence")
    seals = []
    for item in evidence:
        path = require_binding(root, item, "strategy evidence")
        if path.suffix.lower() != ".json":
            continue
        candidate = load_json(path)
        if candidate.get("evidence_version") == SEAL_VERSION:
            seals.append((path, candidate))
    if len(seals) != 1:
        raise ValueError("audit-only request must bind exactly one post-terminal seal")
    return seals[0]


def validate_lineage(
    root: Path,
    dispatch: dict[str, Any],
    ack: dict[str, Any],
    checkpoint_path: Path,
    evidence_path: Path,
    *,
    require_ready: bool = True,
) -> dict[str, Any]:
    active = {
        "request_id": dispatch.get("request_id"),
        "attempt": int(dispatch.get("attempt", 0)),
    }
    if (
        dispatch.get("action") != "joint_numerical_convergence"
        or dispatch.get("status") not in {"pending", "in_progress", "failed"}
        or not isinstance(active["request_id"], str)
        or not active["request_id"]
        or active["attempt"] < 1
    ):
        raise ValueError("audit-only lineage requires the active joint request")
    source_request_id = dispatch.get("strategy_based_on")
    if not isinstance(source_request_id, str) or not source_request_id:
        raise ValueError("audit-only request has no source strategy lineage")
    seal_path, seal = find_seal(root, dispatch)
    target = seal.get("target_request")
    source = seal.get("source_request")
    try:
        first_attempt = int(target.get("attempt", 0)) if isinstance(target, dict) else 0
        max_attempts = int(target.get("max_attempts", 0)) if isinstance(target, dict) else 0
        dispatch_max_attempts = int(dispatch.get("max_attempts", 0))
    except (TypeError, ValueError):
        first_attempt = max_attempts = dispatch_max_attempts = 0
    if (
        not isinstance(target, dict)
        or target.get("request_id") != active["request_id"]
        or first_attempt != 1
        or max_attempts < first_attempt
        or dispatch_max_attempts != max_attempts
        or not first_attempt <= active["attempt"] <= max_attempts
        or int(target.get("strategy_revision", 0))
        != int(dispatch.get("strategy_revision", 0))
        or not isinstance(source, dict)
        or source.get("request_id") != source_request_id
        or int(source.get("attempt", 0)) < 1
        or seal.get("audit_only") is not True
        or seal.get("checkpoint_mutation_authorized") is not False
        or seal.get("training_allowed") is not False
    ):
        raise ValueError("post-terminal seal request identity mismatch")

    history_path = require_binding(root, seal.get("source_dispatch_history"), "source history")
    history = load_json(history_path)
    source_dispatch = history.get("request")
    source_ack = history.get("final_ack")
    if (
        not isinstance(source_dispatch, dict)
        or source_dispatch.get("request_id") != source["request_id"]
        or int(source_dispatch.get("attempt", 0)) != int(source["attempt"])
        or source_dispatch.get("action") != ACTION
        or source_dispatch.get("status") != "failed"
        or source_dispatch.get("terminal_failure") is not True
        or str(source_dispatch.get("failure_class", "")).lower() != "permanent"
        or not isinstance(source_ack, dict)
        or source_ack != seal.get("source_final_ack")
        or source_ack.get("status") != "failed"
        or str(source_ack.get("failure_class", "")).lower() != "permanent"
        or source_ack.get("checks", {}).get("finalization_classification")
        != "execution_integrity_failure"
    ):
        raise ValueError("archived source terminal state is invalid")

    recovery_path = require_binding(
        root, seal.get("live_recovery_observation"), "live recovery observation"
    )
    recovery = load_json(recovery_path)
    if (
        recovery.get("evidence_version") != RECOVERY_VERSION
        or recovery.get("source_request", {}).get("request_id") != source["request_id"]
        or int(recovery.get("source_request", {}).get("attempt", 0))
        != int(source["attempt"])
        or recovery.get("observation_only") is not True
        or recovery.get("checkpoint_reuse_authorized") is not False
    ):
        raise ValueError("live recovery observation is not fail-closed")

    if require_binding(root, seal.get("checkpoint"), "sealed checkpoint") != checkpoint_path.resolve():
        raise ValueError("audit-only checkpoint path differs from seal")
    if require_binding(root, seal.get("worker_evidence"), "sealed worker evidence") != evidence_path.resolve():
        raise ValueError("audit-only worker evidence path differs from seal")
    plan_path = require_binding(root, seal.get("plan"), "sealed plan")
    with checkpoint_path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    validate_raw_results(checkpoint)
    if checkpoint.get("meta", {}).get("request") != source:
        raise ValueError("checkpoint producer request differs from seal")
    if checkpoint.get("meta", {}).get("runtime_hashes") != seal.get("runner_runtime_hashes"):
        raise ValueError("sealed runner closure differs from checkpoint")
    for name, expected in seal["runner_runtime_hashes"].items():
        path = resolve(root, name)
        if not path.is_file() or file_digest(path) != str(expected).upper():
            raise ValueError(f"runner runtime changed after sealing: {name}")
    evidence = load_json(evidence_path)
    if (
        evidence.get("evidence_version") != WORKER_VERSION
        or evidence.get("request") != source
        or evidence.get("plan") != binding(root, plan_path)
        or evidence.get("training_allowed") is not False
        or str(evidence.get("pool_sha256", "")).upper()
        != str(seal.get("pool_sha256", "")).upper()
    ):
        raise ValueError("sealed worker evidence is invalid")
    if (
        require_binding(root, evidence.get("checkpoint"), "worker checkpoint")
        != checkpoint_path.resolve()
        or int(evidence.get("checkpoint", {}).get("tasks", 0)) != 96
    ):
        raise ValueError("worker evidence checkpoint is not complete")
    exact_bound(source_ack.get("evidence"), binding(root, checkpoint_path), "source checkpoint")
    exact_bound(source_ack.get("evidence"), binding(root, evidence_path), "source worker evidence")
    diagnostic_count = 0
    for item in source_ack.get("evidence", []):
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            continue
        path = resolve(root, item["path"])
        if not path.is_file() or path.suffix.lower() != ".json":
            continue
        payload = load_json(path)
        if payload.get("classification") == "execution_integrity_failure":
            require_binding(root, item, "source finalization diagnostic")
            validate_source_diagnostic(
                payload,
                {
                    "request_id": source["request_id"],
                    "attempt": int(source["attempt"]),
                    "action": source_dispatch["action"],
                },
            )
            diagnostic_count += 1
    if diagnostic_count != 1:
        raise ValueError("source final ack lacks one exact integrity diagnostic")

    if (
        ack.get("request_id") != active["request_id"]
        or int(ack.get("attempt", 0)) != active["attempt"]
        or ack.get("status") not in ACTIVE_ACK_STATUSES | {"failed"}
        or ack.get("worker_pid") is not None
    ):
        raise ValueError("active acknowledgement identity is invalid for audit-only recovery")
    if require_ready and (
        ack.get("checks", {}).get("audit_only_recovery") is not True
        or ack.get("checks", {}).get("finalization_ready") is not True
        or ack.get("checks", {}).get("recovery_seal") != binding(root, seal_path)
        or ack.get("checks", {}).get("completed_tasks") != 96
        or ack.get("checks", {}).get("training_allowed") is not False
    ):
        raise ValueError("active acknowledgement is not sealed audit-only recovery")
    return {
        "producer_request": source,
        "active_request": active,
        "seal": binding(root, seal_path),
        "history": binding(root, history_path),
        "checkpoint": binding(root, checkpoint_path),
        "worker_evidence": binding(root, evidence_path),
    }
