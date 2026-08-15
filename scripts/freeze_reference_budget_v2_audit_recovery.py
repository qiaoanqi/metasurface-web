#!/usr/bin/env python3
"""Freeze an explicit lineage for re-auditing a live budget-v2 run.

This does not bless changed strategy evidence. It records the drift and proves
that the expensive solver closure stayed frozen. Only a post-terminal seal may
later authorize raw checkpoint/evidence reuse.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pipeline_supervisor as supervisor  # noqa: E402


VERSION = "paper2-reference-budget-v2-audit-recovery-v1"
RUNNER_PATHS = (
    "rcwa_batch.py",
    "paper2_colorimetry.py",
    "color_utils.py",
    "scripts/run_reference_resolution_budget_v2.py",
)
AUDIT_PATHS = (
    "scripts/prepare_reference_budget_v2_retry.py",
    "scripts/audit_reference_resolution_budget_v2.py",
    "scripts/finalize_paper2_request.py",
    "scripts/paper2_auto_transition.py",
    "pipeline_supervisor.py",
)


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def binding(path: Path) -> dict[str, str]:
    return {"path": relative(path), "sha256": supervisor.file_digest(path)}


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def build_recovery(dispatch_path: Path, ack_path: Path, checkpoint_path: Path) -> dict[str, Any]:
    policy = supervisor.load_policy()
    dispatch = supervisor.load_json(dispatch_path, {}) or {}
    ack = supervisor.load_json(ack_path, {}) or {}
    request = {
        "request_id": dispatch.get("request_id"),
        "attempt": int(dispatch.get("attempt", 0)),
        "action": dispatch.get("action"),
        "strategy_revision": int(dispatch.get("strategy_revision", 0)),
    }
    if (
        request["action"] != "joint_numerical_convergence"
        or dispatch.get("status") != "in_progress"
        or not isinstance(request["request_id"], str)
        or not request["request_id"]
        or request["attempt"] < 1
        or request["strategy_revision"] < 2
    ):
        raise ValueError("audit recovery requires the live budget-v2 joint request")
    if (
        ack.get("request_id") != request["request_id"]
        or int(ack.get("attempt", 0)) != request["attempt"]
        or ack.get("status") not in {"accepted", "claimed", "running", "in_progress"}
        or not ack.get("worker_pid")
        or not supervisor.pid_alive(ack["worker_pid"])
    ):
        raise ValueError("audit recovery requires the matching live worker acknowledgement")
    if not checkpoint_path.is_file():
        raise ValueError("budget-v2 checkpoint is missing")
    with checkpoint_path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    meta = checkpoint.get("meta") if isinstance(checkpoint, dict) else None
    if not isinstance(meta, dict) or meta.get("request") != {
        "request_id": request["request_id"],
        "attempt": request["attempt"],
    }:
        raise ValueError("budget-v2 checkpoint is not bound to the live request")
    frozen_runtime = meta.get("runtime_hashes")
    current_runtime = {path: supervisor.file_digest(ROOT / path) for path in RUNNER_PATHS}
    if frozen_runtime != current_runtime:
        raise ValueError("solver runtime changed; raw checkpoint reuse is forbidden")

    observations = []
    for item in dispatch.get("strategy_evidence", []):
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("live strategy evidence is malformed")
        path = supervisor.workspace_file(item["path"])
        expected = str(item.get("sha256", "")).upper()
        actual = supervisor.file_digest(path) if path is not None and path.is_file() else None
        observations.append(
            {
                "path": item["path"],
                "expected_sha256": expected,
                "actual_sha256": actual,
                "passed": bool(expected and actual == expected),
            }
        )
    drifted = [item for item in observations if not item["passed"]]
    if not drifted:
        raise ValueError("strategy evidence has not drifted; recovery is unnecessary")
    if any(item["path"] in RUNNER_PATHS for item in drifted):
        raise ValueError("solver runtime drifted; this audit-only recovery is invalid")
    protected = supervisor.audit_protected_files(policy)
    if not protected or not all(item.get("passed") is True for item in protected):
        raise ValueError("protected assets changed during the live request")
    pool_sha = str(dispatch.get("payload", {}).get("pool_sha256", "")).upper()
    pool_path = supervisor.workspace_file(dispatch.get("payload", {}).get("pool"))
    if pool_path is None or not pool_path.is_file() or supervisor.file_digest(pool_path) != pool_sha:
        raise ValueError("live request pool binding changed")

    return {
        "schema_version": 1,
        "evidence_version": VERSION,
        "source_request": request,
        "classification": "execution_integrity_failure",
        "reason": "strategy_evidence_changed_after_worker_start",
        "dispatch": binding(dispatch_path),
        "pool": binding(pool_path),
        "checkpoint_path": relative(checkpoint_path),
        "checkpoint_meta_sha256": canonical_sha256(meta),
        "checkpoint_completed_at_freeze": len(checkpoint.get("results", {})),
        "runner_runtime_hashes": current_runtime,
        "strategy_evidence_observations": observations,
        "drifted_strategy_evidence": drifted,
        "recovery_runtime": {path: supervisor.file_digest(ROOT / path) for path in AUDIT_PATHS},
        "protected_files": [
            {"path": item["path"], "md5": item["actual_md5"]} for item in protected
        ],
        "observation_only": True,
        "checkpoint_reuse_authorized": False,
        "worker_evidence_reuse_authorized": False,
        "scientific_outcome_authorized": False,
        "training_allowed": False,
    }


def freeze(output: Path, payload: dict[str, Any]) -> dict[str, Any]:
    if output.is_file():
        existing = supervisor.load_json(output, {}) or {}
        if existing != payload:
            raise ValueError("existing audit-recovery evidence differs")
        return existing
    supervisor.atomic_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dispatch", default=".state/dispatch_request.json")
    parser.add_argument("--ack", default=".state/executor_ack.json")
    parser.add_argument("--checkpoint", default=".state/reference_resolution_budget_v2_checkpoint.pkl")
    parser.add_argument("--output", default=".state/reference_resolution_budget_v2_audit_recovery_v1.json")
    args = parser.parse_args()
    output = ROOT / args.output
    payload = build_recovery(ROOT / args.dispatch, ROOT / args.ack, ROOT / args.checkpoint)
    freeze(output, payload)
    print(json.dumps({"output": relative(output), "source_request": payload["source_request"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
