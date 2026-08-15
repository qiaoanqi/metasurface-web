#!/usr/bin/env python3
"""Finalize a terminal paper-2 worker with an independent audit and ack.

The executor owns process launch, but this script owns the durable handoff.
It never runs while the bound worker is alive, never enables training, and
never activates a pool. Scientific failures remain failures; only a fully
verified reference holdout can register its gate.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pipeline_supervisor as supervisor  # noqa: E402


VERSION = "paper2-finalizer-v1"
JOINT_ACTION = "joint_numerical_convergence"
HOLDOUT_ACTION = "reference_resolution"
TERMINAL_ACK_STATUSES = {"completed", "succeeded", "failed"}
ACTIVE_ACK_STATUSES = {"accepted", "claimed", "running", "in_progress"}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def binding(path: Path) -> dict[str, str]:
    return {"path": relative(path), "sha256": supervisor.file_digest(path)}


def load_dispatch(path: Path) -> dict[str, Any]:
    dispatch = supervisor.load_json(path, {}) or {}
    if (
        dispatch.get("action") not in {JOINT_ACTION, HOLDOUT_ACTION}
        or dispatch.get("status") != "in_progress"
        or not isinstance(dispatch.get("request_id"), str)
        or not dispatch["request_id"]
        or int(dispatch.get("attempt", 0)) < 1
    ):
        raise ValueError("finalizer requires an active paper-2 request")
    return dispatch


def load_matching_ack(path: Path, dispatch: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    ack = supervisor.load_json(path, {}) or {}
    if (
        ack.get("request_id") != dispatch["request_id"]
        or int(ack.get("attempt", 0)) != int(dispatch["attempt"])
        or ack.get("status") not in ACTIVE_ACK_STATUSES | TERMINAL_ACK_STATUSES
    ):
        raise ValueError("finalizer ack is not bound to the active request attempt")
    expected_thread = policy.get("executor_thread_id")
    actual_thread = ack.get("thread_id") or ack.get("target_thread_id")
    if expected_thread and actual_thread != expected_thread:
        raise ValueError("finalizer ack executor thread identity mismatch")
    if ack.get("worker_pid") and supervisor.pid_alive(ack["worker_pid"]):
        raise ValueError("finalizer refuses to run while the worker is alive")
    return ack


def pool_context(policy: dict, dispatch: dict) -> tuple[dict, dict, str]:
    _spec, pool = supervisor.resolve_active_pool(policy)
    pool_sha = str(dispatch.get("payload", {}).get("pool_sha256", "")).upper()
    if not pool_sha:
        pool_sha = str(pool.get("sha256", "")).upper()
    if pool_sha != str(pool.get("sha256", "")).upper():
        raise ValueError("finalizer pool SHA256 does not match the active pool")
    return _spec, pool, pool_sha


def run_auditor(script: str, output: Path, dispatch_path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, str(ROOT / script), "--output", relative(output), "--dispatch", relative(dispatch_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    audit = supervisor.load_json(output, {}) if output.is_file() else {}
    if not isinstance(audit, dict):
        audit = {}
    if not audit:
        raise ValueError(
            f"{script} produced no canonical audit (returncode={result.returncode}): "
            f"{(result.stderr or result.stdout).strip()}"
        )
    return audit


def evidence_bindings(paths: list[Path]) -> list[dict[str, str]]:
    return [binding(path) for path in paths if path.is_file()]


def write_terminal_ack(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Replace only the matching active ack, or replay an identical terminal ack."""
    if not path.is_file():
        raise ValueError("executor ack is missing; refusing to create an unbound terminal ack")
    existing = supervisor.load_json(path, {}) or {}
    same_identity = (
        existing.get("request_id") == payload.get("request_id")
        and int(existing.get("attempt", 0)) == int(payload.get("attempt", 0))
    )
    if not same_identity:
        raise ValueError("executor ack identity collision")
    existing_status = existing.get("status")
    if existing_status in ACTIVE_ACK_STATUSES:
        supervisor.atomic_json(path, payload)
        return payload
    if existing_status in TERMINAL_ACK_STATUSES:
        if existing != payload:
            raise ValueError("existing terminal executor ack differs for the same request attempt")
        return existing
    raise ValueError(f"executor ack has unsupported status: {existing_status}")


def write_once(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return write_terminal_ack(path, payload)


def finalization_diagnostic(
    dispatch_path: Path,
    ack_path: Path,
    dispatch: dict[str, Any],
    error: Exception,
) -> Path:
    path = ROOT / ".state" / "finalization_diagnostics" / (
        f"{dispatch['request_id']}-attempt{int(dispatch['attempt'])}.json"
    )
    payload = {
        "schema_version": 1,
        "evidence_version": "paper2-finalization-diagnostic-v1",
        "request": {
            "request_id": dispatch["request_id"],
            "attempt": int(dispatch["attempt"]),
            "action": dispatch.get("action"),
        },
        "classification": "execution_integrity_failure",
        "error": f"{type(error).__name__}: {error}",
        "dispatch": binding(dispatch_path),
        "active_ack_sha256": supervisor.file_digest(ack_path),
        "finalizer": binding(Path(__file__).resolve()),
        "training_allowed": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        existing = supervisor.load_json(path, {}) or {}
        if existing != payload:
            raise ValueError("finalization diagnostic collision for the same request attempt")
    else:
        supervisor.atomic_json(path, payload)
    return path


def terminal_replay(ack: dict[str, Any], pool_sha: str, policy: dict[str, Any]) -> dict[str, Any]:
    if ack.get("status") in {"completed", "succeeded"}:
        valid, error = supervisor.validate_completed_ack(ack, pool_sha, policy)
    elif ack.get("status") == "failed":
        valid, error = supervisor.validate_failed_ack(ack, pool_sha)
    else:
        raise ValueError("terminal replay requires a terminal ack")
    if not valid:
        raise ValueError(f"existing terminal executor ack failed validation: {error}")
    return ack


def ack_base(dispatch: dict, pool_sha: str, checkpoint: Path) -> dict[str, Any]:
    timestamp = now_iso()
    return {
        "schema_version": 1,
        "finalizer_version": VERSION,
        "thread_id": supervisor.load_policy()["executor_thread_id"],
        "request_id": dispatch["request_id"],
        "attempt": int(dispatch["attempt"]),
        "observed_at": timestamp,
        "heartbeat_at": timestamp,
        "worker_pid": None,
        "checkpoint_path": relative(checkpoint),
        "checks": {
            "pool_sha256": pool_sha,
            "training_allowed": False,
            "finalizer_verified_worker_dead": True,
            "finalizer_version": VERSION,
        },
    }


def finalize_joint(
    dispatch: dict, ack: dict, policy: dict, pool_sha: str, dispatch_path: Path, ack_path: Path
) -> dict[str, Any]:
    checkpoint = ROOT / str(ack.get("checkpoint_path") or ".state/reference_resolution_budget_v2_checkpoint.pkl")
    evidence = ROOT / ".state/reference_resolution_budget_v2.json"
    audit_path = ROOT / ".state/reference_resolution_budget_v2_audit.json"
    audit = run_auditor("scripts/audit_reference_resolution_budget_v2.py", audit_path, dispatch_path)
    checks = dict(audit.get("checks", {}))
    base = ack_base(dispatch, pool_sha, checkpoint)
    base["checks"].update(
        {
            "audit_passed": audit.get("passed") is True,
            "audit_classification": audit.get("classification"),
            "diagnostic_only": True,
        }
    )
    base["checks"]["independent_audit_checks"] = all(value is True for value in checks.values())
    base["evidence"] = evidence_bindings(
        [evidence, audit_path, checkpoint, ROOT / ".state/reference_resolution_budget_v2_plan.json"]
    )
    base["outputs"] = []
    base["paper_hashes"] = []
    base["status"] = "failed"
    base["failure_class"] = "scientific" if audit.get("classification") == "budget_v2_still_insufficient" or audit.get("passed") is True else "permanent"
    base["error"] = f"budget-v2 finalization: {audit.get('classification', 'missing classification')}"
    valid, error = supervisor.validate_failed_ack(base, pool_sha)
    if not valid:
        raise ValueError(f"finalizer produced invalid failure ack: {error}")
    write_terminal_ack(ack_path, base)
    return base


def register_reference_gate(audit_path: Path) -> dict[str, Any]:
    gate_path = supervisor.GATE_STATE
    state = supervisor.load_json(gate_path, {}) or {"schema_version": 1, "gates": {}}
    if state.get("schema_version") != 1 or not isinstance(state.get("gates"), dict):
        raise ValueError("gate state is malformed")
    entry = {
        "passed": True,
        "checked_at": now_iso(),
        "evidence": [binding(audit_path)],
    }
    existing = state["gates"].get("reference_resolution")
    if isinstance(existing, dict) and existing.get("passed") is True:
        if existing.get("evidence") != entry["evidence"]:
            raise ValueError("reference_resolution gate already points to different evidence")
        return state
    updated = dict(state)
    updated["gates"] = dict(state["gates"])
    updated["gates"]["reference_resolution"] = entry
    supervisor.atomic_json(gate_path, updated)
    return updated


def finalize_holdout(
    dispatch: dict, ack: dict, policy: dict, pool: dict, pool_sha: str, dispatch_path: Path, ack_path: Path
) -> dict[str, Any]:
    checkpoint = ROOT / str(ack.get("checkpoint_path") or ".state/reference_resolution_holdout_v2_checkpoint.pkl")
    evidence = ROOT / ".state/reference_resolution_holdout_v2.json"
    audit_path = ROOT / ".state/reference_resolution_holdout_v1_audit.json"
    audit = run_auditor("scripts/audit_reference_resolution_holdout.py", audit_path, dispatch_path)
    base = ack_base(dispatch, pool_sha, checkpoint)
    base["checks"].update(
        {
            "audit_passed": audit.get("passed") is True,
            "audit_classification": audit.get("classification"),
            "independent_reproduction": audit.get("independent_reproduction") is True,
        }
    )
    base["evidence"] = evidence_bindings([evidence, audit_path, checkpoint])
    base["paper_hashes"] = [
        {"path": item["path"], "md5": supervisor.file_digest(ROOT / item["path"], "md5")}
        for item in policy["protected_files"]
    ]
    if audit.get("classification") == "execution_integrity_failure":
        base["status"] = "failed"
        base["failure_class"] = "permanent"
        base["outputs"] = []
        base["error"] = "holdout independent audit execution integrity failure"
        valid, error = supervisor.validate_failed_ack(base, pool_sha)
        if not valid:
            raise ValueError(f"finalizer produced invalid failure ack: {error}")
        write_terminal_ack(ack_path, base)
        return base
    if audit.get("passed") is not True:
        base["status"] = "failed"
        base["failure_class"] = "scientific"
        base["outputs"] = []
        base["error"] = f"holdout scientific result: {audit.get('classification', 'negative')}"
        valid, error = supervisor.validate_failed_ack(base, pool_sha)
        if not valid:
            raise ValueError(f"finalizer produced invalid failure ack: {error}")
        write_terminal_ack(ack_path, base)
        return base

    valid, error = supervisor.verify_gate_payload("reference_resolution", audit, pool)
    if not valid:
        raise ValueError(f"holdout gate payload failed supervisor verification: {error}")
    register_reference_gate(audit_path)
    base["status"] = "completed"
    base["failure_class"] = None
    base["outputs"] = [
        {"path": relative(evidence), "material": "reference_resolution_holdout", "sha256": supervisor.file_digest(evidence)},
        {"path": relative(audit_path), "material": "independent_reference_audit", "sha256": supervisor.file_digest(audit_path)},
    ]
    base["checks"]["reference_gate_registered"] = True
    valid, error = supervisor.validate_completed_ack(base, pool_sha, policy)
    if not valid:
        raise ValueError(f"finalizer produced invalid completed ack: {error}")
    write_terminal_ack(ack_path, base)
    return base


def finalize(dispatch_path: Path, ack_path: Path) -> dict[str, Any]:
    dispatch = load_dispatch(dispatch_path)
    policy = supervisor.load_policy()
    ack = load_matching_ack(ack_path, dispatch, policy)
    _spec, pool, pool_sha = pool_context(policy, dispatch)
    if ack.get("status") in TERMINAL_ACK_STATUSES:
        return terminal_replay(ack, pool_sha, policy)
    if dispatch["action"] == JOINT_ACTION:
        finalizer = lambda: finalize_joint(dispatch, ack, policy, pool_sha, dispatch_path, ack_path)
    else:
        finalizer = lambda: finalize_holdout(dispatch, ack, policy, pool, pool_sha, dispatch_path, ack_path)
    try:
        return finalizer()
    except Exception as exc:
        diagnostic_path = finalization_diagnostic(dispatch_path, ack_path, dispatch, exc)
        base = ack_base(
            dispatch,
            pool_sha,
            ROOT / str(ack.get("checkpoint_path") or ".state/finalization_missing_checkpoint.pkl"),
        )
        base.update(
            {
                "status": "failed",
                "failure_class": "permanent",
                "outputs": [],
                "paper_hashes": [],
                "evidence": [binding(diagnostic_path)],
                "error": "independent finalization failed; see diagnostic evidence",
            }
        )
        valid, error = supervisor.validate_failed_ack(base, pool_sha)
        if not valid:
            raise ValueError(f"finalizer produced invalid failure ack: {error}") from exc
        write_terminal_ack(ack_path, base)
        return base


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dispatch", default=".state/dispatch_request.json")
    parser.add_argument("--ack", default=".state/executor_ack.json")
    args = parser.parse_args()
    result = finalize(ROOT / args.dispatch, ROOT / args.ack)
    print(json.dumps({"status": result["status"], "request_id": result["request_id"], "attempt": result["attempt"]}, sort_keys=True))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
