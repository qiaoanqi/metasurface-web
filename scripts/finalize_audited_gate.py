#!/usr/bin/env python3
"""Finalize a policy-registered worker with an independent gate auditor."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pipeline_supervisor as supervisor  # noqa: E402
from scripts import finalize_paper2_request as common  # noqa: E402


VERSION = "paper2-audited-gate-finalizer-v1"


def action_spec(policy: dict[str, Any], action: str) -> dict[str, Any]:
    matches = [
        item
        for item in policy.get("workflow", {}).get("actions", [])
        if item.get("action") == action
    ]
    if len(matches) != 1:
        raise ValueError("audited finalizer requires exactly one workflow action")
    spec = matches[0]
    required = {
        "gate",
        "evidence_version",
        "auditor_script",
        "worker_evidence",
        "audit_evidence",
        "finalizer",
    }
    if not required <= set(spec):
        raise ValueError("audited finalizer workflow contract is incomplete")
    if spec.get("finalizer") != "scripts/finalize_audited_gate.py":
        raise ValueError("audited finalizer is not registered for this action")
    return spec


def register_provisional_gate(gate: str, audit_path: Path) -> None:
    state = supervisor.load_json(supervisor.GATE_STATE, {}) or {
        "schema_version": 1,
        "gates": {},
    }
    if state.get("schema_version") != 1 or not isinstance(state.get("gates"), dict):
        raise ValueError("gate state is malformed")
    entry = {
        "passed": True,
        "phase": "provisional_until_completed_ack",
        "checked_at": supervisor.now_iso(),
        "evidence": [common.binding(audit_path)],
    }
    existing = state["gates"].get(gate)
    if isinstance(existing, dict) and existing.get("passed") is True:
        if existing.get("evidence") != entry["evidence"]:
            raise ValueError(f"{gate} already points to different evidence")
        return
    updated = dict(state)
    updated["gates"] = dict(state["gates"])
    updated["gates"][gate] = entry
    supervisor.atomic_json(supervisor.GATE_STATE, updated)


def validate_audit(
    audit: dict[str, Any], dispatch: dict[str, Any], spec: dict[str, Any], pool_sha: str
) -> None:
    request = {
        "request_id": dispatch["request_id"],
        "attempt": int(dispatch["attempt"]),
    }
    if (
        audit.get("schema_version") != 1
        or audit.get("evidence_version") != spec["evidence_version"]
        or audit.get("request") != request
        or not isinstance(audit.get("passed"), bool)
        or audit.get("training_allowed") is not False
        or str(audit.get("pool_sha256", "")).upper() != pool_sha
        or audit.get("independent_reproduction") is not True
    ):
        raise ValueError("independent gate audit identity is invalid")
    valid, error = supervisor.verify_file_binding(
        audit.get("worker_evidence"), "audited gate worker evidence"
    )
    if not valid:
        raise ValueError(error or "audited gate worker evidence is invalid")


def paper_hashes(policy: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "path": item["path"],
            "md5": supervisor.file_digest(ROOT / item["path"], "md5"),
        }
        for item in policy["protected_files"]
    ]


def run_auditor(script: str, output: Path) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, str(ROOT / script), "--output", common.relative(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,
        check=False,
    )
    audit = supervisor.load_json(output, {}) if output.is_file() else {}
    if not isinstance(audit, dict) or not audit:
        raise ValueError(
            f"{script} produced no canonical audit (returncode={result.returncode}): "
            f"{(result.stderr or result.stdout).strip()}"
        )
    expected = 0 if audit.get("passed") is True else 2 if audit.get("passed") is False else None
    if expected is None or result.returncode != expected:
        raise ValueError(
            f"{script} returncode/audit outcome mismatch: "
            f"returncode={result.returncode}, passed={audit.get('passed')!r}"
        )
    return audit


def finalize(dispatch_path: Path, ack_path: Path) -> dict[str, Any]:
    dispatch = supervisor.load_json(dispatch_path, {}) or {}
    if (
        dispatch.get("status") != "in_progress"
        or not isinstance(dispatch.get("action"), str)
        or not dispatch["action"]
        or not isinstance(dispatch.get("request_id"), str)
        or not dispatch["request_id"]
        or int(dispatch.get("attempt", 0)) < 1
    ):
        raise ValueError("audited finalizer requires an active paper-2 request")
    policy = supervisor.load_policy()
    ack = common.load_matching_ack(ack_path, dispatch, policy)
    _pool_spec, pool, pool_sha = common.pool_context(policy, dispatch)
    if ack.get("status") in common.TERMINAL_ACK_STATUSES:
        return common.terminal_replay(ack, pool_sha, policy)
    spec = action_spec(policy, dispatch["action"])
    worker_path = supervisor.workspace_file(spec["worker_evidence"])
    audit_path = supervisor.workspace_file(spec["audit_evidence"])
    checkpoint = supervisor.workspace_file(
        ack.get("checkpoint_path") or spec.get("checkpoint") or spec["worker_evidence"]
    )
    if worker_path is None or audit_path is None or checkpoint is None:
        raise ValueError("audited finalizer paths are outside the workspace")

    try:
        common.verify_dispatch_strategy_evidence(dispatch)
        audit = run_auditor(spec["auditor_script"], audit_path)
        validate_audit(audit, dispatch, spec, pool_sha)
        base = common.ack_base(dispatch, pool_sha, checkpoint)
        base["finalizer_version"] = VERSION
        base["checks"].update(
            {
                "finalizer_version": VERSION,
                "audit_passed": audit["passed"],
                "audit_classification": audit.get("classification"),
                "independent_reproduction": True,
                "training_allowed": False,
            }
        )
        evidence_paths = [worker_path, audit_path]
        if checkpoint.is_file() and checkpoint not in evidence_paths:
            evidence_paths.append(checkpoint)
        base["evidence"] = common.evidence_bindings(evidence_paths)

        if audit["passed"] is True:
            valid, error = supervisor.verify_gate_payload(spec["gate"], audit, pool)
            if not valid:
                raise ValueError(f"gate payload failed supervisor verification: {error}")
            register_provisional_gate(spec["gate"], audit_path)
            base["status"] = "completed"
            base["failure_class"] = None
            base["outputs"] = [
                {
                    "path": common.relative(path),
                    "material": "independent_gate_audit" if path == audit_path else "gate_worker_evidence",
                    "sha256": supervisor.file_digest(path),
                }
                for path in (worker_path, audit_path)
            ]
            base["paper_hashes"] = paper_hashes(policy)
            base["checks"]["gate_commit_phase"] = "ack_commits_provisional_gate"
            valid, error = supervisor.validate_completed_ack(base, pool_sha, policy)
            if not valid:
                raise ValueError(f"audited finalizer produced invalid completed ack: {error}")
        else:
            classification = str(audit.get("classification", ""))
            scientific = set(spec.get("scientific_failure_classifications", []))
            permanent = set(spec.get("permanent_failure_classifications", []))
            if classification not in scientific | permanent:
                raise ValueError("audited finalizer received an unregistered failure classification")
            base["status"] = "failed"
            base["failure_class"] = "scientific" if classification in scientific else "permanent"
            base["outputs"] = []
            base["paper_hashes"] = []
            base["error"] = f"independent gate audit: {classification}"
            valid, error = supervisor.validate_failed_ack(base, pool_sha)
            if not valid:
                raise ValueError(f"audited finalizer produced invalid failure ack: {error}")
        common.write_terminal_ack(ack_path, base)
        return base
    except Exception as exc:
        diagnostic = common.finalization_diagnostic(dispatch_path, ack_path, dispatch, exc)
        base = common.ack_base(dispatch, pool_sha, checkpoint)
        base.update(
            {
                "finalizer_version": VERSION,
                "status": "failed",
                "failure_class": "permanent",
                "outputs": [],
                "paper_hashes": [],
                "evidence": common.evidence_bindings(
                    [path for path in (diagnostic, worker_path, checkpoint) if path.is_file()]
                ),
                "error": "independent gate finalization failed; see diagnostic evidence",
            }
        )
        base["checks"]["finalization_classification"] = "execution_integrity_failure"
        valid, error = supervisor.validate_failed_ack(base, pool_sha)
        if not valid:
            raise ValueError(f"audited finalizer produced invalid failure ack: {error}") from exc
        common.write_terminal_ack(ack_path, base)
        return base


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dispatch", default=".state/dispatch_request.json")
    parser.add_argument("--ack", default=".state/executor_ack.json")
    args = parser.parse_args()
    result = finalize(ROOT / args.dispatch, ROOT / args.ack)
    print(
        json.dumps(
            {
                "status": result["status"],
                "request_id": result["request_id"],
                "attempt": result["attempt"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
