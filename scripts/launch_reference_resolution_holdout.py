#!/usr/bin/env python3
"""Fail-closed launcher for the v2-bound 24-case holdout."""

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
from scripts import run_reference_resolution_holdout as holdout  # noqa: E402


VERSION = "paper2-reference-holdout-launch-preflight-v2"


def normalized(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is not a JSON object")
    return payload


def source_matches(source: Any, path: Path) -> bool:
    return bool(
        isinstance(source, dict)
        and str(source.get("path", "")).replace("\\", "/") == normalized(path)
        and str(source.get("sha256", "")).upper() == supervisor.file_digest(path)
    )


def registered_plan_sha(dispatch: dict[str, Any], plan_path: Path) -> str:
    matches = [
        str(item.get("sha256", "")).upper()
        for item in dispatch.get("strategy_evidence", [])
        if isinstance(item, dict)
        and str(item.get("path", "")).replace("\\", "/") == normalized(plan_path)
    ]
    if len(matches) != 1 or len(matches[0]) != 64:
        raise ValueError("dispatch does not bind exactly one v2 holdout plan SHA256")
    return matches[0]


def validate_source_chain(
    plan_path: Path,
    v2_plan_path: Path,
    v2_checkpoint_path: Path,
    v2_evidence_path: Path,
    v2_audit_path: Path,
    expected_plan_sha256: str,
) -> dict[str, Any]:
    plan_sha = supervisor.file_digest(plan_path)
    if plan_sha != expected_plan_sha256.upper():
        raise ValueError("v2 holdout plan SHA256 differs from dispatch")
    plan = holdout.load_plan(plan_path)
    explicit = {
        "source_v2_plan": v2_plan_path,
        "source_v2_checkpoint": v2_checkpoint_path,
        "source_v2_worker_evidence": v2_evidence_path,
        "source_v2_independent_audit": v2_audit_path,
    }
    for name, path in explicit.items():
        if not source_matches(plan.get(name), path):
            raise ValueError(f"v2 holdout plan {name} binding mismatch")
    holdout.load_source_results(plan)
    audit = read_json(v2_audit_path, "independent v2 audit")
    if audit.get("passed") is not True:
        raise ValueError("independent v2 audit is not passed")
    sources = {"plan": {"path": normalized(plan_path), "sha256": plan_sha}}
    sources.update(
        {
            name: {"path": normalized(path), "sha256": supervisor.file_digest(path)}
            for name, path in explicit.items()
        }
    )
    for name in ("archived_v1_holdout_plan", "source_base_checkpoint"):
        path = holdout.resolve_binding(plan[name], name)
        sources[name] = {"path": normalized(path), "sha256": supervisor.file_digest(path)}
    return sources


def strategy_has_sources(
    dispatch: dict[str, Any], sources: dict[str, Any], launcher_path: Path
) -> bool:
    registered = {
        (str(item.get("path", "")).replace("\\", "/"), str(item.get("sha256", "")).upper())
        for item in dispatch.get("strategy_evidence", [])
        if isinstance(item, dict)
    }
    required = {(item["path"], item["sha256"]) for item in sources.values()}
    required.add((normalized(launcher_path), supervisor.file_digest(launcher_path)))
    return required.issubset(registered)


def lock_available(path: Path) -> bool:
    if not path.exists():
        return True
    try:
        owner = json.loads(path.read_text(encoding="utf-8"))
        pid = int(owner.get("pid", -1))
    except (OSError, TypeError, ValueError):
        return True
    return not supervisor.pid_alive(pid)


def build_preflight(args: argparse.Namespace) -> dict[str, Any]:
    dispatch = read_json(ROOT / args.dispatch, "dispatch")
    paths = {
        "plan": ROOT / args.plan,
        "v2_plan": ROOT / args.v2_plan,
        "v2_checkpoint": ROOT / args.v2_checkpoint,
        "v2_evidence": ROOT / args.v2_evidence,
        "v2_audit": ROOT / args.v2_audit,
    }
    sources = validate_source_chain(
        paths["plan"],
        paths["v2_plan"],
        paths["v2_checkpoint"],
        paths["v2_evidence"],
        paths["v2_audit"],
        registered_plan_sha(dispatch, paths["plan"]),
    )
    policy = supervisor.load_policy()
    integrity = supervisor.verify_policy_integrity(policy)
    protected = supervisor.audit_protected_files(policy)
    controller = read_json(supervisor.CONTROLLER_STATE, "controller state")
    audit = read_json(supervisor.AUDIT_RESULT, "pipeline audit")
    launcher_path = Path(__file__).resolve()
    checks = {
        "policy_integrity": integrity.get("passed") is True,
        "paper1_and_legacy_assets_unchanged": all(item.get("passed") for item in protected),
        "dispatch_action": dispatch.get("action") == "reference_resolution",
        "dispatch_nonterminal": dispatch.get("status") in {"pending", "in_progress"},
        "dispatch_target": dispatch.get("target_thread_id") == policy["executor_thread_id"],
        "strategy_revision": int(dispatch.get("strategy_revision", 0)) >= 2,
        "strategy_sources_hash_bound": strategy_has_sources(dispatch, sources, launcher_path),
        "controller_training_locked": controller.get("training_allowed") is False,
        "audit_training_locked": audit.get("training_gates", {}).get("training_allowed") is False,
        "reference_gate_not_preapproved": audit.get("training_gates", {}).get("reference_resolution") is False,
        "holdout_lock_available": lock_available(ROOT / args.lock),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"holdout launch preflight failed: {failed}")
    command = [
        sys.executable,
        normalized(ROOT / "scripts/run_reference_resolution_holdout.py"),
        "--plan",
        args.plan,
        "--checkpoint",
        args.checkpoint,
        "--evidence",
        args.evidence,
        "--lock",
        args.lock,
        "--n-jobs",
        str(args.n_jobs),
        "--request-id",
        str(dispatch["request_id"]),
        "--attempt",
        str(dispatch["attempt"]),
    ]
    return {
        "schema_version": 1,
        "evidence_version": VERSION,
        "passed": True,
        "request_id": dispatch["request_id"],
        "attempt": dispatch["attempt"],
        "pool_sha256": holdout.load_plan(paths["plan"])["pool"]["sha256"],
        "checks": checks,
        "sources": sources,
        "launcher": {"path": normalized(launcher_path), "sha256": supervisor.file_digest(launcher_path)},
        "policy_integrity": integrity,
        "protected_files": protected,
        "command": command,
        "training_allowed": False,
    }


def write_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("existing v2 holdout preflight differs; use a new evidence version")
    else:
        supervisor.atomic_json(path, payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dispatch", default=".state/dispatch_request.json")
    parser.add_argument("--plan", default=".state/reference_resolution_holdout_v2_plan.json")
    parser.add_argument("--v2-plan", default=".state/reference_resolution_budget_v2_plan.json")
    parser.add_argument("--v2-checkpoint", default=".state/reference_resolution_budget_v2_checkpoint.pkl")
    parser.add_argument("--v2-evidence", default=".state/reference_resolution_budget_v2.json")
    parser.add_argument("--v2-audit", default=".state/reference_resolution_budget_v2_audit.json")
    parser.add_argument("--checkpoint", default=".state/reference_resolution_holdout_v2_checkpoint.pkl")
    parser.add_argument("--evidence", default=".state/reference_resolution_holdout_v2.json")
    parser.add_argument(
        "--preflight-evidence",
        default=".state/reference_resolution_holdout_preflight_v2_{request_id}_a{attempt}.json",
    )
    parser.add_argument("--lock", default=".state/reference_resolution_holdout_v2.lock")
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    try:
        preflight = build_preflight(args)
        preflight_path = ROOT / args.preflight_evidence.format(
            request_id=preflight["request_id"], attempt=preflight["attempt"]
        )
        write_once(preflight_path, preflight)
    except Exception as exc:
        print(json.dumps({"passed": False, "error": f"{type(exc).__name__}: {exc}"}))
        return 2
    if args.preflight_only:
        print(json.dumps({"passed": True, "request_id": preflight["request_id"]}))
        return 0
    return subprocess.run(preflight["command"], cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
