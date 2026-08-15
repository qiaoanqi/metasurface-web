#!/usr/bin/env python3
"""Fail-closed launcher for the pre-frozen 32-case reference holdout."""

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
from scripts import run_reference_resolution_escalation as candidate_runner  # noqa: E402
from scripts import run_reference_resolution_holdout as holdout  # noqa: E402


VERSION = "paper2-reference-holdout-launch-preflight-v1"
EXPECTED_PLAN_SHA256 = "B3812A34116AD62CD69BFD8AB806949F3AEEF7549702905C9BEAF38495911CFE"
CANDIDATE_AUDIT_VERSION = "paper2-reference-resolution-audit-v1"
CANDIDATE_PROTOCOL_VERSION = "paper2-reference-protocol-candidate-v1"


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
    if not isinstance(source, dict):
        return False
    return (
        str(source.get("path", "")).replace("\\", "/") == normalized(path)
        and str(source.get("sha256", "")).upper() == supervisor.file_digest(path)
    )


def validate_candidate_chain(
    plan_path: Path,
    evidence_path: Path,
    checkpoint_path: Path,
    audit_path: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    plan_sha = supervisor.file_digest(plan_path)
    if plan_sha != EXPECTED_PLAN_SHA256:
        raise ValueError("pre-frozen holdout plan SHA256 mismatch")
    plan = holdout.load_plan(plan_path)

    evidence = read_json(evidence_path, "candidate evidence")
    if evidence.get("evidence_version") != candidate_runner.VERSION or evidence.get("passed") is not True:
        raise ValueError("eight-case candidate evidence is not passed")
    if evidence.get("pool_sha256") != plan["pool"]["sha256"]:
        raise ValueError("candidate evidence pool SHA256 mismatch")
    holdout.load_candidate(evidence_path, checkpoint_path, plan)

    audit = read_json(audit_path, "independent candidate audit")
    required_checks = (
        "reference_checkpoint_exact_80",
        "reference_evidence_passed",
        "physics_controls_passed",
        "production_1nm_comparison_complete",
    )
    if (
        audit.get("evidence_version") != CANDIDATE_AUDIT_VERSION
        or audit.get("passed") is not True
        or audit.get("replacement_pool_required") is not True
        or audit.get("pool_sha256") != plan["pool"]["sha256"]
        or not all(audit.get("checks", {}).get(name) is True for name in required_checks)
        or not source_matches(audit.get("inputs", {}).get("reference_evidence"), evidence_path)
        or not source_matches(audit.get("inputs", {}).get("reference_checkpoint"), checkpoint_path)
        or not source_matches(audit.get("inputs", {}).get("plan"), ROOT / ".state/reference_resolution_v1_plan.json")
    ):
        raise ValueError("independent candidate audit is missing, failed, or not hash-linked")

    protocol = read_json(protocol_path, "candidate protocol screening")
    if (
        protocol.get("evidence_version") != CANDIDATE_PROTOCOL_VERSION
        or protocol.get("passed") is not True
        or protocol.get("approved") is not False
        or protocol.get("pool_sha256") != plan["pool"]["sha256"]
        or not source_matches(protocol.get("source_evidence"), evidence_path)
        or not source_matches(protocol.get("source_checkpoint"), checkpoint_path)
    ):
        raise ValueError("candidate protocol screening is missing, failed, or not hash-linked")

    return {
        "plan": {"path": normalized(plan_path), "sha256": plan_sha},
        "candidate_evidence": {"path": normalized(evidence_path), "sha256": supervisor.file_digest(evidence_path)},
        "candidate_checkpoint": {"path": normalized(checkpoint_path), "sha256": supervisor.file_digest(checkpoint_path)},
        "candidate_audit": {"path": normalized(audit_path), "sha256": supervisor.file_digest(audit_path)},
        "candidate_protocol": {"path": normalized(protocol_path), "sha256": supervisor.file_digest(protocol_path)},
    }


def strategy_has_sources(dispatch: dict[str, Any], sources: dict[str, Any], launcher_path: Path) -> bool:
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
    paths = {
        "plan": ROOT / args.plan,
        "candidate_evidence": ROOT / args.candidate_evidence,
        "candidate_checkpoint": ROOT / args.candidate_checkpoint,
        "candidate_audit": ROOT / args.candidate_audit,
        "candidate_protocol": ROOT / args.candidate_protocol,
    }
    sources = validate_candidate_chain(
        paths["plan"],
        paths["candidate_evidence"],
        paths["candidate_checkpoint"],
        paths["candidate_audit"],
        paths["candidate_protocol"],
    )
    policy = supervisor.load_policy()
    integrity = supervisor.verify_policy_integrity(policy)
    protected = supervisor.audit_protected_files(policy)
    dispatch = read_json(ROOT / args.dispatch, "dispatch")
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
        "--plan", args.plan,
        "--candidate-evidence", args.candidate_evidence,
        "--candidate-checkpoint", args.candidate_checkpoint,
        "--checkpoint", args.checkpoint,
        "--evidence", args.evidence,
        "--lock", args.lock,
        "--n-jobs", str(args.n_jobs),
    ]
    return {
        "schema_version": 1,
        "evidence_version": VERSION,
        "passed": True,
        "request_id": dispatch["request_id"],
        "attempt": dispatch["attempt"],
        "pool_sha256": read_json(paths["candidate_evidence"], "candidate evidence")["pool_sha256"],
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
            raise ValueError("existing holdout preflight differs; use a new evidence version")
    else:
        supervisor.atomic_json(path, payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dispatch", default=".state/dispatch_request.json")
    parser.add_argument("--plan", default=".state/reference_resolution_holdout_v1_plan.json")
    parser.add_argument("--candidate-evidence", default=".state/reference_resolution_v1.json")
    parser.add_argument("--candidate-checkpoint", default=".state/reference_resolution_v1_checkpoint.pkl")
    parser.add_argument("--candidate-audit", default=".state/reference_resolution_v1_audit.json")
    parser.add_argument("--candidate-protocol", default=".state/reference_protocol_candidate_v1.json")
    parser.add_argument("--checkpoint", default=".state/reference_resolution_holdout_v1_checkpoint.pkl")
    parser.add_argument("--evidence", default=".state/reference_resolution_holdout_v1.json")
    parser.add_argument(
        "--preflight-evidence",
        default=".state/reference_resolution_holdout_preflight_v1_{request_id}_a{attempt}.json",
    )
    parser.add_argument("--lock", default=".state/reference_resolution_holdout_v1.lock")
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
