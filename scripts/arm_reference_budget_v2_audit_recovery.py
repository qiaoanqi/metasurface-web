#!/usr/bin/env python3
"""Seal a completed integrity-failed budget run and arm audit-only recovery."""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pipeline_supervisor as supervisor  # noqa: E402
from scripts.reference_budget_v2_lineage import (  # noqa: E402
    validate_raw_results,
    validate_source_diagnostic,
)


ACTION = "joint_numerical_convergence"
RECOVERY_VERSION = "paper2-reference-budget-v2-audit-recovery-v1"
SEAL_VERSION = "paper2-reference-budget-v2-post-terminal-seal-v1"
SEAL_PATH = ".state/reference_resolution_budget_v2_post_terminal_seal_v1.json"
STATIC_EVIDENCE_PATHS = (
    ".state/reference_resolution_budget_v2_plan.json",
    ".state/reference_resolution_budget_v2_audit_recovery_v1.json",
    SEAL_PATH,
    "scripts/run_reference_resolution_budget_v2.py",
    "scripts/prepare_reference_budget_v2_retry.py",
    "scripts/audit_reference_resolution_budget_v2.py",
    "scripts/finalize_paper2_request.py",
    "scripts/paper2_auto_transition.py",
    "scripts/reference_budget_v2_lineage.py",
    "scripts/freeze_reference_budget_v2_audit_recovery.py",
    "scripts/arm_reference_budget_v2_audit_recovery.py",
    "pipeline_supervisor.py",
    "tests/test_reference_resolution_budget_v2.py",
    "tests/test_reference_budget_v2_retry.py",
    "tests/test_finalize_paper2_request.py",
    "tests/test_paper2_auto_transition.py",
    "tests/test_reference_budget_v2_audit_recovery.py",
    "tests/test_pipeline_supervisor.py",
)


def binding(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/"),
        "sha256": supervisor.file_digest(path),
    }


def strategy_instruction() -> str:
    return (
        "Audit-only recovery of the completed numerical-budget-v2 producer is authorized by the frozen "
        "post-terminal seal. After claiming this request, run python scripts/prepare_reference_budget_v2_retry.py. "
        "It must return audit_existing_evidence and atomically mark the acknowledgement finalization_ready; "
        "never start or resume a worker for this cross-request lineage. The supervisor finalizer then runs the "
        "independent auditor. Require exact archived dispatch, permanent execution-integrity final ack, diagnostic, "
        "96/96 checkpoint, worker evidence, plan, pool, runner closure, seal, and strategy hashes. Preserve source "
        "bytes. Do not change thresholds, register the historical joint gate, launch holdout, activate a pool, "
        "modify paper 1, or enable training."
    )


def exact_evidence_binding(evidence: list, path: Path) -> dict[str, str]:
    expected = binding(path)
    matches = [
        item
        for item in evidence
        if isinstance(item, dict)
        and item.get("path") == expected["path"]
        and str(item.get("sha256", "")).upper() == expected["sha256"]
    ]
    if len(matches) != 1:
        raise ValueError(f"source final ack does not bind exactly once: {expected['path']}")
    return expected


def validate_terminal_source(
    dispatch: dict, ack: dict, recovery: dict, checkpoint_path: Path, evidence_path: Path
) -> None:
    source = {
        "request_id": dispatch.get("request_id"),
        "attempt": int(dispatch.get("attempt", 0)),
    }
    if (
        dispatch.get("action") != ACTION
        or dispatch.get("status") != "failed"
        or dispatch.get("terminal_failure") is not True
        or str(dispatch.get("failure_class", "")).lower() != "permanent"
        or not isinstance(source["request_id"], str)
        or not source["request_id"]
        or source["attempt"] < 1
    ):
        raise ValueError("post-terminal seal requires the permanent failed budget request")
    if (
        ack.get("request_id") != source["request_id"]
        or int(ack.get("attempt", 0)) != source["attempt"]
        or ack.get("status") != "failed"
        or str(ack.get("failure_class", "")).lower() != "permanent"
        or ack.get("checks", {}).get("finalization_classification")
        != "execution_integrity_failure"
    ):
        raise ValueError("source final ack is not the expected integrity failure")
    if (
        recovery.get("evidence_version") != RECOVERY_VERSION
        or recovery.get("source_request", {}).get("request_id") != source["request_id"]
        or int(recovery.get("source_request", {}).get("attempt", 0)) != source["attempt"]
        or recovery.get("observation_only") is not True
        or recovery.get("scientific_outcome_authorized") is not False
        or recovery.get("training_allowed") is not False
    ):
        raise ValueError("live recovery observation is not bound to the source request")
    if not checkpoint_path.is_file() or not evidence_path.is_file():
        raise ValueError("complete checkpoint and worker evidence are required")
    with checkpoint_path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    validate_raw_results(checkpoint)
    meta = checkpoint.get("meta") if isinstance(checkpoint, dict) else None
    results = checkpoint.get("results") if isinstance(checkpoint, dict) else None
    if (
        not isinstance(meta, dict)
        or meta.get("request") != source
        or int(meta.get("expected_tasks", 0)) != 96
        or not isinstance(results, dict)
        or len(results) != 96
    ):
        raise ValueError("cross-request recovery requires the complete 96-task source checkpoint")
    evidence = supervisor.load_json(evidence_path, {}) or {}
    checkpoint_binding = evidence.get("checkpoint")
    expected_checkpoint = binding(checkpoint_path)
    if (
        evidence.get("evidence_version") != "paper2-reference-resolution-budget-v2"
        or evidence.get("request") != source
        or evidence.get("training_allowed") is not False
        or not isinstance(checkpoint_binding, dict)
        or checkpoint_binding.get("path") != expected_checkpoint["path"]
        or str(checkpoint_binding.get("sha256", "")).upper()
        != expected_checkpoint["sha256"]
        or int(checkpoint_binding.get("tasks", 0)) != 96
    ):
        raise ValueError("worker evidence is not bound to the complete source checkpoint")
    final_evidence = ack.get("evidence", [])
    exact_evidence_binding(final_evidence, checkpoint_path)
    exact_evidence_binding(final_evidence, evidence_path)
    diagnostic_items = []
    for item in final_evidence:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            continue
        path = supervisor.workspace_file(item["path"])
        if path is None or not path.is_file() or path.suffix.lower() != ".json":
            continue
        payload = supervisor.load_json(path, {}) or {}
        if payload.get("classification") == "execution_integrity_failure":
            if str(item.get("sha256", "")).upper() != supervisor.file_digest(path):
                raise ValueError("finalization diagnostic hash mismatch")
            validate_source_diagnostic(payload, source)
            diagnostic_items.append(item)
    if len(diagnostic_items) != 1:
        raise ValueError("source final ack must bind exactly one integrity diagnostic")


def create_seal(
    dispatch: dict,
    ack: dict,
    recovery_path: Path,
    checkpoint_path: Path,
    evidence_path: Path,
    history_path: Path,
    revision: int,
) -> dict:
    pool_sha = str(dispatch.get("payload", {}).get("pool_sha256", "")).upper()
    target_request_id = supervisor.make_dispatch_id(
        "paper2_pipeline", ACTION, pool_sha, revision, "retry_same_gate", dispatch["request_id"]
    )
    with checkpoint_path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    return {
        "schema_version": 1,
        "evidence_version": SEAL_VERSION,
        "source_request": {
            "request_id": dispatch["request_id"],
            "attempt": int(dispatch["attempt"]),
        },
        "target_request": {
            "request_id": target_request_id,
            "attempt": 1,
            "strategy_revision": revision,
        },
        "source_dispatch_history": binding(history_path),
        "source_final_ack": ack,
        "live_recovery_observation": binding(recovery_path),
        "checkpoint": binding(checkpoint_path) | {"tasks": len(checkpoint["results"])},
        "worker_evidence": binding(evidence_path),
        "plan": binding(ROOT / ".state/reference_resolution_budget_v2_plan.json"),
        "pool_sha256": pool_sha,
        "runner_runtime_hashes": checkpoint["meta"]["runtime_hashes"],
        "audit_only": True,
        "checkpoint_reuse_authorized": True,
        "worker_evidence_reuse_authorized": True,
        "checkpoint_mutation_authorized": False,
        "scientific_outcome_authorized": False,
        "training_allowed": False,
    }


def apply(policy_path: Path, integrity_path: Path, dispatch_path: Path, ack_path: Path) -> dict:
    policy = supervisor.load_json(policy_path, {}) or {}
    lock = supervisor.load_json(integrity_path, {}) or {}
    dispatch = supervisor.load_json(dispatch_path, {}) or {}
    ack = supervisor.load_json(ack_path, {}) or {}
    if supervisor.file_digest(policy_path) != str(lock.get("policy_sha256", "")).upper():
        raise ValueError("policy does not match integrity lock")
    supervisor_path = ROOT / "pipeline_supervisor.py"
    if supervisor.file_digest(supervisor_path) != str(lock.get("supervisor_sha256", "")).upper():
        raise ValueError("supervisor does not match integrity lock")
    recovery_path = ROOT / ".state/reference_resolution_budget_v2_audit_recovery_v1.json"
    checkpoint_path = ROOT / ".state/reference_resolution_budget_v2_checkpoint.pkl"
    evidence_path = ROOT / ".state/reference_resolution_budget_v2.json"
    recovery = supervisor.load_json(recovery_path, {}) or {}
    validate_terminal_source(dispatch, ack, recovery, checkpoint_path, evidence_path)
    history_path = supervisor.archive_dispatch(dispatch, ack, ACTION)
    if history_path is None or not history_path.is_file():
        raise ValueError("failed source request could not be archived")

    current = policy.get("strategy_override", {})
    if current.get("based_on_request_id") == dispatch["request_id"]:
        revision = int(current.get("revision", 0))
    else:
        revision = max(
            int(current.get("revision", 0)), int(dispatch.get("strategy_revision", 0))
        ) + 1
    seal_path = ROOT / SEAL_PATH
    seal = create_seal(
        dispatch, ack, recovery_path, checkpoint_path, evidence_path, history_path, revision
    )
    if seal_path.is_file():
        if (supervisor.load_json(seal_path, {}) or {}) != seal:
            raise ValueError("existing post-terminal recovery seal differs")
    else:
        supervisor.atomic_json(seal_path, seal)

    evidence_paths = list(STATIC_EVIDENCE_PATHS) + [
        str(history_path.relative_to(ROOT)).replace("\\", "/")
    ]
    evidence = []
    for name in evidence_paths:
        path = ROOT / name
        if not path.is_file():
            raise ValueError(f"audit recovery strategy evidence is missing: {name}")
        evidence.append(binding(path))
    strategy = {
        "enabled": True,
        "decision": "retry_same_gate",
        "revision": revision,
        "action": ACTION,
        "based_on_request_id": dispatch["request_id"],
        "instruction_append": strategy_instruction(),
        "evidence": evidence,
    }
    if current.get("based_on_request_id") == dispatch["request_id"]:
        if current != strategy:
            raise ValueError("a different recovery strategy is already armed")
        return {
            "status": "already_armed",
            "strategy_revision": revision,
            "target_request_id": seal["target_request"]["request_id"],
            "integrity_revision": int(lock["protected_assets_revision"]),
        }

    updated = copy.deepcopy(policy)
    updated["strategy_override"] = strategy
    supervisor.atomic_json(policy_path, updated)
    new_lock = {
        "schema_version": 1,
        "policy_sha256": supervisor.file_digest(policy_path),
        "supervisor_sha256": supervisor.file_digest(supervisor_path),
        "protected_assets_revision": int(lock.get("protected_assets_revision", 0)) + 1,
        "note": lock.get("note", "Intentional policy and supervisor integrity revision."),
    }
    supervisor.atomic_json(integrity_path, new_lock)
    return {
        "status": "armed",
        "strategy_revision": revision,
        "target_request_id": seal["target_request"]["request_id"],
        "based_on_request_id": dispatch["request_id"],
        "integrity_revision": new_lock["protected_assets_revision"],
        "policy_sha256": new_lock["policy_sha256"],
        "supervisor_sha256": new_lock["supervisor_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="pipeline_policy.json")
    parser.add_argument("--integrity", default=".state/pipeline_integrity.json")
    parser.add_argument("--dispatch", default=".state/dispatch_request.json")
    parser.add_argument("--ack", default=".state/executor_ack.json")
    args = parser.parse_args()
    result = apply(
        ROOT / args.policy,
        ROOT / args.integrity,
        ROOT / args.dispatch,
        ROOT / args.ack,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
