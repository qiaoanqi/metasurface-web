#!/usr/bin/env python3
"""Independently reproduce the 32-case production-reference decision."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pipeline_supervisor as supervisor  # noqa: E402
from scripts import run_reference_resolution_holdout as holdout  # noqa: E402


VERSION = "paper2-reference-holdout-audit-v1"


def build_audit(
    plan_path: Path,
    candidate_evidence_path: Path,
    candidate_checkpoint_path: Path,
    holdout_evidence_path: Path,
    holdout_checkpoint_path: Path,
) -> dict:
    plan = holdout.load_plan(plan_path)
    candidate = holdout.load_candidate(candidate_evidence_path, candidate_checkpoint_path, plan)
    with holdout_checkpoint_path.open("rb") as handle:
        extension = pickle.load(handle)
    expected_ids = {task["id"] for task in holdout.build_new_tasks(plan)}
    actual_ids = set(extension.get("results", {}))
    if actual_ids != expected_ids:
        raise ValueError("holdout checkpoint task identities are not exact 240/240")
    combined = dict(candidate["results"])
    combined.update(extension["results"])
    recomputed = holdout.summarize(
        plan,
        combined,
        plan_path,
        candidate_evidence_path,
        candidate_checkpoint_path,
        holdout_checkpoint_path,
    )
    stored = json.loads(holdout_evidence_path.read_text(encoding="utf-8"))
    evidence_exact = stored == recomputed
    policy = supervisor.load_policy()
    integrity = supervisor.verify_policy_integrity(policy)
    protected = supervisor.audit_protected_files(policy)
    checks = {
        "policy_integrity": integrity.get("passed") is True,
        "paper1_and_legacy_assets_unchanged": all(item.get("passed") for item in protected),
        "exact_240_extension_tasks": actual_ids == expected_ids,
        "exact_320_combined_tasks": len(combined) == 320,
        "worker_evidence_exactly_reproduced": evidence_exact,
        "production_reference_approved": recomputed.get("production_reference_approved") is True,
        "lowest_cost_protocol_identified": recomputed.get("protocol_selection", {}).get("lowest_cost_passing_protocol") is not None,
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "evidence_version": VERSION,
        "passed": passed,
        "production_reference_approved": passed,
        "checks": checks,
        "approved_protocol_candidate": recomputed.get("protocol_selection", {}).get("lowest_cost_passing_protocol"),
        "pool_sha256": recomputed["pool_sha256"],
        "thresholds": recomputed["thresholds"],
        "sources": {
            "plan": {"path": str(plan_path.relative_to(ROOT)).replace("\\", "/"), "sha256": supervisor.file_digest(plan_path)},
            "candidate_evidence": {"path": str(candidate_evidence_path.relative_to(ROOT)).replace("\\", "/"), "sha256": supervisor.file_digest(candidate_evidence_path)},
            "candidate_checkpoint": {"path": str(candidate_checkpoint_path.relative_to(ROOT)).replace("\\", "/"), "sha256": supervisor.file_digest(candidate_checkpoint_path)},
            "holdout_evidence": {"path": str(holdout_evidence_path.relative_to(ROOT)).replace("\\", "/"), "sha256": supervisor.file_digest(holdout_evidence_path)},
            "holdout_checkpoint": {"path": str(holdout_checkpoint_path.relative_to(ROOT)).replace("\\", "/"), "sha256": supervisor.file_digest(holdout_checkpoint_path)},
        },
        "protected_files": protected,
        "training_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default=".state/reference_resolution_holdout_v1_plan.json")
    parser.add_argument("--candidate-evidence", default=".state/reference_resolution_v1.json")
    parser.add_argument("--candidate-checkpoint", default=".state/reference_resolution_v1_checkpoint.pkl")
    parser.add_argument("--holdout-evidence", default=".state/reference_resolution_holdout_v1.json")
    parser.add_argument("--holdout-checkpoint", default=".state/reference_resolution_holdout_v1_checkpoint.pkl")
    parser.add_argument("--output", default=".state/reference_resolution_holdout_v1_audit.json")
    args = parser.parse_args()
    output = ROOT / args.output
    audit = build_audit(
        ROOT / args.plan,
        ROOT / args.candidate_evidence,
        ROOT / args.candidate_checkpoint,
        ROOT / args.holdout_evidence,
        ROOT / args.holdout_checkpoint,
    )
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != audit:
            raise SystemExit("existing holdout audit differs")
    else:
        supervisor.atomic_json(output, audit)
    print(json.dumps({"passed": audit["passed"], "protocol": audit["approved_protocol_candidate"]}))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
