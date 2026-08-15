#!/usr/bin/env python3
"""Freeze the deterministic numerical-budget v2 escalation plan."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_supervisor import atomic_json, file_digest, load_json  # noqa: E402
from scripts import run_reference_resolution_escalation as v1  # noqa: E402


VERSION = "paper2-reference-budget-v2-plan"
V1_PLAN_SHA256 = "E8720251ABEF1C0ADD26730404E495B77EBAE2AB5AA99A5236B32CA3286BE634"
PROVISIONAL_PLAN_SHA256 = "D856A0DADDC6A334C7CB16651BD72B486B5EC5D2D35E57728FC532AC98F3F783"
EXTRA_CONFIGS = ((450, 512), (365, 768), (450, 768))
EXPECTED_NEW_TASKS = len(EXTRA_CONFIGS) * len(v1.POLS) * 8 * 2


def _binding(path: Path) -> dict:
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": file_digest(path),
    }


def is_known_provisional_plan(payload: dict, digest: str) -> bool:
    return bool(
        digest.upper() == PROVISIONAL_PLAN_SHA256
        and payload.get("evidence_version") == VERSION
        and "source_v1_audit" not in payload
        and payload.get("source_v1_evidence", {}).get("path")
        == ".state/joint_convergence_v1_1.json"
        and payload.get("expected_new_tasks") == EXPECTED_NEW_TASKS
    )


def build_plan(
    v1_audit_path: Path,
    v1_evidence_path: Path | None = None,
    v1_checkpoint_path: Path | None = None,
    v1_plan_path: Path | None = None,
) -> dict:
    """Freeze v2 only after the independent 80-task v1 audit is terminal.

    The previous draft accidentally mixed the 64-task v1.1 checkpoint with the
    80-task reference runner.  Resolve every source from the independent audit
    and bind the resulting files by digest so the runner cannot mix protocols.
    """
    audit = load_json(v1_audit_path, {}) or {}
    if audit.get("evidence_version") != "paper2-reference-resolution-audit-v1":
        raise ValueError("independent v1 reference audit is not frozen")
    if audit.get("passed") is not False:
        raise ValueError("v1 reference audit must remain an explicit scientific failure")
    if audit.get("classification") in {
        "execution_integrity_failure",
        "worker_evidence_integrity_failure",
    }:
        raise ValueError("v1 reference audit is not a scientific failure")
    audit_checks = audit.get("checks", {})
    for key in (
        "frozen_plan_sha256_and_content",
        "checkpoint_meta_and_runtime_hashes",
        "reference_checkpoint_exact_80",
        "worker_claim_matches_independent_recomputation",
        "physics_controls_passed",
    ):
        if audit_checks.get(key) is not True:
            raise ValueError(f"v1 audit prerequisite is not proven: {key}")

    inputs = audit.get("inputs", {})
    def source_path(name: str, explicit: Path | None) -> Path:
        if explicit is not None:
            return explicit
        item = inputs.get(name, {})
        value = item.get("path")
        if not isinstance(value, str) or not value:
            raise ValueError(f"v1 audit lacks input binding: {name}")
        return ROOT / value

    v1_evidence_path = source_path("reference_evidence", v1_evidence_path)
    v1_checkpoint_path = source_path("reference_checkpoint", v1_checkpoint_path)
    v1_plan_path = source_path("plan", v1_plan_path)
    evidence = load_json(v1_evidence_path, {}) or {}
    if evidence.get("evidence_version") != "paper2-reference-resolution-v1":
        raise ValueError("v1 worker evidence version is not frozen")
    if evidence.get("passed") is not False or len(evidence.get("selection", [])) != 8:
        raise ValueError("v1 failed eight-case selection is not frozen")
    if not v1_checkpoint_path.is_file() or not v1_plan_path.is_file():
        raise ValueError("v1 source checkpoint or plan is missing")
    with v1_checkpoint_path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    if len(checkpoint.get("results", {})) != 80:
        raise ValueError("v1 reference checkpoint must be complete 80/80")
    if file_digest(v1_plan_path) != str(inputs.get("plan", {}).get("sha256", "")).upper():
        raise ValueError("v1 reference plan binding differs from independent audit")
    if file_digest(v1_evidence_path) != str(inputs.get("reference_evidence", {}).get("sha256", "")).upper():
        raise ValueError("v1 worker evidence binding differs from independent audit")
    if file_digest(v1_checkpoint_path) != str(inputs.get("reference_checkpoint", {}).get("sha256", "")).upper():
        raise ValueError("v1 checkpoint binding differs from independent audit")
    return {
        "schema_version": 1,
        "evidence_version": VERSION,
        "plan_valid": True,
        "decision": "transition_after_failure",
        "source_failed_action": "joint_numerical_convergence",
        "pool_sha256": str(evidence["pool_sha256"]).upper(),
        "selection": evidence["selection"],
        "source_v1_audit": _binding(v1_audit_path),
        "source_v1_evidence": _binding(v1_evidence_path),
        "source_v1_checkpoint": _binding(v1_checkpoint_path),
        "source_v1_plan": _binding(v1_plan_path),
        "extra_configs": [list(config) for config in EXTRA_CONFIGS],
        "base_config": list(v1.FINE_CONFIG),
        "steps_nm": [1.0, 0.5],
        "expected_new_tasks": EXPECTED_NEW_TASKS,
        "thresholds": {
            "mean_joint_dE00_lt": v1.MEAN_DE_LIMIT,
            "all_joint_dE00_lt": v1.PER_GEOMETRY_DE_LIMIT,
            "pointwise_conservation_lte": v1.CONSERVATION_LIMIT,
        },
        "decision_rule": (
            "Compare each extra order/grid/corner endpoint against the frozen nG365/Nxy512 endpoint "
            "on both 1 nm and matched 0.5 nm grids. All three spatial axes and every spectral comparison "
            "must satisfy the unchanged mean<1.15 and per-geometry<2.3 rules before a new reference can be approved."
        ),
        "guardrails": [
            "diagnostic_only",
            "never rehabilitate the historical nG131 pool",
            "never change thresholds",
            "never activate a pool or enable training",
            "expand to the independent 32-geometry holdout only after this gate passes",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-audit", default=".state/reference_resolution_v1_audit.json")
    parser.add_argument("--v1-evidence", default=None)
    parser.add_argument("--v1-checkpoint", default=None)
    parser.add_argument("--v1-plan", default=None)
    parser.add_argument("--output", default=".state/reference_resolution_budget_v2_plan.json")
    args = parser.parse_args()
    audit_path = ROOT / args.v1_audit
    payload = build_plan(
        audit_path,
        ROOT / args.v1_evidence if args.v1_evidence else None,
        ROOT / args.v1_checkpoint if args.v1_checkpoint else None,
        ROOT / args.v1_plan if args.v1_plan else None,
    )
    output = ROOT / args.output
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != payload:
            existing_sha = file_digest(output)
            if not is_known_provisional_plan(existing, existing_sha):
                raise SystemExit("existing budget v2 plan differs; use a new version")
            archive = output.with_name(
                f"{output.stem}_provisional_{existing_sha[:12]}{output.suffix}"
            )
            if archive.exists():
                archived = json.loads(archive.read_text(encoding="utf-8"))
                if archived != existing:
                    raise SystemExit("provisional plan archive collision")
            else:
                atomic_json(archive, existing)
            atomic_json(output, payload)
    else:
        atomic_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
