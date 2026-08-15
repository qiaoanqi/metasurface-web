#!/usr/bin/env python3
"""Freeze the v2-bound 32-geometry reference holdout plan."""

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

from pipeline_supervisor import atomic_json, file_digest  # noqa: E402
from scripts import audit_reference_resolution_budget_v2 as v2_audit  # noqa: E402
from scripts import run_reference_resolution_budget_v2 as v2_runner  # noqa: E402
from scripts.reference_protocol_selection import evaluate_protocols  # noqa: E402


VERSION = "paper2-reference-holdout-v2-plan"
ARCHIVED_PLAN_VERSION = "paper2-reference-holdout-v1-plan"
ARCHIVED_PLAN_SHA256 = "B3812A34116AD62CD69BFD8AB806949F3AEEF7549702905C9BEAF38495911CFE"
ARCHIVED_EXISTING_SHA256 = "887D23BC5650C2FA5D7B13FA040AF962EAA89B6FCB2BD10B39ABAC0F01EF83E7"
ARCHIVED_NEW_SHA256 = "DD18455E230FC73661D80B6C3C40779A09D5C9F567D4D652F8A62BB09BF66BA8"
ARCHIVED_COMBINED_SHA256 = "5A9CE8F9C831ADE87E1DD81FFE7EF8574A318B72D624CBB5927433F90A172D4F"
NEW_CASES = 24
TOTAL_CASES = 32
FINAL_REFERENCE = (450, 768, 0.5)
MINIMUM_HOLDOUT_PROTOCOLS = (
    (365, 512, 0.5),
    (365, 768, 0.5),
    (450, 512, 0.5),
    (450, 768, 1.0),
    FINAL_REFERENCE,
)
REQUIRED_V2_AUDIT_CHECKS = (
    "plan_and_source_hashes_verified",
    "v1_scientific_failure_verified",
    "exact_new_task_set",
    "new_spectra_valid",
    "runtime_hashes_verified",
    "order_converged",
    "grid_converged",
    "corner_converged",
    "spectral_450x512_converged",
    "spectral_365x768_converged",
    "spectral_450x768_converged",
)
RUNTIME_PATHS = (
    "rcwa_batch.py",
    "paper2_colorimetry.py",
    "color_utils.py",
    "scripts/run_reference_resolution_escalation.py",
    "scripts/run_reference_resolution_budget_v2.py",
    "scripts/audit_reference_resolution_budget_v2.py",
    "scripts/freeze_reference_holdout_plan.py",
    "scripts/launch_reference_resolution_holdout.py",
    "scripts/run_reference_resolution_holdout.py",
    "scripts/audit_reference_resolution_holdout.py",
    "scripts/reference_protocol_selection.py",
)


def relative_path(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def binding(path: Path) -> dict[str, str]:
    return {"path": relative_path(path), "sha256": file_digest(path)}


def read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is not a JSON object")
    return payload


def selection_sha256(selection: list[dict]) -> str:
    encoded = json.dumps(selection, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def require_binding(expected: Any, path: Path, label: str) -> None:
    if not isinstance(expected, dict) or expected.get("path") != relative_path(path):
        raise ValueError(f"{label} path binding mismatch")
    if str(expected.get("sha256", "")).upper() != file_digest(path):
        raise ValueError(f"{label} SHA256 binding mismatch")


def load_archived_plan(path: Path) -> dict[str, Any]:
    if file_digest(path) != ARCHIVED_PLAN_SHA256:
        raise ValueError("archived v1 holdout plan SHA256 mismatch")
    plan = read_json(path, "archived v1 holdout plan")
    if (
        plan.get("schema_version") != 1
        or plan.get("evidence_version") != ARCHIVED_PLAN_VERSION
        or plan.get("plan_valid") is not True
        or plan.get("created_before_candidate_result") is not True
        or len(plan.get("existing_cases", [])) != 8
        or len(plan.get("new_cases", [])) != NEW_CASES
        or len(plan.get("combined_cases", [])) != TOTAL_CASES
        or plan.get("combined_cases") != plan.get("existing_cases") + plan.get("new_cases")
    ):
        raise ValueError("archived v1 holdout plan content is invalid")
    hashes = (
        (plan["existing_cases"], ARCHIVED_EXISTING_SHA256),
        (plan["new_cases"], ARCHIVED_NEW_SHA256),
        (plan["combined_cases"], ARCHIVED_COMBINED_SHA256),
    )
    if any(selection_sha256(items) != expected for items, expected in hashes):
        raise ValueError("archived v1 holdout selection SHA256 mismatch")
    if len({selection_sha256([item]) for item in plan["combined_cases"]}) != TOTAL_CASES:
        raise ValueError("archived v1 holdout contains duplicate geometries")
    return plan


def validate_v2_source(
    plan_path: Path,
    checkpoint_path: Path,
    evidence_path: Path,
    audit_path: Path,
) -> dict[str, Any]:
    plan = read_json(plan_path, "v2 plan")
    v2_audit.validate_plan(plan, plan_path)
    evidence = read_json(evidence_path, "v2 worker evidence")
    audit = read_json(audit_path, "independent v2 audit")
    if evidence.get("evidence_version") != v2_runner.VERSION:
        raise ValueError("v2 worker evidence version mismatch")
    request = evidence.get("request")
    if (
        not isinstance(request, dict)
        or not isinstance(request.get("request_id"), str)
        or int(request.get("attempt", 0)) < 1
        or audit.get("request") != request
    ):
        raise ValueError("v2 evidence and audit request identities are not bound")
    if evidence.get("passed") is not True or evidence.get("training_allowed") is not False:
        raise ValueError("v2 worker evidence is not a passed diagnostic result")
    require_binding(evidence.get("plan"), plan_path, "v2 worker plan")
    require_binding(evidence.get("checkpoint"), checkpoint_path, "v2 worker checkpoint")
    if evidence.get("pool_sha256") != plan.get("pool_sha256"):
        raise ValueError("v2 worker pool SHA256 mismatch")
    if evidence.get("thresholds") != plan.get("thresholds"):
        raise ValueError("v2 worker thresholds changed")
    if (
        audit.get("evidence_version") != v2_audit.VERSION
        or audit.get("passed") is not True
        or audit.get("classification") != "budget_v2_converged"
        or audit.get("training_allowed") is not False
        or audit.get("pool_sha256") != plan.get("pool_sha256")
        or audit.get("thresholds") != plan.get("thresholds")
        or not all(audit.get("checks", {}).get(name) is True for name in REQUIRED_V2_AUDIT_CHECKS)
        or audit.get("worker_claim")
        != {"passed": True, "matches_independent_recomputation": True}
    ):
        raise ValueError("independent v2 audit is missing, failed, or incomplete")
    require_binding(audit.get("plan"), plan_path, "v2 audit plan")
    audit_checkpoint = dict(audit.get("checkpoint", {}))
    audit_checkpoint.pop("tasks", None)
    require_binding(audit_checkpoint, checkpoint_path, "v2 audit checkpoint")
    if audit.get("checkpoint", {}).get("tasks") != v2_runner.EXPECTED_TASKS:
        raise ValueError("independent v2 audit checkpoint task count mismatch")
    recomputed = v2_audit.build_audit(
        evidence_path, checkpoint_path, plan_path, request
    )
    if audit != recomputed or recomputed.get("passed") is not True:
        raise ValueError("independent v2 audit does not exactly reproduce from bound inputs")
    _source_audit, _source_evidence, baseline = v2_audit.validate_v1_source(plan, plan_path)
    with checkpoint_path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    return {
        "plan": plan,
        "evidence": evidence,
        "audit": audit,
        "baseline_checkpoint": baseline,
        "v2_checkpoint": checkpoint,
    }


def source_result_id(index: int, pol: str, config: tuple[int, int], step: float) -> str:
    if config == v2_runner.BASE_CONFIG:
        return v2_runner.v1.task_id(index, pol, config, step)
    return v2_runner.task_id(index, pol, config, step)


def source_results(source: dict[str, Any]) -> dict[str, dict]:
    combined = dict(source["baseline_checkpoint"]["results"])
    combined.update(source["v2_checkpoint"]["results"])
    return combined


def protocol_manifest(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    specs = [
        {"requested_nG": nG, "Nxy": Nxy, "step_nm": step, "role": role}
        for (nG, Nxy, step), role in zip(
            MINIMUM_HOLDOUT_PROTOCOLS,
            ("base", "order", "grid", "spectral", "final_reference"),
        )
    ]
    key = (
        int(candidate["requested_nG"]),
        int(candidate["Nxy"]),
        float(candidate["wavelength_step_nm"]),
    )
    if key not in MINIMUM_HOLDOUT_PROTOCOLS:
        specs.append(
            {
                "requested_nG": key[0],
                "Nxy": key[1],
                "step_nm": key[2],
                "role": "frozen_production_candidate",
            }
        )
    return specs


def build_plan(
    archived_plan_path: Path,
    v2_plan_path: Path,
    v2_checkpoint_path: Path,
    v2_evidence_path: Path,
    v2_audit_path: Path,
) -> dict[str, Any]:
    archived = load_archived_plan(archived_plan_path)
    source = validate_v2_source(
        v2_plan_path, v2_checkpoint_path, v2_evidence_path, v2_audit_path
    )
    v2_plan = source["plan"]
    if v2_plan.get("selection") != archived["existing_cases"]:
        raise ValueError("v2 source selection differs from the pre-frozen eight cases")
    pool_path = ROOT / archived["pool"]["path"]
    if file_digest(pool_path) != archived["pool"]["sha256"]:
        raise ValueError("archived holdout pool SHA256 mismatch")
    if v2_plan.get("pool_sha256") != archived["pool"]["sha256"]:
        raise ValueError("v2 source and archived holdout pool SHA256 differ")
    evaluation = evaluate_protocols(
        v2_plan["selection"], source_results(source), source_result_id
    )
    candidate = evaluation.get("lowest_cost_passing_protocol")
    if not evaluation.get("any_protocol_passed") or not isinstance(candidate, dict):
        raise ValueError("the initial eight cases do not yield a frozen production candidate")
    manifest = protocol_manifest(candidate)
    combined = list(v2_plan["selection"]) + list(archived["new_cases"])
    thresholds = {
        "mean_joint_dE00_lt": v2_runner.v1.MEAN_DE_LIMIT,
        "all_joint_dE00_lt": v2_runner.v1.PER_GEOMETRY_DE_LIMIT,
        "pointwise_conservation_lte": v2_runner.v1.CONSERVATION_LIMIT,
    }
    if v2_plan.get("thresholds") != thresholds:
        raise ValueError("v2 thresholds differ from the pre-registered limits")
    protocols_per_geometry = len(manifest) * len(v2_runner.POLS)
    return {
        "schema_version": 1,
        "evidence_version": VERSION,
        "plan_valid": True,
        "created_before_holdout_results": True,
        "candidate_frozen_on_initial_eight_only": True,
        "holdout_cannot_reselect_candidate": True,
        "source_gate": "independent_reference_resolution_budget_v2_passed",
        "pool": binding(pool_path),
        "archived_v1_holdout_plan": binding(archived_plan_path),
        "source_v2_plan": binding(v2_plan_path),
        "source_v2_checkpoint": binding(v2_checkpoint_path),
        "source_v2_worker_evidence": binding(v2_evidence_path),
        "source_v2_independent_audit": binding(v2_audit_path),
        "source_base_checkpoint": v2_plan["source_v1_checkpoint"],
        "existing_cases": list(v2_plan["selection"]),
        "new_cases": list(archived["new_cases"]),
        "combined_cases": combined,
        "existing_selection_sha256": ARCHIVED_EXISTING_SHA256,
        "new_selection_sha256": ARCHIVED_NEW_SHA256,
        "combined_selection_sha256": ARCHIVED_COMBINED_SHA256,
        "selection_method": archived["selection_method"],
        "existing_case_count": 8,
        "new_case_count": NEW_CASES,
        "combined_case_count": TOTAL_CASES,
        "polarizations": list(v2_runner.POLS),
        "minimum_holdout_protocols": [list(item) for item in MINIMUM_HOLDOUT_PROTOCOLS],
        "task_protocols": manifest,
        "frozen_candidate": candidate,
        "source_protocol_evaluation": evaluation,
        "final_reference": {
            "requested_nG": FINAL_REFERENCE[0],
            "Nxy": FINAL_REFERENCE[1],
            "wavelength_step_nm": FINAL_REFERENCE[2],
        },
        "expected_source_tasks": 8 * protocols_per_geometry,
        "expected_new_tasks": NEW_CASES * protocols_per_geometry,
        "expected_combined_tasks": TOTAL_CASES * protocols_per_geometry,
        "thresholds": thresholds,
        "primary_gate_population": "24_new_holdout_geometries_only",
        "combined_32_population_scope": "supplemental_reporting_only",
        "decision_rule": (
            "Freeze the lowest measured-cost passing candidate on the initial eight cases, then on the "
            "24 untouched cases require order, grid, corner, spectral, and frozen-candidate-to-final-reference "
            "comparisons to pass unchanged joint DeltaE00 limits. Never reselect on holdout outcomes."
        ),
        "guardrails": [
            "v1 holdout plan is archive-only and accepted only by exact SHA256",
            "v2 independent audit must remain passed and hash-bound",
            "24 holdout cases may confirm or reject but never select a candidate",
            "never change thresholds, skip an axis, activate a pool, or enable training",
        ],
        "runtime_hashes": {path: file_digest(ROOT / path) for path in RUNTIME_PATHS},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archived-plan", default=".state/reference_resolution_holdout_v1_plan.json")
    parser.add_argument("--v2-plan", default=".state/reference_resolution_budget_v2_plan.json")
    parser.add_argument("--v2-checkpoint", default=".state/reference_resolution_budget_v2_checkpoint.pkl")
    parser.add_argument("--v2-evidence", default=".state/reference_resolution_budget_v2.json")
    parser.add_argument("--v2-audit", default=".state/reference_resolution_budget_v2_audit.json")
    parser.add_argument("--output", default=".state/reference_resolution_holdout_v2_plan.json")
    args = parser.parse_args()
    output = ROOT / args.output
    plan = build_plan(
        ROOT / args.archived_plan,
        ROOT / args.v2_plan,
        ROOT / args.v2_checkpoint,
        ROOT / args.v2_evidence,
        ROOT / args.v2_audit,
    )
    if output.exists():
        existing = read_json(output, "existing v2 holdout plan")
        if existing != plan:
            raise SystemExit("existing v2 holdout plan differs; use a new evidence version")
    else:
        atomic_json(output, plan)
    print(json.dumps({"plan_valid": True, "new_cases": NEW_CASES, "tasks": plan["expected_new_tasks"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
