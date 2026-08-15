#!/usr/bin/env python3
"""Freeze the paper-2 multi-fidelity active-learning preregistration."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pipeline_supervisor as supervisor  # noqa: E402


VERSION = "paper2-multifidelity-preregistration-v1"
POOL_PATH = ROOT / "data" / "rcwa_ellip_TiO2_3000_air.pkl"
HOLDOUT_PLAN_PATH = ROOT / ".state" / "reference_resolution_holdout_v1_plan.json"
CHECKPOINT_PATH = ROOT / ".state" / "reference_resolution_budget_v2_checkpoint.pkl"
V2_OUTCOME_PATHS = (
    ROOT / ".state" / "reference_resolution_holdout_v2.json",
    ROOT / ".state" / "reference_resolution_holdout_v2_audit.json",
)

SEED_TRAIN = 96
SEED_MAXIMIN = 64
SEED_RANDOM = 32
DEFAULT_ACTIVE_BATCHES = 2
MAX_ACTIVE_BATCHES = 3
DEFAULT_TRAIN = SEED_TRAIN + DEFAULT_ACTIVE_BATCHES * 32
MAX_TRAIN = SEED_TRAIN + MAX_ACTIVE_BATCHES * 32
VALIDATION = 64
TEST = 96
ACTIVE_BATCH = 32
PASSIVE_BATCHES = 4


def digest(path: Path) -> str:
    return supervisor.file_digest(path)


def binding(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve().relative_to(ROOT)).replace("\\", "/"), "sha256": digest(path)}


def canonical_geometry(values: dict[str, Any] | tuple[float, ...]) -> tuple[float, float, float, float]:
    if isinstance(values, dict):
        values = tuple(float(values[name]) for name in ("L", "W", "H", "P"))
    L, W, H, P = (float(value) for value in values)
    return (max(L, W), min(L, W), H, P)


def geometry_id(values: tuple[float, float, float, float]) -> str:
    encoded = "|".join(f"{float(value):.17g}" for value in values).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()[:24]


def selection_digest(items: list[str]) -> str:
    encoded = json.dumps(items, separators=(",", ":"), sort_keys=False).encode("ascii")
    return hashlib.sha256(encoded).hexdigest().upper()


def load_pool() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not POOL_PATH.is_file():
        raise ValueError(f"low-fidelity pool is missing: {POOL_PATH}")
    with POOL_PATH.open("rb") as handle:
        payload = pickle.load(handle)
    records = payload.get("records", []) if isinstance(payload, dict) else []
    if len(records) != 6000:
        raise ValueError("the preregistration requires the frozen 6000-record low-fidelity pool")
    if payload.get("meta", {}).get("background") != "air":
        raise ValueError("the low-fidelity pool background must be air")
    return payload, records


def geometry_manifest(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for record in records:
        geometry = canonical_geometry(record)
        identifier = geometry_id(geometry)
        entry = groups.setdefault(identifier, {"geometry": geometry, "pols": {}})
        pol = str(record.get("pol", ""))
        if pol not in {"p", "s"} or pol in entry["pols"]:
            raise ValueError("low-fidelity pool does not contain unique p/s pairs")
        entry["pols"][pol] = record
    if len(groups) != 3000 or any(set(item["pols"]) != {"p", "s"} for item in groups.values()):
        raise ValueError("low-fidelity pool geometry/polarization pairing is invalid")
    return groups


def holdout_ids() -> list[str]:
    if not HOLDOUT_PLAN_PATH.is_file():
        raise ValueError("the immutable 32-geometry holdout plan is missing")
    plan = json.loads(HOLDOUT_PLAN_PATH.read_text(encoding="utf-8"))
    cases = plan.get("combined_cases")
    if plan.get("plan_valid") is not True or not isinstance(cases, list) or len(cases) != 32:
        raise ValueError("the immutable holdout plan is not the expected 32-geometry manifest")
    ids = [geometry_id(canonical_geometry(case)) for case in cases]
    if len(set(ids)) != 32:
        raise ValueError("the holdout geometry manifest contains duplicates")
    return ids


def split_counts(identifiers: list[str], pool_sha: str, excluded: set[str]) -> dict[str, int]:
    ranked = sorted(
        identifiers,
        key=lambda identifier: hashlib.sha256(
            f"sha256-ranked-80-10-10-v1|{pool_sha}|{identifier}".encode("ascii")
        ).hexdigest(),
    )
    validation_count = len(ranked) // 10
    test_count = len(ranked) // 10
    labels = {}
    for index, identifier in enumerate(ranked):
        labels[identifier] = (
            "train"
            if index < len(ranked) - validation_count - test_count
            else "validation"
            if index < len(ranked) - test_count
            else "test"
        )
    return {
        name: sum(label == name and identifier not in excluded for identifier, label in labels.items())
        for name in ("train", "validation", "test")
    }


def cost_basis() -> dict[str, Any]:
    if not CHECKPOINT_PATH.is_file():
        raise ValueError("the current numerical-budget checkpoint is missing")
    with CHECKPOINT_PATH.open("rb") as handle:
        checkpoint = pickle.load(handle)
    results = checkpoint.get("results", {}) if isinstance(checkpoint, dict) else {}
    if not isinstance(results, dict) or len(results) < 8:
        raise ValueError("the cost basis needs at least eight completed numerical tasks")
    groups: dict[str, list[float]] = defaultdict(list)
    for result in results.values():
        key = "nG{0}-Nxy{1}-step{2:g}".format(
            int(result["requested_nG"]), int(result["Nxy"]), float(result["step_nm"])
        )
        groups[key].append(float(result["time_s"]))
    summary = {
        key: {
            "tasks_observed": len(values),
            "median_seconds": statistics.median(values),
            "min_seconds": min(values),
            "max_seconds": max(values),
        }
        for key, values in sorted(groups.items())
    }
    task_seconds = [float(result["time_s"]) for result in results.values()]
    return {
        "schema_version": 1,
        "evidence_version": "paper2-multifidelity-cost-basis-v1",
        "observed_at": supervisor.now_iso(),
        "source_checkpoint": binding(CHECKPOINT_PATH),
        "tasks_observed": len(task_seconds),
        "task_seconds_min": min(task_seconds),
        "task_seconds_median": statistics.median(task_seconds),
        "task_seconds_max": max(task_seconds),
        "protocol_groups": summary,
        "workers": 16,
        "paired_polarizations": True,
        "wall_time_formula": "geometry_count * 2 * task_seconds / 16 * 1.20",
        "overhead_factor": 1.20,
        "training_allowed": False,
    }


def build_plan(cost_path: Path, cost: dict[str, Any], pool: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    pool_sha = digest(POOL_PATH)
    groups = geometry_manifest(records)
    all_ids = sorted(groups)
    excluded = set(holdout_ids())
    if not excluded <= set(all_ids):
        raise ValueError("holdout geometries are not all present in the low-fidelity pool")
    counts = split_counts(all_ids, pool_sha, excluded)
    outcome_absent = not any(path.exists() for path in V2_OUTCOME_PATHS)
    if not outcome_absent:
        raise ValueError("the v2 holdout outcome already exists; preregistration must precede it")
    return {
        "schema_version": 1,
        "evidence_version": VERSION,
        "preregistration_revision": 1,
        "created_before_reference_budget_terminal": True,
        "created_before_holdout_results": True,
        "holdout_outcome_paths_absent_at_registration": True,
        "scientific_question": "Can a physics-audited low-to-high-fidelity residual model support dual-polarization elliptical inverse design without treating a coarse pool as truth?",
        "fidelity_roles": {
            "low": {
                "path": str(POOL_PATH.relative_to(ROOT)).replace("\\", "/"),
                "sha256": pool_sha,
                "records": 6000,
                "geometries": 3000,
                "polarizations": ["p", "s"],
                "nG_requested": 131,
                "background": "air",
                "wavelength_step_nm": 5.0,
                "role": "coverage_and_baseline_only",
                "production_truth": False,
            },
            "high": {
                "protocol_source": "approved_v2_reference_holdout_candidate",
                "must_bind_to_independent_reference_audit": True,
                "must_use_air_background": True,
                "must_keep_polarizations_paired": True,
                "common_label_grid_nm": 5.0,
                "production_truth": True,
            },
        },
        "holdout_isolation": {
            "manifest": binding(HOLDOUT_PLAN_PATH),
            "geometry_count": 32,
            "geometry_ids_sha256": selection_digest(sorted(excluded)),
            "exclude_from_selection_training_validation_early_stopping": True,
            "exclude_from_active_acquisition": True,
            "exclude_from_hyperparameter_selection": True,
            "result_independent_of_this_plan": True,
        },
        "geometry_split": {
            "version": "sha256-ranked-80-10-10-v1",
            "source_geometry_count": len(all_ids),
            "source_record_count": len(records),
            "pre_exclusion_counts": {"train": 2400, "validation": 300, "test": 300},
            "eligible_counts_after_holdout_exclusion": counts,
            "assignment_input_pool_sha256": pool_sha,
            "split_before_holdout_exclusion": True,
            "p_s_pair_unit": True,
        },
        "high_fidelity_budget": {
            "seed_train_geometries": SEED_TRAIN,
            "seed_maximin_geometries": SEED_MAXIMIN,
            "seed_fixed_random_geometries": SEED_RANDOM,
            "active_batch_geometries": ACTIVE_BATCH,
            "default_train_geometries": DEFAULT_TRAIN,
            "maximum_train_geometries": MAX_TRAIN,
            "validation_geometries": VALIDATION,
            "test_geometries": TEST,
            "default_total_geometries": DEFAULT_TRAIN + VALIDATION + TEST,
            "maximum_total_geometries": MAX_TRAIN + VALIDATION + TEST,
            "passive_control_batch_geometries": ACTIVE_BATCH,
            "passive_control_batches": PASSIVE_BATCHES,
            "maximum_unique_geometries_with_passive_control": MAX_TRAIN + PASSIVE_BATCHES * ACTIVE_BATCH + VALIDATION + TEST,
            "polarization_records_per_geometry": 2,
            "minimum_active_batches": DEFAULT_ACTIVE_BATCHES,
            "maximum_active_batches": MAX_ACTIVE_BATCHES,
            "never_expand_after_maximum": True,
        },
        "seed_selection": {
            "algorithm": "64 deterministic joint-feature maximin plus 32 deterministic stratified random controls",
            "features": ["L", "W", "H", "P", "r", "fill", "D65_Lab_p", "D65_Lab_s", "fixed_17_band_spectra_p", "fixed_17_band_spectra_s"],
            "feature_fit_scope": "eligible_train_geometries_only",
            "maximin_start": "lexicographically smallest SHA256-ranked eligible train geometry",
            "random_seed": 2026,
            "random_control_is_not_used_for_acquisition": True,
            "p_s_pair_atomic": True,
        },
        "multifidelity_model": {
            "primary": "low_fidelity_plus_high_fidelity_residual",
            "residual_grid_nm": 5.0,
            "residual_basis_components": 16,
            "basis_fit_scope": "high_fidelity_train_seed_only_then_frozen",
            "regressor": "5-member residual ensemble, input geometry plus low-fidelity p/s spectra, hidden 128x3, output 16",
            "ensemble_seeds": [42, 123, 456, 789, 2026],
            "normalization_fit_scope": "eligible_train_only",
            "fixed_baselines": ["low_fidelity_only", "high_fidelity_only", "random_acquisition"],
            "no_model_selection_on_holdout": True,
        },
        "active_acquisition": {
            "batch_size_geometries": ACTIVE_BATCH,
            "score": "0.50 spectral disagreement + 0.30 inverse-target ranking disagreement + 0.20 distance to labeled-HF set",
            "uncertainty": "ensemble spectral variance integrated over both polarizations",
            "boundary_quota": "at least 2 per batch from high-r, fill-extreme, P-extreme, or high p/s split",
            "retrain_after_each_batch": True,
            "train_only": True,
            "validation_and_test_never_acquired": True,
        },
        "stopping": {
            "minimum_batches": DEFAULT_ACTIVE_BATCHES,
            "success": "two consecutive validation checks with mean joint dE00 < 1.15, all 64 validation geometries < 2.3, and paired-bootstrap 95% upper bound of mean < 1.15",
            "plateau": "two consecutive batches with mean improvement < 0.05 dE00 and one-sided 95% upper bound of improvement < 0.10 while success is not met",
            "hard_stop": "maximum 192 high-fidelity active-train geometries",
            "test_reveal_only_after_stop": True,
            "thresholds_are_frozen": True,
        },
        "inverse_design": {
            "target_source": "24 primary frozen-reference geometries plus 64 final-test high-fidelity geometries",
            "target_count": 88,
            "polarization_joint_metric": "max(dE00_p, dE00_s)",
            "success_threshold": 2.3,
            "surrogate_candidate_budget": 10,
            "random_topk_baseline": True,
            "single_fidelity_baseline": True,
            "direct_optimization_baseline": {"targets": 8, "solver_geometry_budget": 20},
            "independent_high_fidelity_rcwa_verification": True,
            "holdout_targets_unlocked_only_after_model_and_stopping_lock": True,
            "holdout_targets_final_evaluation_only": True,
        },
        "failure_and_fallback": {
            "on_plateau_or_hard_stop_without_success": "multifidelity_scientific_negative_or_narrowed_claim",
            "full_high_fidelity_pool": "requires_new_independent_full_pool_necessity_audit",
            "full_pool_automatic_launch": False,
            "threshold_change_allowed": False,
            "training_without_all_gates": False,
        },
        "cost_basis": binding(cost_path),
        "cost_estimate": {
            "formula": cost["wall_time_formula"],
            "default_wall_time_hours": "70-140 for 320 geometries (160 train + 64 validation + 96 test)",
            "maximum_active_wall_time_hours": "77-155 for 352 geometries (192 active-train + 64 validation + 96 test)",
            "active_passive_comparison_wall_time_hours": "100-210 for at most 480 unique geometries",
            "full_3000_geometry_wall_time_hours": "approximately 650-1300 at the observed high-fidelity endpoint; never the default",
            "estimate_is_not_a_scientific_gate": True,
        },
        "training_allowed": False,
        "forbidden": ["holdout_reselection", "threshold_edit", "old_pool_overwrite", "paper1_edit", "automatic_full_pool_launch", "training_before_all_gates"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="protocols/paper2_multifidelity_preregistration_v1.json")
    parser.add_argument("--cost", default="protocols/paper2_multifidelity_cost_basis_v1.json")
    args = parser.parse_args()
    plan_path = ROOT / args.plan
    cost_path = ROOT / args.cost
    pool, records = load_pool()
    cost = cost_basis()
    cost_path.parent.mkdir(parents=True, exist_ok=True)
    supervisor.atomic_json(cost_path, cost)
    plan = build_plan(cost_path, cost, pool, records)
    supervisor.atomic_json(plan_path, plan)
    print(json.dumps({"plan": str(plan_path), "cost": str(cost_path), "pool_sha256": plan["fidelity_roles"]["low"]["sha256"], "passed": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
