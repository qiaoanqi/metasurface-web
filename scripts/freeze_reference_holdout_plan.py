#!/usr/bin/env python3
"""Freeze 24 new stratified cases before reading candidate-reference results."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_supervisor import atomic_json, file_digest  # noqa: E402
from scripts import run_reference_resolution_escalation as base  # noqa: E402


VERSION = "paper2-reference-holdout-v1-plan"
NEW_CASES = 24
TOTAL_CASES = 32


def geometry_key(item: dict) -> tuple[float, float, float, float]:
    return tuple(float(item[name]) for name in ("L", "W", "H", "P"))


def load_geometries(pool_path: Path) -> list[dict]:
    with pool_path.open("rb") as handle:
        payload = pickle.load(handle)
    records = payload.get("records", []) if isinstance(payload, dict) else payload
    paired: dict[tuple[float, float, float, float], dict[str, dict]] = {}
    for record in records:
        if record.get("success"):
            paired.setdefault(geometry_key(record), {})[str(record["pol"])] = record
    geometries = []
    for key, channels in paired.items():
        if set(channels) != {"p", "s"}:
            continue
        L, W, H, P = key
        sharpness = max(
            float(np.max(np.abs(np.diff(np.asarray(channels[pol]["R"], dtype=float)))))
            for pol in ("p", "s")
        )
        geometries.append(
            {
                "L": L,
                "W": W,
                "H": H,
                "P": P,
                "r": max(L, W) / min(L, W),
                "fill": np.pi * (L / 2.0) * (W / 2.0) / (P * P),
                "sharpness_5nm": sharpness,
            }
        )
    return sorted(geometries, key=geometry_key)


def feature_matrix(geometries: list[dict]) -> np.ndarray:
    matrix = np.asarray(
        [
            [
                max(item["L"], item["W"]),
                min(item["L"], item["W"]),
                item["H"],
                item["P"],
                item["r"],
                item["fill"],
                item["sharpness_5nm"],
            ]
            for item in geometries
        ],
        dtype=float,
    )
    lower = np.min(matrix, axis=0)
    scale = np.maximum(np.max(matrix, axis=0) - lower, 1e-15)
    return (matrix - lower) / scale


def choose_holdout(
    geometries: list[dict], existing: list[dict], per_stratum: int = 6
) -> tuple[list[dict], list[float]]:
    existing_keys = {geometry_key(item) for item in existing}
    candidates = [item for item in geometries if geometry_key(item) not in existing_keys]
    if len(candidates) < NEW_CASES:
        raise ValueError("not enough independent pool geometries for the holdout")
    boundaries = np.quantile(
        [item["sharpness_5nm"] for item in candidates], [0.25, 0.5, 0.75]
    ).tolist()
    all_items = candidates + existing
    normalized = feature_matrix(all_items)
    candidate_features = normalized[: len(candidates)]
    anchors = list(normalized[len(candidates) :])
    selected: list[dict] = []
    for stratum in range(4):
        indices = [
            index
            for index, item in enumerate(candidates)
            if int(np.searchsorted(boundaries, item["sharpness_5nm"], side="right")) == stratum
        ]
        for _ in range(per_stratum):
            if not indices:
                raise ValueError(f"sharpness stratum {stratum} is undersized")
            best_index = None
            best_score = -1.0
            for index in indices:
                point = candidate_features[index]
                score = min(float(np.linalg.norm(point - anchor)) for anchor in anchors)
                key = geometry_key(candidates[index])
                if score > best_score + 1e-15 or (
                    abs(score - best_score) <= 1e-15
                    and (best_index is None or key < geometry_key(candidates[best_index]))
                ):
                    best_index = index
                    best_score = score
            chosen = dict(candidates[best_index])
            chosen["selection"] = f"sharpness_quartile_{stratum + 1}_maximin"
            chosen["selection_distance"] = best_score
            selected.append(chosen)
            anchors.append(candidate_features[best_index])
            indices.remove(best_index)
    return selected, boundaries


def selection_sha256(selection: list[dict]) -> str:
    encoded = json.dumps(selection, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def build_plan(pool_path: Path, source_path: Path) -> dict:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("evidence_version") != "paper2-joint-convergence-v1.1":
        raise ValueError("holdout plan requires the frozen v1.1 selection")
    if source.get("pool_sha256") != file_digest(pool_path):
        raise ValueError("holdout source and pool SHA256 differ")
    existing = source.get("selection", [])
    if len(existing) != 8:
        raise ValueError("holdout source must contain exactly eight frozen cases")
    new_cases, boundaries = choose_holdout(load_geometries(pool_path), existing)
    combined = [dict(item) for item in existing] + new_cases
    runtime_paths = (
        "rcwa_batch.py",
        "paper2_colorimetry.py",
        "color_utils.py",
        "scripts/run_reference_resolution_escalation.py",
        "scripts/freeze_reference_holdout_plan.py",
        "scripts/run_reference_resolution_holdout.py",
        "scripts/reference_protocol_selection.py",
        "scripts/run_joint_convergence.py",
        "scripts/run_replacement_pool.py",
    )
    return {
        "schema_version": 1,
        "evidence_version": VERSION,
        "plan_valid": True,
        "created_before_candidate_result": True,
        "pool": {"path": str(pool_path.relative_to(ROOT)).replace("\\", "/"), "sha256": file_digest(pool_path)},
        "selection_source": {"path": str(source_path.relative_to(ROOT)).replace("\\", "/"), "sha256": file_digest(source_path)},
        "existing_cases": existing,
        "new_cases": new_cases,
        "combined_cases": combined,
        "new_selection_sha256": selection_sha256(new_cases),
        "combined_selection_sha256": selection_sha256(combined),
        "selection_method": {
            "name": "four sharpness quartiles plus deterministic maximin coverage",
            "features": ["canonical_long_axis", "canonical_short_axis", "H", "P", "aspect", "fill", "sharpness_5nm"],
            "sharpness_quartile_boundaries": boundaries,
            "new_per_quartile": 6,
            "candidate_result_used": False,
        },
        "existing_case_count": 8,
        "new_case_count": NEW_CASES,
        "combined_case_count": TOTAL_CASES,
        "polarizations": list(base.POLS),
        "configs_1nm": [list(config) for config in base.CONFIGS_1NM],
        "fine_config": list(base.FINE_CONFIG),
        "fine_step_nm": 0.5,
        "expected_new_tasks": NEW_CASES * len(base.POLS) * (len(base.CONFIGS_1NM) + 1),
        "expected_combined_tasks": TOTAL_CASES * len(base.POLS) * (len(base.CONFIGS_1NM) + 1),
        "thresholds": {
            "mean_joint_dE00_lt": base.MEAN_DE_LIMIT,
            "all_joint_dE00_lt": base.PER_GEOMETRY_DE_LIMIT,
            "pointwise_conservation_lte": base.CONSERVATION_LIMIT,
        },
        "decision_rule": "Approve a production reference only when the combined 32-case order, grid, spectral, retained-order, raw-spectrum, and conservation checks all pass unchanged thresholds.",
        "runtime_hashes": {path: file_digest(ROOT / path) for path in runtime_paths},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="data/rcwa_ellip_TiO2_3000_air.pkl")
    parser.add_argument("--source", default=".state/joint_convergence_v1_1.json")
    parser.add_argument("--output", default=".state/reference_resolution_holdout_v1_plan.json")
    args = parser.parse_args()
    output = ROOT / args.output
    plan = build_plan(ROOT / args.pool, ROOT / args.source)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != plan:
            raise SystemExit("existing holdout plan differs")
    else:
        atomic_json(output, plan)
    print(json.dumps({"plan_valid": True, "new_cases": NEW_CASES, "tasks": plan["expected_new_tasks"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
