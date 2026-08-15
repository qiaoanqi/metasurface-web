#!/usr/bin/env python3
"""Independently audit the elevated reference and the old production budget.

This is deliberately separate from the long-running reference worker. It
validates the exact frozen tasks and reuses the preserved nG131/Nxy256/1 nm
spectra from v1.1 to compare the historical production budget with the new
candidate reference. It never edits a pool or registers a gate itself.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from color_utils import delta_e2000  # noqa: E402
from pipeline_supervisor import atomic_json, file_digest, load_json  # noqa: E402
from scripts.run_joint_convergence import retained_order  # noqa: E402
from scripts.run_joint_convergence_v1_1 import (  # noqa: E402
    PRODUCTION,
    task_id as v1_1_task_id,
)
from scripts.run_reference_resolution_escalation import (  # noqa: E402
    CONFIGS_1NM,
    EXPECTED_TASKS,
    FINE_CONFIG,
    GRID_384,
    MEAN_DE_LIMIT,
    ORDER_17,
    ORDER_19,
    PER_GEOMETRY_DE_LIMIT,
    POLS,
    VERSION as REFERENCE_VERSION,
    build_plan as build_reference_plan,
    build_tasks,
    geometry_key,
    labels_on_grid,
    task_id as reference_task_id,
)


VERSION = "paper2-reference-resolution-audit-v1"
POINTWISE_CONSERVATION_LIMIT = 1e-6
EXPECTED_PLAN_SHA256 = "E8720251ABEF1C0ADD26730404E495B77EBAE2AB5AA99A5236B32CA3286BE634"
EXPECTED_RUNTIME_PATHS = (
    "rcwa_batch.py",
    "paper2_colorimetry.py",
    "color_utils.py",
    "scripts/run_reference_resolution_escalation.py",
)
EXPECTED_THRESHOLDS = {
    "mean_joint_dE00_lt": MEAN_DE_LIMIT,
    "all_joint_dE00_lt": PER_GEOMETRY_DE_LIMIT,
    "pointwise_conservation_lte": POINTWISE_CONSERVATION_LIMIT,
}
AXIS_SPECS = {
    "order": (ORDER_17, 1.0, ORDER_19, 1.0),
    "grid": (GRID_384, 1.0, ORDER_19, 1.0),
    "spectral": (FINE_CONFIG, 1.0, FINE_CONFIG, 0.5),
}
REFERENCE_RESULT_SCHEMA = "reference_resolution_v1"
JOINT_V1_1_RESULT_SCHEMA = "joint_convergence_v1_1"
RESULT_SCHEMA_FIELDS = {
    REFERENCE_RESULT_SCHEMA: {"status", "R", "T", "time_s"},
    JOINT_V1_1_RESULT_SCHEMA: {"status", "R", "T"},
}


def load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"pickle payload is not an object: {path}")
    return payload


def exact_array(value, expected) -> bool:
    left = np.asarray(value, dtype=float)
    right = np.asarray(expected, dtype=float)
    return left.shape == right.shape and np.array_equal(left, right)


def relative_path(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def validate_runtime_hashes(runtime_hashes: dict) -> None:
    if set(runtime_hashes) != set(EXPECTED_RUNTIME_PATHS):
        raise ValueError("runtime hash path set does not match the frozen plan")
    for name in EXPECTED_RUNTIME_PATHS:
        if file_digest(ROOT / name) != str(runtime_hashes.get(name, "")).upper():
            raise ValueError(f"runtime hash mismatch: {name}")


def validate_result(
    result: dict,
    expected: dict,
    schema: str = REFERENCE_RESULT_SCHEMA,
) -> None:
    if not isinstance(result, dict) or result.get("status") != "ok":
        raise ValueError(f"task failed or malformed: {expected['id']}")
    if schema not in RESULT_SCHEMA_FIELDS:
        raise ValueError(f"unknown result schema: {schema}")
    expected_fields = set(expected) | RESULT_SCHEMA_FIELDS[schema]
    if set(result) != expected_fields:
        raise ValueError(f"task field set mismatch: {expected['id']}")
    for name in ("id", "geometry_index", "pol", "requested_nG", "retained_nG", "Nxy"):
        if result.get(name) != expected[name]:
            raise ValueError(f"task field mismatch {expected['id']}: {name}")
    if float(result.get("step_nm", -1.0)) != float(expected["step_nm"]):
        raise ValueError(f"task step mismatch: {expected['id']}")
    if geometry_key(result.get("geometry", {})) != geometry_key(expected["geometry"]):
        raise ValueError(f"task geometry mismatch: {expected['id']}")
    if not exact_array(result.get("wavelength_nm"), expected["wavelength_nm"]):
        raise ValueError(f"task wavelength mismatch: {expected['id']}")
    wavelength = np.asarray(expected["wavelength_nm"], dtype=float)
    reflectance = np.asarray(result.get("R"), dtype=float)
    transmittance = np.asarray(result.get("T"), dtype=float)
    if reflectance.shape != wavelength.shape or transmittance.shape != wavelength.shape:
        raise ValueError(f"task spectrum shape mismatch: {expected['id']}")
    if not np.isfinite(reflectance).all() or not np.isfinite(transmittance).all():
        raise ValueError(f"task contains non-finite spectra: {expected['id']}")
    if not (
        np.all(reflectance >= -1e-8)
        and np.all(reflectance <= 1.0 + 1e-8)
        and np.all(transmittance >= -1e-8)
        and np.all(transmittance <= 1.0 + 1e-8)
    ):
        raise ValueError(f"task spectrum is outside physical bounds: {expected['id']}")
    if np.max(np.abs(reflectance + transmittance - 1.0)) > POINTWISE_CONSERVATION_LIMIT:
        raise ValueError(f"task violates energy conservation: {expected['id']}")
    if schema == REFERENCE_RESULT_SCHEMA:
        time_s = float(result.get("time_s", -1.0))
        if not np.isfinite(time_s) or time_s < 0.0:
            raise ValueError(f"task runtime is invalid: {expected['id']}")


def validate_reference_checkpoint(
    checkpoint: dict, pool_sha256: str, expected_meta: dict | None = None
) -> list[dict]:
    meta = checkpoint.get("meta", {})
    if expected_meta is not None and meta != expected_meta:
        raise ValueError("reference checkpoint metadata does not match the frozen plan")
    if meta.get("version") != REFERENCE_VERSION:
        raise ValueError("unexpected reference checkpoint version")
    if str(meta.get("pool_sha256", "")).upper() != pool_sha256.upper():
        raise ValueError("reference checkpoint pool SHA256 mismatch")
    selected = meta.get("selected_geometries", [])
    expected_tasks = build_tasks(selected)
    if len(expected_tasks) != EXPECTED_TASKS:
        raise ValueError("reference checkpoint selection does not produce 80 tasks")
    expected_by_id = {item["id"]: item for item in expected_tasks}
    results = checkpoint.get("results", {})
    if set(results) != set(expected_by_id):
        missing = sorted(set(expected_by_id) - set(results))
        extra = sorted(set(results) - set(expected_by_id))
        raise ValueError(f"reference task identity mismatch: missing={missing}, extra={extra}")
    for identifier, expected in expected_by_id.items():
        validate_result(results[identifier], expected)
    return selected


def validate_frozen_plan(
    plan_path: Path,
    checkpoint_meta: dict,
    v1_checkpoint: dict,
    v1_checkpoint_path: Path,
    expected_plan_sha256: str,
) -> dict:
    actual_plan_sha256 = file_digest(plan_path)
    if actual_plan_sha256 != expected_plan_sha256.upper():
        raise ValueError("frozen reference plan SHA256 mismatch")
    plan = load_json(plan_path, {}) or {}
    baseline = {
        "path": relative_path(v1_checkpoint_path),
        "sha256": file_digest(v1_checkpoint_path),
    }
    if plan.get("baseline_checkpoint") != baseline:
        raise ValueError("frozen plan baseline checkpoint binding mismatch")
    source = plan.get("failed_gate_source", {})
    source_path = ROOT / str(source.get("path", ""))
    if (
        not source_path.is_file()
        or file_digest(source_path) != str(source.get("sha256", "")).upper()
    ):
        raise ValueError("frozen plan failed-gate source binding mismatch")
    selected = v1_checkpoint.get("meta", {}).get("selected_geometries", [])
    expected_meta = {
        "version": REFERENCE_VERSION,
        "pool_sha256": plan.get("pool_sha256"),
        "selected_geometries": selected,
        "selection_source": source,
        "baseline_checkpoint": baseline,
        "expected_tasks": EXPECTED_TASKS,
        "configs_1nm": [list(config) for config in CONFIGS_1NM],
        "fine_config": list(FINE_CONFIG),
        "fine_step_nm": 0.5,
        "thresholds": EXPECTED_THRESHOLDS,
        "runtime_hashes": plan.get("runtime_hashes"),
    }
    if checkpoint_meta != expected_meta:
        raise ValueError("reference checkpoint metadata does not match the frozen plan")
    if plan != build_reference_plan(expected_meta):
        raise ValueError("frozen reference plan content mismatch")
    validate_runtime_hashes(expected_meta["runtime_hashes"])
    return plan


def validate_worker_evidence_binding(
    evidence: dict,
    checkpoint_path: Path,
    selected: list[dict],
    expected_meta: dict,
) -> bool:
    if evidence.get("evidence_version") != REFERENCE_VERSION:
        raise ValueError("unexpected reference evidence version")
    if evidence.get("pool_sha256") != expected_meta["pool_sha256"]:
        raise ValueError("reference evidence pool SHA256 mismatch")
    expected_checkpoint = {
        "path": relative_path(checkpoint_path),
        "sha256": file_digest(checkpoint_path),
        "tasks": EXPECTED_TASKS,
    }
    if evidence.get("checkpoint") != expected_checkpoint:
        raise ValueError("reference evidence checkpoint binding mismatch")
    if evidence.get("input_evidence") != expected_meta["selection_source"]:
        raise ValueError("reference evidence failed-gate source mismatch")
    if evidence.get("baseline_checkpoint") != expected_meta["baseline_checkpoint"]:
        raise ValueError("reference evidence baseline checkpoint mismatch")
    if evidence.get("runtime_hashes") != expected_meta["runtime_hashes"]:
        raise ValueError("reference evidence runtime hashes mismatch")
    if evidence.get("thresholds") != EXPECTED_THRESHOLDS:
        raise ValueError("reference evidence thresholds mismatch")
    if evidence.get("selection") != selected:
        raise ValueError("reference evidence selection mismatch")
    if not isinstance(evidence.get("passed"), bool):
        raise ValueError("reference evidence passed claim must be boolean")
    return evidence["passed"]


def production_results(v1_checkpoint: dict, selected: list[dict]) -> dict:
    meta_selected = v1_checkpoint.get("meta", {}).get("selected_geometries", [])
    if [geometry_key(item) for item in meta_selected] != [geometry_key(item) for item in selected]:
        raise ValueError("v1.1 and reference selections differ")
    results = v1_checkpoint.get("results", {})
    output = {}
    wavelength = np.arange(380.0, 781.0, 1.0)
    for index, geometry in enumerate(selected):
        for pol in POLS:
            identifier = v1_1_task_id(index, pol, PRODUCTION)
            expected = {
                "id": identifier,
                "geometry_index": index,
                "geometry": geometry,
                "pol": pol,
                "requested_nG": PRODUCTION[0],
                "retained_nG": retained_order(PRODUCTION[0], geometry["P"]),
                "Nxy": PRODUCTION[1],
                "step_nm": 1,
                "wavelength_nm": wavelength,
            }
            if identifier not in results:
                raise ValueError(f"v1.1 production task missing: {identifier}")
            validate_result(
                results[identifier], expected, schema=JOINT_V1_1_RESULT_SCHEMA
            )
            output[(index, pol)] = results[identifier]
    return output


def compare_production_to_candidate(
    selected: list[dict], production: dict, reference_results: dict
) -> dict:
    values = []
    rows = []
    for index, _geometry in enumerate(selected):
        channel_values = []
        for pol in POLS:
            old = production[(index, pol)]
            current = reference_results[
                reference_task_id(index, pol, FINE_CONFIG, 0.5)
            ]
            old_lab = labels_on_grid(old["wavelength_nm"], old["R"])
            current_lab = labels_on_grid(current["wavelength_nm"], current["R"])
            value = float(delta_e2000(old_lab, current_lab))
            rows.append({"geometry_index": index, "pol": pol, "dE00": value})
            channel_values.append(value)
        values.append(max(channel_values))
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
        "mean_lt_1_15": bool(np.mean(array) < MEAN_DE_LIMIT),
        "all_lt_2_3": bool(np.all(array < PER_GEOMETRY_DE_LIMIT)),
        "joint_max_by_geometry": array.tolist(),
        "rows": rows,
    }


def compare_reference_axis(
    selected: list[dict],
    results: dict,
    first_config: tuple[int, int],
    first_step: float,
    second_config: tuple[int, int],
    second_step: float,
) -> dict:
    values = []
    rows = []
    for index, _geometry in enumerate(selected):
        channel_values = []
        for pol in POLS:
            first = results[reference_task_id(index, pol, first_config, first_step)]
            second = results[reference_task_id(index, pol, second_config, second_step)]
            first_lab = labels_on_grid(first["wavelength_nm"], first["R"])
            second_lab = labels_on_grid(second["wavelength_nm"], second["R"])
            value = float(delta_e2000(first_lab, second_lab))
            rows.append({"geometry_index": index, "pol": pol, "dE00": value})
            channel_values.append(value)
        values.append(max(channel_values))
    array = np.asarray(values, dtype=float)
    mean_passed = bool(array.size == len(selected) and np.mean(array) < MEAN_DE_LIMIT)
    all_passed = bool(
        array.size == len(selected) and np.all(array < PER_GEOMETRY_DE_LIMIT)
    )
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
        "mean_lt_1_15": mean_passed,
        "all_lt_2_3": all_passed,
        "passed": mean_passed and all_passed,
        "joint_max_by_geometry": array.tolist(),
        "rows": rows,
    }


def recompute_reference_axes(selected: list[dict], results: dict) -> dict:
    return {
        axis: compare_reference_axis(selected, results, *spec)
        for axis, spec in AXIS_SPECS.items()
    }


def physics_controls_pass(physics: dict) -> bool:
    checks = physics.get("independent_checks", {})
    fresnel = checks.get("empty_layer_fresnel", {})
    return bool(
        physics.get("solver_verdict") == "pass"
        and abs(float(fresnel.get("R", np.inf)) - float(fresnel.get("analytic_R", -np.inf))) <= 1e-7
        and abs(float(fresnel.get("rt", np.inf)) - 1.0) <= POINTWISE_CONSERVATION_LIMIT
        and float(checks.get("rotation_max_dR", np.inf)) <= 1e-7
        and float(checks.get("rotation_max_dT", np.inf)) <= 1e-7
        and checks.get("lw_circle_bitwise") is True
    )


def classify(
    reference_passed: bool,
    controls_passed: bool,
    production_matches: bool,
    failed_axes: list[str] | None = None,
    worker_claim_consistent: bool = True,
) -> str:
    if not worker_claim_consistent:
        return "worker_evidence_integrity_failure"
    if not controls_passed:
        return "implementation_control_failure"
    if not reference_passed:
        axes = failed_axes or []
        if "spectral" in axes and len(axes) > 1:
            return "reference_spectral_resolution_blocks_spatial_interpretation"
        if axes == ["spectral"]:
            return "reference_spectral_resolution_insufficient"
        spatial = [axis for axis in ("order", "grid") if axis in axes]
        if spatial:
            return "reference_spatial_budget_insufficient_" + "_and_".join(spatial)
        return "reference_requires_followup"
    if not production_matches:
        return "historical_production_budget_rejected"
    return "historical_5nm_sampling_rejected"


def build_audit(
    evidence_path: Path,
    checkpoint_path: Path,
    v1_checkpoint_path: Path,
    physics_path: Path,
    plan_path: Path,
    *,
    expected_plan_sha256: str = EXPECTED_PLAN_SHA256,
) -> dict:
    evidence = load_json(evidence_path, {}) or {}
    pool_sha256 = str(evidence.get("pool_sha256", "")).upper()
    if not pool_sha256:
        raise ValueError("reference evidence lacks pool SHA256")
    checkpoint = load_pickle(checkpoint_path)
    v1_checkpoint = load_pickle(v1_checkpoint_path)
    plan = validate_frozen_plan(
        plan_path,
        checkpoint.get("meta", {}),
        v1_checkpoint,
        v1_checkpoint_path,
        expected_plan_sha256,
    )
    if str(plan.get("pool_sha256", "")).upper() != pool_sha256:
        raise ValueError("reference plan pool SHA256 mismatch")
    selected = validate_reference_checkpoint(
        checkpoint, pool_sha256, checkpoint.get("meta", {})
    )
    worker_claim = validate_worker_evidence_binding(
        evidence, checkpoint_path, selected, checkpoint["meta"]
    )
    reference_axes = recompute_reference_axes(selected, checkpoint["results"])
    failed_axes = [axis for axis, result in reference_axes.items() if not result["passed"]]
    reference_passed = not failed_axes
    worker_claim_consistent = worker_claim is reference_passed
    production = production_results(v1_checkpoint, selected)
    comparison = compare_production_to_candidate(selected, production, checkpoint["results"])
    physics = load_json(physics_path, {}) or {}
    controls_passed = physics_controls_pass(physics)
    production_matches = comparison["mean_lt_1_15"] and comparison["all_lt_2_3"]
    checks = {
        "frozen_plan_sha256_and_content": True,
        "checkpoint_meta_and_runtime_hashes": True,
        "reference_checkpoint_exact_80": True,
        "worker_claim_matches_independent_recomputation": worker_claim_consistent,
        "physics_controls_passed": controls_passed,
        "production_1nm_comparison_complete": comparison["count"] == len(selected),
        "reference_order_converged": reference_axes["order"]["passed"],
        "reference_grid_converged": reference_axes["grid"]["passed"],
        "reference_spectral_grid_converged": reference_axes["spectral"]["passed"],
    }
    passed = all(checks.values())
    runtime_paths = (
        "scripts/audit_reference_resolution_result.py",
        "scripts/run_reference_resolution_escalation.py",
        "scripts/run_joint_convergence_v1_1.py",
        "rcwa_batch.py",
        "paper2_colorimetry.py",
        "color_utils.py",
    )
    return {
        "schema_version": 1,
        "evidence_version": VERSION,
        "passed": passed,
        "pool_sha256": pool_sha256,
        "classification": classify(
            reference_passed,
            controls_passed,
            production_matches,
            failed_axes,
            worker_claim_consistent,
        ),
        "failure_axes": failed_axes,
        "replacement_pool_required": bool(passed),
        "checks": checks,
        "worker_claim": {
            "passed": worker_claim,
            "matches_independent_recomputation": worker_claim_consistent,
        },
        "thresholds": EXPECTED_THRESHOLDS,
        "reference_axes": reference_axes,
        "production_nG131_nxy256_1nm_vs_candidate_nG365_nxy512_0p5nm": comparison,
        "inputs": {
            "reference_evidence": {"path": str(evidence_path.relative_to(ROOT)).replace("\\", "/"), "sha256": file_digest(evidence_path)},
            "reference_checkpoint": {"path": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"), "sha256": file_digest(checkpoint_path)},
            "v1_1_checkpoint": {"path": str(v1_checkpoint_path.relative_to(ROOT)).replace("\\", "/"), "sha256": file_digest(v1_checkpoint_path)},
            "physics_audit": {"path": str(physics_path.relative_to(ROOT)).replace("\\", "/"), "sha256": file_digest(physics_path)},
            "plan": {"path": str(plan_path.relative_to(ROOT)).replace("\\", "/"), "sha256": file_digest(plan_path)},
        },
        "runtime_hashes": {path: file_digest(ROOT / path) for path in runtime_paths},
        "decision_scope": (
            "Passing validates a candidate reference only. The historical 5 nm pool remains immutable and cannot "
            "be used for final training; a new versioned pool is required."
        ),
    }


def persist_audit(output: Path, audit: dict) -> None:
    if not output.exists():
        atomic_json(output, audit)
        return
    existing = load_json(output, {}) or {}
    replace_execution_failure = bool(
        existing.get("passed") is False
        and existing.get("classification") == "execution_integrity_failure"
        and audit.get("classification") != "execution_integrity_failure"
    )
    if existing != audit and not replace_execution_failure:
        raise ValueError("existing reference audit differs; bump the evidence version")
    if replace_execution_failure:
        atomic_json(output, audit)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-evidence", default=".state/reference_resolution_v1.json")
    parser.add_argument("--reference-checkpoint", default=".state/reference_resolution_v1_checkpoint.pkl")
    parser.add_argument("--v1-checkpoint", default=".state/joint_convergence_v1_1_checkpoint.pkl")
    parser.add_argument("--physics-audit", default=".state/physics_audit.json")
    parser.add_argument("--plan", default=".state/reference_resolution_v1_plan.json")
    parser.add_argument("--output", default=".state/reference_resolution_v1_audit.json")
    args = parser.parse_args()
    paths = {
        name: ROOT / value
        for name, value in (
            ("evidence", args.reference_evidence),
            ("checkpoint", args.reference_checkpoint),
            ("v1_checkpoint", args.v1_checkpoint),
            ("physics", args.physics_audit),
            ("plan", args.plan),
        )
    }
    missing = [str(path.relative_to(ROOT)) for path in paths.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"required audit inputs are missing: {missing}")
    try:
        audit = build_audit(
            paths["evidence"], paths["checkpoint"], paths["v1_checkpoint"],
            paths["physics"], paths["plan"],
        )
    except Exception as exc:
        audit = {
            "schema_version": 1,
            "evidence_version": VERSION,
            "passed": False,
            "classification": "execution_integrity_failure",
            "error": f"{type(exc).__name__}: {exc}",
            "inputs": {
                name: {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": file_digest(path)}
                for name, path in paths.items()
            },
        }
    output = ROOT / args.output
    try:
        persist_audit(output, audit)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({"passed": audit["passed"], "classification": audit["classification"]}))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
