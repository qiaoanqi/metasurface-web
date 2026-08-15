#!/usr/bin/env python3
"""Independently audit the diagnostic numerical-budget v2 run.

The runner is allowed to produce a summary, but this module treats that
summary as an untrusted claim.  It verifies every hash-bound input and task,
then recomputes all six comparison axes from raw spectra.
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

from pipeline_supervisor import atomic_json, file_digest, load_json  # noqa: E402
from scripts import run_reference_resolution_escalation as v1  # noqa: E402


VERSION = "paper2-reference-resolution-budget-v2-audit"
PLAN_VERSION = "paper2-reference-budget-v2-plan"
POLS = ("p", "s")
EXTRA_CONFIGS = ((450, 512), (365, 768), (450, 768))
BASE_CONFIG = (365, 512)
STEPS = (1.0, 0.5)
EXPECTED_TASKS = len(EXTRA_CONFIGS) * len(POLS) * 8 * len(STEPS)
RUNTIME_PATHS = (
    "rcwa_batch.py",
    "paper2_colorimetry.py",
    "color_utils.py",
    "scripts/run_reference_resolution_budget_v2.py",
)
SPATIAL_CONFIGS = {
    "order": EXTRA_CONFIGS[0],
    "grid": EXTRA_CONFIGS[1],
    "corner": EXTRA_CONFIGS[2],
}


def spatial_axis_specs() -> list[tuple[str, tuple[int, int], float]]:
    return [
        (f"{axis}_365x512_to_{config[0]}x{config[1]}_{'0p5nm' if step == 0.5 else '1nm'}", config, step)
        for axis, config in SPATIAL_CONFIGS.items()
        for step in STEPS
    ]


def relative_path(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def binding(path: Path) -> dict:
    return {"path": relative_path(path), "sha256": file_digest(path)}


def require_binding(expected: dict, path: Path, label: str) -> None:
    if not isinstance(expected, dict) or expected.get("path") != relative_path(path):
        raise ValueError(f"{label} path binding mismatch")
    if str(expected.get("sha256", "")).upper() != file_digest(path):
        raise ValueError(f"{label} SHA256 binding mismatch")


def geometry_key(geometry: dict) -> tuple[float, float, float, float]:
    return tuple(float(geometry[name]) for name in ("L", "W", "H", "P"))


def exact_array(actual: object, expected: object) -> bool:
    left = np.asarray(actual, dtype=float)
    right = np.asarray(expected, dtype=float)
    return left.shape == right.shape and np.array_equal(left, right)


def runtime_hashes_match(runtime_hashes: object) -> bool:
    expected = {path: file_digest(ROOT / path) for path in RUNTIME_PATHS}
    return isinstance(runtime_hashes, dict) and runtime_hashes == expected


def worker_claim_matches(worker_claim: object, recomputed: bool) -> bool:
    return isinstance(worker_claim, bool) and worker_claim is recomputed


def task_id(index: int, pol: str, config: tuple[int, int], step: float) -> str:
    token = "0p5" if step == 0.5 else "1"
    return f"refbudget-v2-g{index:02d}-{pol}-ng{config[0]}-nxy{config[1]}-step{token}"


def build_tasks(selected: list[dict]) -> list[dict]:
    if len(selected) != 8:
        raise ValueError("budget v2 requires eight frozen cases")
    tasks = []
    for index, geometry in enumerate(selected):
        for pol in POLS:
            for config in EXTRA_CONFIGS:
                for step in STEPS:
                    wavelength = v1.WL_HALF_NM if step == 0.5 else v1.WL_1NM
                    tasks.append(
                        {
                            "id": task_id(index, pol, config, step),
                            "geometry_index": index,
                            "geometry": geometry,
                            "pol": pol,
                            "requested_nG": config[0],
                            "retained_nG": v1.retained_order(config[0], geometry["P"]),
                            "Nxy": config[1],
                            "step_nm": step,
                            "wavelength_nm": wavelength,
                        }
                    )
    return tasks


def validate_result(result: dict, expected: dict) -> None:
    if not isinstance(result, dict) or result.get("status") != "ok":
        raise ValueError(f"task failed or malformed: {expected['id']}")
    expected_fields = set(expected) | {"status", "R", "T", "time_s"}
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
    if np.max(np.abs(reflectance + transmittance - 1.0)) > v1.CONSERVATION_LIMIT:
        raise ValueError(f"task violates energy conservation: {expected['id']}")
    time_s = float(result.get("time_s", -1.0))
    if not np.isfinite(time_s) or time_s < 0.0:
        raise ValueError(f"task runtime is invalid: {expected['id']}")


def validate_results(results: dict, tasks: list[dict]) -> dict:
    expected_by_id = {task["id"]: task for task in tasks}
    failures = []
    maximum_error = 0.0
    if set(results) != set(expected_by_id):
        failures.append("task_id_set_mismatch")
    for identifier, result in results.items():
        expected = expected_by_id.get(identifier)
        if expected is None:
            continue
        try:
            validate_result(result, expected)
            r = np.asarray(result["R"], dtype=float)
            t = np.asarray(result["T"], dtype=float)
            maximum_error = max(maximum_error, float(np.max(np.abs(r + t - 1.0))))
        except (TypeError, ValueError, KeyError) as exc:
            failures.append(f"{identifier}: {exc}")
    return {
        "passed": not failures and maximum_error <= v1.CONSERVATION_LIMIT,
        "records": len(results),
        "expected_records": len(expected_by_id),
        "pointwise_conservation_error_max": maximum_error,
        "failures": failures,
    }


def validate_v1_source(plan: dict, plan_path: Path) -> tuple[dict, dict, dict]:
    audit_binding = plan.get("source_v1_audit")
    evidence_binding = plan.get("source_v1_evidence")
    checkpoint_binding = plan.get("source_v1_checkpoint")
    source_plan_binding = plan.get("source_v1_plan")
    for item, label in (
        (audit_binding, "v1 audit"),
        (evidence_binding, "v1 evidence"),
        (checkpoint_binding, "v1 checkpoint"),
        (source_plan_binding, "v1 plan"),
    ):
        if not isinstance(item, dict):
            raise ValueError(f"missing {label} binding")
    audit_path = ROOT / audit_binding["path"]
    evidence_path = ROOT / evidence_binding["path"]
    checkpoint_path = ROOT / checkpoint_binding["path"]
    source_plan_path = ROOT / source_plan_binding["path"]
    for path, label in (
        (audit_path, "v1 audit"),
        (evidence_path, "v1 evidence"),
        (checkpoint_path, "v1 checkpoint"),
        (source_plan_path, "v1 plan"),
    ):
        if not path.is_file():
            raise ValueError(f"missing {label}: {relative_path(path)}")
    require_binding(audit_binding, audit_path, "v1 audit")
    require_binding(evidence_binding, evidence_path, "v1 evidence")
    require_binding(checkpoint_binding, checkpoint_path, "v1 checkpoint")
    require_binding(source_plan_binding, source_plan_path, "v1 plan")
    audit = load_json(audit_path, {}) or {}
    if audit.get("evidence_version") != "paper2-reference-resolution-audit-v1":
        raise ValueError("unexpected v1 audit version")
    if audit.get("passed") is not False or audit.get("classification") in {
        "execution_integrity_failure",
        "worker_evidence_integrity_failure",
    }:
        raise ValueError("v1 source is not a terminal scientific failure")
    for key in (
        "frozen_plan_sha256_and_content",
        "checkpoint_meta_and_runtime_hashes",
        "reference_checkpoint_exact_80",
        "worker_claim_matches_independent_recomputation",
        "physics_controls_passed",
    ):
        if audit.get("checks", {}).get(key) is not True:
            raise ValueError(f"v1 source prerequisite is unproven: {key}")
    evidence = load_json(evidence_path, {}) or {}
    if evidence.get("evidence_version") != "paper2-reference-resolution-v1":
        raise ValueError("unexpected v1 worker evidence version")
    if evidence.get("passed") is not False or evidence.get("pool_sha256") != plan.get("pool_sha256"):
        raise ValueError("v1 worker evidence is not bound to the frozen failed pool")
    if evidence.get("selection") != plan.get("selection"):
        raise ValueError("v1 worker selection differs from v2 plan")
    audit_inputs = audit.get("inputs", {})
    for key, expected in (
        ("reference_evidence", evidence_binding),
        ("reference_checkpoint", checkpoint_binding),
        ("plan", source_plan_binding),
    ):
        if audit_inputs.get(key) != expected:
            raise ValueError(f"v1 audit {key} binding differs from v2 plan")
    with checkpoint_path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    baseline_tasks = v1.build_tasks(plan["selection"])
    if len(checkpoint.get("results", {})) != len(baseline_tasks):
        raise ValueError("v1 checkpoint is not complete 80/80")
    baseline_validation = validate_results(checkpoint.get("results", {}), baseline_tasks)
    if not baseline_validation["passed"]:
        raise ValueError("v1 checkpoint raw spectra are invalid")
    if evidence.get("checkpoint", {}).get("sha256", "").upper() != file_digest(checkpoint_path):
        raise ValueError("v1 worker evidence checkpoint hash mismatch")
    return audit, evidence, checkpoint


def validate_plan(plan: dict, plan_path: Path) -> None:
    if plan.get("schema_version") != 1 or plan.get("evidence_version") != PLAN_VERSION:
        raise ValueError("unexpected v2 plan version")
    if plan.get("plan_valid") is not True or plan.get("source_failed_action") != "joint_numerical_convergence":
        raise ValueError("v2 plan is not an explicit post-failure diagnostic plan")
    if plan.get("extra_configs") != [list(x) for x in EXTRA_CONFIGS]:
        raise ValueError("v2 extra configuration set drift")
    if plan.get("base_config") != list(BASE_CONFIG) or plan.get("steps_nm") != [1.0, 0.5]:
        raise ValueError("v2 baseline or spectral steps drift")
    if plan.get("expected_new_tasks") != EXPECTED_TASKS or len(plan.get("selection", [])) != 8:
        raise ValueError("v2 task count or selection drift")
    expected_thresholds = {
        "mean_joint_dE00_lt": v1.MEAN_DE_LIMIT,
        "all_joint_dE00_lt": v1.PER_GEOMETRY_DE_LIMIT,
        "pointwise_conservation_lte": v1.CONSERVATION_LIMIT,
    }
    if plan.get("thresholds") != expected_thresholds:
        raise ValueError("v2 thresholds changed")
    pool_sha = str(plan.get("pool_sha256", ""))
    if len(pool_sha) != 64 or pool_sha != pool_sha.upper() or any(
        char not in "0123456789ABCDEF" for char in pool_sha
    ):
        raise ValueError("v2 pool hash is not canonical")
    if not plan_path.is_file():
        raise ValueError("v2 plan file is missing")


def comparison(
    selected: list[dict],
    left: dict,
    right: dict,
    left_id,
    right_id,
) -> dict:
    values = []
    rows = []
    for index, _geometry in enumerate(selected):
        channels = []
        for pol in POLS:
            a = left[left_id(index, pol)]
            b = right[right_id(index, pol)]
            value = float(
                v1.delta_e2000(
                    v1.labels_on_grid(a["wavelength_nm"], a["R"]),
                    v1.labels_on_grid(b["wavelength_nm"], b["R"]),
                )
            )
            channels.append(value)
            rows.append({"geometry_index": index, "pol": pol, "dE00": value})
        values.append(max(channels))
    array = np.asarray(values, dtype=float)
    mean_passed = bool(array.size == len(selected) and np.mean(array) < v1.MEAN_DE_LIMIT)
    all_passed = bool(array.size == len(selected) and np.all(array < v1.PER_GEOMETRY_DE_LIMIT))
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)) if array.size else None,
        "max": float(np.max(array)) if array.size else None,
        "mean_lt_1_15": mean_passed,
        "all_lt_2_3": all_passed,
        "passed": mean_passed and all_passed,
        "joint_max_by_geometry": array.tolist(),
        "rows": rows,
    }


def build_audit(
    evidence_path: Path,
    checkpoint_path: Path,
    plan_path: Path,
) -> dict:
    plan = load_json(plan_path, {}) or {}
    validate_plan(plan, plan_path)
    _source_audit, _source_evidence, baseline = validate_v1_source(plan, plan_path)
    evidence = load_json(evidence_path, {}) or {}
    if evidence.get("evidence_version") != "paper2-reference-resolution-budget-v2":
        raise ValueError("unexpected v2 worker evidence version")
    require_binding(evidence.get("plan"), plan_path, "v2 evidence plan")
    require_binding(evidence.get("checkpoint"), checkpoint_path, "v2 evidence checkpoint")
    if evidence.get("pool_sha256") != plan.get("pool_sha256"):
        raise ValueError("v2 evidence pool binding mismatch")
    if evidence.get("thresholds") != plan.get("thresholds"):
        raise ValueError("v2 evidence threshold binding mismatch")
    if not runtime_hashes_match(evidence.get("runtime_hashes")):
        raise ValueError("v2 evidence runtime hash mismatch")
    if evidence.get("training_allowed") is not False:
        raise ValueError("v2 diagnostic evidence cannot enable training")
    with checkpoint_path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    tasks = build_tasks(plan["selection"])
    expected_meta = {
        "version": "paper2-reference-resolution-budget-v2",
        "plan_sha256": file_digest(plan_path),
        "pool_sha256": plan["pool_sha256"],
        "selected_geometries": plan["selection"],
        "expected_tasks": EXPECTED_TASKS,
        "tasks": [
            {key: task[key] for key in ("id", "geometry_index", "pol", "requested_nG", "Nxy", "step_nm")}
            for task in tasks
        ],
        "runtime_hashes": {path: file_digest(ROOT / path) for path in RUNTIME_PATHS},
    }
    if checkpoint.get("meta") != expected_meta:
        raise ValueError("v2 checkpoint metadata or runtime hash mismatch")
    validation = validate_results(checkpoint.get("results", {}), tasks)
    if not validation["passed"]:
        raise ValueError("v2 checkpoint raw spectra are invalid")
    baseline_results = baseline["results"]
    candidate_results = checkpoint["results"]
    def candidate_id(config: tuple[int, int], step: float):
        return lambda index, pol: task_id(index, pol, config, step)
    axes = {
        name: comparison(
            plan["selection"],
            baseline_results,
            candidate_results,
            lambda index, pol, step=step: v1.task_id(index, pol, BASE_CONFIG, step),
            candidate_id(config, step),
        )
        for name, config, step in spatial_axis_specs()
    }
    axes.update({
        "spectral_450x512": comparison(
            plan["selection"], candidate_results, candidate_results,
            candidate_id(EXTRA_CONFIGS[0], 1.0), candidate_id(EXTRA_CONFIGS[0], 0.5)
        ),
        "spectral_365x768": comparison(
            plan["selection"], candidate_results, candidate_results,
            candidate_id(EXTRA_CONFIGS[1], 1.0), candidate_id(EXTRA_CONFIGS[1], 0.5)
        ),
        "spectral_450x768": comparison(
            plan["selection"], candidate_results, candidate_results,
            candidate_id(EXTRA_CONFIGS[2], 1.0), candidate_id(EXTRA_CONFIGS[2], 0.5)
        ),
    })
    checks = {
        "plan_and_source_hashes_verified": True,
        "v1_scientific_failure_verified": True,
        "exact_new_task_set": validation["records"] == EXPECTED_TASKS,
        "new_spectra_valid": validation["passed"],
        "runtime_hashes_verified": runtime_hashes_match(expected_meta["runtime_hashes"]),
        "order_converged": all(
            axes[name]["passed"] for name, _config, _step in spatial_axis_specs()
            if name.startswith("order_")
        ),
        "grid_converged": all(
            axes[name]["passed"] for name, _config, _step in spatial_axis_specs()
            if name.startswith("grid_")
        ),
        "corner_converged": all(
            axes[name]["passed"] for name, _config, _step in spatial_axis_specs()
            if name.startswith("corner_")
        ),
        "spectral_450x512_converged": axes["spectral_450x512"]["passed"],
        "spectral_365x768_converged": axes["spectral_365x768"]["passed"],
        "spectral_450x768_converged": axes["spectral_450x768"]["passed"],
    }
    passed = all(checks.values())
    worker_claim = evidence.get("passed")
    if not isinstance(worker_claim, bool):
        raise ValueError("v2 worker passed claim must be boolean")
    if not worker_claim_matches(worker_claim, passed):
        raise ValueError("v2 worker passed claim disagrees with independent recomputation")
    return {
        "schema_version": 1,
        "evidence_version": VERSION,
        "passed": passed,
        "classification": "budget_v2_converged" if passed else "budget_v2_still_insufficient",
        "pool_sha256": plan["pool_sha256"],
        "checks": checks,
        "worker_claim": {"passed": worker_claim, "matches_independent_recomputation": True},
        "thresholds": plan["thresholds"],
        "comparisons": axes,
        "plan": binding(plan_path),
        "checkpoint": binding(checkpoint_path) | {"tasks": len(candidate_results)},
        "source_v1_audit": plan["source_v1_audit"],
        "source_v1_evidence": plan["source_v1_evidence"],
        "source_v1_checkpoint": plan["source_v1_checkpoint"],
        "runtime_hashes": expected_meta["runtime_hashes"],
        "training_allowed": False,
        "decision_scope": "Diagnostic only; never activates a pool or authorizes training.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default=".state/reference_resolution_budget_v2_plan.json")
    parser.add_argument("--evidence", default=".state/reference_resolution_budget_v2.json")
    parser.add_argument("--checkpoint", default=".state/reference_resolution_budget_v2_checkpoint.pkl")
    parser.add_argument("--output", default=".state/reference_resolution_budget_v2_audit.json")
    args = parser.parse_args()
    paths = {name: ROOT / value for name, value in (
        ("plan", args.plan), ("evidence", args.evidence), ("checkpoint", args.checkpoint)
    )}
    try:
        missing = [relative_path(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise ValueError(f"required v2 audit inputs are missing: {missing}")
        audit = build_audit(paths["evidence"], paths["checkpoint"], paths["plan"])
    except Exception as exc:
        audit = {
            "schema_version": 1,
            "evidence_version": VERSION,
            "passed": False,
            "classification": "execution_integrity_failure",
            "error": f"{type(exc).__name__}: {exc}",
            "inputs": {
                name: binding(path) for name, path in paths.items() if path.is_file()
            },
            "training_allowed": False,
        }
    output = ROOT / args.output
    if output.exists():
        existing = load_json(output, {}) or {}
        if existing != audit:
            raise SystemExit("existing v2 audit differs; use a new evidence version")
    else:
        atomic_json(output, audit)
    print(json.dumps({"passed": audit["passed"], "classification": audit["classification"]}, sort_keys=True))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
