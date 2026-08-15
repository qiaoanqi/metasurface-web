#!/usr/bin/env python3
"""Run the frozen, diagnostic-only numerical-budget v2 escalation."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_supervisor import atomic_json, audit_protected_files, file_digest, load_json  # noqa: E402
from scripts import run_reference_resolution_escalation as v1  # noqa: E402
from scripts.freeze_reference_budget_v2 import EXTRA_CONFIGS, VERSION as PLAN_VERSION  # noqa: E402


VERSION = "paper2-reference-resolution-budget-v2"
ACTION = "joint_numerical_convergence"
POLS = v1.POLS
BASE_CONFIG = v1.FINE_CONFIG
STEPS = (1.0, 0.5)
EXPECTED_TASKS = len(EXTRA_CONFIGS) * len(POLS) * 8 * len(STEPS)
SPATIAL_CONFIGS = {
    "order": EXTRA_CONFIGS[0],
    "grid": EXTRA_CONFIGS[1],
    "corner": EXTRA_CONFIGS[2],
}


def dispatch_identity(path: Path) -> dict:
    dispatch = load_json(path, {}) or {}
    if (
        dispatch.get("action") != ACTION
        or dispatch.get("status") != "in_progress"
        or int(dispatch.get("strategy_revision", 0)) < 2
        or not isinstance(dispatch.get("request_id"), str)
        or not dispatch["request_id"]
        or int(dispatch.get("attempt", 0)) < 1
    ):
        raise ValueError("budget v2 requires an active revision-2 joint request")
    return {
        "request_id": dispatch["request_id"],
        "attempt": int(dispatch["attempt"]),
    }


def spatial_axis_specs() -> list[tuple[str, tuple[int, int], float]]:
    return [
        (f"{axis}_365x512_to_{config[0]}x{config[1]}_{'0p5nm' if step == 0.5 else '1nm'}", config, step)
        for axis, config in SPATIAL_CONFIGS.items()
        for step in STEPS
    ]


def task_id(index: int, pol: str, config: tuple[int, int], step: float) -> str:
    token = "0p5" if step == 0.5 else "1"
    return f"refbudget-v2-g{index:02d}-{pol}-ng{config[0]}-nxy{config[1]}-step{token}"


def build_tasks(selected: list[dict]) -> list[dict]:
    if len(selected) != 8:
        raise ValueError("budget v2 requires the frozen eight cases")
    tasks = []
    for index, geometry in enumerate(selected):
        for pol in POLS:
            for config in EXTRA_CONFIGS:
                for step in STEPS:
                    wavelength = v1.WL_HALF_NM if step == 0.5 else v1.WL_1NM
                    tasks.append({
                        "id": task_id(index, pol, config, step),
                        "geometry_index": index,
                        "geometry": geometry,
                        "pol": pol,
                        "requested_nG": config[0],
                        "retained_nG": v1.retained_order(config[0], geometry["P"]),
                        "Nxy": config[1],
                        "step_nm": step,
                        "wavelength_nm": wavelength,
                    })
    return tasks


def run_task(task: dict) -> dict:
    # Reuse the frozen solver entry point, but use versioned task identities.
    return v1.run_task(task)


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


def validate_checkpoint_results(results: dict, tasks: list[dict]) -> None:
    expected_by_id = {task["id"]: task for task in tasks}
    if set(results) - set(expected_by_id):
        raise ValueError("budget v2 checkpoint contains unknown task ids")
    for identifier, result in results.items():
        validate_result(result, expected_by_id[identifier])


def load_inputs(
    plan_path: Path,
    v1_audit_path: Path,
    v1_evidence_path: Path,
    v1_checkpoint_path: Path,
) -> tuple[dict, dict, dict]:
    plan = load_json(plan_path, {}) or {}
    if plan.get("evidence_version") != PLAN_VERSION or plan.get("plan_valid") is not True:
        raise ValueError("budget v2 plan is not valid")
    require_binding(plan.get("source_v1_audit"), v1_audit_path, "v1 audit")
    require_binding(plan.get("source_v1_evidence"), v1_evidence_path, "v1 evidence")
    require_binding(plan.get("source_v1_checkpoint"), v1_checkpoint_path, "v1 checkpoint")
    audit = load_json(v1_audit_path, {}) or {}
    if audit.get("evidence_version") != "paper2-reference-resolution-audit-v1":
        raise ValueError("independent v1 reference audit is required before budget v2")
    if audit.get("passed") is not False or audit.get("classification") in {
        "execution_integrity_failure",
        "worker_evidence_integrity_failure",
    }:
        raise ValueError("budget v2 requires a terminal scientific v1 failure")
    evidence = load_json(v1_evidence_path, {}) or {}
    if evidence.get("evidence_version") != "paper2-reference-resolution-v1":
        raise ValueError("v1 worker evidence version is not frozen")
    if evidence.get("passed") is not False:
        raise ValueError("budget v2 requires the failed v1 audit to remain explicit")
    with v1_checkpoint_path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    if len(checkpoint.get("results", {})) != 80:
        raise ValueError("v1 reference checkpoint must be complete before budget v2")
    if str(plan.get("pool_sha256", "")).upper() != str(evidence.get("pool_sha256", "")).upper():
        raise ValueError("budget v2 pool binding mismatch")
    if plan.get("selection") != evidence.get("selection"):
        raise ValueError("budget v2 geometry selection drift")
    baseline_tasks = v1.build_tasks(plan["selection"])
    if len(baseline_tasks) != 80:
        raise ValueError("v1 reference task protocol is not 80 tasks")
    validate_checkpoint_results(checkpoint.get("results", {}), baseline_tasks)
    return plan, evidence, checkpoint


def validate_results(results: dict, tasks: list[dict]) -> dict:
    maximum_error = 0.0
    failures = []
    expected_by_id = {task["id"]: task for task in tasks}
    valid = set(results) == set(expected_by_id)
    for identifier in sorted(set(expected_by_id) - set(results)):
        failures.append({"id": identifier, "error": "missing task"})
    for identifier in sorted(set(results) - set(expected_by_id)):
        failures.append({"id": identifier, "error": "unknown task"})
    for identifier, result in results.items():
        if identifier not in expected_by_id:
            valid = False
            continue
        try:
            validate_result(result, expected_by_id[identifier])
            reflectance = np.asarray(result["R"], dtype=float)
            transmittance = np.asarray(result["T"], dtype=float)
            maximum_error = max(maximum_error, float(np.max(np.abs(reflectance + transmittance - 1.0))))
        except (TypeError, ValueError, KeyError) as exc:
            valid = False
            failures.append({"id": identifier, "error": str(exc)})
    return {
        "passed": valid and maximum_error <= v1.CONSERVATION_LIMIT and not failures,
        "records": len(results),
        "expected_records": len(expected_by_id),
        "pointwise_conservation_error_max": maximum_error,
        "failures": failures,
    }


def comparison(selected: list[dict], results: dict, left_config: tuple[int, int], left_step: float, right_config: tuple[int, int], right_step: float) -> dict:
    def result_id(index: int, pol: str, config: tuple[int, int], step: float) -> str:
        if config in EXTRA_CONFIGS:
            return task_id(index, pol, config, step)
        return v1.task_id(index, pol, config, step)

    values = []
    rows = []
    for index, _geometry in enumerate(selected):
        channels = []
        for pol in POLS:
            left = results[result_id(index, pol, left_config, left_step)]
            right = results[result_id(index, pol, right_config, right_step)]
            left_lab = v1.labels_on_grid(left["wavelength_nm"], left["R"])
            right_lab = v1.labels_on_grid(right["wavelength_nm"], right["R"])
            value = float(v1.delta_e2000(left_lab, right_lab))
            channels.append(value)
            rows.append({"geometry_index": index, "pol": pol, "dE00": value})
        values.append(max(channels))
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)) if array.size else None,
        "max": float(np.max(array)) if array.size else None,
        "mean_lt_1_15": bool(array.size == len(selected) and np.mean(array) < v1.MEAN_DE_LIMIT),
        "all_lt_2_3": bool(array.size == len(selected) and np.all(array < v1.PER_GEOMETRY_DE_LIMIT)),
        "passed": bool(array.size == len(selected) and np.mean(array) < v1.MEAN_DE_LIMIT and np.all(array < v1.PER_GEOMETRY_DE_LIMIT)),
        "joint_max_by_geometry": array.tolist(),
        "rows": rows,
    }


def summarize(
    plan: dict,
    evidence: dict,
    baseline: dict,
    results: dict,
    checkpoint_path: Path,
    runtime_hashes: dict,
    request: dict,
) -> dict:
    selected = plan["selection"]
    tasks = build_tasks(selected)
    spectra = validate_results(results, tasks)
    common = {
        "schema_version": 1,
        "evidence_version": VERSION,
        "request": request,
        "pool_sha256": plan["pool_sha256"],
        "spectra": spectra,
        "thresholds": plan["thresholds"],
        "plan": binding(ROOT / ".state/reference_resolution_budget_v2_plan.json"),
        "source_v1_audit": plan["source_v1_audit"],
        "source_v1_evidence": plan["source_v1_evidence"],
        "source_v1_checkpoint": plan["source_v1_checkpoint"],
        "source_v1_plan": plan["source_v1_plan"],
        "checkpoint": {
            "path": relative_path(checkpoint_path),
            "sha256": file_digest(checkpoint_path),
            "tasks": len(results),
        },
        "runtime_hashes": runtime_hashes,
        "protected_files": audit_protected_files(load_json(ROOT / "pipeline_policy.json", {}) or {}),
        "training_allowed": False,
        "decision_scope": "Diagnostic only; never rehabilitates the historical nG131 pool or authorizes training.",
    }
    if not spectra["passed"]:
        return {
            **common,
            "passed": False,
            "classification": "execution_or_spectrum_failure",
            "checks": {
                "exact_new_task_set": spectra["records"] == EXPECTED_TASKS,
                "new_spectra_valid": False,
                "runtime_hashes_verified": all(
                    file_digest(ROOT / path) == value for path, value in runtime_hashes.items()
                ),
                "order_converged": False,
                "grid_converged": False,
                "corner_converged": False,
                "spectral_450x512_converged": False,
                "spectral_365x768_converged": False,
                "spectral_450x768_converged": False,
            },
            "comparisons": {},
        }
    combined = dict(baseline["results"])
    combined.update(results)
    axes = {
        name: comparison(selected, combined, BASE_CONFIG, step, config, step)
        for name, config, step in spatial_axis_specs()
    }
    axes.update({
        "spectral_450x512": comparison(selected, combined, EXTRA_CONFIGS[0], 1.0, EXTRA_CONFIGS[0], 0.5),
        "spectral_365x768": comparison(selected, combined, EXTRA_CONFIGS[1], 1.0, EXTRA_CONFIGS[1], 0.5),
        "spectral_450x768": comparison(selected, combined, EXTRA_CONFIGS[2], 1.0, EXTRA_CONFIGS[2], 0.5),
    })
    checks = {
        "exact_new_task_set": spectra["records"] == EXPECTED_TASKS,
        "new_spectra_valid": spectra["passed"],
        "runtime_hashes_verified": all(file_digest(ROOT / path) == value for path, value in runtime_hashes.items()),
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
    return {
        **common,
        "passed": passed,
        "classification": "budget_v2_converged" if passed else "budget_v2_still_insufficient",
        "checks": checks,
        "comparisons": axes,
    }


def run(args: argparse.Namespace) -> dict:
    request = dispatch_identity(ROOT / args.dispatch)
    plan_path = ROOT / args.plan
    v1_audit_path = ROOT / args.v1_audit
    v1_evidence_path = ROOT / args.v1_evidence
    v1_checkpoint_path = ROOT / args.v1_checkpoint
    plan, evidence, baseline = load_inputs(
        plan_path, v1_audit_path, v1_evidence_path, v1_checkpoint_path
    )
    tasks = build_tasks(plan["selection"])
    meta = {
        "version": VERSION,
        "request": request,
        "plan_sha256": file_digest(plan_path),
        "pool_sha256": plan["pool_sha256"],
        "selected_geometries": plan["selection"],
        "expected_tasks": len(tasks),
        "tasks": [{key: task[key] for key in ("id", "geometry_index", "pol", "requested_nG", "Nxy", "step_nm")} for task in tasks],
        "runtime_hashes": {path: file_digest(ROOT / path) for path in ("rcwa_batch.py", "paper2_colorimetry.py", "color_utils.py", "scripts/run_reference_resolution_budget_v2.py")},
    }
    checkpoint_path = ROOT / args.checkpoint
    if checkpoint_path.exists():
        with checkpoint_path.open("rb") as handle:
            checkpoint = pickle.load(handle)
        if checkpoint.get("meta") != meta:
            raise ValueError("budget v2 checkpoint metadata mismatch")
    else:
        checkpoint = {"meta": meta, "results": {}}
        v1.atomic_pickle(checkpoint_path, checkpoint)
    validate_checkpoint_results(checkpoint.get("results", {}), tasks)
    pending = [task for task in tasks if task["id"] not in checkpoint["results"]]
    with Pool(max(1, int(args.n_jobs))) as workers:
        for result in workers.imap_unordered(run_task, pending, chunksize=1):
            checkpoint["results"][result["id"]] = result
            v1.atomic_pickle(checkpoint_path, checkpoint)
    result = summarize(
        plan,
        evidence,
        baseline,
        checkpoint["results"],
        checkpoint_path,
        meta["runtime_hashes"],
        request,
    )
    output = ROOT / args.evidence
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != result:
            raise SystemExit("existing budget v2 evidence differs; use a new version")
    else:
        atomic_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default=".state/reference_resolution_budget_v2_plan.json")
    parser.add_argument("--v1-audit", default=".state/reference_resolution_v1_audit.json")
    parser.add_argument("--v1-evidence", default=".state/reference_resolution_v1.json")
    parser.add_argument("--v1-checkpoint", default=".state/reference_resolution_v1_checkpoint.pkl")
    parser.add_argument("--checkpoint", default=".state/reference_resolution_budget_v2_checkpoint.pkl")
    parser.add_argument("--evidence", default=".state/reference_resolution_budget_v2.json")
    parser.add_argument("--dispatch", default=".state/dispatch_request.json")
    parser.add_argument("--n-jobs", type=int, default=16)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"passed": result["passed"], "classification": result["classification"], "checks": result["checks"]}, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
