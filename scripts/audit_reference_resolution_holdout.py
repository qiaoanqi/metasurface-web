#!/usr/bin/env python3
"""Independently audit the v2-bound 24-case holdout."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pipeline_supervisor as supervisor  # noqa: E402
from scripts import freeze_reference_holdout_plan as freezer  # noqa: E402
from scripts import run_reference_resolution_budget_v2 as v2_runner  # noqa: E402


VERSION = "paper2-reference-holdout-audit-v1"
WORKER_VERSION = "paper2-reference-holdout-v2"
POLS = ("p", "s")
FINAL_REFERENCE = (450, 768, 0.5)
FINAL_REFERENCE_SPEC = {
    "requested_nG": FINAL_REFERENCE[0],
    "Nxy": FINAL_REFERENCE[1],
    "wavelength_step_nm": FINAL_REFERENCE[2],
}
COMPARISON_NAMES = (
    "order_365x768_to_450x768_0p5nm",
    "grid_450x512_to_450x768_0p5nm",
    "corner_365x512_to_450x768_0p5nm",
    "spectral_450x768_1nm_to_0p5nm",
    "frozen_candidate_to_final_reference",
)


def dispatch_identity(dispatch: dict[str, Any]) -> dict[str, Any]:
    if (
        dispatch.get("action") != "reference_resolution"
        or dispatch.get("status") not in {"pending", "in_progress"}
        or not isinstance(dispatch.get("request_id"), str)
        or not dispatch["request_id"]
        or int(dispatch.get("attempt", 0)) < 1
    ):
        raise ValueError("active reference_resolution dispatch identity is invalid")
    return {
        "request_id": dispatch["request_id"],
        "attempt": int(dispatch["attempt"]),
    }


def checkpoint_request_identity(request: dict[str, Any]) -> dict[str, str]:
    return {"request_id": str(request["request_id"])}


def reusable_worker_request(stored: Any, active_request: dict[str, Any]) -> dict[str, Any]:
    request = stored.get("request") if isinstance(stored, dict) else None
    if (
        not isinstance(request, dict)
        or request.get("request_id") != active_request["request_id"]
        or int(request.get("attempt", 0)) < 1
        or int(request.get("attempt", 0)) > active_request["attempt"]
    ):
        raise ValueError("holdout worker evidence is not reusable by the active request attempt")
    return request


def retry_replaces_execution_failure(existing: Any, candidate: dict[str, Any]) -> bool:
    if not isinstance(existing, dict):
        return False
    previous = existing.get("request")
    current = candidate.get("request")
    return bool(
        isinstance(previous, dict)
        and isinstance(current, dict)
        and previous.get("request_id") == current.get("request_id")
        and 1 <= int(previous.get("attempt", 0)) <= int(current.get("attempt", 0))
        and existing.get("passed") is False
        and existing.get("classification") == "execution_integrity_failure"
    )


def write_retry_safe_audit(path: Path, payload: dict[str, Any]) -> None:
    if not path.exists():
        supervisor.atomic_json(path, payload)
        return
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing == payload:
        return
    if retry_replaces_execution_failure(existing, payload):
        supervisor.atomic_json(path, payload)
        return
    raise ValueError("existing holdout audit differs; use a new evidence version")


def relative_path(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def binding(path: Path) -> dict[str, str]:
    return {"path": relative_path(path), "sha256": supervisor.file_digest(path)}


def resolve_binding(item: Any, label: str) -> Path:
    if not isinstance(item, dict) or not isinstance(item.get("path"), str):
        raise ValueError(f"missing {label} binding")
    path = (ROOT / item["path"]).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} path escapes the workspace") from exc
    if relative_path(path) != str(item["path"]).replace("\\", "/"):
        raise ValueError(f"{label} path is not canonical")
    if not path.is_file() or supervisor.file_digest(path) != str(item.get("sha256", "")).upper():
        raise ValueError(f"{label} SHA256 binding mismatch")
    return path


def source_id(index: int, pol: str, config: tuple[int, int], step: float) -> str:
    if config == v2_runner.BASE_CONFIG:
        return v2_runner.v1.task_id(index, pol, config, step)
    return v2_runner.task_id(index, pol, config, step)


def extension_id(index: int, pol: str, config: tuple[int, int], step: float) -> str:
    token = "0p5" if step == 0.5 else "1"
    return f"refhold-v2-g{index:02d}-{pol}-ng{config[0]}-nxy{config[1]}-step{token}"


def result_id(index: int, pol: str, config: tuple[int, int], step: float) -> str:
    return source_id(index, pol, config, step) if index < 8 else extension_id(index, pol, config, step)


def independent_protocol_manifest(candidate: dict) -> list[dict]:
    roles = ("base", "order", "grid", "spectral", "final_reference")
    manifest = [
        {"requested_nG": item[0], "Nxy": item[1], "step_nm": item[2], "role": role}
        for item, role in zip(freezer.MINIMUM_HOLDOUT_PROTOCOLS, roles)
    ]
    key = (
        int(candidate["requested_nG"]),
        int(candidate["Nxy"]),
        float(candidate["wavelength_step_nm"]),
    )
    if key not in freezer.MINIMUM_HOLDOUT_PROTOCOLS:
        manifest.append(
            {
                "requested_nG": key[0],
                "Nxy": key[1],
                "step_nm": key[2],
                "role": "frozen_production_candidate",
            }
        )
    return manifest


def validate_plan(plan: dict, path: Path) -> None:
    if (
        plan.get("schema_version") != 1
        or plan.get("evidence_version") != freezer.VERSION
        or plan.get("plan_valid") is not True
        or plan.get("created_before_holdout_results") is not True
        or plan.get("candidate_frozen_on_initial_eight_only") is not True
        or plan.get("holdout_cannot_reselect_candidate") is not True
    ):
        raise ValueError("invalid v2 holdout plan state")
    if (
        len(plan.get("existing_cases", [])) != 8
        or len(plan.get("new_cases", [])) != 24
        or len(plan.get("combined_cases", [])) != 32
        or plan.get("combined_cases") != plan.get("existing_cases") + plan.get("new_cases")
    ):
        raise ValueError("v2 holdout geometry manifest drift")
    expected_hashes = (
        (plan["existing_cases"], freezer.ARCHIVED_EXISTING_SHA256),
        (plan["new_cases"], freezer.ARCHIVED_NEW_SHA256),
        (plan["combined_cases"], freezer.ARCHIVED_COMBINED_SHA256),
    )
    if any(freezer.selection_sha256(items) != digest for items, digest in expected_hashes):
        raise ValueError("v2 holdout selection hash mismatch")
    candidate = plan.get("frozen_candidate")
    if not isinstance(candidate, dict) or candidate.get("passed") is not True:
        raise ValueError("v2 holdout frozen candidate is invalid")
    manifest = independent_protocol_manifest(candidate)
    if plan.get("task_protocols") != manifest:
        raise ValueError("v2 holdout protocol manifest drift")
    per_geometry = len(manifest) * len(POLS)
    if (
        plan.get("expected_source_tasks") != 8 * per_geometry
        or plan.get("expected_new_tasks") != 24 * per_geometry
        or plan.get("expected_combined_tasks") != 32 * per_geometry
    ):
        raise ValueError("v2 holdout task counts drift")
    expected_thresholds = {
        "mean_joint_dE00_lt": v2_runner.v1.MEAN_DE_LIMIT,
        "all_joint_dE00_lt": v2_runner.v1.PER_GEOMETRY_DE_LIMIT,
        "pointwise_conservation_lte": v2_runner.v1.CONSERVATION_LIMIT,
    }
    if plan.get("thresholds") != expected_thresholds:
        raise ValueError("v2 holdout thresholds changed")
    if set(plan.get("runtime_hashes", {})) != set(freezer.RUNTIME_PATHS):
        raise ValueError("v2 holdout runtime hash manifest is incomplete")
    for runtime, digest in plan["runtime_hashes"].items():
        if supervisor.file_digest(ROOT / runtime) != str(digest).upper():
            raise ValueError(f"v2 holdout runtime hash mismatch: {runtime}")
    if not path.is_file():
        raise ValueError("v2 holdout plan file is missing")


def build_task(plan: dict, index: int, pol: str, protocol: dict) -> dict:
    geometry = plan["combined_cases"][index]
    config = (int(protocol["requested_nG"]), int(protocol["Nxy"]))
    step = float(protocol["step_nm"])
    wavelength = v2_runner.v1.WL_HALF_NM if step == 0.5 else v2_runner.v1.WL_1NM
    return {
        "id": result_id(index, pol, config, step),
        "geometry_index": index,
        "geometry": geometry,
        "pol": pol,
        "requested_nG": config[0],
        "retained_nG": v2_runner.v1.retained_order(config[0], geometry["P"]),
        "Nxy": config[1],
        "step_nm": step,
        "wavelength_nm": wavelength,
    }


def build_tasks(plan: dict, indices: range) -> list[dict]:
    return [
        build_task(plan, index, pol, protocol)
        for index in indices
        for pol in POLS
        for protocol in plan["task_protocols"]
    ]


def geometry_key(item: dict) -> tuple[float, float, float, float]:
    return tuple(float(item[name]) for name in ("L", "W", "H", "P"))


def validate_result(result: dict, task: dict) -> None:
    if not isinstance(result, dict) or result.get("status") != "ok":
        raise ValueError(f"failed or malformed task: {task['id']}")
    if set(result) != set(task) | {"status", "R", "T", "time_s"}:
        raise ValueError(f"task field set mismatch: {task['id']}")
    for name in ("id", "geometry_index", "pol", "requested_nG", "retained_nG", "Nxy"):
        if result.get(name) != task[name]:
            raise ValueError(f"task identity mismatch {task['id']}: {name}")
    if float(result.get("step_nm", -1.0)) != float(task["step_nm"]):
        raise ValueError(f"task step mismatch: {task['id']}")
    if geometry_key(result.get("geometry", {})) != geometry_key(task["geometry"]):
        raise ValueError(f"task geometry mismatch: {task['id']}")
    expected_wavelength = np.asarray(task["wavelength_nm"], dtype=float)
    actual_wavelength = np.asarray(result.get("wavelength_nm"), dtype=float)
    R = np.asarray(result.get("R"), dtype=float)
    T = np.asarray(result.get("T"), dtype=float)
    if actual_wavelength.shape != expected_wavelength.shape or not np.array_equal(
        actual_wavelength, expected_wavelength
    ):
        raise ValueError(f"task wavelength grid mismatch: {task['id']}")
    if R.shape != expected_wavelength.shape or T.shape != expected_wavelength.shape:
        raise ValueError(f"task spectrum shape mismatch: {task['id']}")
    if not np.isfinite(R).all() or not np.isfinite(T).all():
        raise ValueError(f"task spectrum contains non-finite values: {task['id']}")
    if not (
        np.all(R >= -1e-8)
        and np.all(R <= 1.0 + 1e-8)
        and np.all(T >= -1e-8)
        and np.all(T <= 1.0 + 1e-8)
    ):
        raise ValueError(f"task spectrum is outside physical bounds: {task['id']}")
    if np.max(np.abs(R + T - 1.0)) > v2_runner.v1.CONSERVATION_LIMIT:
        raise ValueError(f"task violates energy conservation: {task['id']}")
    runtime = float(result.get("time_s", -1.0))
    if not np.isfinite(runtime) or runtime < 0.0:
        raise ValueError(f"task runtime is invalid: {task['id']}")


def validate_results(results: dict, tasks: list[dict]) -> dict:
    expected = {task["id"]: task for task in tasks}
    failures = []
    maximum_error = 0.0
    if set(results) != set(expected):
        failures.append("task_id_set_mismatch")
    for identifier, result in results.items():
        task = expected.get(identifier)
        if task is None:
            continue
        try:
            validate_result(result, task)
            R = np.asarray(result["R"], dtype=float)
            T = np.asarray(result["T"], dtype=float)
            maximum_error = max(maximum_error, float(np.max(np.abs(R + T - 1.0))))
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(f"{identifier}: {exc}")
    return {
        "passed": not failures and maximum_error <= v2_runner.v1.CONSERVATION_LIMIT,
        "records": len(results),
        "expected_records": len(expected),
        "pointwise_conservation_error_max": maximum_error,
        "failures": failures,
    }


def independent_threshold_summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)) if array.size else None,
        "max": float(np.max(array)) if array.size else None,
        "mean_lt_1_15": bool(
            array.size and np.mean(array) < v2_runner.v1.MEAN_DE_LIMIT
        ),
        "all_lt_2_3": bool(
            array.size and np.all(array < v2_runner.v1.PER_GEOMETRY_DE_LIMIT)
        ),
        "joint_max_by_geometry": array.tolist(),
    }


def independent_candidate_evaluation(selected: list[dict], results: dict) -> dict:
    configs = {
        "BASE": (365, 512),
        "ORDER": (450, 512),
        "GRID": (365, 768),
        "CORNER": (450, 768),
    }
    evaluations = []
    for name, config in configs.items():
        for step in (1.0, 0.5):
            values = []
            runtimes = []
            for index, _geometry in enumerate(selected):
                channels = []
                for pol in POLS:
                    source = results[source_id(index, pol, config, step)]
                    reference = results[source_id(index, pol, (450, 768), 0.5)]
                    channels.append(
                        float(
                            v2_runner.v1.delta_e2000(
                                v2_runner.v1.labels_on_grid(source["wavelength_nm"], source["R"]),
                                v2_runner.v1.labels_on_grid(reference["wavelength_nm"], reference["R"]),
                            )
                        )
                    )
                    runtimes.append(float(source["time_s"]))
                values.append(max(channels))
            summary = independent_threshold_summary(values)
            mean_runtime = float(np.mean(runtimes))
            summary.update(
                {
                    "config_name": name,
                    "requested_nG": config[0],
                    "Nxy": config[1],
                    "wavelength_step_nm": step,
                    "mean_task_seconds_estimate": mean_runtime,
                    "estimated_wall_hours_16_workers_6000": mean_runtime * 6000 / 16 / 3600,
                    "passed": summary["mean_lt_1_15"] and summary["all_lt_2_3"],
                }
            )
            evaluations.append(summary)
    passing = [item for item in evaluations if item["passed"]]
    candidate = min(
        passing,
        key=lambda item: (
            item["estimated_wall_hours_16_workers_6000"],
            item["requested_nG"],
            item["Nxy"],
            item["wavelength_step_nm"],
        ),
    ) if passing else None
    return {
        "selection_population": "initial_eight_cases_only",
        "holdout_used_for_selection": False,
        "reference": {
            "config_name": "CORNER",
            "requested_nG": 450,
            "Nxy": 768,
            "wavelength_step_nm": 0.5,
        },
        "thresholds": {
            "mean_joint_dE00_lt": v2_runner.v1.MEAN_DE_LIMIT,
            "all_joint_dE00_lt": v2_runner.v1.PER_GEOMETRY_DE_LIMIT,
        },
        "evaluations": evaluations,
        "lowest_cost_passing_protocol": candidate,
        "any_protocol_passed": bool(passing),
    }


def load_source_results(plan: dict) -> dict[str, dict]:
    paths = {
        name: resolve_binding(plan[name], name)
        for name in (
            "source_v2_plan",
            "source_v2_checkpoint",
            "source_v2_worker_evidence",
            "source_v2_independent_audit",
        )
    }
    source = freezer.validate_v2_source(
        paths["source_v2_plan"],
        paths["source_v2_checkpoint"],
        paths["source_v2_worker_evidence"],
        paths["source_v2_independent_audit"],
    )
    all_results = freezer.source_results(source)
    evaluation = independent_candidate_evaluation(plan["existing_cases"], all_results)
    if evaluation != plan.get("source_protocol_evaluation"):
        raise ValueError("initial-eight protocol evaluation does not independently reproduce")
    if evaluation.get("lowest_cost_passing_protocol") != plan.get("frozen_candidate"):
        raise ValueError("frozen candidate does not independently reproduce")
    tasks = build_tasks(plan, range(8))
    selected = {task["id"]: all_results[task["id"]] for task in tasks}
    if not validate_results(selected, tasks)["passed"]:
        raise ValueError("source task subset is invalid")
    return selected


def comparison_specs(plan: dict):
    candidate = plan["frozen_candidate"]
    candidate_key = (
        int(candidate["requested_nG"]),
        int(candidate["Nxy"]),
        float(candidate["wavelength_step_nm"]),
    )
    return (
        (COMPARISON_NAMES[0], (365, 768, 0.5), FINAL_REFERENCE),
        (COMPARISON_NAMES[1], (450, 512, 0.5), FINAL_REFERENCE),
        (COMPARISON_NAMES[2], (365, 512, 0.5), FINAL_REFERENCE),
        (COMPARISON_NAMES[3], (450, 768, 1.0), FINAL_REFERENCE),
        (COMPARISON_NAMES[4], candidate_key, FINAL_REFERENCE),
    )


def independent_comparison(results: dict, indices: range, left, right) -> dict:
    values = []
    rows = []
    for index in indices:
        channels = []
        for pol in POLS:
            a = results[result_id(index, pol, left[:2], left[2])]
            b = results[result_id(index, pol, right[:2], right[2])]
            value = float(
                v2_runner.v1.delta_e2000(
                    v2_runner.v1.labels_on_grid(a["wavelength_nm"], a["R"]),
                    v2_runner.v1.labels_on_grid(b["wavelength_nm"], b["R"]),
                )
            )
            channels.append(value)
            rows.append({"geometry_index": index, "pol": pol, "dE00": value})
        values.append(max(channels))
    array = np.asarray(values, dtype=float)
    mean_passed = bool(array.size == len(indices) and np.mean(array) < v2_runner.v1.MEAN_DE_LIMIT)
    all_passed = bool(
        array.size == len(indices) and np.all(array < v2_runner.v1.PER_GEOMETRY_DE_LIMIT)
    )
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


def evaluate_population(plan: dict, results: dict, indices: range) -> dict[str, dict]:
    return {
        name: independent_comparison(results, indices, left, right)
        for name, left, right in comparison_specs(plan)
    }


def classify(valid: bool, holdout: dict[str, dict]) -> str:
    if not valid:
        return "execution_integrity_failure"
    if not all(holdout[name]["passed"] for name in COMPARISON_NAMES[:4]):
        return "reference_budget_insufficient"
    if not holdout[COMPARISON_NAMES[4]]["passed"]:
        return "production_candidate_holdout_negative"
    return "reference_holdout_passed"


def expected_checkpoint_meta(
    plan: dict,
    plan_path: Path,
    tasks: list[dict],
    request: dict[str, Any],
) -> dict:
    return {
        "version": WORKER_VERSION,
        "request": checkpoint_request_identity(request),
        "plan": {"path": relative_path(plan_path), "sha256": supervisor.file_digest(plan_path)},
        "source_bindings": {
            name: plan[name]
            for name in (
                "source_v2_plan",
                "source_v2_checkpoint",
                "source_v2_worker_evidence",
                "source_v2_independent_audit",
                "source_base_checkpoint",
            )
        },
        "expected_tasks": len(tasks),
        "task_protocols": plan["task_protocols"],
        "tasks": [
            {key: task[key] for key in ("id", "geometry_index", "pol", "requested_nG", "Nxy", "step_nm")}
            for task in tasks
        ],
        "runtime_hashes": plan["runtime_hashes"],
    }


def recompute_worker_evidence(
    plan: dict,
    source_results: dict,
    extension_results: dict,
    plan_path: Path,
    checkpoint_path: Path,
    request: dict[str, Any],
) -> dict:
    extension_validation = validate_results(extension_results, build_tasks(plan, range(8, 32)))
    combined = dict(source_results)
    combined.update(extension_results)
    combined_validation = validate_results(combined, build_tasks(plan, range(32)))
    valid = extension_validation["passed"] and combined_validation["passed"]
    holdout = evaluate_population(plan, combined, range(8, 32)) if valid else {}
    supplemental = evaluate_population(plan, combined, range(32)) if valid else {}
    classification = classify(valid, holdout)
    passed = classification == "reference_holdout_passed"
    checks = {
        "v2_source_audit_passed_and_hash_bound": True,
        "candidate_frozen_on_initial_eight_only": True,
        "exact_new_task_set": extension_validation["records"] == plan["expected_new_tasks"],
        "exact_combined_task_set": combined_validation["records"] == plan["expected_combined_tasks"],
        "raw_spectra_and_conservation_valid": valid,
        "holdout_order_converged": bool(holdout.get(COMPARISON_NAMES[0], {}).get("passed")),
        "holdout_grid_converged": bool(holdout.get(COMPARISON_NAMES[1], {}).get("passed")),
        "holdout_corner_converged": bool(holdout.get(COMPARISON_NAMES[2], {}).get("passed")),
        "holdout_spectral_converged": bool(holdout.get(COMPARISON_NAMES[3], {}).get("passed")),
        "frozen_candidate_passed_holdout": bool(holdout.get(COMPARISON_NAMES[4], {}).get("passed")),
    }
    return {
        "schema_version": 1,
        "evidence_version": WORKER_VERSION,
        "passed": passed,
        "production_reference_approved": passed,
        "classification": classification,
        "request": request,
        "final_reference": FINAL_REFERENCE_SPEC,
        "pool_sha256": plan["pool"]["sha256"],
        "primary_gate_population": "24_new_holdout_geometries_only",
        "combined_32_population_scope": "supplemental_reporting_only",
        "selection": {
            "existing_case_count": 8,
            "new_case_count": 24,
            "combined_case_count": 32,
            "combined_selection_sha256": plan["combined_selection_sha256"],
        },
        "frozen_candidate": plan["frozen_candidate"],
        "task_protocols": plan["task_protocols"],
        "thresholds": plan["thresholds"],
        "extension_validation": extension_validation,
        "combined_validation": combined_validation,
        "checks": checks,
        "holdout_comparisons": holdout,
        "combined_32_supplemental_comparisons": supplemental,
        "plan": binding(plan_path),
        "source_v2_plan": plan["source_v2_plan"],
        "source_v2_checkpoint": plan["source_v2_checkpoint"],
        "source_v2_worker_evidence": plan["source_v2_worker_evidence"],
        "source_v2_independent_audit": plan["source_v2_independent_audit"],
        "source_base_checkpoint": plan["source_base_checkpoint"],
        "holdout_checkpoint": binding(checkpoint_path) | {"tasks": len(extension_results)},
        "training_allowed": False,
    }


def build_audit(
    plan_path: Path,
    worker_path: Path,
    checkpoint_path: Path,
    dispatch_path: Path,
) -> dict:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_plan(plan, plan_path)
    dispatch = json.loads(dispatch_path.read_text(encoding="utf-8"))
    active_request = dispatch_identity(dispatch)
    source_results = load_source_results(plan)
    with checkpoint_path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    extension_tasks = build_tasks(plan, range(8, 32))
    if checkpoint.get("meta") != expected_checkpoint_meta(
        plan, plan_path, extension_tasks, active_request
    ):
        raise ValueError("holdout checkpoint metadata, task identity, or runtime hash mismatch")
    extension_results = checkpoint.get("results", {})
    extension_validation = validate_results(extension_results, extension_tasks)
    stored = json.loads(worker_path.read_text(encoding="utf-8"))
    request = reusable_worker_request(stored, active_request)
    recomputed = recompute_worker_evidence(
        plan,
        source_results,
        extension_results,
        plan_path,
        checkpoint_path,
        request,
    )
    worker_exact = stored == recomputed
    policy = supervisor.load_policy()
    integrity = supervisor.verify_policy_integrity(policy)
    protected = supervisor.audit_protected_files(policy)
    checks = {
        "policy_integrity": integrity.get("passed") is True,
        "paper1_and_legacy_assets_unchanged": all(item.get("passed") for item in protected),
        "v2_plan_and_all_source_hashes_verified": True,
        "candidate_independently_refrozen_on_initial_eight": True,
        "holdout_did_not_reselect_candidate": True,
        "exact_extension_task_set": (
            extension_validation["records"] == plan["expected_new_tasks"]
            and extension_validation["passed"]
        ),
        "worker_evidence_exactly_reproduced": worker_exact,
        "production_reference_approved": recomputed["production_reference_approved"] is True,
    }
    passed = all(checks.values())
    classification = recomputed["classification"]
    if not worker_exact or not extension_validation["passed"]:
        classification = "execution_integrity_failure"
    return {
        "schema_version": 1,
        "evidence_version": VERSION,
        "protocol_revision": "v2_bound_holdout",
        "passed": passed,
        "production_reference_approved": passed,
        "classification": classification,
        "request": request,
        "final_reference": FINAL_REFERENCE_SPEC,
        "checks": checks,
        "approved_protocol_candidate": plan["frozen_candidate"] if passed else None,
        "pool_sha256": plan["pool"]["sha256"],
        "thresholds": plan["thresholds"],
        "primary_gate_population": "24_new_holdout_geometries_only",
        "combined_32_population_scope": "supplemental_reporting_only",
        "independent_holdout_comparisons": recomputed["holdout_comparisons"],
        "combined_32_supplemental_comparisons": recomputed[
            "combined_32_supplemental_comparisons"
        ],
        "sources": {
            "plan": binding(plan_path),
            "source_v2_plan": plan["source_v2_plan"],
            "source_v2_checkpoint": plan["source_v2_checkpoint"],
            "source_v2_worker_evidence": plan["source_v2_worker_evidence"],
            "source_v2_independent_audit": plan["source_v2_independent_audit"],
            "source_base_checkpoint": plan["source_base_checkpoint"],
            "holdout_evidence": binding(worker_path),
            "holdout_checkpoint": binding(checkpoint_path),
        },
        "worker_evidence": binding(worker_path),
        "independent_reproduction": True,
        "auditor_runtime_hashes": {
            path: supervisor.file_digest(ROOT / path)
            for path in supervisor.AUDITOR_RUNTIME_PATHS["reference_resolution"]
        },
        "protected_files": protected,
        "training_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default=".state/reference_resolution_holdout_v2_plan.json")
    parser.add_argument("--holdout-evidence", default=".state/reference_resolution_holdout_v2.json")
    parser.add_argument("--holdout-checkpoint", default=".state/reference_resolution_holdout_v2_checkpoint.pkl")
    parser.add_argument("--output", default=".state/reference_resolution_holdout_v1_audit.json")
    parser.add_argument("--dispatch", default=".state/dispatch_request.json")
    args = parser.parse_args()
    output = ROOT / args.output
    try:
        audit = build_audit(
            ROOT / args.plan,
            ROOT / args.holdout_evidence,
            ROOT / args.holdout_checkpoint,
            ROOT / args.dispatch,
        )
    except Exception as exc:
        print(json.dumps({
            "passed": False,
            "classification": "execution_integrity_failure",
            "error": f"{type(exc).__name__}: {exc}",
        }))
        return 2
    try:
        write_retry_safe_audit(output, audit)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({"passed": audit["passed"], "classification": audit["classification"]}))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
