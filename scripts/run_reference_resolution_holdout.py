#!/usr/bin/env python3
"""Run the v2-bound 24-case confirmation holdout."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_supervisor import atomic_json, file_digest  # noqa: E402
from scripts import freeze_reference_holdout_plan as freezer  # noqa: E402
from scripts import run_reference_resolution_budget_v2 as v2_runner  # noqa: E402
from scripts.run_replacement_pool import RunLock  # noqa: E402


VERSION = "paper2-reference-holdout-v2"
PLAN_VERSION = freezer.VERSION
POLS = tuple(v2_runner.POLS)
BASE_CONFIG = tuple(v2_runner.BASE_CONFIG)
FINAL_REFERENCE = freezer.FINAL_REFERENCE
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


def request_identity(request_id: str, attempt: int) -> dict[str, Any]:
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("holdout request_id is required")
    attempt = int(attempt)
    if attempt < 1:
        raise ValueError("holdout attempt must be positive")
    return {"request_id": request_id, "attempt": attempt}


def checkpoint_request_identity(request: dict[str, Any]) -> dict[str, str]:
    """Return the stable request identity shared by all retry attempts."""
    return {"request_id": str(request["request_id"])}


def retry_equivalent_evidence(existing: Any, candidate: dict[str, Any]) -> bool:
    if not isinstance(existing, dict):
        return False
    previous = existing.get("request")
    current = candidate.get("request")
    if not isinstance(previous, dict) or not isinstance(current, dict):
        return False
    if (
        previous.get("request_id") != current.get("request_id")
        or int(previous.get("attempt", 0)) < 1
        or int(previous.get("attempt", 0)) > int(current.get("attempt", 0))
    ):
        return False
    expected = dict(candidate)
    expected["request"] = previous
    return existing == expected


def write_retry_safe_evidence(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        atomic_json(path, payload)
        return payload
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing == payload:
        return existing
    if retry_equivalent_evidence(existing, payload):
        return existing
    raise ValueError("existing v2 holdout evidence differs; use a new version")


def normalized(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def resolve_binding(item: Any, label: str) -> Path:
    if not isinstance(item, dict) or not isinstance(item.get("path"), str):
        raise ValueError(f"missing {label} binding")
    path = (ROOT / item["path"]).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} path escapes the workspace") from exc
    if not path.is_file() or file_digest(path) != str(item.get("sha256", "")).upper():
        raise ValueError(f"{label} SHA256 binding mismatch")
    if normalized(path) != str(item["path"]).replace("\\", "/"):
        raise ValueError(f"{label} path is not canonical")
    return path


def protocol_key(item: dict) -> tuple[int, int, float]:
    return int(item["requested_nG"]), int(item["Nxy"]), float(item["step_nm"])


def expected_protocol_manifest(candidate: dict) -> list[dict]:
    return freezer.protocol_manifest(candidate)


def load_plan(path: Path) -> dict:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != 1 or plan.get("evidence_version") != PLAN_VERSION:
        raise ValueError("unexpected v2 holdout plan version")
    if (
        plan.get("plan_valid") is not True
        or plan.get("created_before_holdout_results") is not True
        or plan.get("candidate_frozen_on_initial_eight_only") is not True
        or plan.get("holdout_cannot_reselect_candidate") is not True
    ):
        raise ValueError("holdout plan does not enforce pre-holdout candidate freezing")
    if (
        len(plan.get("existing_cases", [])) != 8
        or len(plan.get("new_cases", [])) != freezer.NEW_CASES
        or len(plan.get("combined_cases", [])) != freezer.TOTAL_CASES
        or plan.get("combined_cases") != plan.get("existing_cases") + plan.get("new_cases")
    ):
        raise ValueError("v2 holdout case counts or order are invalid")
    hashes = (
        (plan["existing_cases"], freezer.ARCHIVED_EXISTING_SHA256),
        (plan["new_cases"], freezer.ARCHIVED_NEW_SHA256),
        (plan["combined_cases"], freezer.ARCHIVED_COMBINED_SHA256),
    )
    if any(freezer.selection_sha256(items) != expected for items, expected in hashes):
        raise ValueError("v2 holdout selection SHA256 mismatch")
    candidate = plan.get("frozen_candidate")
    if not isinstance(candidate, dict) or candidate.get("passed") is not True:
        raise ValueError("v2 holdout plan lacks a passing frozen candidate")
    manifest = expected_protocol_manifest(candidate)
    if plan.get("task_protocols") != manifest:
        raise ValueError("v2 holdout task protocol manifest drift")
    if plan.get("minimum_holdout_protocols") != [list(x) for x in freezer.MINIMUM_HOLDOUT_PROTOCOLS]:
        raise ValueError("v2 holdout minimum reference matrix drift")
    tasks_per_geometry = len(manifest) * len(POLS)
    expected_counts = {
        "expected_source_tasks": 8 * tasks_per_geometry,
        "expected_new_tasks": freezer.NEW_CASES * tasks_per_geometry,
        "expected_combined_tasks": freezer.TOTAL_CASES * tasks_per_geometry,
    }
    if any(plan.get(name) != value for name, value in expected_counts.items()):
        raise ValueError("v2 holdout task counts drift")
    expected_thresholds = {
        "mean_joint_dE00_lt": v2_runner.v1.MEAN_DE_LIMIT,
        "all_joint_dE00_lt": v2_runner.v1.PER_GEOMETRY_DE_LIMIT,
        "pointwise_conservation_lte": v2_runner.v1.CONSERVATION_LIMIT,
    }
    if plan.get("thresholds") != expected_thresholds:
        raise ValueError("v2 holdout thresholds changed")
    for name in (
        "archived_v1_holdout_plan",
        "source_v2_plan",
        "source_v2_checkpoint",
        "source_v2_worker_evidence",
        "source_v2_independent_audit",
        "source_base_checkpoint",
    ):
        resolve_binding(plan.get(name), name)
    for runtime_path, expected in plan.get("runtime_hashes", {}).items():
        runtime = (ROOT / runtime_path).resolve()
        if file_digest(runtime) != str(expected).upper():
            raise ValueError(f"holdout runtime hash mismatch: {runtime_path}")
    if set(plan.get("runtime_hashes", {})) != set(freezer.RUNTIME_PATHS):
        raise ValueError("holdout runtime hash manifest is incomplete")
    return plan


def holdout_task_id(index: int, pol: str, config: tuple[int, int], step: float) -> str:
    token = "0p5" if step == 0.5 else "1"
    return f"refhold-v2-g{index:02d}-{pol}-ng{config[0]}-nxy{config[1]}-step{token}"


def result_id(index: int, pol: str, config: tuple[int, int], step: float) -> str:
    if index < 8:
        return freezer.source_result_id(index, pol, config, step)
    return holdout_task_id(index, pol, config, step)


def build_task(index: int, geometry: dict, pol: str, protocol: dict) -> dict:
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
        build_task(index, plan["combined_cases"][index], pol, protocol)
        for index in indices
        for pol in POLS
        for protocol in plan["task_protocols"]
    ]


def build_new_tasks(plan: dict) -> list[dict]:
    return build_tasks(plan, range(8, freezer.TOTAL_CASES))


def build_source_tasks(plan: dict) -> list[dict]:
    return build_tasks(plan, range(8))


def build_combined_tasks(plan: dict) -> list[dict]:
    return build_tasks(plan, range(freezer.TOTAL_CASES))


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
    if source["plan"].get("selection") != plan["existing_cases"]:
        raise ValueError("v2 source geometry selection differs from holdout plan")
    evaluation = freezer.evaluate_protocols(
        plan["existing_cases"], freezer.source_results(source), freezer.source_result_id
    )
    if evaluation != plan.get("source_protocol_evaluation"):
        raise ValueError("frozen source protocol evaluation does not reproduce")
    if evaluation.get("lowest_cost_passing_protocol") != plan.get("frozen_candidate"):
        raise ValueError("frozen candidate differs from initial-eight evaluation")
    all_source = freezer.source_results(source)
    expected = build_source_tasks(plan)
    selected = {task["id"]: all_source[task["id"]] for task in expected}
    validation = validate_results(selected, expected)
    if not validation["passed"]:
        raise ValueError("hash-bound source spectra are invalid")
    return selected


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
            v2_runner.validate_result(result, task)
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


def comparison(
    plan: dict,
    results: dict,
    indices: range,
    left: tuple[int, int, float],
    right: tuple[int, int, float],
) -> dict:
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


def comparison_specs(plan: dict) -> list[tuple[str, tuple[int, int, float], tuple[int, int, float]]]:
    candidate = plan["frozen_candidate"]
    candidate_key = (
        int(candidate["requested_nG"]),
        int(candidate["Nxy"]),
        float(candidate["wavelength_step_nm"]),
    )
    return [
        (COMPARISON_NAMES[0], (365, 768, 0.5), FINAL_REFERENCE),
        (COMPARISON_NAMES[1], (450, 512, 0.5), FINAL_REFERENCE),
        (COMPARISON_NAMES[2], (365, 512, 0.5), FINAL_REFERENCE),
        (COMPARISON_NAMES[3], (450, 768, 1.0), FINAL_REFERENCE),
        (COMPARISON_NAMES[4], candidate_key, FINAL_REFERENCE),
    ]


def evaluate_population(plan: dict, results: dict, indices: range) -> dict[str, dict]:
    return {
        name: comparison(plan, results, indices, left, right)
        for name, left, right in comparison_specs(plan)
    }


def classify(validation_passed: bool, holdout: dict[str, dict]) -> str:
    if not validation_passed:
        return "execution_integrity_failure"
    if not all(holdout[name]["passed"] for name in COMPARISON_NAMES[:4]):
        return "reference_budget_insufficient"
    if not holdout[COMPARISON_NAMES[4]]["passed"]:
        return "production_candidate_holdout_negative"
    return "reference_holdout_passed"


def summarize(
    plan: dict,
    source_results: dict,
    extension_results: dict,
    plan_path: Path,
    checkpoint_path: Path,
    request: dict[str, Any],
) -> dict:
    extension_validation = validate_results(extension_results, build_new_tasks(plan))
    combined = dict(source_results)
    combined.update(extension_results)
    combined_validation = validate_results(combined, build_combined_tasks(plan))
    validation_passed = extension_validation["passed"] and combined_validation["passed"]
    holdout = evaluate_population(plan, combined, range(8, freezer.TOTAL_CASES)) if validation_passed else {}
    supplemental = evaluate_population(plan, combined, range(freezer.TOTAL_CASES)) if validation_passed else {}
    classification = classify(validation_passed, holdout)
    passed = classification == "reference_holdout_passed"
    checks = {
        "v2_source_audit_passed_and_hash_bound": True,
        "candidate_frozen_on_initial_eight_only": True,
        "exact_new_task_set": extension_validation["records"] == plan["expected_new_tasks"],
        "exact_combined_task_set": combined_validation["records"] == plan["expected_combined_tasks"],
        "raw_spectra_and_conservation_valid": validation_passed,
        "holdout_order_converged": bool(holdout.get(COMPARISON_NAMES[0], {}).get("passed")),
        "holdout_grid_converged": bool(holdout.get(COMPARISON_NAMES[1], {}).get("passed")),
        "holdout_corner_converged": bool(holdout.get(COMPARISON_NAMES[2], {}).get("passed")),
        "holdout_spectral_converged": bool(holdout.get(COMPARISON_NAMES[3], {}).get("passed")),
        "frozen_candidate_passed_holdout": bool(holdout.get(COMPARISON_NAMES[4], {}).get("passed")),
    }
    return {
        "schema_version": 1,
        "evidence_version": VERSION,
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
            "new_case_count": freezer.NEW_CASES,
            "combined_case_count": freezer.TOTAL_CASES,
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
        "plan": {"path": normalized(plan_path), "sha256": file_digest(plan_path)},
        "source_v2_plan": plan["source_v2_plan"],
        "source_v2_checkpoint": plan["source_v2_checkpoint"],
        "source_v2_worker_evidence": plan["source_v2_worker_evidence"],
        "source_v2_independent_audit": plan["source_v2_independent_audit"],
        "source_base_checkpoint": plan["source_base_checkpoint"],
        "holdout_checkpoint": {
            "path": normalized(checkpoint_path),
            "sha256": file_digest(checkpoint_path),
            "tasks": len(extension_results),
        },
        "training_allowed": False,
    }


def run(args: argparse.Namespace) -> dict:
    plan_path = ROOT / args.plan
    checkpoint_path = ROOT / args.checkpoint
    evidence_path = ROOT / args.evidence
    plan = load_plan(plan_path)
    request = request_identity(args.request_id, args.attempt)
    sources = load_source_results(plan)
    tasks = build_new_tasks(plan)
    meta = {
        "version": VERSION,
        "request": checkpoint_request_identity(request),
        "plan": {"path": normalized(plan_path), "sha256": file_digest(plan_path)},
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
    with RunLock(ROOT / args.lock, file_digest(plan_path)):
        if checkpoint_path.exists():
            with checkpoint_path.open("rb") as handle:
                checkpoint = pickle.load(handle)
            if checkpoint.get("meta") != meta:
                raise ValueError("v2 holdout checkpoint protocol mismatch")
        else:
            checkpoint = {"meta": meta, "results": {}}
            v2_runner.v1.atomic_pickle(checkpoint_path, checkpoint)
        expected_ids = {task["id"] for task in tasks}
        unknown_ids = set(checkpoint.get("results", {})) - expected_ids
        if unknown_ids:
            raise ValueError("v2 holdout checkpoint contains unknown task identities")
        partial_tasks = [task for task in tasks if task["id"] in checkpoint.get("results", {})]
        partial_results = {task["id"]: checkpoint["results"][task["id"]] for task in partial_tasks}
        if not validate_results(partial_results, partial_tasks)["passed"]:
            raise ValueError("v2 holdout partial checkpoint is invalid")
        pending = [task for task in tasks if task["id"] not in checkpoint["results"]]
        with Pool(max(1, int(args.n_jobs))) as workers:
            for result in workers.imap_unordered(v2_runner.run_task, pending, chunksize=1):
                checkpoint["results"][result["id"]] = result
                v2_runner.v1.atomic_pickle(checkpoint_path, checkpoint)
        evidence = summarize(
            plan,
            sources,
            checkpoint["results"],
            plan_path,
            checkpoint_path,
            request,
        )
        evidence = write_retry_safe_evidence(evidence_path, evidence)
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default=".state/reference_resolution_holdout_v2_plan.json")
    parser.add_argument("--checkpoint", default=".state/reference_resolution_holdout_v2_checkpoint.pkl")
    parser.add_argument("--evidence", default=".state/reference_resolution_holdout_v2.json")
    parser.add_argument("--lock", default=".state/reference_resolution_holdout_v2.lock")
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"passed": result["passed"], "classification": result["classification"]}, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
