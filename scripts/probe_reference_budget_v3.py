#!/usr/bin/env python3
"""Run a pre-registered high-budget engineering probe for the worst v2 case.

This probe informs the next numerical strategy only. It is not a gate, pool
source, holdout, or training input.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_supervisor import atomic_json, file_digest, load_json  # noqa: E402
from scripts import run_reference_resolution_escalation as v1  # noqa: E402


VERSION = "paper2-reference-budget-v3-probe-v1"
PLAN_PATH = ROOT / ".state/reference_resolution_budget_v2_plan.json"
AUDIT_PATH = ROOT / ".state/reference_resolution_budget_v2_audit.json"
SOURCE_CHECKPOINT = ROOT / ".state/reference_resolution_budget_v2_checkpoint.pkl"
GEOMETRY_INDEX = 2
CONFIGS = ((550, 1024), (650, 1024))
STEPS = (1.0, 0.5)
POLS = ("p", "s")
EXPECTED_TASKS = len(CONFIGS) * len(STEPS) * len(POLS)
MEAN_LIMIT = v1.MEAN_DE_LIMIT
PER_GEOMETRY_LIMIT = v1.PER_GEOMETRY_DE_LIMIT


def task_id(pol: str, config: tuple[int, int], step: float) -> str:
    token = "0p5" if step == 0.5 else "1"
    return f"probe-v3-g{GEOMETRY_INDEX:02d}-{pol}-ng{config[0]}-nxy{config[1]}-step{token}"


def build_tasks() -> list[dict]:
    plan = load_json(PLAN_PATH, {}) or {}
    selected = plan.get("selection", [])
    if len(selected) != 8:
        raise ValueError("v3 probe requires the frozen eight-case selection")
    geometry = selected[GEOMETRY_INDEX]
    tasks = []
    for pol in POLS:
        for config in CONFIGS:
            for step in STEPS:
                wavelength = v1.WL_HALF_NM if step == 0.5 else v1.WL_1NM
                tasks.append(
                    {
                        "id": task_id(pol, config, step),
                        "geometry_index": GEOMETRY_INDEX,
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


def run_task(task: dict) -> dict:
    return v1.run_task(task)


def validate_result(result: dict, expected: dict) -> None:
    if not isinstance(result, dict) or result.get("status") != "ok":
        raise ValueError(f"probe task failed: {expected['id']}")
    for key in ("id", "geometry_index", "pol", "requested_nG", "retained_nG", "Nxy", "step_nm"):
        if result.get(key) != expected[key]:
            raise ValueError(f"probe task field mismatch: {expected['id']}:{key}")
    wavelength = np.asarray(expected["wavelength_nm"], dtype=float)
    for key in ("wavelength_nm", "R", "T"):
        values = np.asarray(result.get(key), dtype=float)
        if key == "wavelength_nm" and not np.array_equal(values, wavelength):
            raise ValueError(f"probe wavelength mismatch: {expected['id']}")
        if key != "wavelength_nm" and (values.shape != wavelength.shape or not np.isfinite(values).all()):
            raise ValueError(f"probe spectrum mismatch: {expected['id']}")
    if np.max(np.abs(np.asarray(result["R"]) + np.asarray(result["T"]) - 1.0)) > v1.CONSERVATION_LIMIT:
        raise ValueError(f"probe conservation failure: {expected['id']}")


def binding(path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": file_digest(path)}


def comparison(results: dict, left: tuple[int, float], right: tuple[int, float]) -> dict:
    values = []
    rows = []
    for pol in POLS:
        def lookup(config: int, step: float) -> dict:
            return results[task_id(pol, (config, 1024), step)]
        a = lookup(*left)
        b = lookup(*right)
        value = float(v1.delta_e2000(
            v1.labels_on_grid(a["wavelength_nm"], a["R"]),
            v1.labels_on_grid(b["wavelength_nm"], b["R"]),
        ))
        values.append(value)
        rows.append({"pol": pol, "dE00": value})
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
        "mean_lt_1_15": bool(np.mean(array) < MEAN_LIMIT),
        "all_lt_2_3": bool(np.all(array < PER_GEOMETRY_LIMIT)),
        "passed": bool(np.mean(array) < MEAN_LIMIT and np.all(array < PER_GEOMETRY_LIMIT)),
        "rows": rows,
    }


def summarize(results: dict, tasks: list[dict], checkpoint: Path, request: dict | None) -> dict:
    for task in tasks:
        validate_result(results[task["id"]], task)
    comparisons = {
        "order_550_to_650_1nm": comparison(results, (550, 1.0), (650, 1.0)),
        "order_550_to_650_0p5nm": comparison(results, (550, 0.5), (650, 0.5)),
        "spectral_550": comparison(results, (550, 1.0), (550, 0.5)),
        "spectral_650": comparison(results, (650, 1.0), (650, 0.5)),
    }
    passed = all(item["passed"] for item in comparisons.values())
    return {
        "schema_version": 1,
        "evidence_version": VERSION,
        "request": request,
        "source_v2_plan": binding(PLAN_PATH),
        "source_v2_audit": binding(AUDIT_PATH),
        "source_v2_checkpoint": binding(SOURCE_CHECKPOINT),
        "geometry_index": GEOMETRY_INDEX,
        "configs": [list(config) for config in CONFIGS],
        "steps_nm": list(STEPS),
        "expected_tasks": EXPECTED_TASKS,
        "checkpoint": binding(checkpoint) | {"tasks": len(results)},
        "runtime_hashes": {
            path: file_digest(ROOT / path)
            for path in ("rcwa_batch.py", "paper2_colorimetry.py", "color_utils.py")
        },
        "thresholds": {
            "mean_joint_dE00_lt": MEAN_LIMIT,
            "all_joint_dE00_lt": PER_GEOMETRY_LIMIT,
            "pointwise_conservation_lte": v1.CONSERVATION_LIMIT,
        },
        "comparisons": comparisons,
        "passed": passed,
        "classification": "candidate_budget_supported" if passed else "candidate_budget_insufficient",
        "engineering_only": True,
        "training_allowed": False,
        "decision_scope": "Select a pre-registered v3 budget; never register a gate or authorize training.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--checkpoint", default=".state/reference_resolution_budget_v3_probe_checkpoint.pkl")
    parser.add_argument("--output", default=".state/reference_resolution_budget_v3_probe.json")
    parser.add_argument("--request", default=None)
    args = parser.parse_args()
    tasks = build_tasks()
    checkpoint_path = ROOT / args.checkpoint
    if checkpoint_path.is_file():
        with checkpoint_path.open("rb") as handle:
            state = pickle.load(handle)
        if state.get("task_ids") != [task["id"] for task in tasks]:
            raise ValueError("v3 probe checkpoint task set differs")
    else:
        state = {"version": VERSION, "task_ids": [task["id"] for task in tasks], "results": {}}
    results = state.setdefault("results", {})
    pending = [task for task in tasks if task["id"] not in results]
    with Pool(max(1, int(args.n_jobs))) as workers:
        for result in workers.imap_unordered(run_task, pending, chunksize=1):
            results[result["id"]] = result
            temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
            with temporary.open("wb") as handle:
                pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
            temporary.replace(checkpoint_path)
    request = json.loads(args.request) if args.request else None
    audit = summarize(results, tasks, checkpoint_path, request)
    atomic_json(ROOT / args.output, audit)
    print(json.dumps({"status": "passed" if audit["passed"] else "candidate_budget_insufficient", "records": len(results), "output": args.output}, sort_keys=True))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
