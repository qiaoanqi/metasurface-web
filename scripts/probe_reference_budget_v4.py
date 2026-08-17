#!/usr/bin/env python3
"""Run the pre-registered v4 order-budget engineering probe.

This probe reuses the frozen nG650/0.5 nm anchor and computes only the
pre-registered nG750 and nG850 tasks. It cannot register a gate or authorize
training.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_supervisor import atomic_json, file_digest, load_json  # noqa: E402
from scripts import run_reference_resolution_escalation as v1  # noqa: E402


VERSION = "paper2-reference-budget-v4-probe-v1"
PROTOCOL_PATH = ROOT / "protocols/paper2_reference_budget_v4_probe_v1.json"
PLAN_PATH = ROOT / ".state/reference_resolution_budget_v2_plan.json"
SOURCE_RESULT = ROOT / ".state/reference_resolution_budget_v3_probe.json"
SOURCE_CHECKPOINT = ROOT / ".state/reference_resolution_budget_v3_probe_checkpoint.pkl"
POLARIZATIONS = ("p", "s")


def binding(path: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": file_digest(path),
    }


def load_protocol() -> dict:
    protocol = load_json(PROTOCOL_PATH, {}) or {}
    if protocol.get("evidence_version") != VERSION:
        raise ValueError("v4 probe protocol version differs")
    for key in ("source_v3_probe", "source_v3_checkpoint", "source_v3_protocol", "source_v3_runner"):
        item = protocol.get(key, {})
        path = ROOT / str(item.get("path", ""))
        if not path.is_file() or file_digest(path) != item.get("sha256"):
            raise ValueError(f"v4 probe frozen source differs: {key}")
    if protocol.get("polarizations") != list(POLARIZATIONS):
        raise ValueError("v4 probe polarization contract differs")
    if protocol.get("expected_new_tasks") != 6:
        raise ValueError("v4 probe task-count contract differs")
    implementations = protocol.get("implementation_hashes", [])
    if {item.get("path") for item in implementations} != {
        "scripts/probe_reference_budget_v4.py",
        "scripts/audit_reference_budget_v4_probe.py",
    }:
        raise ValueError("v4 probe implementation contract differs")
    for item in implementations:
        path = ROOT / str(item.get("path", ""))
        if not path.is_file() or file_digest(path) != item.get("sha256"):
            raise ValueError(f"v4 probe implementation differs: {item.get('path')}")
    return protocol


def task_id(pol: str, nG: int, step: float) -> str:
    token = "0p5" if step == 0.5 else "1"
    return f"probe-v4-g02-{pol}-ng{nG}-nxy1024-step{token}"


def source_task_id(pol: str) -> str:
    return f"probe-v3-g02-{pol}-ng650-nxy1024-step0p5"


def build_tasks(protocol: dict | None = None) -> list[dict]:
    protocol = protocol or load_protocol()
    plan = load_json(PLAN_PATH, {}) or {}
    geometry_index = int(protocol["geometry_index"])
    selected = plan.get("selection", [])
    if len(selected) != 8 or geometry_index != 2:
        raise ValueError("v4 probe requires frozen geometry index 2 of eight")
    geometry = selected[geometry_index]
    tasks = []
    for pol in POLARIZATIONS:
        for config in protocol["new_configs"]:
            nG = int(config["nG_requested"])
            Nxy = int(config["Nxy"])
            retained = v1.retained_order(nG, geometry["P"])
            if retained != int(config["retained_nG"]):
                raise ValueError(f"v4 retained order differs for nG={nG}")
            for step in config["steps_nm"]:
                step = float(step)
                wavelength = v1.WL_HALF_NM if step == 0.5 else v1.WL_1NM
                tasks.append(
                    {
                        "id": task_id(pol, nG, step),
                        "geometry_index": geometry_index,
                        "geometry": geometry,
                        "pol": pol,
                        "requested_nG": nG,
                        "retained_nG": retained,
                        "Nxy": Nxy,
                        "step_nm": step,
                        "wavelength_nm": wavelength,
                    }
                )
    if len(tasks) != int(protocol["expected_new_tasks"]):
        raise ValueError("v4 probe generated task count differs")
    return tasks


def run_task(task: dict) -> dict:
    return v1.run_task(task)


def validate_result(result: dict, expected: dict) -> None:
    if not isinstance(result, dict) or result.get("status") != "ok":
        raise ValueError(f"v4 probe task failed: {expected['id']}")
    for key in ("id", "geometry_index", "pol", "requested_nG", "retained_nG", "Nxy", "step_nm"):
        if result.get(key) != expected[key]:
            raise ValueError(f"v4 probe task field mismatch: {expected['id']}:{key}")
    wavelength = np.asarray(expected["wavelength_nm"], dtype=float)
    if not np.array_equal(np.asarray(result.get("wavelength_nm"), dtype=float), wavelength):
        raise ValueError(f"v4 probe wavelength mismatch: {expected['id']}")
    arrays = [np.asarray(result.get(key), dtype=float) for key in ("R", "T")]
    if any(value.shape != wavelength.shape or not np.isfinite(value).all() for value in arrays):
        raise ValueError(f"v4 probe spectrum mismatch: {expected['id']}")
    if any(np.any(value < -1e-8) or np.any(value > 1.0 + 1e-8) for value in arrays):
        raise ValueError(f"v4 probe spectrum range failure: {expected['id']}")
    if np.max(np.abs(arrays[0] + arrays[1] - 1.0)) > v1.CONSERVATION_LIMIT:
        raise ValueError(f"v4 probe conservation failure: {expected['id']}")


def atomic_pickle(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(10):
            try:
                os.replace(temporary, path)
                break
            except OSError as exc:
                transient = getattr(exc, "winerror", None) in {5, 32} or exc.errno in {13, 16}
                if not transient or attempt == 9:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def compare(left: dict, right: dict) -> dict:
    rows = []
    for pol in POLARIZATIONS:
        a, b = left[pol], right[pol]
        value = float(
            v1.delta_e2000(
                v1.labels_on_grid(a["wavelength_nm"], a["R"]),
                v1.labels_on_grid(b["wavelength_nm"], b["R"]),
            )
        )
        rows.append({"pol": pol, "dE00": value})
    values = np.asarray([row["dE00"] for row in rows], dtype=float)
    mean_limit = v1.MEAN_DE_LIMIT
    all_limit = v1.PER_GEOMETRY_DE_LIMIT
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
        "mean_lt_1_15": bool(np.mean(values) < mean_limit),
        "all_lt_2_3": bool(np.all(values < all_limit)),
        "passed": bool(np.mean(values) < mean_limit and np.all(values < all_limit)),
        "rows": rows,
    }


def load_source_anchor(protocol: dict) -> dict[str, dict]:
    if binding(SOURCE_CHECKPOINT) != protocol["source_v3_checkpoint"]:
        raise ValueError("v4 source checkpoint binding differs")
    with SOURCE_CHECKPOINT.open("rb") as handle:
        source = pickle.load(handle)
    results = source.get("results", {})
    anchor = {}
    for pol in POLARIZATIONS:
        item = results.get(source_task_id(pol))
        if not isinstance(item, dict) or item.get("status") != "ok":
            raise ValueError(f"v4 source anchor missing: {pol}")
        anchor[pol] = item
    return anchor


def summarize(results: dict, tasks: list[dict], checkpoint: Path, protocol: dict) -> dict:
    expected = {task["id"]: task for task in tasks}
    if set(results) != set(expected):
        raise ValueError("v4 probe result task set differs")
    for key, task in expected.items():
        validate_result(results[key], task)
    anchor = load_source_anchor(protocol)

    def group(nG: int, step: float) -> dict[str, dict]:
        return {pol: results[task_id(pol, nG, step)] for pol in POLARIZATIONS}

    comparisons = {
        "order_650_to_750_0p5nm_diagnostic": compare(anchor, group(750, 0.5)),
        "order_750_to_850_0p5nm": compare(group(750, 0.5), group(850, 0.5)),
        "spectral_850": compare(group(850, 1.0), group(850, 0.5)),
    }
    passed = comparisons["order_750_to_850_0p5nm"]["passed"] and comparisons["spectral_850"]["passed"]
    return {
        "schema_version": 1,
        "evidence_version": VERSION,
        "protocol": binding(PROTOCOL_PATH),
        "source_v3_probe": binding(SOURCE_RESULT),
        "source_v3_checkpoint": binding(SOURCE_CHECKPOINT),
        "plan": binding(PLAN_PATH),
        "geometry_index": int(protocol["geometry_index"]),
        "expected_new_tasks": int(protocol["expected_new_tasks"]),
        "checkpoint": binding(checkpoint) | {"tasks": len(results)},
        "runtime_hashes": {
            path: file_digest(ROOT / path)
            for path in ("rcwa_batch.py", "paper2_colorimetry.py", "color_utils.py")
        },
        "thresholds": protocol["thresholds"],
        "candidate_reference": protocol["candidate_reference"],
        "comparisons": comparisons,
        "passed": bool(passed),
        "classification": "candidate_budget_supported" if passed else "candidate_budget_insufficient",
        "engineering_only": True,
        "training_allowed": False,
        "gate_registration_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--checkpoint", default=".state/reference_resolution_budget_v4_probe_checkpoint.pkl")
    parser.add_argument("--output", default=".state/reference_resolution_budget_v4_probe.json")
    args = parser.parse_args()
    protocol = load_protocol()
    tasks = build_tasks(protocol)
    checkpoint_path = ROOT / args.checkpoint
    task_ids = [task["id"] for task in tasks]
    if checkpoint_path.is_file():
        with checkpoint_path.open("rb") as handle:
            state = pickle.load(handle)
        if state.get("version") != VERSION or state.get("protocol_sha256") != file_digest(PROTOCOL_PATH):
            raise ValueError("v4 probe checkpoint provenance differs")
        if state.get("task_ids") != task_ids:
            raise ValueError("v4 probe checkpoint task set differs")
    else:
        state = {
            "version": VERSION,
            "protocol_sha256": file_digest(PROTOCOL_PATH),
            "task_ids": task_ids,
            "results": {},
        }
    results = state.setdefault("results", {})
    pending = [task for task in tasks if task["id"] not in results]
    with Pool(max(1, int(args.n_jobs))) as workers:
        for result in workers.imap_unordered(run_task, pending, chunksize=1):
            results[result["id"]] = result
            atomic_pickle(checkpoint_path, state)
    evidence = summarize(results, tasks, checkpoint_path, protocol)
    atomic_json(ROOT / args.output, evidence)
    print(json.dumps({
        "status": "passed" if evidence["passed"] else "candidate_budget_insufficient",
        "records": len(results),
        "output": args.output,
    }, sort_keys=True))
    return 0 if evidence["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
