#!/usr/bin/env python3
"""Run the formal eight-geometry reference-resolution budget gate."""
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


VERSION = "paper2-reference-budget-v4"
PROTOCOL_PATH = ROOT / "protocols/paper2_reference_budget_v4.json"
PLAN_PATH = ROOT / ".state/reference_resolution_budget_v2_plan.json"
PROBE_PROTOCOL = ROOT / "protocols/paper2_reference_budget_v4_probe_v1.json"
PROBE_AUDIT = ROOT / ".state/reference_resolution_budget_v4_probe_audit.json"
PROBE_CHECKPOINT = ROOT / ".state/reference_resolution_budget_v4_probe_checkpoint.pkl"
POLS = ("p", "s")
CONFIGS = ((750, 1024, 0.5), (850, 1024, 1.0), (850, 1024, 0.5))
CHUNK_SIZE = 40


def binding(path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": file_digest(path)}


def task_id(index: int, pol: str, config: tuple[int, int, float]) -> str:
    nG, nxy, step = config
    token = "0p5" if step == 0.5 else "1"
    return f"refbudget-v4-g{index:02d}-{pol}-ng{nG}-nxy{nxy}-step{token}"


def block_id(full_id: str, index: int) -> str:
    return f"{full_id}-block{index:03d}"


def load_protocol() -> dict:
    protocol = load_json(PROTOCOL_PATH, {}) or {}
    if protocol.get("evidence_version") != VERSION:
        raise ValueError("formal v4 protocol version differs")
    for key, path in (("plan", PLAN_PATH), ("probe_protocol", PROBE_PROTOCOL), ("probe_audit", PROBE_AUDIT), ("probe_checkpoint", PROBE_CHECKPOINT)):
        item = protocol.get(key, {})
        if item.get("path") != str(path.relative_to(ROOT)).replace("\\", "/") or item.get("sha256") != file_digest(path):
            raise ValueError(f"formal v4 {key} binding differs")
    if protocol.get("polarizations") != list(POLS) or protocol.get("configs") != [list(c) for c in CONFIGS]:
        raise ValueError("formal v4 task configuration differs")
    if protocol.get("expected_tasks") != 48 or protocol.get("chunking", {}).get("chunk_size_wavelengths") != CHUNK_SIZE:
        raise ValueError("formal v4 task-count or chunking contract differs")
    if protocol.get("training_allowed") is not False or protocol.get("gate_registration_allowed") is not True:
        raise ValueError("formal v4 safety flags differ")
    for item in protocol.get("implementation_hashes", []):
        path = ROOT / item["path"]
        if not path.is_file() or file_digest(path) != item["sha256"]:
            raise ValueError(f"formal v4 implementation differs: {item['path']}")
    return protocol


def build_tasks(protocol: dict | None = None) -> list[dict]:
    protocol = protocol or load_protocol()
    plan = load_json(PLAN_PATH, {}) or {}
    selected = plan.get("selection", [])
    if len(selected) != 8:
        raise ValueError("formal v4 requires the frozen eight geometries")
    tasks = []
    for index, geometry in enumerate(selected):
        for pol in POLS:
            for nG, nxy, step in CONFIGS:
                wavelength = v1.WL_HALF_NM if step == 0.5 else v1.WL_1NM
                tasks.append({
                    "id": task_id(index, pol, (nG, nxy, step)),
                    "geometry_index": index,
                    "geometry": geometry,
                    "pol": pol,
                    "requested_nG": nG,
                    "retained_nG": v1.retained_order(nG, geometry["P"]),
                    "Nxy": nxy,
                    "step_nm": step,
                    "wavelength_nm": wavelength,
                })
    if len(tasks) != 48:
        raise ValueError("formal v4 generated task count differs")
    return tasks


def build_blocks(tasks: list[dict]) -> list[dict]:
    blocks = []
    for full in tasks:
        wavelengths = np.asarray(full["wavelength_nm"], dtype=float)
        for index, start in enumerate(range(0, len(wavelengths), CHUNK_SIZE)):
            stop = min(start + CHUNK_SIZE, len(wavelengths))
            block = dict(full)
            block.update({"id": block_id(full["id"], index), "full_task_id": full["id"], "block_index": index,
                          "start_index": start, "stop_index": stop, "wavelength_nm": wavelengths[start:stop]})
            blocks.append(block)
    return blocks


def run_block(task: dict) -> dict:
    return v1.run_task(task)


def atomic_pickle(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with tmp.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def validate_result(result: dict, expected: dict) -> None:
    if result.get("status") != "ok":
        raise ValueError(f"formal v4 task failed: {expected['id']}")
    for key in ("id", "geometry_index", "pol", "requested_nG", "retained_nG", "Nxy", "step_nm"):
        if result.get(key) != expected[key]:
            raise ValueError(f"formal v4 field mismatch: {expected['id']}:{key}")
    if tuple(float(result["geometry"][k]) for k in ("L", "W", "H", "P")) != tuple(float(expected["geometry"][k]) for k in ("L", "W", "H", "P")):
        raise ValueError(f"formal v4 geometry mismatch: {expected['id']}")
    wavelength = np.asarray(expected["wavelength_nm"], dtype=float)
    arrays = [np.asarray(result.get(key), dtype=float) for key in ("wavelength_nm", "R", "T")]
    if any(value.shape != wavelength.shape or not np.isfinite(value).all() for value in arrays):
        raise ValueError(f"formal v4 spectrum shape/finite failure: {expected['id']}")
    if not np.array_equal(arrays[0], wavelength):
        raise ValueError(f"formal v4 wavelength mismatch: {expected['id']}")
    if np.max(np.abs(arrays[1] + arrays[2] - 1.0)) > float(load_protocol()["thresholds"]["pointwise_conservation_lte"]):
        raise ValueError(f"formal v4 conservation failure: {expected['id']}")


def stitch(block_results: dict[str, dict], full: dict) -> dict:
    pieces = []
    expected = np.asarray(full["wavelength_nm"], dtype=float)
    for index, start in enumerate(range(0, len(expected), CHUNK_SIZE)):
        item = block_results[block_id(full["id"], index)]
        stop = min(start + CHUNK_SIZE, len(expected))
        if item.get("status") != "ok" or item.get("full_task_id") != full["id"] or item.get("start_index") != start or item.get("stop_index") != stop:
            raise ValueError(f"formal v4 block metadata failure: {full['id']}:{index}")
        pieces.append(item)
    result = dict(full)
    result.update({"status": "ok", "R": np.concatenate([np.asarray(x["R"], dtype=float) for x in pieces]),
                   "T": np.concatenate([np.asarray(x["T"], dtype=float) for x in pieces]),
                   "wavelength_nm": expected, "time_s": float(sum(float(x.get("time_s", 0.0)) for x in pieces))})
    validate_result(result, full)
    return result


def seed_probe(results: dict[str, dict], tasks: list[dict], protocol: dict) -> None:
    with PROBE_CHECKPOINT.open("rb") as handle:
        probe = pickle.load(handle)
    probe_results = probe.get("results", {})
    by_formal = {task["id"]: task for task in tasks if task["geometry_index"] == 2}
    for formal, task in by_formal.items():
        token = "0p5" if task["step_nm"] == 0.5 else "1"
        probe_id = f"probe-v4-g02-{task['pol']}-ng{task['requested_nG']}-nxy1024-step{token}"
        if probe_id in probe_results:
            candidate = dict(probe_results[probe_id])
            candidate["id"] = formal
            candidate["geometry_index"] = task["geometry_index"]
            candidate["geometry"] = task["geometry"]
            candidate["retained_nG"] = task["retained_nG"]
            validate_result(candidate, task)
            results[formal] = candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--checkpoint", default=".state/reference_resolution_budget_v4_checkpoint.pkl")
    parser.add_argument("--evidence", default=".state/reference_resolution_budget_v4.json")
    args = parser.parse_args()
    protocol = load_protocol()
    tasks = build_tasks(protocol)
    blocks = build_blocks(tasks)
    task_by_id = {task["id"]: task for task in tasks}
    block_by_id = {block["id"]: block for block in blocks}
    checkpoint = ROOT / args.checkpoint
    if checkpoint.is_file():
        with checkpoint.open("rb") as handle:
            state = pickle.load(handle)
        if state.get("version") != VERSION or state.get("protocol_sha256") != file_digest(PROTOCOL_PATH):
            raise ValueError("formal v4 checkpoint provenance differs")
    else:
        state = {"version": VERSION, "protocol_sha256": file_digest(PROTOCOL_PATH),
                 "runtime_hashes": protocol["runtime_hashes"], "results": {}, "blocks": {}}
    results = state.setdefault("results", {})
    block_results = state.setdefault("blocks", {})
    if not results and not block_results:
        seed_probe(results, tasks, protocol)
        atomic_pickle(checkpoint, state)
    pending = [block for block in blocks if block["id"] not in block_results and block["full_task_id"] not in results]
    with Pool(max(1, int(args.n_jobs)), maxtasksperchild=1) as workers:
        for result in workers.imap_unordered(run_block, pending, chunksize=1):
            identifier = result["id"]
            expected = block_by_id[identifier]
            if result.get("status") == "ok":
                result["full_task_id"] = expected["full_task_id"]
                result["block_index"] = expected["block_index"]
                result["start_index"] = expected["start_index"]
                result["stop_index"] = expected["stop_index"]
                block_results[identifier] = result
            else:
                block_results[identifier] = result
            atomic_pickle(checkpoint, state)
    for full_id, full in task_by_id.items():
        if full_id in results:
            continue
        needed = [block_id(full_id, i) for i, _ in enumerate(range(0, len(full["wavelength_nm"]), CHUNK_SIZE))]
        if not all(key in block_results and block_results[key].get("status") == "ok" for key in needed):
            continue
        results[full_id] = stitch(block_results, full)
        atomic_pickle(checkpoint, state)
    complete = len(results) == len(tasks) and all(task["id"] in results for task in tasks)
    evidence = {"schema_version": 1, "evidence_version": VERSION, "protocol": binding(PROTOCOL_PATH),
                "plan": binding(PLAN_PATH), "probe_audit": binding(PROBE_AUDIT),
                "checkpoint": binding(checkpoint) | {"tasks": len(results)}, "records": len(results),
                "expected_records": len(tasks), "complete": complete, "training_allowed": False,
                "gate_registration_allowed": True, "seeded_geometry_index": 2}
    atomic_json(ROOT / args.evidence, evidence)
    print(json.dumps({"status": "complete" if complete else "running", "records": len(results), "blocks": len(block_results)}, sort_keys=True))
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
