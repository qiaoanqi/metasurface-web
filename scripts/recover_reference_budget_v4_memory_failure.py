#!/usr/bin/env python3
"""Recover the two v4 spectra that exhausted memory in monolithic workers."""
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
from scripts import probe_reference_budget_v4 as v4  # noqa: E402
from scripts import run_reference_resolution_escalation as v1  # noqa: E402


VERSION = "paper2-reference-budget-v4-memory-recovery-v1"
PROTOCOL_PATH = ROOT / "protocols/paper2_reference_budget_v4_memory_recovery_v1.json"


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
                return
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


def atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def load_protocol() -> dict:
    protocol = load_json(PROTOCOL_PATH, {}) or {}
    if protocol.get("evidence_version") != VERSION:
        raise ValueError("v4 memory-recovery protocol version differs")
    for key in ("source_failed_checkpoint", "source_protocol", "source_runner", "source_auditor"):
        item = protocol.get(key, {})
        path = ROOT / str(item.get("path", ""))
        if not path.is_file() or file_digest(path) != item.get("sha256"):
            raise ValueError(f"v4 memory-recovery frozen source differs: {key}")
    implementations = protocol.get("implementation_hashes", [])
    if len(implementations) != 1 or implementations[0].get("path") != "scripts/recover_reference_budget_v4_memory_failure.py":
        raise ValueError("v4 memory-recovery implementation manifest differs")
    if file_digest(ROOT / implementations[0]["path"]) != implementations[0].get("sha256"):
        raise ValueError("v4 memory-recovery implementation hash differs")
    return protocol


def validate_source(protocol: dict) -> tuple[dict, bytes]:
    checkpoint_path = ROOT / protocol["source_failed_checkpoint"]["path"]
    payload = checkpoint_path.read_bytes()
    if file_digest(checkpoint_path) != protocol["source_failed_checkpoint"]["sha256"]:
        raise ValueError("v4 failed checkpoint hash differs")
    state = pickle.loads(payload)
    v4_protocol = v4.load_protocol()
    tasks = {task["id"]: task for task in v4.build_tasks(v4_protocol)}
    results = state.get("results", {})
    if set(results) != set(tasks):
        raise ValueError("v4 failed checkpoint task set differs")
    failed_ids = set(protocol["failed_task_ids"])
    observed_failed = {key for key, value in results.items() if value.get("status") != "ok"}
    if observed_failed != failed_ids:
        raise ValueError("v4 failed checkpoint failure set differs")
    required = protocol["required_failure"]
    for key in failed_ids:
        item = results[key]
        if item.get("status") != required["status"] or not str(item.get("error", "")).startswith(required["error_prefix"]):
            raise ValueError(f"v4 failed checkpoint error differs: {key}")
        for field in ("requested_nG", "retained_nG", "Nxy", "step_nm"):
            if item.get(field) != required[field]:
                raise ValueError(f"v4 failed checkpoint field differs: {key}:{field}")
    for key in set(tasks) - failed_ids:
        v4.validate_result(results[key], tasks[key])
    return state, payload


def block_id(pol: str, index: int) -> str:
    return f"recovery-v4-g02-{pol}-ng850-step0p5-block{index:02d}"


def build_block_tasks(protocol: dict) -> tuple[list[dict], dict[str, dict]]:
    source_tasks = {task["id"]: task for task in v4.build_tasks(v4.load_protocol())}
    full_tasks = {key: source_tasks[key] for key in protocol["failed_task_ids"]}
    size = int(protocol["chunking"]["chunk_size_wavelengths"])
    tasks = []
    for full in full_tasks.values():
        wavelength = np.asarray(full["wavelength_nm"], dtype=float)
        for index, start in enumerate(range(0, len(wavelength), size)):
            stop = min(start + size, len(wavelength))
            task = dict(full)
            task["id"] = block_id(full["pol"], index)
            task["source_task_id"] = full["id"]
            task["block_index"] = index
            task["start_index"] = start
            task["stop_index"] = stop
            task["wavelength_nm"] = wavelength[start:stop]
            tasks.append(task)
    expected_chunks = int(protocol["chunking"]["expected_total_chunks"])
    if len(tasks) != expected_chunks or len({task["id"] for task in tasks}) != expected_chunks:
        raise ValueError("v4 recovery block task set differs")
    return tasks, full_tasks


def run_block(task: dict) -> dict:
    return v1.run_task(task)


def validate_block(result: dict, expected: dict) -> None:
    if not isinstance(result, dict) or result.get("status") != "ok" or result.get("id") != expected["id"]:
        raise ValueError(f"v4 recovery block failed: {expected['id']}")
    for field in (
        "source_task_id", "block_index", "start_index", "stop_index", "geometry_index",
        "pol", "requested_nG", "retained_nG", "Nxy", "step_nm",
    ):
        if result.get(field) != expected[field]:
            raise ValueError(f"v4 recovery block field differs: {expected['id']}:{field}")
    wavelength = np.asarray(expected["wavelength_nm"], dtype=float)
    if not np.array_equal(np.asarray(result.get("wavelength_nm"), dtype=float), wavelength):
        raise ValueError(f"v4 recovery block wavelength differs: {expected['id']}")
    reflection = np.asarray(result.get("R"), dtype=float)
    transmission = np.asarray(result.get("T"), dtype=float)
    if reflection.shape != wavelength.shape or transmission.shape != wavelength.shape:
        raise ValueError(f"v4 recovery block shape differs: {expected['id']}")
    if not np.isfinite(reflection).all() or not np.isfinite(transmission).all():
        raise ValueError(f"v4 recovery block is non-finite: {expected['id']}")
    if np.max(np.abs(reflection + transmission - 1.0)) > v1.CONSERVATION_LIMIT:
        raise ValueError(f"v4 recovery block conservation differs: {expected['id']}")


def stitch(pol: str, blocks: dict[str, dict], tasks: list[dict], full: dict) -> dict:
    expected = sorted((task for task in tasks if task["pol"] == pol), key=lambda item: item["block_index"])
    pieces = []
    for task in expected:
        result = blocks[task["id"]]
        validate_block(result, task)
        pieces.append(result)
    recovered = dict(full)
    recovered.update(
        {
            "status": "ok",
            "wavelength_nm": np.concatenate([np.asarray(item["wavelength_nm"], dtype=float) for item in pieces]),
            "R": np.concatenate([np.asarray(item["R"], dtype=float) for item in pieces]),
            "T": np.concatenate([np.asarray(item["T"], dtype=float) for item in pieces]),
            "time_s": float(sum(float(item.get("time_s", 0.0)) for item in pieces)),
            "recovery": {
                "evidence_version": VERSION,
                "blocks": len(pieces),
                "maxtasksperchild": 1,
            },
        }
    )
    v4.validate_result(recovered, full)
    return recovered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()
    protocol = load_protocol()
    source_state, source_bytes = validate_source(protocol)
    raw_seal_path = ROOT / protocol["raw_failure_seal"]
    if raw_seal_path.is_file():
        if file_digest(raw_seal_path) != protocol["source_failed_checkpoint"]["sha256"]:
            raise ValueError("v4 raw failure seal differs")
    else:
        atomic_bytes(raw_seal_path, source_bytes)
    tasks, full_tasks = build_block_tasks(protocol)
    recovery_path = ROOT / protocol["recovery_checkpoint"]
    protocol_sha256 = file_digest(PROTOCOL_PATH)
    block_ids = [task["id"] for task in tasks]
    if recovery_path.is_file():
        with recovery_path.open("rb") as handle:
            recovery_state = pickle.load(handle)
        if recovery_state.get("version") != VERSION or recovery_state.get("protocol_sha256") != protocol_sha256:
            raise ValueError("v4 recovery checkpoint provenance differs")
        if recovery_state.get("block_ids") != block_ids:
            raise ValueError("v4 recovery checkpoint task set differs")
    else:
        recovery_state = {
            "version": VERSION,
            "protocol_sha256": protocol_sha256,
            "block_ids": block_ids,
            "results": {},
        }
    results = recovery_state.setdefault("results", {})
    expected_by_id = {task["id"]: task for task in tasks}
    for key in list(results):
        if key not in expected_by_id:
            raise ValueError("v4 recovery checkpoint contains unknown block")
        if results[key].get("status") == "ok":
            validate_block(results[key], expected_by_id[key])
    pending = [task for task in tasks if results.get(task["id"], {}).get("status") != "ok"]
    n_jobs = int(args.n_jobs or protocol["chunking"]["default_n_jobs"])
    with Pool(max(1, n_jobs), maxtasksperchild=1) as workers:
        for result in workers.imap_unordered(run_block, pending, chunksize=1):
            results[result["id"]] = result
            atomic_pickle(recovery_path, recovery_state)
    if set(results) != set(expected_by_id):
        raise ValueError("v4 recovery block set is incomplete")
    for key, task in expected_by_id.items():
        validate_block(results[key], task)
    recovered_state = pickle.loads(source_bytes)
    for pol in ("p", "s"):
        full_id = v4.task_id(pol, 850, 0.5)
        recovered_state["results"][full_id] = stitch(pol, results, tasks, full_tasks[full_id])
    checkpoint_path = ROOT / protocol["source_failed_checkpoint"]["path"]
    if file_digest(checkpoint_path) != protocol["source_failed_checkpoint"]["sha256"]:
        raise ValueError("v4 live failed checkpoint changed before recovery commit")
    atomic_pickle(checkpoint_path, recovered_state)
    evidence = {
        "schema_version": 1,
        "evidence_version": VERSION,
        "status": "completed",
        "protocol": {"path": str(PROTOCOL_PATH.relative_to(ROOT)).replace("\\", "/"), "sha256": protocol_sha256},
        "source_failed_checkpoint": protocol["source_failed_checkpoint"],
        "raw_failure_seal": {"path": protocol["raw_failure_seal"], "sha256": file_digest(raw_seal_path)},
        "recovery_checkpoint": {"path": protocol["recovery_checkpoint"], "sha256": file_digest(recovery_path), "blocks": len(results)},
        "recovered_checkpoint": {"path": protocol["source_failed_checkpoint"]["path"], "sha256": file_digest(checkpoint_path), "tasks": len(recovered_state["results"])},
        "replaced_task_ids": protocol["failed_task_ids"],
        "training_allowed": False,
        "gate_registration_allowed": False,
    }
    atomic_json(ROOT / protocol["recovery_evidence"], evidence)
    print(json.dumps({"status": "completed", "blocks": len(results), "checkpoint": evidence["recovered_checkpoint"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
