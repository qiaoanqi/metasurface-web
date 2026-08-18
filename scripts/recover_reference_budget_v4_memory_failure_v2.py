#!/usr/bin/env python3
"""Finalize v4 memory-recovery blocks with an immutable metadata repair."""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import recover_reference_budget_v4_memory_failure as base  # noqa: E402
from scripts import run_reference_resolution_escalation as rcwa_runner  # noqa: E402


VERSION = "paper2-reference-budget-v4-memory-recovery-v2"
PROTOCOL_PATH = ROOT / "protocols/paper2_reference_budget_v4_memory_recovery_v2.json"
V1_RECOVERY_PATH = ROOT / "protocols/paper2_reference_budget_v4_memory_recovery_v1.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def load_protocol() -> dict:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("evidence_version") != VERSION:
        raise ValueError("v4 memory-recovery v2 protocol version differs")
    for key in ("source_failed_checkpoint", "source_protocol", "source_runner", "source_auditor", "source_recovery_checkpoint_v1"):
        item = protocol[key]
        path = ROOT / item["path"]
        if not path.is_file() or digest(path) != item["sha256"]:
            raise ValueError(f"v4 memory-recovery v2 frozen source differs: {key}")
    for item in protocol.get("implementation_hashes", []):
        path = ROOT / item["path"]
        if not path.is_file() or digest(path) != item["sha256"]:
            raise ValueError(f"v4 memory-recovery v2 implementation differs: {item['path']}")
    return protocol


def attach_block_metadata(result: dict, expected: dict) -> dict:
    repaired = dict(result)
    for field in (
        "source_task_id", "block_index", "start_index", "stop_index", "geometry_index",
        "pol", "requested_nG", "retained_nG", "Nxy", "step_nm",
    ):
        repaired[field] = expected[field]
    return repaired


def seed_v2_checkpoint(protocol: dict) -> None:
    target = ROOT / protocol["recovery_checkpoint"]
    if target.is_file():
        with target.open("rb") as handle:
            existing = pickle.load(handle)
        if existing.get("version") == VERSION and existing.get("protocol_sha256") == digest(PROTOCOL_PATH):
            return
    source = ROOT / protocol["source_recovery_checkpoint_v1"]["path"]
    with source.open("rb") as handle:
        old = pickle.load(handle)
    tasks, _ = base.build_block_tasks(protocol)
    expected = {task["id"]: task for task in tasks}
    old_results = old.get("results", {})
    if set(old_results) != set(expected):
        raise ValueError("v4 v1 recovery block set differs")
    repaired = {
        "version": VERSION,
        "protocol_sha256": digest(PROTOCOL_PATH),
        "block_ids": [task["id"] for task in tasks],
        "results": {key: attach_block_metadata(old_results[key], expected[key]) for key in expected},
    }
    target.write_bytes(pickle.dumps(repaired, protocol=pickle.HIGHEST_PROTOCOL))


def run_block(task: dict) -> dict:
    result = rcwa_runner.run_task(task)
    return attach_block_metadata(result, task)


def main() -> int:
    protocol = load_protocol()
    seed_v2_checkpoint(protocol)
    base.VERSION = VERSION
    base.PROTOCOL_PATH = PROTOCOL_PATH
    base.load_protocol = load_protocol
    base.run_block = run_block
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
