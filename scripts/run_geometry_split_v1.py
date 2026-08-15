#!/usr/bin/env python3
"""Freeze deterministic geometry-level train/validation/test assignments."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pipeline_supervisor as supervisor  # noqa: E402
from scripts import run_joint_convergence_v2 as joint  # noqa: E402
from scripts import run_replacement_pool as replacement  # noqa: E402


VERSION = "paper2-geometry-split-worker-v1"
SPLIT_VERSION = "sha256-ranked-80-10-10-v1"
SPLIT_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}


def binding(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/"),
        "sha256": supervisor.file_digest(path),
    }


def load_geometry_ids(pool_path: Path) -> list[str]:
    with pool_path.open("rb") as handle:
        payload = pickle.load(handle)
    records = payload.get("records", []) if isinstance(payload, dict) else []
    groups: dict[str, dict[str, tuple[float, float, float, float]]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("replacement pool contains a non-object record")
        geometry = tuple(float(record[name]) for name in ("L", "W", "H", "P"))
        if geometry[0] < geometry[1]:
            raise ValueError("replacement pool geometry axes are not canonical")
        identifier = str(record.get("geometry_id", ""))
        if identifier != replacement.geometry_id(geometry):
            raise ValueError("replacement pool geometry_id is not canonical")
        pol = str(record.get("pol", ""))
        if pol not in {"p", "s"}:
            raise ValueError("replacement pool polarization is invalid")
        group = groups.setdefault(identifier, {})
        if pol in group:
            raise ValueError("replacement pool repeats a geometry/polarization record")
        group[pol] = geometry
    if not groups or any(set(group) != {"p", "s"} for group in groups.values()):
        raise ValueError("replacement pool does not contain exact p/s geometry pairs")
    if any(group["p"] != group["s"] for group in groups.values()):
        raise ValueError("replacement pool p/s geometry values differ")
    if len(records) != 2 * len(groups):
        raise ValueError("replacement pool record count differs from paired geometries")
    return sorted(groups)


def build_assignments(geometry_ids: list[str], pool_sha256: str) -> list[dict[str, str]]:
    ranked = sorted(
        geometry_ids,
        key=lambda identifier: hashlib.sha256(
            f"{SPLIT_VERSION}|{pool_sha256}|{identifier}".encode("ascii")
        ).hexdigest(),
    )
    validation_count = len(ranked) // 10
    test_count = len(ranked) // 10
    train_count = len(ranked) - validation_count - test_count
    assignments = []
    for index, identifier in enumerate(ranked):
        if index < train_count:
            split = "train"
        elif index < train_count + validation_count:
            split = "validation"
        else:
            split = "test"
        assignments.append({"geometry_id": identifier, "split": split})
    return assignments


def summarize(active_path: Path, request: dict[str, Any]) -> dict[str, Any]:
    context = joint.load_active_context(active_path)
    geometry_ids = load_geometry_ids(context["pool_path"])
    expected = int(context["active"]["pool_spec"].get("n_samples", 0))
    if expected and len(geometry_ids) != expected:
        raise ValueError("geometry count differs from the active pool specification")
    assignments = build_assignments(geometry_ids, context["pool_sha256"])
    counts = {
        name: sum(item["split"] == name for item in assignments)
        for name in ("train", "validation", "test")
    }
    checks = {
        "active_pool_hash_verified": True,
        "canonical_axes_verified": True,
        "exact_dual_polarization_pairs": True,
        "stable_geometry_ids_verified": True,
        "geometry_level_no_leakage": len({item["geometry_id"] for item in assignments})
        == len(assignments),
        "split_counts_exact": counts["validation"] == len(assignments) // 10
        and counts["test"] == len(assignments) // 10
        and sum(counts.values()) == len(assignments),
    }
    return {
        "schema_version": 1,
        "evidence_version": VERSION,
        "request": request,
        "generated_at": supervisor.now_iso(),
        "passed": all(checks.values()),
        "classification": "geometry_split_frozen"
        if all(checks.values())
        else "geometry_split_invalid",
        "pool_sha256": context["pool_sha256"],
        "active_pool": binding(active_path),
        "split_version": SPLIT_VERSION,
        "ratios": SPLIT_RATIOS,
        "geometry_count": len(geometry_ids),
        "record_count": 2 * len(geometry_ids),
        "counts": counts,
        "assignments_sha256": supervisor.json_payload_digest(assignments),
        "assignments": assignments,
        "checks": checks,
        "runtime_hashes": {
            name: supervisor.file_digest(ROOT / name)
            for name in (
                "pipeline_supervisor.py",
                "scripts/run_geometry_split_v1.py",
                "scripts/run_joint_convergence_v2.py",
                "scripts/run_replacement_pool.py",
            )
        },
        "training_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active", default=".state/active_pool.json")
    parser.add_argument("--output", default=".state/geometry_split_v1.json")
    args = parser.parse_args()
    request = supervisor.current_request_identity("geometry_split_freeze")
    output = ROOT / args.output
    payload = summarize(ROOT / args.active, request)
    if output.exists():
        existing = supervisor.load_json(output, {}) or {}
        comparable = dict(payload)
        comparable["generated_at"] = existing.get("generated_at")
        if existing != comparable:
            raise SystemExit("existing geometry split evidence differs")
    else:
        supervisor.atomic_json(output, payload)
    print(json.dumps({"passed": payload["passed"], "counts": payload["counts"]}, sort_keys=True))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
