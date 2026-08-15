#!/usr/bin/env python3
"""Independently reproduce the frozen geometry-level split."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pipeline_supervisor as supervisor  # noqa: E402
from scripts import run_geometry_split_v1 as split  # noqa: E402


VERSION = "paper2-geometry-split-v1"


def binding(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/"),
        "sha256": supervisor.file_digest(path),
    }


def build_audit(worker_path: Path, active_path: Path) -> dict:
    worker = supervisor.load_json(worker_path, {}) or {}
    if worker.get("evidence_version") != split.VERSION:
        raise ValueError("geometry split worker evidence version is invalid")
    supervisor.reusable_evidence_request(worker.get("request"), "geometry_split_freeze")
    request = supervisor.current_request_identity("geometry_split_freeze")
    reproduced = split.summarize(active_path, request)
    comparable_worker = copy.deepcopy(worker)
    comparable_reproduced = copy.deepcopy(reproduced)
    comparable_worker.pop("generated_at", None)
    comparable_reproduced.pop("generated_at", None)
    comparable_worker["request"] = request
    if comparable_worker != comparable_reproduced:
        raise ValueError("geometry split worker evidence differs from independent reproduction")
    audit = copy.deepcopy(reproduced)
    audit.update(
        {
            "evidence_version": VERSION,
            "request": request,
            "worker_evidence": binding(worker_path),
            "independent_reproduction": True,
            "auditor_runtime_hashes": {
                name: supervisor.file_digest(ROOT / name)
                for name in supervisor.AUDITOR_RUNTIME_PATHS["geometry_split_freeze"]
            },
            "training_allowed": False,
        }
    )
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", default=".state/geometry_split_v1.json")
    parser.add_argument("--active", default=".state/active_pool.json")
    parser.add_argument("--output", default=".state/geometry_split_v1_audit.json")
    args = parser.parse_args()
    output = ROOT / args.output
    audit = build_audit(ROOT / args.worker, ROOT / args.active)
    if output.exists():
        existing = supervisor.load_json(output, {}) or {}
        comparable = dict(audit)
        comparable["generated_at"] = existing.get("generated_at")
        if existing != comparable:
            raise SystemExit("existing geometry split audit differs")
    else:
        supervisor.atomic_json(output, audit)
    print(json.dumps({"passed": audit["passed"], "classification": audit["classification"]}))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
