#!/usr/bin/env python3
"""Independently evaluate raw circular-control spectra and invariants."""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pipeline_supervisor as supervisor  # noqa: E402
from scripts import run_circular_control_v1 as circular  # noqa: E402
from scripts import run_joint_convergence_v2 as joint  # noqa: E402


VERSION = "paper2-circular-control-v1"


def binding(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/"),
        "sha256": supervisor.file_digest(path),
    }


def build_audit(worker_path: Path, checkpoint_path: Path, active_path: Path) -> dict:
    worker = supervisor.load_json(worker_path, {}) or {}
    if worker.get("evidence_version") != circular.VERSION:
        raise ValueError("circular control worker evidence version is invalid")
    supervisor.reusable_evidence_request(worker.get("request"), "circular_control")
    request = supervisor.current_request_identity("circular_control")
    context = joint.load_active_context(active_path)
    expected_meta = circular.build_meta(context, request)
    with checkpoint_path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    producer_request = checkpoint.get("meta", {}).get("request")
    if not isinstance(producer_request, dict):
        raise ValueError("circular control producer request is missing")
    expected_meta["request"] = producer_request
    circular.validate_checkpoint(checkpoint, expected_meta, complete=True)
    reproduced = circular.summarize(checkpoint, checkpoint_path)
    comparable_worker = copy.deepcopy(worker)
    comparable_reproduced = copy.deepcopy(reproduced)
    comparable_worker.pop("generated_at", None)
    comparable_reproduced.pop("generated_at", None)
    if comparable_worker != comparable_reproduced:
        raise ValueError("circular control worker summary differs from raw checkpoint")
    audit = copy.deepcopy(reproduced)
    audit.update(
        {
            "evidence_version": VERSION,
            "request": request,
            "producer_request": producer_request,
            "worker_evidence": binding(worker_path),
            "independent_reproduction": True,
            "auditor_runtime_hashes": {
                name: supervisor.file_digest(ROOT / name)
                for name in supervisor.AUDITOR_RUNTIME_PATHS["circular_control"]
            },
            "training_allowed": False,
        }
    )
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", default=".state/circular_control_v1.json")
    parser.add_argument("--checkpoint", default=".state/circular_control_v1_checkpoint.pkl")
    parser.add_argument("--active", default=".state/active_pool.json")
    parser.add_argument("--output", default=".state/circular_control_v1_audit.json")
    args = parser.parse_args()
    output = ROOT / args.output
    audit = build_audit(ROOT / args.worker, ROOT / args.checkpoint, ROOT / args.active)
    if output.exists():
        existing = supervisor.load_json(output, {}) or {}
        comparable = dict(audit)
        comparable["generated_at"] = existing.get("generated_at")
        if existing != comparable:
            raise SystemExit("existing circular control audit differs")
    else:
        supervisor.atomic_json(output, audit)
    print(json.dumps({"passed": audit["passed"], "classification": audit["classification"]}))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
