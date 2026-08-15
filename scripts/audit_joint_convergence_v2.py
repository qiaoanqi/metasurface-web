#!/usr/bin/env python3
"""Independently reproduce the joint-v2 worker evidence."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pipeline_supervisor as supervisor  # noqa: E402
from scripts import run_joint_convergence_v2 as joint  # noqa: E402


VERSION = "paper2-joint-convergence-audit-v1"


def binding(path: Path) -> dict:
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": supervisor.file_digest(path),
    }


def build_audit(worker_path: Path, active_path: Path) -> dict:
    worker = supervisor.load_json(worker_path, {}) or {}
    if worker.get("evidence_version") != joint.VERSION:
        raise ValueError("joint-v2 worker evidence version is invalid")
    worker_request = supervisor.reusable_evidence_request(
        worker.get("request"), "joint_numerical_convergence"
    )
    request = supervisor.current_request_identity("joint_numerical_convergence")
    context = joint.load_active_context(active_path)
    reference = joint.load_reference(context)
    recomputed = joint.build_evidence(context, reference, worker_request)
    if worker != recomputed:
        raise ValueError("joint-v2 worker evidence differs from independent recomputation")
    audit = copy.deepcopy(worker)
    audit["evidence_version"] = VERSION
    audit["request"] = request
    audit["worker_evidence"] = binding(worker_path)
    audit["independent_reproduction"] = True
    audit["independent_evaluation"] = copy.deepcopy(recomputed["evaluation"])
    audit["auditor_runtime_hashes"] = {
        path: supervisor.file_digest(ROOT / path)
        for path in supervisor.AUDITOR_RUNTIME_PATHS["joint_numerical_convergence"]
    }
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", default=".state/joint_convergence_v2.json")
    parser.add_argument("--active", default=".state/active_pool.json")
    parser.add_argument("--output", default=".state/joint_convergence_v2_audit.json")
    args = parser.parse_args()
    output = ROOT / args.output
    audit = build_audit(ROOT / args.worker, ROOT / args.active)
    if output.exists():
        if supervisor.load_json(output, {}) != audit:
            raise SystemExit("existing joint-v2 audit differs")
    else:
        supervisor.atomic_json(output, audit)
    print(json.dumps({"passed": audit["passed"], "classification": audit["classification"]}))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
