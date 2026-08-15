#!/usr/bin/env python3
"""Evaluate every measured 8-case candidate directly against the fine reference."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_supervisor import atomic_json, file_digest  # noqa: E402
from scripts import run_reference_resolution_escalation as base  # noqa: E402
from scripts.reference_protocol_selection import evaluate_protocols  # noqa: E402


VERSION = "paper2-reference-protocol-candidate-v1"


def build(checkpoint_path: Path, evidence_path: Path) -> dict:
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("evidence_version") != base.VERSION or evidence.get("passed") is not True:
        raise ValueError("candidate reference evidence is not passed")
    with checkpoint_path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    if len(checkpoint.get("results", {})) != 80:
        raise ValueError("candidate reference checkpoint is not complete 80/80")
    if file_digest(checkpoint_path) != evidence.get("checkpoint", {}).get("sha256"):
        raise ValueError("candidate checkpoint hash mismatch")
    selected = evidence.get("selection", [])
    evaluation = evaluate_protocols(
        selected,
        checkpoint["results"],
        lambda index, pol, config, step: base.task_id(index, pol, config, step),
    )
    return {
        "schema_version": 1,
        "evidence_version": VERSION,
        "passed": evaluation["any_protocol_passed"],
        "approved": False,
        "scope": "eight-case cost screening only; 32-case holdout remains mandatory",
        "pool_sha256": evidence["pool_sha256"],
        "evaluation": evaluation,
        "source_evidence": {"path": str(evidence_path.relative_to(ROOT)).replace("\\", "/"), "sha256": file_digest(evidence_path)},
        "source_checkpoint": {"path": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"), "sha256": file_digest(checkpoint_path)},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=".state/reference_resolution_v1_checkpoint.pkl")
    parser.add_argument("--evidence", default=".state/reference_resolution_v1.json")
    parser.add_argument("--output", default=".state/reference_protocol_candidate_v1.json")
    args = parser.parse_args()
    output = ROOT / args.output
    result = build(ROOT / args.checkpoint, ROOT / args.evidence)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != result:
            raise SystemExit("existing candidate protocol evaluation differs")
    else:
        atomic_json(output, result)
    print(json.dumps({"passed": result["passed"], "approved": False}))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
