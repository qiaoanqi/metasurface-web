#!/usr/bin/env python3
"""Independently audit a completed replacement-pool worker artifact."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pipeline_supervisor as supervisor  # noqa: E402
from scripts import run_replacement_pool as replacement  # noqa: E402


VERSION = "paper2-replacement-pool-audit-v1"
AUDIT_FIELDS = (
    "records", "expected_records", "geometries", "complete_pairs",
    "duplicate_keys", "R_plus_T_mean", "R_plus_T_min", "R_plus_T_max",
    "pointwise_conservation_error_max", "R_min", "R_max", "T_min", "T_max",
)


def binding(path: Path) -> dict:
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": supervisor.file_digest(path),
    }


def build_audit(worker_path: Path) -> dict:
    worker = supervisor.load_json(worker_path, {}) or {}
    if (
        worker.get("schema_version") != 1
        or worker.get("evidence_version") != replacement.EVIDENCE_VERSION
        or worker.get("passed") is not True
        or worker.get("training_allowed") is not False
    ):
        raise ValueError("replacement worker evidence contract is invalid")
    supervisor.reusable_evidence_request(
        worker.get("request"), "replacement_pool_generation"
    )
    request = supervisor.current_request_identity("replacement_pool_generation")
    worker_binding = binding(worker_path)
    if not supervisor.failed_ack_authorizes_worker(
        worker_binding, request, "replacement_pool_generation"
    ):
        raise ValueError("replacement worker is not named by the terminal failed ack")
    approved = worker.get("approved_protocol")
    valid, error = supervisor.verify_file_binding(approved, "replacement protocol")
    if not valid:
        raise ValueError(error)
    protocol_path = supervisor.workspace_file(approved["path"])
    if protocol_path is None:
        raise ValueError("replacement protocol path escapes the workspace")
    context = replacement.validate_protocol(protocol_path)
    if worker.get("pool_spec") != context["protocol"]["pool_spec"]:
        raise ValueError("replacement worker pool_spec differs from the approved protocol")
    if worker.get("reference_gate_evidence") != context["protocol"]["source_reference_gate"]:
        raise ValueError("replacement worker reference gate differs from the approved protocol")
    pool_path = context["output"]
    pool_sha256 = supervisor.file_digest(pool_path)
    if (
        worker.get("pool_sha256") != pool_sha256
        or worker.get("pool_md5") != supervisor.file_digest(pool_path, "md5")
        or worker.get("size_bytes") != pool_path.stat().st_size
    ):
        raise ValueError("replacement worker pool digest or size is invalid")
    activation_id = hashlib.sha256(
        f"{approved['sha256']}|{pool_sha256}".encode("ascii")
    ).hexdigest()[:24]
    if worker.get("activation_id") != activation_id:
        raise ValueError("replacement worker activation identity is invalid")
    checkpoint = worker.get("checkpoint")
    valid, error = supervisor.verify_file_binding(checkpoint, "replacement checkpoint")
    if not valid or not isinstance(checkpoint.get("failure_events"), int):
        raise ValueError(error or "replacement checkpoint failure ledger is invalid")
    runtime_paths = {
        "rcwa_batch.py", "paper2_colorimetry_fine.py", "scripts/run_replacement_pool.py"
    }
    valid, error = supervisor.runtime_hashes_match(worker.get("runtime_hashes"), runtime_paths)
    if not valid:
        raise ValueError(error)
    independent = supervisor.audit_pool(pool_path, worker["pool_spec"])
    if independent.get("passed") is not True:
        raise ValueError("independent replacement strict audit failed")
    expected_audit = {name: independent.get(name) for name in AUDIT_FIELDS}
    if worker.get("audit") != expected_audit:
        raise ValueError("replacement worker audit differs from independent recomputation")
    valid, error = supervisor.verify_protected_snapshot(worker.get("protected_files"))
    if not valid:
        raise ValueError(error)
    audit = copy.deepcopy(worker)
    audit["request"] = request
    audit["evidence_version"] = VERSION
    audit["worker_evidence"] = worker_binding
    audit["independent_reproduction"] = True
    audit["independent_audit"] = expected_audit
    audit["auditor_runtime_hashes"] = {
        path: supervisor.file_digest(ROOT / path)
        for path in supervisor.AUDITOR_RUNTIME_PATHS["replacement_pool_generation"]
    }
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", default=".state/replacement_pool_v1.json")
    parser.add_argument("--output", default=".state/replacement_pool_v1_audit.json")
    args = parser.parse_args()
    output = ROOT / args.output
    audit = build_audit(ROOT / args.worker)
    if output.exists():
        if supervisor.load_json(output, {}) != audit:
            raise SystemExit("existing replacement-pool audit differs")
    else:
        supervisor.atomic_json(output, audit)
    print(json.dumps({"passed": audit["passed"], "pool_sha256": audit["pool_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
