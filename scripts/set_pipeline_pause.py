#!/usr/bin/env python3
"""Atomically bind or clear an unattended pause after one active request."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pipeline_supervisor as supervisor  # noqa: E402
from scripts import policy_integrity_transaction as transaction  # noqa: E402


POLICY_PATH = ROOT / "pipeline_policy.json"
INTEGRITY_PATH = ROOT / ".state" / "pipeline_integrity.json"
HASH_BOUND_RUNTIME = (
    ROOT / "pipeline_supervisor.py",
    ROOT / "tests" / "test_pipeline_supervisor.py",
)


def build_after(before: dict, request_id: str, enabled: bool) -> dict:
    policy = copy.deepcopy(before)
    operations = policy.setdefault("operations", {})
    operations["pause_after_request"] = {
        "enabled": enabled,
        "request_id": request_id,
        "reason": "user_requested_safe_pause",
        "resume_requires": "explicit_user_authorization",
    }
    evidence = policy.get("strategy_override", {}).get("evidence", [])
    hashes = {
        str(path.relative_to(ROOT)).replace("\\", "/"): supervisor.file_digest(path)
        for path in HASH_BOUND_RUNTIME
    }
    for item in evidence:
        path = item.get("path") if isinstance(item, dict) else None
        if path in hashes:
            item["sha256"] = hashes[path]
    return policy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    before_policy = supervisor.load_json(POLICY_PATH, {}) or {}
    before_integrity = supervisor.load_json(INTEGRITY_PATH, {}) or {}
    after_policy = build_after(before_policy, args.request_id, not args.resume)
    if after_policy == before_policy:
        print(json.dumps({"status": "already_set", "paused": not args.resume}, sort_keys=True))
        return 0
    after_integrity = copy.deepcopy(before_integrity)
    after_integrity["protected_assets_revision"] = int(
        before_integrity.get("protected_assets_revision", 0)
    ) + 1
    after_integrity["policy_sha256"] = transaction.json_file_sha256(after_policy)
    after_integrity["supervisor_sha256"] = supervisor.file_digest(
        ROOT / "pipeline_supervisor.py"
    )
    after_integrity["note"] = (
        "safe pause bound to active request; explicit authorization required to resume"
        if not args.resume
        else "safe pause cleared by explicit authorization"
    )
    result = transaction.apply_policy_integrity_transaction(
        POLICY_PATH,
        INTEGRITY_PATH,
        before_policy,
        before_integrity,
        after_policy,
        after_integrity,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "paused": not args.resume,
                "request_id": args.request_id,
                "policy_sha256": result["policy_sha256"],
                "integrity_sha256": result["integrity_sha256"],
                "supervisor_sha256": after_integrity["supervisor_sha256"],
                "revision": after_integrity["protected_assets_revision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
