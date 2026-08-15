#!/usr/bin/env python3
"""Apply the two pre-registered paper 2 strategy transitions once evidence exists."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.reference_v1_outcome import validate_audit as validate_v1_audit  # noqa: E402
from pipeline_supervisor import file_digest  # noqa: E402
from scripts.reference_budget_v2_lineage import validate_lineage  # noqa: E402

STATE = ROOT / ".state"
DISPATCH = STATE / "dispatch_request.json"
V1_AUDIT = STATE / "reference_resolution_v1_audit.json"
V2_AUDIT = STATE / "reference_resolution_budget_v2_audit.json"
ACK = STATE / "executor_ack.json"


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def terminal_scientific_request(dispatch: dict[str, Any]) -> bool:
    return bool(
        dispatch.get("action") == "joint_numerical_convergence"
        and dispatch.get("status") == "failed"
        and dispatch.get("terminal_failure") is True
        and str(dispatch.get("failure_class", "")).lower() == "scientific"
        and isinstance(dispatch.get("request_id"), str)
        and dispatch["request_id"]
        and int(dispatch.get("attempt", 0)) >= 1
    )


def request_identity(dispatch: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_id": dispatch["request_id"],
        "attempt": int(dispatch["attempt"]),
    }


def reusable_request(previous: object, active: dict[str, Any]) -> bool:
    return bool(
        isinstance(previous, dict)
        and previous.get("request_id") == active.get("request_id")
        and 1 <= int(previous.get("attempt", 0)) <= int(active.get("attempt", 0))
    )


def authorized_worker_request(previous: object, active: dict, strategy_based_on: str | None) -> bool:
    if reusable_request(previous, active):
        return True
    return bool(
        isinstance(previous, dict)
        and isinstance(strategy_based_on, str)
        and strategy_based_on
        and previous.get("request_id") == strategy_based_on
        and int(previous.get("attempt", 0)) >= 1
    )


def run_command(script: str) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(ROOT / script)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"{script} failed: {detail}")
    return {
        "script": script,
        "returncode": completed.returncode,
    }


def v1_transition_ready(audit: dict[str, Any]) -> bool:
    if not audit:
        return False
    validate_v1_audit(audit)
    return True


def v2_transition_decision(
    audit: dict[str, Any], dispatch: dict[str, Any], ack: dict[str, Any] | None = None
) -> str:
    if not audit:
        return "waiting_for_v2_audit"
    if audit.get("evidence_version") != "paper2-reference-resolution-budget-v2-audit":
        raise ValueError("unexpected v2 audit version")
    active = request_identity(dispatch)
    authorization = audit.get("authorization_request", audit.get("request"))
    if authorization != active:
        raise ValueError("v2 audit authorization is not bound to the terminal request attempt")
    if audit.get("request") != active:
        raise ValueError("v2 audit request is not the terminal request attempt")
    producer_request = audit.get("producer_request", audit.get("request"))
    if not authorized_worker_request(
        producer_request, active, dispatch.get("strategy_based_on")
    ):
        raise ValueError("v2 audit worker evidence has no authorized request lineage")
    if ack:
        expected = {
            "path": str(V2_AUDIT.relative_to(ROOT)).replace("\\", "/"),
            "sha256": file_digest(V2_AUDIT),
        }
        if not any(
            isinstance(item, dict)
            and item.get("path") == expected["path"]
            and str(item.get("sha256", "")).upper() == expected["sha256"]
            for item in ack.get("evidence", [])
        ):
            raise ValueError("terminal ack does not bind the canonical v2 audit")
    if (
        isinstance(producer_request, dict)
        and producer_request.get("request_id") != active.get("request_id")
    ):
        lineage = validate_lineage(
            ROOT,
            dispatch,
            ack or {},
            ROOT / ".state/reference_resolution_budget_v2_checkpoint.pkl",
            ROOT / ".state/reference_resolution_budget_v2.json",
        )
        if audit.get("recovery_lineage") != lineage:
            raise ValueError("v2 audit recovery lineage differs from the sealed source")
    if (
        audit.get("passed") is True
        and audit.get("classification") == "budget_v2_converged"
        and audit.get("training_allowed") is False
    ):
        return "advance_to_holdout"
    if (
        audit.get("passed") is False
        and audit.get("classification") == "budget_v2_still_insufficient"
    ):
        return "terminal_scientific_negative"
    raise ValueError("v2 audit is not an approved pass or a registered scientific negative")


def advance_once() -> dict[str, Any]:
    dispatch = load_json(DISPATCH)
    if not terminal_scientific_request(dispatch):
        return {"status": "idle", "reason": "no_terminal_scientific_joint_request"}

    revision = int(dispatch.get("strategy_revision", 0))
    if revision < 2:
        audit = load_json(V1_AUDIT)
        if not v1_transition_ready(audit):
            return {"status": "waiting", "reason": "waiting_for_v1_audit"}
        commands = [
            run_command("scripts/freeze_reference_budget_v2.py"),
            run_command("scripts/advance_reference_budget_v2_strategy.py"),
        ]
        return {
            "status": "advanced",
            "transition": "reference_budget_v2",
            "based_on_request_id": dispatch["request_id"],
            "commands": commands,
        }

    ack = load_json(ACK)
    decision = v2_transition_decision(load_json(V2_AUDIT), dispatch, ack or None)
    if decision == "waiting_for_v2_audit":
        return {"status": "waiting", "reason": decision}
    if decision == "terminal_scientific_negative":
        return {
            "status": "stopped",
            "reason": decision,
            "based_on_request_id": dispatch["request_id"],
        }
    commands = [
        run_command("scripts/freeze_reference_holdout_plan.py"),
        run_command("scripts/advance_reference_holdout_strategy.py"),
    ]
    return {
        "status": "advanced",
        "transition": "reference_resolution_holdout",
        "based_on_request_id": dispatch["request_id"],
        "commands": commands,
    }


def main() -> int:
    result = advance_once()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
