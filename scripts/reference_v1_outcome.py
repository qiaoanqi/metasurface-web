"""Shared contract for a terminal 80-task reference-resolution outcome."""
from __future__ import annotations

from typing import Any


AUDIT_VERSION = "paper2-reference-resolution-audit-v1"
WORKER_VERSION = "paper2-reference-resolution-v1"
REQUIRED_CHECKS = (
    "frozen_plan_sha256_and_content",
    "checkpoint_meta_and_runtime_hashes",
    "reference_checkpoint_exact_80",
    "worker_claim_matches_independent_recomputation",
    "physics_controls_passed",
)
SCIENTIFIC_CLASSIFICATIONS = {
    "historical_5nm_sampling_rejected",
    "historical_production_budget_rejected",
    "reference_requires_followup",
    "reference_spatial_budget_insufficient_grid",
    "reference_spatial_budget_insufficient_order",
    "reference_spatial_budget_insufficient_order_and_grid",
    "reference_spectral_resolution_blocks_spatial_interpretation",
    "reference_spectral_resolution_insufficient",
}


def validate_audit(audit: dict[str, Any]) -> None:
    """Accept either converged or unconverged reference evidence, never bad execution."""
    if audit.get("evidence_version") != AUDIT_VERSION:
        raise ValueError("independent v1 reference audit is not frozen")
    if not isinstance(audit.get("passed"), bool):
        raise ValueError("v1 reference audit lacks a boolean terminal result")
    if audit.get("classification") not in SCIENTIFIC_CLASSIFICATIONS:
        raise ValueError("v1 reference audit is not a transition-eligible scientific outcome")
    for key in REQUIRED_CHECKS:
        if audit.get("checks", {}).get(key) is not True:
            raise ValueError(f"v1 audit prerequisite is not proven: {key}")


def validate_worker_evidence(audit: dict[str, Any], evidence: dict[str, Any]) -> None:
    validate_audit(audit)
    if evidence.get("evidence_version") != WORKER_VERSION:
        raise ValueError("v1 worker evidence version is not frozen")
    if not isinstance(evidence.get("passed"), bool):
        raise ValueError("v1 worker evidence lacks a boolean terminal result")
    if evidence["passed"] is not audit["passed"]:
        raise ValueError("v1 worker claim differs from the independent audit")
