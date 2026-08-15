#!/usr/bin/env python3
"""Audit an activated replacement pool against the frozen 32-case reference."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import paper2_colorimetry_fine as colorimetry  # noqa: E402
import pipeline_supervisor as supervisor  # noqa: E402
from color_utils import delta_e2000  # noqa: E402
from scripts import run_reference_resolution_escalation as base  # noqa: E402
from scripts import run_reference_resolution_holdout as holdout  # noqa: E402
from scripts import run_replacement_pool as replacement  # noqa: E402


VERSION = "paper2-joint-convergence-v2"
MEAN_LIMIT = 1.15
PER_GEOMETRY_LIMIT = 2.3
CONSERVATION_LIMIT = 1e-6
LABEL_TOLERANCE = 1e-10


def load_active_context(active_path: Path) -> dict:
    policy = supervisor.load_policy()
    if supervisor.verify_policy_integrity(policy).get("passed") is not True:
        raise ValueError("pipeline policy integrity is not verified")
    active = supervisor.load_json(active_path, {}) or {}
    if active.get("active") is not True or active.get("evidence_version") != "paper2-active-pool-v1":
        raise ValueError("a versioned replacement pool is not active")
    approved = active.get("approved_protocol", {})
    protocol_path = replacement.canonical_workspace_path(str(approved.get("path", "")))
    if not protocol_path.is_file() or replacement.file_digest(protocol_path) != str(approved.get("sha256", "")).upper():
        raise ValueError("active pool approved protocol hash mismatch")
    protocol_context = replacement.validate_protocol(protocol_path)
    protocol = protocol_context["protocol"]
    if active.get("pool_spec") != protocol.get("pool_spec"):
        raise ValueError("active pool spec differs from the approved protocol")
    pool_path = replacement.canonical_workspace_path(
        str(active["pool_spec"]["path"]), require_replacement_dir=True
    )
    pool_sha256 = supervisor.file_digest(pool_path)
    if pool_sha256 != str(active.get("pool_sha256", "")).upper():
        raise ValueError("active replacement pool SHA256 mismatch")
    audit = supervisor.audit_pool(pool_path, active["pool_spec"])
    if audit.get("passed") is not True:
        raise ValueError("active replacement pool strict audit failed")
    return {
        "policy": policy,
        "active": active,
        "active_path": active_path,
        "protocol": protocol,
        "protocol_path": protocol_path,
        "pool_path": pool_path,
        "pool_sha256": pool_sha256,
        "pool_audit": audit,
    }


def load_reference(context: dict) -> dict:
    source = context["protocol"]["source_reference_gate"]
    audit_path = replacement.canonical_workspace_path(source["path"])
    if supervisor.file_digest(audit_path) != str(source["sha256"]).upper():
        raise ValueError("approved reference gate evidence hash mismatch")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("evidence_version") != "paper2-reference-holdout-audit-v1" or audit.get("passed") is not True:
        raise ValueError("approved reference gate is not the 32-case audit")
    sources = audit.get("sources", {})

    def source_path(name: str) -> Path:
        item = sources[name]
        path = replacement.canonical_workspace_path(item["path"])
        if supervisor.file_digest(path) != str(item["sha256"]).upper():
            raise ValueError(f"reference source hash mismatch: {name}")
        return path

    plan_path = source_path("plan")
    candidate_path = source_path("candidate_checkpoint")
    extension_path = source_path("holdout_checkpoint")
    plan = holdout.load_plan(plan_path)
    with candidate_path.open("rb") as handle:
        candidate = pickle.load(handle)
    with extension_path.open("rb") as handle:
        extension = pickle.load(handle)
    results = dict(candidate.get("results", {}))
    results.update(extension.get("results", {}))
    if len(results) != 320:
        raise ValueError("approved reference does not contain exact 320 raw spectra")
    return {
        "audit": audit,
        "audit_path": audit_path,
        "plan": plan,
        "plan_path": plan_path,
        "candidate_path": candidate_path,
        "extension_path": extension_path,
        "results": results,
    }


def geometry_key(values) -> tuple[float, float, float, float]:
    if isinstance(values, dict):
        return tuple(float(values[name]) for name in ("L", "W", "H", "P"))
    return tuple(float(value) for value in values)


def canonical_geometry(item: dict) -> tuple[tuple[float, float, float, float], bool]:
    L, W, H, P = geometry_key(item)
    return (max(L, W), min(L, W), H, P), L < W


def load_pool_records(path: Path) -> dict[tuple[tuple[float, ...], str], dict]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    records = payload.get("records", []) if isinstance(payload, dict) else []
    indexed = {}
    for record in records:
        key = (geometry_key(record), str(record.get("pol")))
        if key in indexed:
            raise ValueError("duplicate replacement pool geometry/polarization key")
        indexed[key] = record
    return indexed


def reference_result(reference: dict, index: int, pol: str, axes_swapped: bool) -> dict:
    source_pol = ({"p": "s", "s": "p"}[pol]) if axes_swapped else pol
    identifier = holdout.result_id(index, source_pol, base.FINE_CONFIG, 0.5)
    result = reference["results"].get(identifier)
    if not isinstance(result, dict) or result.get("status") != "ok":
        raise ValueError(f"missing approved reference result: {identifier}")
    return result


def record_labels_are_exact(record: dict, recomputed: dict[str, np.ndarray]) -> bool:
    if record.get("label_provenance_version") != colorimetry.COLORIMETRY_VERSION:
        return False
    for stored_name, computed_name in (
        ("xyz", "xyz"),
        ("lab", "lab"),
        ("srgb_display", "srgb_display"),
    ):
        stored = np.asarray(record.get(stored_name), dtype=float)
        expected = np.asarray(recomputed[computed_name], dtype=float)
        if stored.shape != expected.shape or not np.allclose(
            stored, expected, rtol=0.0, atol=LABEL_TOLERANCE
        ):
            return False
    return True


def evaluate(context: dict, reference: dict) -> dict:
    records = load_pool_records(context["pool_path"])
    rows = []
    missing = []
    label_failures = []
    conservation_max = 0.0
    reference_conservation_max = 0.0
    for index, old_geometry in enumerate(reference["plan"]["combined_cases"]):
        canonical, axes_swapped = canonical_geometry(old_geometry)
        channel_errors = []
        for pol in base.POLS:
            record = records.get((canonical, pol))
            if record is None:
                missing.append({"geometry_index": index, "geometry": canonical, "pol": pol})
                continue
            wavelength = np.asarray(record["wl_nm"], dtype=float)
            R = np.asarray(record["R"], dtype=float)
            T = np.asarray(record["T"], dtype=float)
            labels = colorimetry.spectrum_to_labels_d65(R, wavelength)
            if not record_labels_are_exact(record, labels):
                label_failures.append({"geometry_index": index, "pol": pol})
            ref = reference_result(reference, index, pol, axes_swapped)
            ref_wavelength = np.asarray(ref["wavelength_nm"], dtype=float)
            ref_R = np.asarray(ref["R"], dtype=float)
            ref_T = np.asarray(ref["T"], dtype=float)
            ref_labels = colorimetry.spectrum_to_labels_d65(ref_R, ref_wavelength)
            error = float(delta_e2000(labels["lab"], ref_labels["lab"]))
            channel_errors.append(error)
            conservation = float(np.max(np.abs(R + T - 1.0)))
            ref_conservation = float(np.max(np.abs(ref_R + ref_T - 1.0)))
            conservation_max = max(conservation_max, conservation)
            reference_conservation_max = max(reference_conservation_max, ref_conservation)
            rows.append(
                {
                    "geometry_index": index,
                    "geometry_id": record.get("geometry_id"),
                    "pol": pol,
                    "axes_swapped_from_reference": axes_swapped,
                    "dE00": error,
                    "pool_pointwise_conservation_error_max": conservation,
                    "reference_pointwise_conservation_error_max": ref_conservation,
                    "nG_requested": record.get("nG_requested"),
                    "nG_retained": record.get("nG_retained"),
                    "Nxy": record.get("Nxy"),
                    "wavelength_step_nm": record.get("wavelength_step_nm"),
                }
            )
        if len(channel_errors) == 2:
            rows[-1]["joint_geometry_dE00"] = max(channel_errors)
            rows[-2]["joint_geometry_dE00"] = max(channel_errors)
    joint = [
        max(row["dE00"] for row in rows if row["geometry_index"] == index)
        for index in range(32)
        if sum(row["geometry_index"] == index for row in rows) == 2
    ]
    values = np.asarray(joint, dtype=float)
    checks = {
        "exact_32_complete_p_s_geometries": len(joint) == 32 and not missing,
        "derived_labels_exact": not label_failures,
        "pool_conservation": conservation_max <= CONSERVATION_LIMIT,
        "reference_conservation": reference_conservation_max <= CONSERVATION_LIMIT,
        "mean_joint_dE00": bool(values.size == 32 and np.mean(values) < MEAN_LIMIT),
        "all_joint_dE00": bool(values.size == 32 and np.all(values < PER_GEOMETRY_LIMIT)),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "joint_dE00": {
            "count": len(joint),
            "mean": float(np.mean(values)) if values.size else None,
            "median": float(np.median(values)) if values.size else None,
            "max": float(np.max(values)) if values.size else None,
            "values": joint,
        },
        "pointwise_conservation_error_max": conservation_max,
        "reference_pointwise_conservation_error_max": reference_conservation_max,
        "missing": missing,
        "label_failures": label_failures,
        "rows": rows,
    }


def build_evidence(context: dict, reference: dict) -> dict:
    evaluation = evaluate(context, reference)
    protected = supervisor.audit_protected_files(context["policy"])
    checks = {
        "active_pool_strict_audit": context["pool_audit"].get("passed") is True,
        "paper1_and_legacy_assets_unchanged": all(item.get("passed") for item in protected),
        "replacement_vs_reference": evaluation["passed"],
    }
    runtime_paths = (
        "paper2_colorimetry_fine.py",
        "scripts/run_joint_convergence_v2.py",
    )
    return {
        "schema_version": 1,
        "evidence_version": VERSION,
        "passed": all(checks.values()),
        "classification": "passed" if all(checks.values()) else "replacement_pool_numerical_mismatch",
        "pool_sha256": context["pool_sha256"],
        "checks": checks,
        "thresholds": {
            "mean_joint_dE00_lt": MEAN_LIMIT,
            "all_joint_dE00_lt": PER_GEOMETRY_LIMIT,
            "pointwise_conservation_lte": CONSERVATION_LIMIT,
            "stored_label_atol": LABEL_TOLERANCE,
        },
        "evaluation": evaluation,
        "active_pool": {"path": str(context["active_path"].relative_to(ROOT)).replace("\\", "/"), "sha256": supervisor.file_digest(context["active_path"])},
        "approved_protocol": {"path": str(context["protocol_path"].relative_to(ROOT)).replace("\\", "/"), "sha256": supervisor.file_digest(context["protocol_path"])},
        "reference_gate": {"path": str(reference["audit_path"].relative_to(ROOT)).replace("\\", "/"), "sha256": supervisor.file_digest(reference["audit_path"])},
        "reference_raw_spectra": {
            "candidate_checkpoint_sha256": supervisor.file_digest(reference["candidate_path"]),
            "holdout_checkpoint_sha256": supervisor.file_digest(reference["extension_path"]),
        },
        "runtime_hashes": {path: supervisor.file_digest(ROOT / path) for path in runtime_paths},
        "protected_files": protected,
        "training_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active", default=".state/active_pool.json")
    parser.add_argument("--output", default=".state/joint_convergence_v2.json")
    args = parser.parse_args()
    output = ROOT / args.output
    context = load_active_context(ROOT / args.active)
    reference = load_reference(context)
    evidence = build_evidence(context, reference)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != evidence:
            raise SystemExit("existing joint-v2 evidence differs; use a new version")
    else:
        supervisor.atomic_json(output, evidence)
    print(json.dumps({"passed": evidence["passed"], "classification": evidence["classification"]}))
    return 0 if evidence["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
