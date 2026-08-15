#!/usr/bin/env python3
"""Freeze the exact replacement-pool protocol after the 32-case gate passes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import paper2_colorimetry_fine as colorimetry  # noqa: E402
import pipeline_supervisor as supervisor  # noqa: E402
from rcwa_batch import generate_params_elliptical  # noqa: E402
from scripts import run_replacement_pool as runner  # noqa: E402
from scripts.run_joint_convergence import retained_order  # noqa: E402


def registered_reference_audit(path: Path) -> dict:
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("evidence_version") != "paper2-reference-holdout-audit-v1":
        raise ValueError("replacement protocol requires the independent 32-case audit")
    if audit.get("passed") is not True or audit.get("production_reference_approved") is not True:
        raise ValueError("32-case production reference is not approved")
    gate_state = supervisor.load_json(supervisor.GATE_STATE, {}) or {}
    gate = gate_state.get("gates", {}).get("reference_resolution", {})
    expected = {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": supervisor.file_digest(path),
    }
    if gate.get("passed") is not True or expected not in gate.get("evidence", []):
        raise ValueError("32-case audit is not the registered reference_resolution gate")
    return audit


def step_token(step: float) -> str:
    return str(step).replace(".", "p").rstrip("0").rstrip("p")


def build_protocol(audit_path: Path) -> dict:
    audit = registered_reference_audit(audit_path)
    policy = supervisor.load_policy()
    if supervisor.verify_policy_integrity(policy).get("passed") is not True:
        raise ValueError("pipeline policy integrity is not verified")
    selected = audit.get("approved_protocol_candidate")
    if not isinstance(selected, dict) or selected.get("passed") is not True:
        raise ValueError("32-case audit does not contain a passing protocol candidate")
    nG = int(selected["requested_nG"])
    Nxy = int(selected["Nxy"])
    step = float(selected["wavelength_step_nm"])
    wavelength = colorimetry.wavelength_grid(step)
    baseline = policy["pool"]
    baseline_meta = baseline["expected_meta"]
    samples = int(baseline["expected_records"]) // 2
    seed = int(baseline_meta["seed"])
    params = runner.canonicalize_params(generate_params_elliptical(samples, seed=seed))
    retained = {retained_order(nG, values[3]) for values in params}
    if len(retained) != 1:
        raise ValueError("retained order unexpectedly varies across the square-lattice pool")
    retained_nG = retained.pop()
    version_name = f"ng{nG}_nxy{Nxy}_step{step_token(step)}_v1"
    output_path = f"data/replacement/rcwa_ellip_TiO2_3000_air_{version_name}.pkl"
    protocol_path = ".state/replacement_protocol_v1.json"
    quality_rule = str(baseline_meta["quality_rule"])
    expected_meta = {
        "seed": seed,
        "nG": nG,
        "Nxy": Nxy,
        "material": baseline["material"],
        "substrate": baseline["substrate"],
        "background": "air",
        "pols": ["p", "s"],
        "n_samples": samples,
        "sampler_version": baseline_meta["sampler_version"],
        "quality_rule": quality_rule,
        "wavelength_step_nm": step,
        "colorimetry_version": colorimetry.COLORIMETRY_VERSION,
        "axis_canonicalization": "L=max(raw_axes), W=min(raw_axes); recompute p/s on canonical axes",
    }
    pool_spec = {
        "path": output_path,
        "expected_records": samples * 2,
        "wavelength_nm": wavelength.tolist(),
        "required_record_fields": sorted(
            runner.BASE_REQUIRED_FIELDS | runner.REPLACEMENT_REQUIRED_FIELDS
        ),
        "polarizations": ["p", "s"],
        "material": baseline["material"],
        "substrate": baseline["substrate"],
        "nG_requested": nG,
        "lossless": True,
        "range_tolerance": baseline["range_tolerance"],
        "pointwise_conservation_tolerance": baseline["pointwise_conservation_tolerance"],
        "stored_value_tolerance": baseline["stored_value_tolerance"],
        "quality_tolerance": baseline["quality_tolerance"],
        "expected_meta": expected_meta,
        "resume_command": (
            "python scripts/run_replacement_pool.py "
            f"--approved-protocol {protocol_path} --n-jobs 16 --resume"
        ),
    }
    runtime_paths = (
        "rcwa_batch.py",
        "paper2_colorimetry_fine.py",
        "scripts/run_replacement_pool.py",
    )
    return {
        "schema_version": 1,
        "evidence_version": runner.PROTOCOL_VERSION,
        "approved": True,
        "automatic_launch_authorized": True,
        "source_reference_gate": {
            "path": str(audit_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": supervisor.file_digest(audit_path),
        },
        "pool_spec": pool_spec,
        "samples": samples,
        "seed": seed,
        "material": baseline["material"],
        "substrate": baseline["substrate"],
        "background": "air",
        "sampler_version": baseline_meta["sampler_version"],
        "quality_rule": quality_rule,
        "nG_requested": nG,
        "nG_retained": retained_nG,
        "Nxy": Nxy,
        "wavelength_step_nm": step,
        "max_same_config_attempts": 2,
        "geometry_manifest_sha256": runner.geometry_manifest_hash(params),
        "runtime_hashes": {name: supervisor.file_digest(ROOT / name) for name in runtime_paths},
        "cost_estimate": {
            "source": "32-case measured task runtimes scaled to 6000 records on 16 workers",
            "wall_hours": selected["estimated_wall_hours_16_workers_6000"],
            "mean_task_seconds": selected["mean_task_seconds_estimate"],
        },
        "decision_rule": "lowest measured-cost protocol that directly passes unchanged joint DeltaE thresholds against nG365/Nxy512/0.5nm on all 32 frozen geometries",
        "training_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", default=".state/reference_resolution_holdout_v1_audit.json")
    parser.add_argument("--output", default=".state/replacement_protocol_v1.json")
    args = parser.parse_args()
    output = ROOT / args.output
    protocol = build_protocol(ROOT / args.audit)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != protocol:
            raise SystemExit("existing approved replacement protocol differs")
    else:
        supervisor.atomic_json(output, protocol)
    print(json.dumps({"approved": True, "pool": protocol["pool_spec"]["path"], "cost_hours": protocol["cost_estimate"]["wall_hours"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
