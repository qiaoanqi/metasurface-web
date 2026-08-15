#!/usr/bin/env python3
"""Independently reproduce cross-solver evaluation and physical controls."""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pipeline_supervisor as supervisor  # noqa: E402
from scripts import run_cross_solver_validation as legacy  # noqa: E402
from scripts import run_cross_solver_validation_v2 as cross  # noqa: E402
from scripts import run_joint_convergence_v2 as joint  # noqa: E402


VERSION = "paper2-cross-solver-audit-v1"


def binding(path: Path) -> dict:
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": supervisor.file_digest(path),
    }


def expected_meta(context: dict) -> tuple[list[dict], list[dict], dict]:
    geometries, _ = cross.load_pool(context["pool_path"])
    selected = legacy.select_cross_solver_geometries(geometries)
    tasks, stress = cross.build_tasks(selected, context["protocol"])
    production = {
        "nG_requested": int(context["protocol"]["nG_requested"]),
        "nG_retained": int(context["protocol"]["nG_retained"]),
        "Nxy": int(context["protocol"]["Nxy"]),
    }
    runtime_paths = (
        "rcwa_batch.py",
        "paper2_colorimetry_fine.py",
        "scripts/run_cross_solver_validation.py",
        "scripts/run_cross_solver_validation_v2.py",
    )
    expected = {
        "version": cross.VERSION,
        "pool_sha256": context["pool_sha256"],
        "approved_protocol_sha256": supervisor.file_digest(context["protocol_path"]),
        "selected_geometries": selected,
        "production": production,
        "stress_configs": stress,
        "expected_tasks": len(tasks),
        "runtime_hashes": {
            path: supervisor.file_digest(ROOT / path) for path in runtime_paths
        },
        "packages": {
            "rcwa": importlib.metadata.version("rcwa"),
            "grcwa": importlib.metadata.version("grcwa"),
        },
        "thresholds": {
            "per_spectrum_R_T_rmse_lte": cross.PER_SPECTRUM_RMSE_LIMIT,
            "mean_spectrum_R_T_rmse_lte": cross.MEAN_SPECTRUM_RMSE_LIMIT,
            "mean_joint_dE00_lt": cross.MEAN_JOINT_DE00_LIMIT,
            "per_geometry_joint_dE00_lt": cross.PER_GEOMETRY_JOINT_DE00_LIMIT,
            "energy_error_lte": cross.ENERGY_LIMIT,
            "analytic_and_symmetry_error_lte": cross.INVARIANT_LIMIT,
        },
    }
    return tasks, stress, expected


def build_audit(worker_path: Path, checkpoint_path: Path, active_path: Path) -> dict:
    worker = supervisor.load_json(worker_path, {}) or {}
    if worker.get("evidence_version") != cross.VERSION:
        raise ValueError("cross-v2 worker evidence version is invalid")
    supervisor.reusable_evidence_request(
        worker.get("request"), "cross_solver_spectrum_validation"
    )
    request = supervisor.current_request_identity("cross_solver_spectrum_validation")
    context = joint.load_active_context(active_path)
    if worker.get("pool_sha256") != context["pool_sha256"]:
        raise ValueError("cross-v2 worker evidence pool differs from the active pool")
    valid, error = supervisor.verify_file_binding(
        worker.get("raw_checkpoint"), "cross-v2 worker checkpoint"
    )
    if not valid:
        raise ValueError("cross-v2 worker checkpoint binding is invalid")
    bound_checkpoint = supervisor.workspace_file(worker["raw_checkpoint"]["path"])
    if bound_checkpoint is None or bound_checkpoint.resolve() != checkpoint_path.resolve():
        raise ValueError("cross-v2 auditor checkpoint differs from the worker binding")
    with checkpoint_path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    tasks, stress, expected = expected_meta(context)
    if int(worker.get("raw_checkpoint", {}).get("tasks", -1)) != len(checkpoint.get("results", {})):
        raise ValueError("cross-v2 worker checkpoint task count is invalid")
    if checkpoint.get("meta") != expected:
        raise ValueError("cross-v2 checkpoint metadata differs from the frozen runtime design")
    cross.validate_checkpoint_results(checkpoint, tasks, require_complete=True)
    evaluation = cross.evaluate_results(checkpoint["results"], stress)
    if worker.get("evaluation") != evaluation:
        raise ValueError("cross-v2 worker evaluation differs from raw checkpoint recomputation")
    production = expected["production"]
    controls = cross.run_controls(production["nG_requested"], production["Nxy"])
    if controls.get("passed") is not True:
        raise ValueError("independent cross-v2 physical controls failed")
    if worker.get("controls") != controls:
        raise ValueError("cross-v2 worker controls differ from independent recomputation")
    audit = copy.deepcopy(worker)
    audit["evidence_version"] = VERSION
    audit["request"] = request
    audit["worker_evidence"] = binding(worker_path)
    audit["independent_evaluation"] = copy.deepcopy(evaluation)
    audit["independent_controls"] = controls
    audit["independent_reproduction"] = True
    audit["auditor_runtime_hashes"] = {
        path: supervisor.file_digest(ROOT / path)
        for path in supervisor.AUDITOR_RUNTIME_PATHS[
            "cross_solver_spectrum_validation"
        ]
    }
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", default=".state/cross_solver_v2.json")
    parser.add_argument("--checkpoint", default=".state/cross_solver_v2_checkpoint.pkl")
    parser.add_argument("--active", default=".state/active_pool.json")
    parser.add_argument("--output", default=".state/cross_solver_v2_audit.json")
    args = parser.parse_args()
    output = ROOT / args.output
    audit = build_audit(ROOT / args.worker, ROOT / args.checkpoint, ROOT / args.active)
    if output.exists():
        if supervisor.load_json(output, {}) != audit:
            raise SystemExit("existing cross-v2 audit differs")
    else:
        supervisor.atomic_json(output, audit)
    print(json.dumps({"passed": audit["passed"], "classification": audit["classification"]}))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
