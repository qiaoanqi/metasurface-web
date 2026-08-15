#!/usr/bin/env python3
"""Run a matched-budget circular-pillar polarization control."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import paper2_colorimetry_fine as colorimetry  # noqa: E402
import pipeline_supervisor as supervisor  # noqa: E402
from color_utils import delta_e2000  # noqa: E402
from rcwa_batch import rcwa_spectrum  # noqa: E402
from scripts import run_joint_convergence_v2 as joint  # noqa: E402
from scripts import run_reference_resolution_escalation as checkpoint_io  # noqa: E402


VERSION = "paper2-circular-control-worker-v1"
GEOMETRY_COUNT = 12
POLARIZATION_SPECTRUM_MAX_ABS_LIMIT = 1e-7
POLARIZATION_DE00_LIMIT = 0.01


def binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/"),
        "sha256": supervisor.file_digest(path),
    }


def load_source_geometries(pool_path: Path) -> list[dict[str, Any]]:
    with pool_path.open("rb") as handle:
        payload = pickle.load(handle)
    records = payload.get("records", []) if isinstance(payload, dict) else []
    grouped: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("active pool contains a non-object record")
        identifier = str(record.get("geometry_id", ""))
        pol = str(record.get("pol", ""))
        geometry = tuple(float(record[name]) for name in ("L", "W", "H", "P"))
        if not identifier or pol not in {"p", "s"} or geometry[0] < geometry[1]:
            raise ValueError("active pool geometry identity is invalid")
        group = grouped.setdefault(identifier, {"geometry": geometry, "polarizations": set()})
        if group["geometry"] != geometry or pol in group["polarizations"]:
            raise ValueError("active pool geometry pairing is inconsistent")
        group["polarizations"].add(pol)
    if any(item["polarizations"] != {"p", "s"} for item in grouped.values()):
        raise ValueError("active pool lacks exact p/s geometry pairs")
    candidates = []
    for identifier, item in grouped.items():
        L, W, H, P = item["geometry"]
        diameter = float(np.sqrt(L * W))
        candidates.append(
            {
                "source_geometry_id": identifier,
                "D": diameter,
                "H": H,
                "P": P,
                "source_aspect_ratio": L / W,
            }
        )
    candidates.sort(
        key=lambda item: (
            item["source_aspect_ratio"],
            item["P"],
            item["H"],
            item["source_geometry_id"],
        )
    )
    if len(candidates) < GEOMETRY_COUNT:
        raise ValueError("active pool has too few geometries for the circular control")
    indices = np.linspace(0, len(candidates) - 1, GEOMETRY_COUNT, dtype=int)
    selected = [candidates[int(index)] for index in indices]
    if len({item["source_geometry_id"] for item in selected}) != GEOMETRY_COUNT:
        raise ValueError("circular control selection contains duplicate source geometries")
    for index, item in enumerate(selected):
        raw = (
            f"{item['source_geometry_id']}|{item['D']:.17g}|{item['H']:.17g}|"
            f"{item['P']:.17g}"
        ).encode("ascii")
        item["control_id"] = f"circle-{index:02d}-{hashlib.sha256(raw).hexdigest()[:12]}"
    return selected


def build_meta(context: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    protocol = context["protocol"]
    step = float(protocol["wavelength_step_nm"])
    wavelength = np.arange(380.0, 780.0 + step / 2.0, step, dtype=float)
    conservation = float(protocol["pool_spec"]["pointwise_conservation_tolerance"])
    return {
        "version": VERSION,
        "request": request,
        "pool_sha256": context["pool_sha256"],
        "approved_protocol": binding(context["protocol_path"]),
        "protocol": {
            "nG_requested": int(protocol["nG_requested"]),
            "nG_retained": int(protocol["nG_retained"]),
            "Nxy": int(protocol["Nxy"]),
            "wavelength_step_nm": step,
            "material": protocol["material"],
            "substrate": protocol["substrate"],
            "background": protocol["background"],
            "max_attempts": int(protocol["max_same_config_attempts"]),
        },
        "wavelength_nm": wavelength,
        "geometries": load_source_geometries(context["pool_path"]),
        "thresholds": {
            "pointwise_conservation_lte": conservation,
            "polarization_spectrum_max_abs_lte": POLARIZATION_SPECTRUM_MAX_ABS_LIMIT,
            "polarization_dE00_lte": POLARIZATION_DE00_LIMIT,
        },
        "runtime_hashes": {
            name: supervisor.file_digest(ROOT / name)
            for name in (
                "pipeline_supervisor.py",
                "scripts/run_circular_control_v1.py",
                "scripts/run_joint_convergence_v2.py",
                "rcwa_batch.py",
                "paper2_colorimetry_fine.py",
                "color_utils.py",
            )
        },
    }


def solve_geometry(payload: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    geometry, meta = payload
    protocol = meta["protocol"]
    wavelength = np.asarray(meta["wavelength_nm"], dtype=float)
    started = time.perf_counter()
    errors = []
    for attempt in range(1, protocol["max_attempts"] + 1):
        spectra = {}
        try:
            for pol in ("p", "s"):
                R, T = rcwa_spectrum(
                    geometry["D"],
                    geometry["H"],
                    geometry["P"],
                    wavelength,
                    nG_req=protocol["nG_requested"],
                    Nxy=protocol["Nxy"],
                    material=protocol["material"],
                    substrate=protocol["substrate"],
                    W_nm=geometry["D"],
                    pol=pol,
                    background=protocol["background"],
                )
                R = np.asarray(R, dtype=float)
                T = np.asarray(T, dtype=float)
                if R.shape != wavelength.shape or T.shape != wavelength.shape:
                    raise ValueError("circular control spectrum shape is invalid")
                if not np.isfinite(R).all() or not np.isfinite(T).all():
                    raise ValueError("circular control spectrum is non-finite")
                spectra[pol] = {"R": R, "T": T}
            return {
                "id": geometry["control_id"],
                "geometry": geometry,
                "status": "ok",
                "attempt": attempt,
                "spectra": spectra,
                "time_s": time.perf_counter() - started,
            }
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    return {
        "id": geometry["control_id"],
        "geometry": geometry,
        "status": "failed",
        "errors": errors,
        "time_s": time.perf_counter() - started,
    }


def result_metrics(result: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    if result.get("status") != "ok":
        return {"id": result.get("id"), "valid": False, "error": result.get("errors")}
    wavelength = np.asarray(meta["wavelength_nm"], dtype=float)
    spectra = result["spectra"]
    arrays = {
        f"{pol}_{field}": np.asarray(spectra[pol][field], dtype=float)
        for pol in ("p", "s")
        for field in ("R", "T")
    }
    valid = all(
        array.shape == wavelength.shape
        and np.isfinite(array).all()
        and np.all(array >= -1e-8)
        and np.all(array <= 1.0 + 1e-8)
        for array in arrays.values()
    )
    if not valid:
        return {"id": result["id"], "valid": False, "error": "invalid raw spectrum"}
    labels = {
        pol: colorimetry.spectrum_to_labels_d65(arrays[f"{pol}_R"], wavelength)
        for pol in ("p", "s")
    }
    max_energy = max(
        float(np.max(np.abs(arrays[f"{pol}_R"] + arrays[f"{pol}_T"] - 1.0)))
        for pol in ("p", "s")
    )
    return {
        "id": result["id"],
        "valid": True,
        "max_pointwise_conservation_error": max_energy,
        "polarization_R_max_abs": float(np.max(np.abs(arrays["p_R"] - arrays["s_R"]))),
        "polarization_T_max_abs": float(np.max(np.abs(arrays["p_T"] - arrays["s_T"]))),
        "polarization_dE00": float(delta_e2000(labels["p"]["lab"], labels["s"]["lab"])),
    }


def validate_checkpoint(checkpoint: dict[str, Any], meta: dict[str, Any], complete: bool) -> None:
    if not isinstance(checkpoint, dict) or checkpoint.get("meta") != meta:
        raise ValueError("circular control checkpoint metadata differs")
    results = checkpoint.get("results")
    if not isinstance(results, dict):
        raise ValueError("circular control checkpoint lacks results")
    expected = {item["control_id"] for item in meta["geometries"]}
    if not set(results) <= expected or (complete and set(results) != expected):
        raise ValueError("circular control checkpoint task set differs")
    for key, result in results.items():
        if result.get("id") != key or result.get("geometry", {}).get("control_id") != key:
            raise ValueError("circular control checkpoint task identity differs")
        result_metrics(result, meta)


def summarize(checkpoint: dict[str, Any], checkpoint_path: Path) -> dict[str, Any]:
    meta = checkpoint["meta"]
    metrics = [result_metrics(checkpoint["results"][item["control_id"]], meta) for item in meta["geometries"]]
    thresholds = meta["thresholds"]
    checks = {
        "exact_frozen_geometry_set": len(metrics) == GEOMETRY_COUNT,
        "no_task_failures": all(item.get("valid") is True for item in metrics),
        "pointwise_conservation": all(
            item.get("max_pointwise_conservation_error", float("inf"))
            <= thresholds["pointwise_conservation_lte"]
            for item in metrics
        ),
        "circular_polarization_spectrum_symmetry": all(
            max(
                item.get("polarization_R_max_abs", float("inf")),
                item.get("polarization_T_max_abs", float("inf")),
            )
            <= thresholds["polarization_spectrum_max_abs_lte"]
            for item in metrics
        ),
        "circular_polarization_color_symmetry": all(
            item.get("polarization_dE00", float("inf"))
            <= thresholds["polarization_dE00_lte"]
            for item in metrics
        ),
    }
    return {
        "schema_version": 1,
        "evidence_version": VERSION,
        "request": meta["request"],
        "generated_at": supervisor.now_iso(),
        "passed": all(checks.values()),
        "classification": "circular_control_passed"
        if all(checks.values())
        else "circular_control_failed",
        "pool_sha256": meta["pool_sha256"],
        "approved_protocol": meta["approved_protocol"],
        "protocol": meta["protocol"],
        "thresholds": thresholds,
        "selected_geometries": meta["geometries"],
        "raw_checkpoint": binding(checkpoint_path) | {"tasks": len(checkpoint["results"])},
        "metrics": metrics,
        "checks": checks,
        "runtime_hashes": meta["runtime_hashes"],
        "training_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active", default=".state/active_pool.json")
    parser.add_argument("--checkpoint", default=".state/circular_control_v1_checkpoint.pkl")
    parser.add_argument("--output", default=".state/circular_control_v1.json")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    context = joint.load_active_context(ROOT / args.active)
    request = supervisor.current_request_identity("circular_control")
    meta = build_meta(context, request)
    checkpoint_path = ROOT / args.checkpoint
    if checkpoint_path.exists():
        if not args.resume:
            raise SystemExit("circular control checkpoint exists; use --resume")
        with checkpoint_path.open("rb") as handle:
            checkpoint = pickle.load(handle)
        validate_checkpoint(checkpoint, meta, complete=False)
    else:
        checkpoint = {"meta": meta, "results": {}}
        checkpoint_io.atomic_pickle(checkpoint_path, checkpoint)
    pending = [item for item in meta["geometries"] if item["control_id"] not in checkpoint["results"]]
    if pending:
        with mp.Pool(processes=max(1, min(args.n_jobs, len(pending)))) as pool:
            for result in pool.imap_unordered(solve_geometry, [(item, meta) for item in pending]):
                checkpoint["results"][result["id"]] = result
                checkpoint_io.atomic_pickle(checkpoint_path, checkpoint)
    validate_checkpoint(checkpoint, meta, complete=True)
    evidence = summarize(checkpoint, checkpoint_path)
    output = ROOT / args.output
    if output.exists():
        existing = supervisor.load_json(output, {}) or {}
        comparable = dict(evidence)
        comparable["generated_at"] = existing.get("generated_at")
        if existing != comparable:
            raise SystemExit("existing circular control evidence differs")
    else:
        supervisor.atomic_json(output, evidence)
    print(json.dumps({"passed": evidence["passed"], "classification": evidence["classification"]}))
    return 0 if evidence["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
