#!/usr/bin/env python3
"""Escalate the paper 2 numerical reference after the v1.1 gate failed.

This diagnostic does not rehabilitate the nG=131 production pool. It tests
whether a higher-order, finer-grid, finer-wavelength reference is itself
stable enough to define a replacement production protocol.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from color_utils import CIE_X, CIE_Y, CIE_Z, delta_e2000  # noqa: E402
from paper2_colorimetry import D65_SPD  # noqa: E402
from pipeline_supervisor import atomic_json, file_digest  # noqa: E402
from rcwa_batch import rcwa_spectrum  # noqa: E402
from scripts.run_joint_convergence import retained_order  # noqa: E402


VERSION = "paper2-reference-resolution-v1"
POLS = ("p", "s")
BASELINE = (251, 384)
ORDER_17 = (290, 512)
ORDER_19 = (365, 512)
GRID_384 = (365, 384)
CONFIGS_1NM = ((290, 384), ORDER_17, GRID_384, ORDER_19)
FINE_CONFIG = ORDER_19
WL_5NM = np.arange(380.0, 785.0, 5.0)
WL_1NM = np.arange(380.0, 781.0, 1.0)
WL_HALF_NM = np.arange(380.0, 780.0 + 0.25, 0.5)
MEAN_DE_LIMIT = 1.15
PER_GEOMETRY_DE_LIMIT = 2.3
CONSERVATION_LIMIT = 1e-6
EXPECTED_TASKS = 8 * len(POLS) * (len(CONFIGS_1NM) + 1)


def geometry_key(geometry: dict) -> tuple[float, float, float, float]:
    return tuple(float(geometry[name]) for name in ("L", "W", "H", "P"))


def step_token(step_nm: float) -> str:
    return "0p5" if step_nm == 0.5 else str(int(step_nm))


def task_id(index: int, pol: str, config: tuple[int, int], step_nm: float) -> str:
    return (
        f"refesc-g{index:02d}-{pol}-ng{config[0]}-nxy{config[1]}"
        f"-step{step_token(step_nm)}"
    )


def build_tasks(selected: list[dict]) -> list[dict]:
    if len(selected) != 8:
        raise ValueError("reference escalation requires the frozen eight v1.1 cases")
    tasks = []
    for index, geometry in enumerate(selected):
        for pol in POLS:
            for config in CONFIGS_1NM:
                tasks.append(
                    {
                        "id": task_id(index, pol, config, 1.0),
                        "geometry_index": index,
                        "geometry": geometry,
                        "pol": pol,
                        "requested_nG": config[0],
                        "retained_nG": retained_order(config[0], geometry["P"]),
                        "Nxy": config[1],
                        "step_nm": 1.0,
                        "wavelength_nm": WL_1NM,
                    }
                )
            tasks.append(
                {
                    "id": task_id(index, pol, FINE_CONFIG, 0.5),
                    "geometry_index": index,
                    "geometry": geometry,
                    "pol": pol,
                    "requested_nG": FINE_CONFIG[0],
                    "retained_nG": retained_order(FINE_CONFIG[0], geometry["P"]),
                    "Nxy": FINE_CONFIG[1],
                    "step_nm": 0.5,
                    "wavelength_nm": WL_HALF_NM,
                }
            )
    return tasks


def run_task(task: dict) -> dict:
    geometry = task["geometry"]
    started = time.perf_counter()
    base = {
        key: task[key]
        for key in (
            "id",
            "geometry_index",
            "geometry",
            "pol",
            "requested_nG",
            "retained_nG",
            "Nxy",
            "step_nm",
        )
    }
    base["wavelength_nm"] = np.asarray(task["wavelength_nm"], dtype=float)
    try:
        reflectance, transmittance = rcwa_spectrum(
            geometry["L"],
            geometry["H"],
            geometry["P"],
            base["wavelength_nm"],
            nG_req=task["requested_nG"],
            Nxy=task["Nxy"],
            material="TiO2",
            substrate="SiO2",
            W_nm=geometry["W"],
            pol=task["pol"],
            background="air",
        )
        base.update(
            status="ok",
            R=np.asarray(reflectance, dtype=float),
            T=np.asarray(transmittance, dtype=float),
            time_s=time.perf_counter() - started,
        )
    except Exception as exc:
        base.update(
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            time_s=time.perf_counter() - started,
        )
    return base


def atomic_pickle(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(10):
            try:
                os.replace(temporary, path)
                return
            except OSError as exc:
                transient = (
                    isinstance(exc, PermissionError)
                    or getattr(exc, "winerror", None) in {5, 32}
                    or exc.errno in {13, 16}
                )
                if not transient or attempt == 9:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def labels_on_grid(wavelength_nm: np.ndarray, reflectance: np.ndarray) -> np.ndarray:
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    reflectance = np.asarray(reflectance, dtype=float)
    cmf = np.column_stack(
        [np.interp(wavelength_nm, WL_5NM, values) for values in (CIE_X, CIE_Y, CIE_Z)]
    )
    spd = np.interp(wavelength_nm, WL_5NM, D65_SPD)
    normalization = float(np.trapezoid(spd * cmf[:, 1], wavelength_nm))
    white = np.trapezoid(spd[:, None] * cmf, wavelength_nm, axis=0) / normalization
    xyz = (
        np.trapezoid(reflectance[:, None] * spd[:, None] * cmf, wavelength_nm, axis=0)
        / normalization
    )
    ratio = xyz / white
    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = np.where(ratio > epsilon, np.cbrt(ratio), (kappa * ratio + 16.0) / 116.0)
    return np.array(
        [116.0 * f[1] - 16.0, 500.0 * (f[0] - f[1]), 200.0 * (f[1] - f[2])],
        dtype=float,
    )


def threshold_summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)) if array.size else None,
        "max": float(np.max(array)) if array.size else None,
        "mean_lt_1_15": bool(array.size and np.mean(array) < MEAN_DE_LIMIT),
        "all_lt_2_3": bool(array.size and np.all(array < PER_GEOMETRY_DE_LIMIT)),
        "joint_max_by_geometry": array.tolist(),
    }


def joint_comparison(
    selected: list[dict], first: dict, second: dict, first_id, second_id
) -> dict:
    joint = []
    for index, _geometry in enumerate(selected):
        channel_values = []
        for pol in POLS:
            left = first[first_id(index, pol)]
            right = second[second_id(index, pol)]
            left_lab = labels_on_grid(left["wavelength_nm"], left["R"])
            right_lab = labels_on_grid(right["wavelength_nm"], right["R"])
            channel_values.append(float(delta_e2000(left_lab, right_lab)))
        joint.append(max(channel_values))
    summary = threshold_summary(joint)
    summary["complete_p_s_geometries"] = len(joint)
    return summary


def validate_spectra(results: dict) -> dict:
    maximum_error = 0.0
    valid = len(results) == EXPECTED_TASKS
    for result in results.values():
        if result.get("status") != "ok":
            valid = False
            continue
        wavelength = np.asarray(result.get("wavelength_nm"), dtype=float)
        reflectance = np.asarray(result.get("R"), dtype=float)
        transmittance = np.asarray(result.get("T"), dtype=float)
        if reflectance.shape != wavelength.shape or transmittance.shape != wavelength.shape:
            valid = False
            continue
        if not (
            np.isfinite(reflectance).all()
            and np.isfinite(transmittance).all()
            and np.all(reflectance >= -1e-8)
            and np.all(reflectance <= 1.0 + 1e-8)
            and np.all(transmittance >= -1e-8)
            and np.all(transmittance <= 1.0 + 1e-8)
        ):
            valid = False
        maximum_error = max(
            maximum_error, float(np.max(np.abs(reflectance + transmittance - 1.0)))
        )
    return {
        "passed": valid and maximum_error <= CONSERVATION_LIMIT,
        "records": len(results),
        "pointwise_conservation_error_max": maximum_error,
    }


def load_inputs(v1_evidence_path: Path, v1_checkpoint_path: Path) -> tuple[dict, dict]:
    evidence = json.loads(v1_evidence_path.read_text(encoding="utf-8"))
    with v1_checkpoint_path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    if evidence.get("evidence_version") != "paper2-joint-convergence-v1.1":
        raise ValueError("unexpected v1.1 evidence version")
    if evidence.get("passed") is not False:
        raise ValueError("reference escalation requires the preserved failed v1.1 evidence")
    if len(evidence.get("selection", [])) != 8:
        raise ValueError("v1.1 evidence does not contain the frozen eight cases")
    if len(checkpoint.get("results", {})) != 64:
        raise ValueError("v1.1 checkpoint must be complete 64/64")
    return evidence, checkpoint


def build_meta(v1_evidence_path: Path, v1_checkpoint_path: Path) -> dict:
    evidence, _checkpoint = load_inputs(v1_evidence_path, v1_checkpoint_path)
    runtime_paths = (
        "rcwa_batch.py",
        "paper2_colorimetry.py",
        "color_utils.py",
        "scripts/run_reference_resolution_escalation.py",
    )
    return {
        "version": VERSION,
        "pool_sha256": evidence["pool_sha256"],
        "selected_geometries": evidence["selection"],
        "selection_source": {
            "path": str(v1_evidence_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": file_digest(v1_evidence_path),
        },
        "baseline_checkpoint": {
            "path": str(v1_checkpoint_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": file_digest(v1_checkpoint_path),
        },
        "expected_tasks": EXPECTED_TASKS,
        "configs_1nm": [list(config) for config in CONFIGS_1NM],
        "fine_config": list(FINE_CONFIG),
        "fine_step_nm": 0.5,
        "thresholds": {
            "mean_joint_dE00_lt": MEAN_DE_LIMIT,
            "all_joint_dE00_lt": PER_GEOMETRY_DE_LIMIT,
            "pointwise_conservation_lte": CONSERVATION_LIMIT,
        },
        "runtime_hashes": {path: file_digest(ROOT / path) for path in runtime_paths},
    }


def build_plan(meta: dict) -> dict:
    return {
        "schema_version": 1,
        "evidence_version": f"{VERSION}-plan",
        "plan_valid": True,
        "pool_sha256": meta["pool_sha256"],
        "failed_gate_source": meta["selection_source"],
        "baseline_checkpoint": meta["baseline_checkpoint"],
        "selection_count": len(meta["selected_geometries"]),
        "polarizations": list(POLS),
        "expected_tasks": EXPECTED_TASKS,
        "configs_1nm": meta["configs_1nm"],
        "fine_config": meta["fine_config"],
        "fine_grid_nm": {"start": 380.0, "stop": 780.0, "step": 0.5, "samples": 801},
        "thresholds": meta["thresholds"],
        "decision_rule": (
            "The elevated reference passes only if 17x17-to-19x19 order, "
            "Nxy384-to-512 grid, and 1.0-to-0.5 nm spectral comparisons all "
            "meet the unchanged joint color thresholds. This result cannot pass "
            "or rehabilitate the failed nG131 production pool."
        ),
        "conditional_escalation": {
            "spectral_failure": (
                "Treat 1 nm spatial comparisons as provisional and add matched 0.5 nm "
                "runs for every compared endpoint before interpreting order or grid stability."
            ),
            "order_failure": "Add requested nG=450, retained 441, at Nxy=512.",
            "grid_failure": "Add requested nG=365, retained 361, at Nxy=768.",
            "order_and_grid_failure": (
                "Add the single-axis endpoints plus requested nG=450 at Nxy=768; "
                "do not replace the factor design with a diagonal-only comparison."
            ),
            "near_threshold_or_nonmonotonic": (
                "Add 0.25 nm only for the anomalous frozen cases before any conclusion."
            ),
            "after_operational_pass": (
                "Expand the frozen protocol to 32 stratified holdout geometries before "
                "declaring a production reference for the full design domain."
            ),
        },
        "estimated_wall_hours_16_workers": {"low": 8, "high": 12},
        "runtime_hashes": meta["runtime_hashes"],
    }


def summarize(meta: dict, results: dict, baseline: dict, checkpoint_path: Path) -> dict:
    selected = meta["selected_geometries"]
    baseline_results = baseline["results"]
    spectra = validate_spectra(results)
    failures = [result for result in results.values() if result.get("status") != "ok"]
    checkpoint_label = (
        str(checkpoint_path.relative_to(ROOT))
        if checkpoint_path.is_relative_to(ROOT)
        else str(checkpoint_path)
    ).replace("\\", "/")
    common = {
        "schema_version": 1,
        "evidence_version": VERSION,
        "pool_sha256": meta["pool_sha256"],
        "spectra": spectra,
        "failures": failures,
        "selection": selected,
        "thresholds": meta["thresholds"],
        "decision_scope": (
            "Diagnostic only. Passing identifies a candidate reference; it does not pass "
            "the failed nG131 production gate or authorize training."
        ),
        "interpretation_guard": (
            "If the 1.0-to-0.5 nm comparison fails, all 1 nm spatial comparisons are "
            "provisional until their endpoints are recomputed on a matched finer grid."
        ),
        "checkpoint": {
            "path": checkpoint_label,
            "sha256": file_digest(checkpoint_path),
            "tasks": len(results),
        },
        "input_evidence": meta["selection_source"],
        "baseline_checkpoint": meta["baseline_checkpoint"],
        "runtime_hashes": meta["runtime_hashes"],
    }
    if failures or len(results) != EXPECTED_TASKS or not spectra["passed"]:
        return {
            **common,
            "passed": False,
            "classification": "execution_or_spectrum_failure",
            "checks": {
                "all_tasks_completed": len(results) == EXPECTED_TASKS,
                "no_task_failures": not failures,
                "spectra_and_conservation_valid": spectra["passed"],
                "reference_order_converged": False,
                "reference_grid_converged": False,
                "reference_spectral_grid_converged": False,
            },
            "comparisons": {},
        }

    def current_id(config: tuple[int, int], step: float):
        return lambda index, pol: task_id(index, pol, config, step)

    def baseline_id(index: int, pol: str) -> str:
        return f"supp-g{index:02d}-{pol}-ng251-nxy384-step1"

    progression_15_to_17 = joint_comparison(
        selected,
        baseline_results,
        results,
        baseline_id,
        current_id(ORDER_17, 1.0),
    )
    order_17_to_19 = joint_comparison(
        selected,
        results,
        results,
        current_id(ORDER_17, 1.0),
        current_id(ORDER_19, 1.0),
    )
    grid_384_to_512 = joint_comparison(
        selected,
        results,
        results,
        current_id(GRID_384, 1.0),
        current_id(ORDER_19, 1.0),
    )
    spectral_1_to_half = joint_comparison(
        selected,
        results,
        results,
        current_id(FINE_CONFIG, 1.0),
        current_id(FINE_CONFIG, 0.5),
    )
    comparisons = {
        "baseline_15x15_to_17x17": progression_15_to_17,
        "order_17x17_to_19x19": order_17_to_19,
        "grid_Nxy384_to_512_at_19x19": grid_384_to_512,
        "spectral_1nm_to_0p5nm_at_19x19_Nxy512": spectral_1_to_half,
    }
    stable_names = (
        "order_17x17_to_19x19",
        "grid_Nxy384_to_512_at_19x19",
        "spectral_1nm_to_0p5nm_at_19x19_Nxy512",
    )
    checks = {
        "all_tasks_completed": len(results) == EXPECTED_TASKS,
        "no_task_failures": not failures,
        "spectra_and_conservation_valid": spectra["passed"],
        "actual_orders_recorded": all(
            int(result.get("retained_nG", -1))
            == retained_order(int(result["requested_nG"]), float(result["geometry"]["P"]))
            for result in results.values()
        ),
        "runtime_hashes_verified": all(
            file_digest(ROOT / path) == digest for path, digest in meta["runtime_hashes"].items()
        ),
        "reference_order_converged": (
            comparisons[stable_names[0]]["mean_lt_1_15"]
            and comparisons[stable_names[0]]["all_lt_2_3"]
        ),
        "reference_grid_converged": (
            comparisons[stable_names[1]]["mean_lt_1_15"]
            and comparisons[stable_names[1]]["all_lt_2_3"]
        ),
        "reference_spectral_grid_converged": (
            comparisons[stable_names[2]]["mean_lt_1_15"]
            and comparisons[stable_names[2]]["all_lt_2_3"]
        ),
    }
    passed = all(checks.values())
    return {
        **common,
        "passed": passed,
        "classification": (
            "elevated_reference_converged" if passed else "reference_requires_further_escalation"
        ),
        "checks": checks,
        "comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-evidence", default=".state/joint_convergence_v1_1.json")
    parser.add_argument("--v1-checkpoint", default=".state/joint_convergence_v1_1_checkpoint.pkl")
    parser.add_argument("--checkpoint", default=".state/reference_resolution_v1_checkpoint.pkl")
    parser.add_argument("--evidence", default=".state/reference_resolution_v1.json")
    parser.add_argument("--plan-output", default=".state/reference_resolution_v1_plan.json")
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    v1_evidence_path = ROOT / args.v1_evidence
    v1_checkpoint_path = ROOT / args.v1_checkpoint
    checkpoint_path = ROOT / args.checkpoint
    evidence_path = ROOT / args.evidence
    plan_path = ROOT / args.plan_output
    meta = build_meta(v1_evidence_path, v1_checkpoint_path)
    plan = build_plan(meta)
    atomic_json(plan_path, plan)
    if args.plan_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    _v1_evidence, baseline = load_inputs(v1_evidence_path, v1_checkpoint_path)
    if checkpoint_path.exists():
        with checkpoint_path.open("rb") as handle:
            checkpoint = pickle.load(handle)
        if checkpoint.get("meta") != meta:
            raise SystemExit("checkpoint protocol mismatch")
    else:
        checkpoint = {"meta": meta, "results": {}}
        atomic_pickle(checkpoint_path, checkpoint)
    tasks = [task for task in build_tasks(meta["selected_geometries"]) if task["id"] not in checkpoint["results"]]
    with Pool(max(1, args.n_jobs)) as workers:
        for result in workers.imap_unordered(run_task, tasks, chunksize=1):
            checkpoint["results"][result["id"]] = result
            atomic_pickle(checkpoint_path, checkpoint)

    evidence = summarize(meta, checkpoint["results"], baseline, checkpoint_path)
    if evidence_path.exists():
        existing = json.loads(evidence_path.read_text(encoding="utf-8"))
        if existing != evidence:
            raise SystemExit("existing reference-resolution evidence differs")
    else:
        atomic_json(evidence_path, evidence)
    print(json.dumps({"passed": evidence["passed"], "checks": evidence["checks"]}, sort_keys=True))
    return 0 if evidence["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
