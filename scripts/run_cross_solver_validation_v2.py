#!/usr/bin/env python3
"""Matched cross-solver gate bound to the activated replacement protocol."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import paper2_colorimetry_fine as colorimetry  # noqa: E402
import pipeline_supervisor as supervisor  # noqa: E402
from color_utils import delta_e2000  # noqa: E402
from scripts import run_cross_solver_validation as legacy  # noqa: E402
from scripts import run_joint_convergence_v2 as joint_v2  # noqa: E402
from scripts import run_replacement_pool as replacement  # noqa: E402
from scripts.run_joint_convergence import retained_order  # noqa: E402


VERSION = "paper2-cross-solver-v2"
WAVELENGTHS_NM = np.arange(380.0, 785.0, 5.0)
POLS = ("p", "s")
GEOMETRY_COUNT = 12
STRESS_GEOMETRY_COUNT = 4
PER_SPECTRUM_RMSE_LIMIT = 0.05
MEAN_SPECTRUM_RMSE_LIMIT = 0.03
MEAN_JOINT_DE00_LIMIT = 1.15
PER_GEOMETRY_JOINT_DE00_LIMIT = 2.3
ENERGY_LIMIT = 1e-6
INVARIANT_LIMIT = 1e-7


def harmonics_for(requested_nG: int, period_nm: float) -> tuple[int, int]:
    retained = retained_order(requested_nG, period_nm)
    side = int(round(math.sqrt(retained)))
    if side * side != retained or side % 2 != 1:
        raise ValueError(f"retained order {retained} is not an odd square")
    return side, side


def load_pool(path: Path) -> tuple[list[dict], list[dict]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    records = payload.get("records", []) if isinstance(payload, dict) else []
    paired = {}
    for record in records:
        key = legacy.geometry_key(record)
        paired.setdefault(key, {})[str(record["pol"])] = record
    geometries = []
    for key in sorted(paired):
        channels = paired[key]
        if set(channels) != set(POLS):
            continue
        L, W, H, P = key
        target = WAVELENGTHS_NM
        sharpness = max(
            float(
                np.max(
                    np.abs(
                        np.diff(
                            np.interp(
                                target,
                                np.asarray(channels[pol]["wl_nm"], dtype=float),
                                np.asarray(channels[pol]["R"], dtype=float),
                            )
                        )
                    )
                )
            )
            for pol in POLS
        )
        geometries.append(
            {
                "L": L,
                "W": W,
                "H": H,
                "P": P,
                "r": max(L, W) / min(L, W),
                "fill": float(np.pi * (L / 2.0) * (W / 2.0) / (P * P)),
                "sharpness_5nm": sharpness,
            }
        )
    return geometries, records


def stress_configs(production_nG: int, production_Nxy: int) -> list[dict]:
    configs = []
    if production_nG < 365:
        configs.append({"name": "order_axis", "nG_requested": 365, "Nxy": production_Nxy})
    if production_Nxy < 512:
        configs.append({"name": "grid_axis", "nG_requested": production_nG, "Nxy": 512})
    if production_nG < 365 or production_Nxy < 512:
        configs.append({"name": "reference_corner", "nG_requested": 365, "Nxy": 512})
    else:
        configs.extend(
            [
                {"name": "order_axis", "nG_requested": 450, "Nxy": production_Nxy},
                {"name": "grid_axis", "nG_requested": production_nG, "Nxy": 768},
                {"name": "higher_corner", "nG_requested": 450, "Nxy": 768},
            ]
        )
    unique = []
    seen = set()
    for config in configs:
        key = (config["nG_requested"], config["Nxy"])
        if key not in seen and key != (production_nG, production_Nxy):
            unique.append(config)
            seen.add(key)
    return unique


def task_id(index: int, pol: str, mode: str) -> str:
    return f"crossv2-g{index:02d}-{pol}-{mode}"


def build_tasks(selected: list[dict], protocol: dict) -> tuple[list[dict], list[dict]]:
    production = {
        "name": "production",
        "nG_requested": int(protocol["nG_requested"]),
        "Nxy": int(protocol["Nxy"]),
    }
    stress = stress_configs(production["nG_requested"], production["Nxy"])
    tasks = []
    for index, geometry in enumerate(selected):
        for pol in POLS:
            tasks.append(
                {
                    "id": task_id(index, pol, production["name"]),
                    "mode": production["name"],
                    "geometry_index": index,
                    "geometry": geometry,
                    "pol": pol,
                    **production,
                }
            )
            if geometry.get("stress"):
                for config in stress:
                    tasks.append(
                        {
                            "id": task_id(index, pol, config["name"]),
                            "mode": config["name"],
                            "geometry_index": index,
                            "geometry": geometry,
                            "pol": pol,
                            **config,
                        }
                    )
    return tasks, stress


def solve_spectrum(solver: str, geometry: dict, pol: str, nG_requested: int, Nxy: int):
    harmonics = harmonics_for(nG_requested, geometry["P"])
    reflection = []
    transmission = []
    for wavelength in WAVELENGTHS_NM:
        if solver == "grcwa":
            R, T = legacy.solve_grcwa_point(
                geometry, pol, wavelength, nG_requested, Nxy
            )
        elif solver == "thirdparty":
            R, T = legacy.solve_thirdparty_point(
                geometry, pol, wavelength, harmonics, Nxy
            )
        else:
            raise ValueError(f"unsupported solver: {solver}")
        reflection.append(R)
        transmission.append(T)
    return np.asarray(reflection, dtype=float), np.asarray(transmission, dtype=float)


def run_task(task: dict) -> dict:
    started = time.perf_counter()
    try:
        grcwa_R, grcwa_T = solve_spectrum(
            "grcwa", task["geometry"], task["pol"], task["nG_requested"], task["Nxy"]
        )
        third_R, third_T = solve_spectrum(
            "thirdparty", task["geometry"], task["pol"], task["nG_requested"], task["Nxy"]
        )
        return {
            **task,
            "status": "ok",
            "wavelength_nm": WAVELENGTHS_NM,
            "retained_nG": retained_order(task["nG_requested"], task["geometry"]["P"]),
            "harmonics": harmonics_for(task["nG_requested"], task["geometry"]["P"]),
            "grcwa_R": grcwa_R,
            "grcwa_T": grcwa_T,
            "thirdparty_R": third_R,
            "thirdparty_T": third_T,
            "time_s": time.perf_counter() - started,
        }
    except Exception as exc:
        return {
            **task,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "time_s": time.perf_counter() - started,
        }


def validate_checkpoint_results(
    checkpoint: dict, tasks: list[dict], *, require_complete: bool = False
) -> None:
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("results"), dict):
        raise ValueError("cross-solver v2 checkpoint lacks a results object")
    expected = {task["id"]: task for task in tasks}
    results = checkpoint["results"]
    if not set(results) <= set(expected):
        raise ValueError("cross-solver v2 checkpoint contains an unknown task id")
    if require_complete and set(results) != set(expected):
        raise ValueError("cross-solver v2 checkpoint task set is incomplete")
    scientific_keys = set()
    for key, result in results.items():
        task = expected[key]
        if not isinstance(result, dict) or result.get("id") != key:
            raise ValueError("cross-solver v2 checkpoint key differs from its internal id")
        for field in ("mode", "geometry_index", "geometry", "pol", "nG_requested", "Nxy"):
            if result.get(field) != task[field]:
                raise ValueError(f"cross-solver v2 checkpoint task field differs: {field}")
        identity = (result["geometry_index"], result["pol"], result["mode"])
        if identity in scientific_keys:
            raise ValueError("cross-solver v2 checkpoint contains a duplicate scientific task")
        scientific_keys.add(identity)
        if result.get("status") == "failed":
            if not isinstance(result.get("error"), str) or not result["error"]:
                raise ValueError("cross-solver v2 failed task lacks an error ledger")
            continue
        if result.get("status") != "ok":
            raise ValueError("cross-solver v2 checkpoint task status is invalid")
        wavelength = np.asarray(result.get("wavelength_nm"), dtype=float)
        if wavelength.shape != WAVELENGTHS_NM.shape or not np.array_equal(wavelength, WAVELENGTHS_NM):
            raise ValueError("cross-solver v2 checkpoint wavelength grid differs")
        for field in ("grcwa_R", "grcwa_T", "thirdparty_R", "thirdparty_T"):
            values = np.asarray(result.get(field), dtype=float)
            if (
                values.shape != WAVELENGTHS_NM.shape
                or not np.isfinite(values).all()
                or np.any(values < -1e-8)
                or np.any(values > 1.0 + 1e-8)
            ):
                raise ValueError(f"cross-solver v2 checkpoint spectrum is invalid: {field}")


def spectrum_metrics(left_R, left_T, right_R, right_T) -> dict:
    arrays = [np.asarray(value, dtype=float) for value in (left_R, left_T, right_R, right_T)]
    valid = all(
        value.shape == WAVELENGTHS_NM.shape
        and np.isfinite(value).all()
        and np.all(value >= -1e-8)
        and np.all(value <= 1.0 + 1e-8)
        for value in arrays
    )
    if not valid:
        return {"valid": False}
    left_R, left_T, right_R, right_T = arrays
    labels_left = colorimetry.spectrum_to_labels_d65(left_R, WAVELENGTHS_NM)
    labels_right = colorimetry.spectrum_to_labels_d65(right_R, WAVELENGTHS_NM)
    return {
        "valid": True,
        "R_rmse": float(np.sqrt(np.mean((left_R - right_R) ** 2))),
        "T_rmse": float(np.sqrt(np.mean((left_T - right_T) ** 2))),
        "R_max": float(np.max(np.abs(left_R - right_R))),
        "T_max": float(np.max(np.abs(left_T - right_T))),
        "dE00": float(delta_e2000(labels_left["lab"], labels_right["lab"])),
        "left_energy_error_max": float(np.max(np.abs(left_R + left_T - 1.0))),
        "right_energy_error_max": float(np.max(np.abs(right_R + right_T - 1.0))),
    }


def joint_summary(rows: list[dict], expected_geometries: int) -> dict:
    grouped = {}
    for row in rows:
        if row.get("valid"):
            grouped.setdefault(int(row["geometry_index"]), {})[row["pol"]] = row["dE00"]
    joint = [max(channels.values()) for channels in grouped.values() if set(channels) == set(POLS)]
    values = np.asarray(joint, dtype=float)
    return {
        "complete_geometries": len(joint),
        "expected_geometries": expected_geometries,
        "values": joint,
        "mean": float(np.mean(values)) if values.size else None,
        "max": float(np.max(values)) if values.size else None,
        "mean_lt_1_15": bool(values.size == expected_geometries and np.mean(values) < MEAN_JOINT_DE00_LIMIT),
        "all_lt_2_3": bool(values.size == expected_geometries and np.all(values < PER_GEOMETRY_JOINT_DE00_LIMIT)),
    }


def comparison_gate(rows: list[dict], expected_geometries: int) -> dict:
    valid = [row for row in rows if row.get("valid")]
    joint = joint_summary(valid, expected_geometries)
    R_rmse = np.asarray([row["R_rmse"] for row in valid], dtype=float)
    T_rmse = np.asarray([row["T_rmse"] for row in valid], dtype=float)
    checks = {
        "all_spectra_valid": len(valid) == expected_geometries * 2,
        "per_spectrum_R_rmse": bool(R_rmse.size and np.all(R_rmse <= PER_SPECTRUM_RMSE_LIMIT)),
        "per_spectrum_T_rmse": bool(T_rmse.size and np.all(T_rmse <= PER_SPECTRUM_RMSE_LIMIT)),
        "mean_R_rmse": bool(R_rmse.size and np.mean(R_rmse) <= MEAN_SPECTRUM_RMSE_LIMIT),
        "mean_T_rmse": bool(T_rmse.size and np.mean(T_rmse) <= MEAN_SPECTRUM_RMSE_LIMIT),
        "joint_mean": joint["mean_lt_1_15"],
        "joint_all": joint["all_lt_2_3"],
        "left_energy": bool(valid and all(row["left_energy_error_max"] <= ENERGY_LIMIT for row in valid)),
        "right_energy": bool(valid and all(row["right_energy_error_max"] <= ENERGY_LIMIT for row in valid)),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "joint_dE00": joint,
        "R_rmse_mean": float(np.mean(R_rmse)) if R_rmse.size else None,
        "T_rmse_mean": float(np.mean(T_rmse)) if T_rmse.size else None,
        "rows": rows,
    }


def evaluate_results(results: dict, stress: list[dict]) -> dict:
    failures = [item for item in results.values() if item.get("status") != "ok"]
    by_key = {
        (item["geometry_index"], item["pol"], item["mode"]): item
        for item in results.values()
        if item.get("status") == "ok"
    }
    cross_solver = {}
    convergence = {}
    modes = ["production"] + [item["name"] for item in stress]
    for mode in modes:
        rows = []
        for (index, pol, current_mode), result in by_key.items():
            if current_mode == mode:
                rows.append(
                    {
                        "geometry_index": index,
                        "pol": pol,
                        **spectrum_metrics(
                            result["grcwa_R"], result["grcwa_T"],
                            result["thirdparty_R"], result["thirdparty_T"],
                        ),
                    }
                )
        expected = GEOMETRY_COUNT if mode == "production" else STRESS_GEOMETRY_COUNT
        cross_solver[mode] = comparison_gate(rows, expected)
    for config in stress:
        mode = config["name"]
        for solver in ("grcwa", "thirdparty"):
            rows = []
            for index in range(GEOMETRY_COUNT):
                for pol in POLS:
                    base_result = by_key.get((index, pol, "production"))
                    high_result = by_key.get((index, pol, mode))
                    if base_result is None or high_result is None:
                        continue
                    rows.append(
                        {
                            "geometry_index": index,
                            "pol": pol,
                            **spectrum_metrics(
                                base_result[f"{solver}_R"], base_result[f"{solver}_T"],
                                high_result[f"{solver}_R"], high_result[f"{solver}_T"],
                            ),
                        }
                    )
            summary = joint_summary(rows, STRESS_GEOMETRY_COUNT)
            convergence[f"{solver}_{mode}"] = {
                "passed": summary["mean_lt_1_15"] and summary["all_lt_2_3"],
                "joint_dE00": summary,
                "rows": rows,
            }
    checks = {
        "no_task_failures": not failures,
        "production_cross_solver": cross_solver.get("production", {}).get("passed") is True,
        "all_stress_cross_solver": all(
            cross_solver.get(item["name"], {}).get("passed") is True for item in stress
        ),
        "both_solvers_converged": all(item.get("passed") for item in convergence.values()),
    }
    if failures:
        classification = "thirdparty_or_runtime_unavailable"
    elif not checks["both_solvers_converged"]:
        classification = "solver_self_convergence_failed"
    elif not checks["production_cross_solver"] or not checks["all_stress_cross_solver"]:
        classification = "converged_cross_solver_divergence"
    else:
        classification = "passed"
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "classification": classification,
        "cross_solver": cross_solver,
        "convergence": convergence,
        "failures": failures,
    }


def run_controls(production_nG: int, production_Nxy: int) -> dict:
    wavelength = 550.0
    circle = {"L": 200.0, "W": 200.0, "H": 300.0, "P": 400.0}
    ellipse = {"L": 240.0, "W": 120.0, "H": 350.0, "P": 450.0}
    rotated = {"L": 120.0, "W": 240.0, "H": 350.0, "P": 450.0}
    harmonics = harmonics_for(production_nG, circle["P"])
    controls = {}
    for solver in ("grcwa", "thirdparty"):
        def point(geometry, pol, empty=False):
            if solver == "grcwa":
                return legacy.solve_grcwa_point(
                    geometry, pol, wavelength, production_nG, production_Nxy, empty=empty
                )
            return legacy.solve_thirdparty_point(
                geometry, pol, wavelength, harmonics, production_Nxy, empty=empty
            )

        empty = point(circle, "p", empty=True)
        circle_p = point(circle, "p")
        circle_s = point(circle, "s")
        ellipse_p = point(ellipse, "p")
        rotated_s = point(rotated, "s")
        n_substrate = float(legacy.rcwa_batch.n_cauchy(wavelength, "SiO2"))
        analytic = float(((1.0 - n_substrate) / (1.0 + n_substrate)) ** 2)
        controls[solver] = {
            "fresnel_error": abs(empty[0] - analytic),
            "empty_energy_error": abs(sum(empty) - 1.0),
            "circle_max_difference": max(abs(circle_p[0] - circle_s[0]), abs(circle_p[1] - circle_s[1])),
            "rotation_max_difference": max(abs(ellipse_p[0] - rotated_s[0]), abs(ellipse_p[1] - rotated_s[1])),
        }
    checks = {
        f"{solver}_{name}": value <= (ENERGY_LIMIT if name == "empty_energy_error" else INVARIANT_LIMIT)
        for solver, values in controls.items()
        for name, value in values.items()
    }
    return {"passed": all(checks.values()), "checks": checks, "values": controls}


def joint_v2_ready(context: dict) -> bool:
    gates, _ = supervisor.verify_gate_evidence(context["policy"], context["pool_audit"])
    return gates.get("joint_numerical_convergence") is True


def summarize(context: dict, meta: dict, checkpoint: dict, checkpoint_path: Path) -> dict:
    evaluation = evaluate_results(checkpoint["results"], meta["stress_configs"])
    controls = run_controls(meta["production"]["nG_requested"], meta["production"]["Nxy"])
    runtime_ok = all(
        supervisor.file_digest(ROOT / path) == expected
        for path, expected in meta["runtime_hashes"].items()
    )
    protected = supervisor.audit_protected_files(context["policy"])
    checks = {
        "controls": controls["passed"],
        "matched_results": evaluation["passed"],
        "runtime_hashes_verified": runtime_ok,
        "paper1_and_legacy_assets_unchanged": all(item.get("passed") for item in protected),
    }
    classification = evaluation["classification"]
    if not controls["passed"]:
        classification = "implementation_control_failure"
    elif not runtime_ok:
        classification = "runtime_provenance_failure"
    return {
        "schema_version": 1,
        "evidence_version": VERSION,
        "passed": all(checks.values()),
        "pool_sha256": context["pool_sha256"],
        "checks": checks,
        "classification": classification,
        "thresholds": meta["thresholds"],
        "protocol": {
            "production": meta["production"],
            "stress_configs": meta["stress_configs"],
            "geometry_count": GEOMETRY_COUNT,
            "stress_geometry_count": STRESS_GEOMETRY_COUNT,
            "polarizations": list(POLS),
            "wavelength_nm": WAVELENGTHS_NM.tolist(),
            "background": "air",
            "incident": "air",
            "transmission_halfspace": "SiO2",
        },
        "selected_geometries": meta["selected_geometries"],
        "controls": controls,
        "evaluation": evaluation,
        "raw_checkpoint": {
            "path": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": supervisor.file_digest(checkpoint_path),
            "tasks": len(checkpoint["results"]),
        },
        "active_pool": {
            "path": str(context["active_path"].relative_to(ROOT)).replace("\\", "/"),
            "sha256": supervisor.file_digest(context["active_path"]),
        },
        "approved_protocol": {
            "path": str(context["protocol_path"].relative_to(ROOT)).replace("\\", "/"),
            "sha256": supervisor.file_digest(context["protocol_path"]),
        },
        "runtime_hashes": meta["runtime_hashes"],
        "packages": meta["packages"],
        "protected_files": protected,
        "training_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active", default=".state/active_pool.json")
    parser.add_argument("--checkpoint", default=".state/cross_solver_v2_checkpoint.pkl")
    parser.add_argument("--evidence", default=".state/cross_solver_v2.json")
    parser.add_argument("--lock", default=".state/cross_solver_v2.lock")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    context = joint_v2.load_active_context(ROOT / args.active)
    geometries, _records = load_pool(context["pool_path"])
    selected = legacy.select_cross_solver_geometries(geometries)
    tasks, stress = build_tasks(selected, context["protocol"])
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
    meta = {
        "version": VERSION,
        "pool_sha256": context["pool_sha256"],
        "approved_protocol_sha256": supervisor.file_digest(context["protocol_path"]),
        "selected_geometries": selected,
        "production": production,
        "stress_configs": stress,
        "expected_tasks": len(tasks),
        "runtime_hashes": {path: supervisor.file_digest(ROOT / path) for path in runtime_paths},
        "packages": {
            "rcwa": importlib.metadata.version("rcwa"),
            "grcwa": importlib.metadata.version("grcwa"),
        },
        "thresholds": {
            "per_spectrum_R_T_rmse_lte": PER_SPECTRUM_RMSE_LIMIT,
            "mean_spectrum_R_T_rmse_lte": MEAN_SPECTRUM_RMSE_LIMIT,
            "mean_joint_dE00_lt": MEAN_JOINT_DE00_LIMIT,
            "per_geometry_joint_dE00_lt": PER_GEOMETRY_JOINT_DE00_LIMIT,
            "energy_error_lte": ENERGY_LIMIT,
            "analytic_and_symmetry_error_lte": INVARIANT_LIMIT,
        },
    }
    if args.plan_only:
        print(json.dumps(meta, indent=2, sort_keys=True))
        return 0
    if not joint_v2_ready(context):
        raise SystemExit("joint numerical convergence v2 is not registered for the active pool")
    checkpoint_path = ROOT / args.checkpoint
    evidence_path = ROOT / args.evidence
    with replacement.RunLock(ROOT / args.lock, context["pool_sha256"]):
        if checkpoint_path.exists():
            with checkpoint_path.open("rb") as handle:
                checkpoint = pickle.load(handle)
            if checkpoint.get("meta") != meta:
                raise SystemExit("cross-solver v2 checkpoint protocol mismatch")
            try:
                validate_checkpoint_results(checkpoint, tasks)
            except (TypeError, ValueError) as exc:
                raise SystemExit(str(exc)) from exc
        else:
            checkpoint = {"meta": meta, "results": {}}
            legacy.atomic_pickle(checkpoint_path, checkpoint)
        pending = [task for task in tasks if task["id"] not in checkpoint["results"]]
        with Pool(max(1, args.n_jobs)) as workers:
            for result in workers.imap_unordered(run_task, pending, chunksize=1):
                checkpoint["results"][result["id"]] = result
                legacy.atomic_pickle(checkpoint_path, checkpoint)
        try:
            validate_checkpoint_results(checkpoint, tasks, require_complete=True)
        except (TypeError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        evidence = summarize(context, meta, checkpoint, checkpoint_path)
        if evidence_path.exists():
            existing = json.loads(evidence_path.read_text(encoding="utf-8"))
            if existing != evidence:
                raise SystemExit("existing cross-solver v2 evidence differs")
        else:
            supervisor.atomic_json(evidence_path, evidence)
    print(json.dumps({"passed": evidence["passed"], "classification": evidence["classification"]}))
    return 0 if evidence["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
