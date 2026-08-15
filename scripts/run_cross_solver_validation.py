#!/usr/bin/env python3
"""Pre-registered matched cross-solver spectrum gate for paper 2.

The production side is the immutable grcwa pool at 11x11/Nxy256. The
independent side is rcwa 1.0.48 with the same epsilon grid, material model,
half-spaces, and Fourier rectangle. Four stress geometries also compare both
solvers at 13x13/Nxy384 so solver disagreement is not confused with lack of
numerical convergence.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import progressbar


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

progressbar.ProgressBar = lambda *args, **kwargs: type(
    "NoBar",
    (),
    {"start": lambda self: self, "update": lambda self, value: None, "finish": lambda self: None},
)()

from grcwa.rcwa import obj as GrcwaObject  # noqa: E402
from rcwa import Crystal, Layer, LayerStack, Solver, Source  # noqa: E402
from rcwa.shorthand import complexArray  # noqa: E402

import rcwa_batch  # noqa: E402
from color_utils import delta_e2000  # noqa: E402
from paper2_colorimetry import spectrum_to_labels_d65  # noqa: E402
from pipeline_supervisor import atomic_json, file_digest, load_json  # noqa: E402
from scripts.run_joint_convergence import retained_order  # noqa: E402


VERSION = "paper2-cross-solver-v1"
POOL_PATH = "data/rcwa_ellip_TiO2_3000_air.pkl"
WAVELENGTHS_NM = np.arange(380.0, 785.0, 5.0)
POLARIZATIONS = ("p", "s")
GEOMETRY_COUNT = 12
BASE_NG_REQUESTED = 131
BASE_HARMONICS = (11, 11)
BASE_NXY = 256
STRESS_NG_REQUESTED = 201
STRESS_HARMONICS = (13, 13)
STRESS_NXY = 384
STRESS_TAGS = {
    "global_sharpness_1", "global_sharpness_2", "high_aspect", "high_fill"
}

# Frozen before observing matched cross-solver results.
PER_SPECTRUM_RMSE_LIMIT = 0.05
MEAN_SPECTRUM_RMSE_LIMIT = 0.03
MEAN_JOINT_DE00_LIMIT = 1.15
PER_GEOMETRY_JOINT_DE00_LIMIT = 2.3
ENERGY_LIMIT = 1e-6
INVARIANT_LIMIT = 1e-7


def geometry_key(record):
    return tuple(float(record[name]) for name in ("L", "W", "H", "P"))


def pool_geometries(records):
    paired = {}
    for record in records:
        paired.setdefault(geometry_key(record), {})[str(record["pol"])] = record
    geometries = []
    for key in sorted(paired):
        channels = paired[key]
        if set(channels) != set(POLARIZATIONS):
            continue
        L, W, H, P = key
        geometries.append(
            {
                "L": L,
                "W": W,
                "H": H,
                "P": P,
                "r": max(L, W) / min(L, W),
                "fill": float(np.pi * (L / 2.0) * (W / 2.0) / P**2),
                "sharpness_5nm": max(
                    float(np.max(np.abs(np.diff(np.asarray(channels[pol]["R"], float)))))
                    for pol in POLARIZATIONS
                ),
            }
        )
    return geometries, paired


def _maximin_pick(candidates, selected):
    names = ("L", "W", "H", "P", "r", "fill", "sharpness_5nm")
    universe = list(candidates) + list(selected)
    features = np.asarray([[item[name] for name in names] for item in universe], float)
    spans = np.ptp(features, axis=0)
    spans[spans == 0.0] = 1.0
    normalized = (features - np.mean(features, axis=0)) / spans
    candidate_count = len(candidates)
    if not selected:
        index = int(np.argmax(np.linalg.norm(normalized[:candidate_count], axis=1)))
        return candidates[index]
    distances = np.linalg.norm(
        normalized[:candidate_count, None, :] - normalized[candidate_count:][None, :, :],
        axis=2,
    )
    return candidates[int(np.argmax(np.min(distances, axis=1)))]


def select_cross_solver_geometries(geometries, count=GEOMETRY_COUNT):
    if len(geometries) < count or count != 12:
        raise ValueError("cross-solver v1 requires at least 12 geometries and count=12")
    ranked = sorted(
        geometries,
        key=lambda item: (
            item["sharpness_5nm"], item["L"], item["W"], item["H"], item["P"]
        ),
    )
    strata = [list(group) for group in np.array_split(np.asarray(ranked, dtype=object), 4)]
    stratum_by_key = {
        geometry_key(item): index for index, group in enumerate(strata) for item in group
    }
    selected = []
    used = set()

    def add(tag, geometry):
        key = geometry_key(geometry)
        if key in used:
            return
        item = dict(geometry)
        item["selection"] = tag
        item["sharpness_stratum"] = int(stratum_by_key[key])
        item["stress"] = tag in STRESS_TAGS
        selected.append(item)
        used.add(key)

    descending_sharp = list(reversed(ranked))
    add("global_sharpness_1", descending_sharp[0])
    add("global_sharpness_2", descending_sharp[1])
    add("near_circle", min(geometries, key=lambda item: (abs(item["r"] - 1.0), geometry_key(item))))
    add("high_aspect", max(geometries, key=lambda item: (item["r"], tuple(-x for x in geometry_key(item)))))
    add("high_fill", max(geometries, key=lambda item: (item["fill"], tuple(-x for x in geometry_key(item)))))

    for stratum_index, group in enumerate(strata):
        while sum(item["sharpness_stratum"] == stratum_index for item in selected) < 3:
            candidates = [item for item in group if geometry_key(item) not in used]
            if not candidates:
                raise ValueError(f"unable to fill sharpness stratum {stratum_index}")
            pick = _maximin_pick(candidates, selected)
            add(f"stratum_{stratum_index}_maximin", pick)
    if len(selected) != count:
        raise ValueError(f"forced cases violate 3-per-stratum plan: selected {len(selected)}")
    if sum(bool(item["stress"]) for item in selected) != 4:
        raise ValueError("stress subset must contain exactly four distinct geometries")
    return selected


def polarization_vector(pol):
    if pol == "s":
        return complexArray([1.0, 0.0])
    if pol == "p":
        return complexArray([0.0, 1.0])
    raise ValueError(f"unsupported polarization: {pol}")


def epsilon_grid(geometry, wavelength_nm, nxy, empty=False):
    axis = np.linspace(-geometry["P"] / 2.0, geometry["P"] / 2.0, nxy, endpoint=False)
    x_grid, y_grid = np.meshgrid(axis, axis)
    if empty:
        epsilon = np.ones((nxy, nxy), float)
    else:
        mask = (
            x_grid**2 / (geometry["L"] / 2.0) ** 2
            + y_grid**2 / (geometry["W"] / 2.0) ** 2
            <= 1.0
        )
        n_pillar = rcwa_batch.n_complex(wavelength_nm, "TiO2")
        epsilon = np.where(mask, n_pillar**2, 1.0)
    n_substrate = float(rcwa_batch.n_cauchy(wavelength_nm, "SiO2"))
    return epsilon, n_substrate


def solve_grcwa_point(geometry, pol, wavelength_nm, nG_requested, nxy, empty=False):
    epsilon, n_substrate = epsilon_grid(geometry, wavelength_nm, nxy, empty=empty)
    wavelength_um = wavelength_nm / 1000.0
    period_um = geometry["P"] / 1000.0
    block = GrcwaObject(
        int(nG_requested), [period_um, 0.0], [0.0, period_um],
        1.0 / wavelength_um, 0.0, 0.0, verbose=0,
    )
    block.Add_LayerUniform(0.0, 1.0)
    block.Add_LayerGrid(geometry["H"] / 1000.0, nxy, nxy)
    block.Add_LayerUniform(0.0, n_substrate**2)
    block.Init_Setup(Gmethod=1)
    block.GridLayer_geteps(np.asarray(epsilon).ravel())
    if pol == "s":
        block.MakeExcitationPlanewave(0.0, 0.0, 1.0, 0.0)
    elif pol == "p":
        block.MakeExcitationPlanewave(1.0, 0.0, 0.0, 0.0)
    else:
        raise ValueError(f"unsupported polarization: {pol}")
    reflection, transmission = block.RT_Solve(normalize=1)
    return float(np.real(reflection)), float(np.real(transmission))


def solve_thirdparty_point(geometry, pol, wavelength_nm, harmonics, nxy, empty=False):
    epsilon, n_substrate = epsilon_grid(geometry, wavelength_nm, nxy, empty=empty)
    crystal = Crystal(
        complexArray([geometry["P"], 0.0, 0.0]),
        complexArray([0.0, geometry["P"], 0.0]),
        er=epsilon,
        ur=np.ones_like(epsilon),
    )
    incident = Layer(er=1.0, ur=1.0)
    substrate = Layer(er=n_substrate**2, ur=1.0)
    patterned = Layer(crystal=crystal, thickness=geometry["H"])
    source = Source(
        wavelength=wavelength_nm, theta=0.0, phi=0.0,
        pTEM=polarization_vector(pol), layer=incident,
    )
    stack = LayerStack(patterned, incident_layer=incident, transmission_layer=substrate)
    result = Solver(stack, source, harmonics).solve(check_convergence=False)
    return float(np.real(result["RTot"])), float(np.real(result["TTot"]))


def solve_spectrum(solver, geometry, pol, nG_requested, harmonics, nxy):
    reflection = []
    transmission = []
    for wavelength_nm in WAVELENGTHS_NM:
        if solver == "grcwa":
            r_value, t_value = solve_grcwa_point(
                geometry, pol, wavelength_nm, nG_requested, nxy
            )
        elif solver == "thirdparty":
            r_value, t_value = solve_thirdparty_point(
                geometry, pol, wavelength_nm, harmonics, nxy
            )
        else:
            raise ValueError(f"unsupported solver: {solver}")
        reflection.append(r_value)
        transmission.append(t_value)
    return np.asarray(reflection, float), np.asarray(transmission, float)


def task_id(index, pol, mode):
    return f"cross-g{index:02d}-{pol}-{mode}"


def build_tasks(selected, paired):
    tasks = []
    for index, geometry in enumerate(selected):
        channels = paired[geometry_key(geometry)]
        for pol in POLARIZATIONS:
            record = channels[pol]
            tasks.append(
                {
                    "id": task_id(index, pol, "base"), "mode": "base",
                    "geometry_index": index, "geometry": geometry, "pol": pol,
                    "base_grcwa_R": np.asarray(record["R"], float),
                    "base_grcwa_T": np.asarray(record["T"], float),
                }
            )
            if geometry["stress"]:
                tasks.append(
                    {
                        "id": task_id(index, pol, "stress"), "mode": "stress",
                        "geometry_index": index, "geometry": geometry, "pol": pol,
                    }
                )
    return tasks


def run_task(task):
    started = time.perf_counter()
    try:
        if task["mode"] == "base":
            third_R, third_T = solve_spectrum(
                "thirdparty", task["geometry"], task["pol"],
                BASE_NG_REQUESTED, BASE_HARMONICS, BASE_NXY,
            )
            return {
                **task, "status": "ok", "base_thirdparty_R": third_R,
                "base_thirdparty_T": third_T, "time_s": time.perf_counter() - started,
            }
        high_grcwa_R, high_grcwa_T = solve_spectrum(
            "grcwa", task["geometry"], task["pol"],
            STRESS_NG_REQUESTED, STRESS_HARMONICS, STRESS_NXY,
        )
        high_third_R, high_third_T = solve_spectrum(
            "thirdparty", task["geometry"], task["pol"],
            STRESS_NG_REQUESTED, STRESS_HARMONICS, STRESS_NXY,
        )
        return {
            **task, "status": "ok", "high_grcwa_R": high_grcwa_R,
            "high_grcwa_T": high_grcwa_T, "high_thirdparty_R": high_third_R,
            "high_thirdparty_T": high_third_T, "time_s": time.perf_counter() - started,
        }
    except Exception as exc:
        return {
            **task, "status": "failed", "error": f"{type(exc).__name__}: {exc}",
            "time_s": time.perf_counter() - started,
        }


def atomic_pickle(path, payload):
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with tmp.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(10):
            try:
                os.replace(tmp, path)
                return
            except OSError as exc:
                transient = getattr(exc, "winerror", None) in {5, 32} or exc.errno in {13, 16}
                if not transient or attempt == 9:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def spectrum_metrics(left_R, left_T, right_R, right_T):
    arrays = [np.asarray(value, float) for value in (left_R, left_T, right_R, right_T)]
    valid = all(
        value.shape == (81,) and np.isfinite(value).all()
        and np.all((value >= -1e-8) & (value <= 1.0 + 1e-8))
        for value in arrays
    )
    if not valid:
        return {"valid": False}
    left_R, left_T, right_R, right_T = arrays
    error_R = left_R - right_R
    error_T = left_T - right_T
    return {
        "valid": True,
        "R_rmse": float(np.sqrt(np.mean(error_R**2))),
        "T_rmse": float(np.sqrt(np.mean(error_T**2))),
        "R_mae": float(np.mean(np.abs(error_R))),
        "T_mae": float(np.mean(np.abs(error_T))),
        "R_max": float(np.max(np.abs(error_R))),
        "T_max": float(np.max(np.abs(error_T))),
        "dE00": float(delta_e2000(
            spectrum_to_labels_d65(left_R)["lab"],
            spectrum_to_labels_d65(right_R)["lab"],
        )),
        "left_energy_error_max": float(np.max(np.abs(left_R + left_T - 1.0))),
        "right_energy_error_max": float(np.max(np.abs(right_R + right_T - 1.0))),
    }


def joint_summary(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(int(row["geometry_index"]), {})[row["pol"]] = float(row["dE00"])
    joint = [
        max(channels[pol] for pol in POLARIZATIONS)
        for channels in grouped.values() if set(channels) == set(POLARIZATIONS)
    ]
    values = np.asarray(joint, float)
    return {
        "complete_geometries": len(joint), "values": joint,
        "mean": float(np.mean(values)) if values.size else None,
        "max": float(np.max(values)) if values.size else None,
        "mean_lt_1_15": bool(values.size and np.mean(values) < MEAN_JOINT_DE00_LIMIT),
        "all_lt_2_3": bool(values.size and np.all(values < PER_GEOMETRY_JOINT_DE00_LIMIT)),
    }


def comparison_gate(rows, expected_tasks):
    valid_rows = [row for row in rows if row.get("valid")]
    joint = joint_summary(valid_rows)
    rmse_R = np.asarray([row["R_rmse"] for row in valid_rows], float)
    rmse_T = np.asarray([row["T_rmse"] for row in valid_rows], float)
    checks = {
        "all_spectra_valid": len(valid_rows) == expected_tasks,
        "per_spectrum_R_rmse": bool(rmse_R.size and np.all(rmse_R <= PER_SPECTRUM_RMSE_LIMIT)),
        "per_spectrum_T_rmse": bool(rmse_T.size and np.all(rmse_T <= PER_SPECTRUM_RMSE_LIMIT)),
        "mean_R_rmse": bool(rmse_R.size and np.mean(rmse_R) <= MEAN_SPECTRUM_RMSE_LIMIT),
        "mean_T_rmse": bool(rmse_T.size and np.mean(rmse_T) <= MEAN_SPECTRUM_RMSE_LIMIT),
        "complete_joint_geometries": joint["complete_geometries"] * 2 == expected_tasks,
        "mean_joint_dE00": joint["mean_lt_1_15"],
        "all_joint_dE00": joint["all_lt_2_3"],
        "left_energy": bool(valid_rows and all(row["left_energy_error_max"] <= ENERGY_LIMIT for row in valid_rows)),
        "right_energy": bool(valid_rows and all(row["right_energy_error_max"] <= ENERGY_LIMIT for row in valid_rows)),
    }
    return {
        "passed": all(checks.values()), "checks": checks, "joint_dE00": joint,
        "R_rmse_mean": float(np.mean(rmse_R)) if rmse_R.size else None,
        "T_rmse_mean": float(np.mean(rmse_T)) if rmse_T.size else None,
        "R_rmse_max": float(np.max(rmse_R)) if rmse_R.size else None,
        "T_rmse_max": float(np.max(rmse_T)) if rmse_T.size else None,
        "rows": rows,
    }


def evaluate_results(results, expected_tasks, stress_geometry_count=4):
    failures = [item for item in results.values() if item.get("status") != "ok"]
    base_rows = []
    high_rows = []
    grcwa_convergence_rows = []
    thirdparty_convergence_rows = []
    base_by_pair = {}
    for result in results.values():
        if result.get("status") == "ok" and result["mode"] == "base":
            base_by_pair[(result["geometry_index"], result["pol"])] = result
            base_rows.append({
                "geometry_index": result["geometry_index"], "pol": result["pol"],
                **spectrum_metrics(
                    result["base_grcwa_R"], result["base_grcwa_T"],
                    result["base_thirdparty_R"], result["base_thirdparty_T"],
                ),
            })
    for result in results.values():
        if result.get("status") != "ok" or result["mode"] != "stress":
            continue
        base = base_by_pair.get((result["geometry_index"], result["pol"]))
        if not base:
            continue
        common = {"geometry_index": result["geometry_index"], "pol": result["pol"]}
        high_rows.append({
            **common,
            **spectrum_metrics(
                result["high_grcwa_R"], result["high_grcwa_T"],
                result["high_thirdparty_R"], result["high_thirdparty_T"],
            ),
        })
        grcwa_convergence_rows.append({
            **common,
            **spectrum_metrics(
                base["base_grcwa_R"], base["base_grcwa_T"],
                result["high_grcwa_R"], result["high_grcwa_T"],
            ),
        })
        thirdparty_convergence_rows.append({
            **common,
            **spectrum_metrics(
                base["base_thirdparty_R"], base["base_thirdparty_T"],
                result["high_thirdparty_R"], result["high_thirdparty_T"],
            ),
        })

    base_gate = comparison_gate(base_rows, GEOMETRY_COUNT * 2)
    high_gate = comparison_gate(high_rows, stress_geometry_count * 2)

    def convergence_gate(rows):
        valid_rows = [row for row in rows if row.get("valid")]
        joint = joint_summary(valid_rows)
        return {
            "passed": len(valid_rows) == stress_geometry_count * 2
            and joint["complete_geometries"] == stress_geometry_count
            and joint["mean_lt_1_15"] and joint["all_lt_2_3"],
            "joint_dE00": joint, "rows": rows,
        }

    grcwa_convergence = convergence_gate(grcwa_convergence_rows)
    thirdparty_convergence = convergence_gate(thirdparty_convergence_rows)
    checks = {
        "all_tasks_completed": len(results) == expected_tasks,
        "no_task_failures": not failures,
        "base_cross_solver": base_gate["passed"],
        "grcwa_stress_converged": grcwa_convergence["passed"],
        "thirdparty_stress_converged": thirdparty_convergence["passed"],
        "high_order_cross_solver": high_gate["passed"],
    }
    if failures:
        classification = "thirdparty_or_runtime_unavailable"
    elif not grcwa_convergence["passed"] or not thirdparty_convergence["passed"]:
        classification = "uncertain_solver_not_converged"
    elif not base_gate["passed"] or not high_gate["passed"]:
        classification = "converged_cross_solver_divergence"
    else:
        classification = "passed"
    return {
        "passed": all(checks.values()), "checks": checks,
        "classification": classification, "base_cross_solver": base_gate,
        "high_order_cross_solver": high_gate, "grcwa_convergence": grcwa_convergence,
        "thirdparty_convergence": thirdparty_convergence, "failures": failures,
    }


def run_controls():
    wavelength_nm = 550.0
    circle = {"L": 200.0, "W": 200.0, "H": 300.0, "P": 400.0}
    ellipse = {"L": 240.0, "W": 120.0, "H": 350.0, "P": 450.0}
    rotated = {"L": 120.0, "W": 240.0, "H": 350.0, "P": 450.0}
    n_substrate = float(rcwa_batch.n_cauchy(wavelength_nm, "SiO2"))
    fresnel_R = float(((1.0 - n_substrate) / (1.0 + n_substrate)) ** 2)
    controls = {}
    for name, point_solver in (
        ("grcwa", lambda geometry, pol, empty=False: solve_grcwa_point(
            geometry, pol, wavelength_nm, BASE_NG_REQUESTED, BASE_NXY, empty=empty
        )),
        ("thirdparty", lambda geometry, pol, empty=False: solve_thirdparty_point(
            geometry, pol, wavelength_nm, BASE_HARMONICS, BASE_NXY, empty=empty
        )),
    ):
        empty_R, empty_T = point_solver(circle, "p", empty=True)
        circle_p = point_solver(circle, "p")
        circle_s = point_solver(circle, "s")
        ellipse_p = point_solver(ellipse, "p")
        rotated_s = point_solver(rotated, "s")
        controls[name] = {
            "fresnel_R": empty_R, "fresnel_T": empty_T,
            "fresnel_R_analytic": fresnel_R, "fresnel_error": abs(empty_R - fresnel_R),
            "circle_p": list(circle_p), "circle_s": list(circle_s),
            "circle_max_difference": max(abs(circle_p[0] - circle_s[0]), abs(circle_p[1] - circle_s[1])),
            "rotation_ellipse_p": list(ellipse_p), "rotation_swapped_s": list(rotated_s),
            "rotation_max_difference": max(abs(ellipse_p[0] - rotated_s[0]), abs(ellipse_p[1] - rotated_s[1])),
        }
    checks = {
        f"{solver}_{name}": value <= INVARIANT_LIMIT
        for solver, result in controls.items()
        for name, value in (
            ("fresnel", result["fresnel_error"]),
            ("circle", result["circle_max_difference"]),
            ("rotation", result["rotation_max_difference"]),
        )
    }
    checks["both_empty_energy"] = all(
        abs(result["fresnel_R"] + result["fresnel_T"] - 1.0) <= ENERGY_LIMIT
        for result in controls.values()
    )
    return {"passed": all(checks.values()), "checks": checks, "values": controls}


def summarize(checkpoint, checkpoint_path, controls):
    evaluation = evaluate_results(checkpoint["results"], checkpoint["meta"]["expected_tasks"])
    runtime_hashes = checkpoint["meta"]["runtime_hashes"]
    runtime_ok = all(file_digest(ROOT / path) == expected for path, expected in runtime_hashes.items())
    checks = {
        "controls": controls["passed"], "matched_results": evaluation["passed"],
        "runtime_hashes_verified": runtime_ok,
    }
    classification = evaluation["classification"]
    if not controls["passed"]:
        classification = "implementation_control_failure"
    elif not runtime_ok:
        classification = "runtime_provenance_failure"
    return {
        "schema_version": 1, "evidence_version": VERSION,
        "passed": all(checks.values()), "pool_sha256": checkpoint["meta"]["pool_sha256"],
        "checks": checks, "classification": classification,
        "thresholds": checkpoint["meta"]["thresholds"],
        "protocol": checkpoint["meta"]["protocol"],
        "selected_geometries": checkpoint["meta"]["selected_geometries"],
        "controls": controls, "evaluation": evaluation,
        "raw_checkpoint": {
            "path": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": file_digest(checkpoint_path), "tasks": len(checkpoint["results"]),
        },
        "runtime_hashes": runtime_hashes, "packages": checkpoint["meta"]["packages"],
    }


def joint_gate_ready(pool_sha256: str | None = None) -> bool:
    """Require the currently configured joint gate and the exact input pool.

    The production pool and convergence evidence are versioned independently.
    Binding both here prevents a future strategy revision from accidentally
    running this legacy runner against an older pool or stale evidence file.
    """
    audit = load_json(ROOT / ".state" / "audit_result.json", {}) or {}
    if audit.get("training_gates", {}).get("joint_numerical_convergence") is not True:
        return False
    detail = audit.get("gate_evidence", {}).get("joint_numerical_convergence", {})
    if detail.get("verified") is not True:
        return False
    policy = load_json(ROOT / "pipeline_policy.json", {}) or {}
    expected_version = None
    for item in policy.get("workflow", {}).get("actions", []):
        if item.get("gate") == "joint_numerical_convergence":
            expected_version = item.get("evidence_version")
            break
    if not expected_version:
        return False
    for evidence in detail.get("evidence", []):
        path = ROOT / str(evidence.get("path", ""))
        payload = load_json(path, {}) if path.suffix.lower() == ".json" else {}
        if (
            payload.get("evidence_version") == expected_version
            and (pool_sha256 is None or payload.get("pool_sha256") == pool_sha256)
        ):
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default=POOL_PATH)
    parser.add_argument("--checkpoint", default=".state/cross_solver_v1_checkpoint.pkl")
    parser.add_argument("--evidence", default=".state/cross_solver_v1.json")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    pool_path = ROOT / args.pool
    with pool_path.open("rb") as handle:
        payload = pickle.load(handle)
    geometries, paired = pool_geometries(payload["records"])
    selected = select_cross_solver_geometries(geometries)
    tasks = build_tasks(selected, paired)
    runtime_hashes = {
        "scripts/run_cross_solver_validation.py": file_digest(ROOT / "scripts/run_cross_solver_validation.py"),
        "rcwa_batch.py": file_digest(ROOT / "rcwa_batch.py"),
        "paper2_colorimetry.py": file_digest(ROOT / "paper2_colorimetry.py"),
        "color_utils.py": file_digest(ROOT / "color_utils.py"),
    }
    thresholds = {
        "per_spectrum_R_T_rmse": PER_SPECTRUM_RMSE_LIMIT,
        "mean_spectrum_R_T_rmse": MEAN_SPECTRUM_RMSE_LIMIT,
        "mean_joint_dE00": MEAN_JOINT_DE00_LIMIT,
        "per_geometry_joint_dE00": PER_GEOMETRY_JOINT_DE00_LIMIT,
        "energy_error_max": ENERGY_LIMIT,
        "analytic_and_symmetry_error_max": INVARIANT_LIMIT,
    }
    meta = {
        "version": VERSION, "pool": args.pool.replace("\\", "/"),
        "pool_sha256": file_digest(pool_path), "selected_geometries": selected,
        "expected_tasks": len(tasks), "runtime_hashes": runtime_hashes,
        "packages": {
            "rcwa": importlib.metadata.version("rcwa"),
            "grcwa": importlib.metadata.version("grcwa"),
        },
        "thresholds": thresholds,
        "protocol": {
            "production_source": "immutable grcwa pool spectra",
            "independent_solver": "rcwa package",
            "geometry_count": len(selected), "sharpness_strata": 4,
            "geometries_per_stratum": 3,
            "stress_geometry_count": sum(bool(item["stress"]) for item in selected),
            "polarizations": list(POLARIZATIONS),
            "wavelength_nm": WAVELENGTHS_NM.tolist(),
            "base": {
                "grcwa_nG_requested": BASE_NG_REQUESTED,
                "grcwa_nG_retained": retained_order(BASE_NG_REQUESTED, selected[0]["P"]),
                "rcwa_harmonics": list(BASE_HARMONICS), "Nxy": BASE_NXY,
            },
            "stress": {
                "grcwa_nG_requested": STRESS_NG_REQUESTED,
                "grcwa_nG_retained": retained_order(STRESS_NG_REQUESTED, selected[0]["P"]),
                "rcwa_harmonics": list(STRESS_HARMONICS), "Nxy": STRESS_NXY,
            },
            "background": "air", "incident": "air",
            "transmission_halfspace": "SiO2",
            "material_model": "rcwa_batch.n_complex/n_cauchy exact reuse",
            "grid_endpoint": False,
            "polarization_mapping": {"s": "TE [1,0]", "p": "TM [0,1]"},
        },
    }
    if args.plan_only:
        print(json.dumps(meta, indent=2, ensure_ascii=True))
        return
    pool_sha256 = file_digest(pool_path)
    if not joint_gate_ready(pool_sha256):
        raise SystemExit("joint numerical convergence v1.1 gate is not verified")

    checkpoint_path = ROOT / args.checkpoint
    if checkpoint_path.exists():
        with checkpoint_path.open("rb") as handle:
            checkpoint = pickle.load(handle)
        if checkpoint.get("meta") != meta:
            raise SystemExit("checkpoint protocol mismatch; use a new versioned path")
    else:
        checkpoint = {"meta": meta, "results": {}}
        atomic_pickle(checkpoint_path, checkpoint)

    pending = [task for task in tasks if task["id"] not in checkpoint["results"]]
    with Pool(args.n_jobs) as workers:
        for result in workers.imap_unordered(run_task, pending, chunksize=1):
            checkpoint["results"][result["id"]] = result
            atomic_pickle(checkpoint_path, checkpoint)

    controls = run_controls()
    evidence = summarize(checkpoint, checkpoint_path, controls)
    evidence_path = ROOT / args.evidence
    if evidence_path.exists():
        existing = json.loads(evidence_path.read_text(encoding="utf-8"))
        if existing != evidence:
            raise SystemExit("existing cross-solver evidence differs; bump the version")
    else:
        atomic_json(evidence_path, evidence)
    print(json.dumps({"passed": evidence["passed"], "checks": evidence["checks"]}))
    if not evidence["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
