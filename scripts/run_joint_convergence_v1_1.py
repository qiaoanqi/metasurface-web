#!/usr/bin/env python3
"""Supplementary joint-convergence checks with complete 1 nm color integration.

This is deliberately a separate version from the running v1 checkpoint.  Its
evidence is content-stable: no wall-clock fields are written to the JSON.
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
from paper2_colorimetry import D65_SPD, spectrum_to_labels_d65  # noqa: E402
from pipeline_supervisor import atomic_json, file_digest, load_json  # noqa: E402
from rcwa_batch import rcwa_spectrum  # noqa: E402
from scripts.run_joint_convergence import retained_order  # noqa: E402

VERSION = "paper2-joint-convergence-v1.1"
POLS = ("p", "s")
PRODUCTION = (131, 256)
REFERENCE = (251, 384)
REFERENCE_ORDER = (201, 384)
NXY_STABILITY = (251, 256)
CONFIGS = (PRODUCTION, REFERENCE_ORDER, NXY_STABILITY, REFERENCE)
EXPECTED_RETAINED = {131: 121, 201: 169, 251: 225}
WL_1NM = np.arange(380.0, 781.0, 1.0)
WL_5NM = np.arange(380.0, 785.0, 5.0)
CMF_1NM = np.column_stack([np.interp(WL_1NM, WL_5NM, x) for x in (CIE_X, CIE_Y, CIE_Z)])
SPD_1NM = np.interp(WL_1NM, WL_5NM, D65_SPD)
NORM_1NM = float(np.trapezoid(SPD_1NM * CMF_1NM[:, 1], WL_1NM))
WHITE_1NM = np.trapezoid(SPD_1NM[:, None] * CMF_1NM, WL_1NM, axis=0) / NORM_1NM
FROZEN_RUNTIME_HASHES = {"scripts/run_joint_convergence.py": "D27CF3D271B77C6653B6612114F0530338BAE4CFED847E614E37F77846D48DE4", "paper2_colorimetry.py": "710D81D34118AFDE4AA70037993DC3E1DBB38D7F3D9A572A09A7A8E8486AC5CB", "rcwa_batch.py": "8FD7CF97CE2C1BF73BACC5FAC3D5253F3BC8B21A7ECD2A1B23F0416A09D077B8", ".state/joint_convergence_v1_io_recovery_v1.json": "9AAE417355128D13DE2465E568DE8681788C260220C816CF256AB0CC897F2F8F"}


def geometry_key(record):
    return tuple(float(record[name]) for name in ("L", "W", "H", "P"))


def pool_geometries(records):
    paired = {}
    for rec in records:
        paired.setdefault(geometry_key(rec), {})[str(rec["pol"])] = rec
    out = []
    for key in sorted(paired):
        ch = paired[key]
        if set(ch) != set(POLS):
            continue
        L, W, H, P = key
        out.append({"L": L, "W": W, "H": H, "P": P,
                    "r": max(L, W) / min(L, W),
                    "fill": float(np.pi * (L / 2) * (W / 2) / P**2),
                    "sharpness_5nm": max(float(np.max(np.abs(np.diff(np.asarray(ch[p]["R"], float))))) for p in POLS)})
    return out


def select_supplemental(geometries, sharp_count=4, boundary_count=4):
    """Select global sharpness top-4 and distinct H/fill/P/r boundary points."""
    ranked = sorted(geometries, key=lambda g: (-g["sharpness_5nm"], g["L"], g["W"], g["H"], g["P"]))
    chosen, used = [], set()
    for g in ranked[:sharp_count]:
        item = dict(g); item["selection"] = "global_sharpness_top4"; chosen.append(item); used.add(geometry_key(g))
    criteria = [("high_H", "H"), ("high_fill", "fill"), ("large_P", "P"), ("aspect_ratio_boundary", "r")]
    for tag, field in criteria[:boundary_count]:
        candidates = sorted(geometries, key=lambda g: (-g[field], g["L"], g["W"], g["H"], g["P"]))
        pick = next(g for g in candidates if geometry_key(g) not in used)
        item = dict(pick); item["selection"] = tag; chosen.append(item); used.add(geometry_key(pick))
    return chosen


def task_id(index, pol, config, step=1):
    return f"supp-g{index:02d}-{pol}-ng{config[0]}-nxy{config[1]}-step{step}"


def build_tasks(selected):
    tasks = []
    for i, g in enumerate(selected):
        for pol in POLS:
            for config in CONFIGS:
                tasks.append({"id": task_id(i, pol, config), "geometry_index": i, "geometry": g,
                              "pol": pol, "requested_nG": config[0], "retained_nG": retained_order(config[0], g["P"]),
                              "Nxy": config[1], "step_nm": 1, "wavelength_nm": WL_1NM})
    return tasks


def run_task(task):
    g = task["geometry"]
    base = {k: task[k] for k in ("id", "geometry_index", "geometry", "pol", "requested_nG", "retained_nG", "Nxy", "step_nm")}
    base["wavelength_nm"] = np.asarray(WL_1NM, float)
    try:
        r, t = rcwa_spectrum(g["L"], g["H"], g["P"], WL_1NM, nG_req=task["requested_nG"], Nxy=task["Nxy"],
                             material="TiO2", substrate="SiO2", W_nm=g["W"], pol=task["pol"], background="air")
        base.update(status="ok", R=np.asarray(r, float), T=np.asarray(t, float))
    except Exception as exc:
        base.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    return base


def atomic_pickle(path, payload):
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with tmp.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
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


def labels_1nm(reflectance):
    r = np.asarray(reflectance, float)
    xyz = np.trapezoid(r[:, None] * SPD_1NM[:, None] * CMF_1NM, WL_1NM, axis=0) / NORM_1NM
    ratio = xyz / WHITE_1NM
    eps, kap = 216.0 / 24389.0, 24389.0 / 27.0
    f = np.where(ratio > eps, np.cbrt(ratio), (kap * ratio + 16.0) / 116.0)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])], float)

def threshold_summary(values):
    values = np.asarray(values, float)
    return {"count": int(values.size), "mean": float(np.mean(values)) if values.size else None, "max": float(np.max(values)) if values.size else None,
            "mean_lt_1_15": bool(values.size and np.mean(values) < 1.15), "all_lt_2_3": bool(values.size and np.all(values < 2.3))}

def joint_summary(rows, value_key):
    """Reduce p/s to a per-geometry joint maximum before thresholding."""
    grouped = {}
    for row in rows:
        grouped.setdefault(int(row["geometry_index"]), {})[row["pol"]] = float(row[value_key])
    joint = [max(channels[p] for p in POLS) for channels in grouped.values() if set(channels) == set(POLS)]
    summary = threshold_summary(joint)
    summary["joint_max_by_geometry"] = joint
    summary["complete_p_s_geometries"] = len(joint)
    return summary


def orders_recorded(results):
    return all(
        EXPECTED_RETAINED.get(int(result.get("requested_nG", -1)))
        == int(result.get("retained_nG", -1))
        for result in results.values()
    )


def spectrum_validation(results, samples):
    valid = True
    pointwise_max = 0.0
    for result in results.values():
        if result.get("status") != "ok":
            valid = False
            continue
        reflectance = np.asarray(result.get("R"), dtype=float)
        transmittance = np.asarray(result.get("T"), dtype=float)
        if reflectance.shape != (samples,) or transmittance.shape != (samples,):
            valid = False
            continue
        if not (np.isfinite(reflectance).all() and np.isfinite(transmittance).all()):
            valid = False
            continue
        if not (
            np.all((reflectance >= 0.0) & (reflectance <= 1.0))
            and np.all((transmittance >= 0.0) & (transmittance <= 1.0))
        ):
            valid = False
        pointwise_max = max(
            pointwise_max, float(np.max(np.abs(reflectance + transmittance - 1.0)))
        )
    valid = valid and pointwise_max <= 1e-6
    return {
        "passed": valid,
        "records": len(results),
        "samples_per_spectrum": samples,
        "pointwise_conservation_error_max": pointwise_max,
    }


def summarize(checkpoint, checkpoint_path, pool_sha256, v1_checkpoint):
    results = checkpoint["results"]
    frozen_runtime = checkpoint["meta"].get("runtime_hashes", FROZEN_RUNTIME_HASHES)
    failures = [r for r in results.values() if r["status"] != "ok"]
    rows = []
    for i, g in enumerate(checkpoint["meta"]["selected_geometries"]):
        for pol in POLS:
            ref = results.get(task_id(i, pol, REFERENCE))
            if not ref or ref["status"] != "ok": continue
            ref_lab = labels_1nm(ref["R"])
            for config in CONFIGS[:-1]:
                current = results.get(task_id(i, pol, config))
                if not current or current["status"] != "ok": continue
                current_lab = labels_1nm(current["R"])
                rows.append({"geometry_index": i, "pol": pol, "config": list(config),
                             "dE00_1nm_vs_reference": float(delta_e2000(current_lab, ref_lab)),
                             "dE00_1nm_vs_5nm_same_config": float(delta_e2000(current_lab, spectrum_to_labels_d65(np.asarray(current["R"])[::5])["lab"]))})
    comparison_summary = {}
    same_config_1nm_vs_5nm = {}
    for config in CONFIGS[:-1]:
        config_rows = [r for r in rows if tuple(r["config"]) == config]
        key = f"ng{config[0]}_nxy{config[1]}"
        comparison_summary[key] = joint_summary(config_rows, "dE00_1nm_vs_reference")
        same_config_1nm_vs_5nm[key] = joint_summary(config_rows, "dE00_1nm_vs_5nm_same_config")
    with v1_checkpoint.open("rb") as handle: v1 = pickle.load(handle)
    v1_results = v1.get("results", {})
    supplement_spectra = spectrum_validation(results, 401)
    v1_spectra = spectrum_validation(v1_results, 81)
    def v1_values(config):
        rows = []
        for i in range(32):
            for pol in POLS:
                a = v1_results.get(f"g{i:02d}-{pol}-ng{config[0]}-nxy{config[1]}-step5"); b = v1_results.get(f"g{i:02d}-{pol}-ng251-nxy384-step5")
                if a and b and a.get("status") == b.get("status") == "ok": rows.append({"geometry_index": i, "pol": pol, "value": float(delta_e2000(spectrum_to_labels_d65(a["R"])["lab"], spectrum_to_labels_d65(b["R"])["lab"]))})
        return joint_summary(rows, "value")
    v1_checks = {"expected_400": v1.get("meta", {}).get("expected_tasks") == 400, "all_completed": len(v1_results) == 400, "no_failures": not [r for r in v1_results.values() if r.get("status") != "ok"], "orders_recorded": orders_recorded(v1_results), "spectra": v1_spectra, "production_5nm": v1_values(PRODUCTION), "reference_order_5nm": v1_values(REFERENCE_ORDER), "nxy_stability_5nm": v1_values(NXY_STABILITY)}
    checks = {"all_tasks_completed": len(results) == checkpoint["meta"]["expected_tasks"], "no_task_failures": not failures,
              "raw_1nm_R_T_preserved": all("R" in r and "T" in r for r in results.values() if r["status"] == "ok"),
              "actual_orders_recorded": orders_recorded(results),
              "supplement_spectra_valid": supplement_spectra["passed"],
              "full_401_point_integration": True, "global_top4_present": sum(g["selection"] == "global_sharpness_top4" for g in checkpoint["meta"]["selected_geometries"]) == 4,
              "boundary_points_present": {g["selection"] for g in checkpoint["meta"]["selected_geometries"]} >= {"high_H", "high_fill", "large_P", "aspect_ratio_boundary"},
              "production_mean_lt_1_15": comparison_summary["ng131_nxy256"]["mean_lt_1_15"], "production_all_lt_2_3": comparison_summary["ng131_nxy256"]["all_lt_2_3"],
              "reference_order_mean_lt_1_15": comparison_summary["ng201_nxy384"]["mean_lt_1_15"], "reference_order_all_lt_2_3": comparison_summary["ng201_nxy384"]["all_lt_2_3"],
              "nxy_stability_mean_lt_1_15": comparison_summary["ng251_nxy256"]["mean_lt_1_15"], "nxy_stability_all_lt_2_3": comparison_summary["ng251_nxy256"]["all_lt_2_3"],
              "supplement_same_config_1nm_vs_5nm": all(v["mean_lt_1_15"] and v["all_lt_2_3"] for v in same_config_1nm_vs_5nm.values()),
              "v1_expected_400": v1_checks["expected_400"], "v1_all_completed": v1_checks["all_completed"], "v1_no_failures": v1_checks["no_failures"], "v1_orders_recorded": v1_checks["orders_recorded"], "v1_spectra_valid": v1_checks["spectra"]["passed"],
              "v1_production_5nm": v1_checks["production_5nm"]["mean_lt_1_15"] and v1_checks["production_5nm"]["all_lt_2_3"], "v1_reference_order_5nm": v1_checks["reference_order_5nm"]["mean_lt_1_15"] and v1_checks["reference_order_5nm"]["all_lt_2_3"], "v1_nxy_stability_5nm": v1_checks["nxy_stability_5nm"]["mean_lt_1_15"] and v1_checks["nxy_stability_5nm"]["all_lt_2_3"],
              "runtime_hashes_verified": all(file_digest(ROOT / p) == h for p, h in frozen_runtime.items())}
    checkpoint_label = str(checkpoint_path.relative_to(ROOT)).replace("\\", "/") if checkpoint_path.is_relative_to(ROOT) else str(checkpoint_path).replace("\\", "/")
    return {"schema_version": 1, "evidence_version": VERSION, "passed": all(checks.values()), "pool_sha256": pool_sha256,
            "selection": checkpoint["meta"]["selected_geometries"], "checks": checks, "comparisons": rows, "comparison_summary": comparison_summary, "same_config_1nm_vs_5nm": same_config_1nm_vs_5nm, "supplement_spectrum_validation": supplement_spectra, "v1_comparison_summary": v1_checks, "failures": failures,
            "integration": {"grid_nm": {"start": 380, "stop": 780, "step": 1, "samples": 401}, "cmf_convention": "frozen 5 nm CIE CMF and D65 tables linearly interpolated to 1 nm; all 401 points integrated", "lab_source": "unclipped XYZ"},
            "v1_checkpoint": {"path": ".state/joint_convergence_v1_checkpoint.pkl", "sha256": file_digest(v1_checkpoint)},
            "v1_1_checkpoint": {"path": checkpoint_label, "sha256": file_digest(checkpoint_path), "tasks": len(results)},
            "runtime_hashes": frozen_runtime,
            "runtime_hashes_verified": all(file_digest(ROOT / p) == h for p, h in frozen_runtime.items()),
            "io_recovery": load_json(ROOT / ".state/joint_convergence_v1_io_recovery_v1.json", {}),
            "implementation": {"path": "scripts/run_joint_convergence_v1_1.py", "sha256": file_digest(ROOT / "scripts/run_joint_convergence_v1_1.py")}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pool", default="data/rcwa_ellip_TiO2_3000_air.pkl"); ap.add_argument("--checkpoint", default=".state/joint_convergence_v1_1_checkpoint.pkl"); ap.add_argument("--evidence", default=".state/joint_convergence_v1_1.json"); ap.add_argument("--n-jobs", type=int, default=16); ap.add_argument("--plan-only", action="store_true"); args = ap.parse_args()
    pool_path = ROOT / args.pool
    with pool_path.open("rb") as f: payload = pickle.load(f)
    selected = select_supplemental(pool_geometries(payload["records"]))
    runtime_hashes = dict(FROZEN_RUNTIME_HASHES)
    runtime_hashes["color_utils.py"] = file_digest(ROOT / "color_utils.py")
    runtime_hashes["scripts/run_joint_convergence_v1_1.py"] = file_digest(ROOT / "scripts/run_joint_convergence_v1_1.py")
    meta = {"version": VERSION, "pool": args.pool.replace("\\", "/"), "pool_sha256": file_digest(pool_path), "selected_geometries": selected, "expected_tasks": len(selected) * 2 * len(CONFIGS), "runtime_hashes": runtime_hashes, "protocol": {"configs": [list(x) for x in CONFIGS], "reference": list(REFERENCE), "polarizations": list(POLS), "full_1nm_points": 401}}
    if args.plan_only: print(json.dumps(meta, indent=2)); return
    v1_path = ROOT / ".state/joint_convergence_v1_checkpoint.pkl"
    if not v1_path.exists(): raise SystemExit("v1 checkpoint missing; refuse v1.1 start")
    with v1_path.open("rb") as handle: v1_preflight = pickle.load(handle)
    v1_results_preflight = v1_preflight.get("results", {})
    if v1_preflight.get("meta", {}).get("expected_tasks") != 400 or len(v1_results_preflight) != 400 or any(r.get("status") != "ok" for r in v1_results_preflight.values()):
        raise SystemExit("v1 checkpoint is not complete 400/400 with zero failures; refuse v1.1 start")
    cp = ROOT / args.checkpoint
    if cp.exists():
        with cp.open("rb") as f: checkpoint = pickle.load(f)
        if checkpoint.get("meta") != meta: raise SystemExit("checkpoint protocol mismatch")
    else: checkpoint = {"meta": meta, "results": {}}; atomic_pickle(cp, checkpoint)
    tasks = [t for t in build_tasks(selected) if t["id"] not in checkpoint["results"]]
    with Pool(args.n_jobs) as workers:
        for result in workers.imap_unordered(run_task, tasks, chunksize=1):
            checkpoint["results"][result["id"]] = result; atomic_pickle(cp, checkpoint)
    evidence = summarize(checkpoint, cp, meta["pool_sha256"], v1_path)
    ep = ROOT / args.evidence
    if ep.exists() and json.loads(ep.read_text(encoding="utf-8")) != evidence: raise SystemExit("existing v1.1 evidence differs")
    if not ep.exists(): atomic_json(ep, evidence)
    print(json.dumps({"passed": evidence["passed"], "checks": evidence["checks"]}, sort_keys=True))
    if not evidence["passed"]: raise SystemExit(2)

if __name__ == "__main__": main()
