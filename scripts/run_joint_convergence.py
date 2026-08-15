#!/usr/bin/env python3
"""Run the versioned paper 2 joint nG/Nxy convergence gate.

The mutable checkpoint preserves every raw R/T spectrum. The final JSON is a
content-stable evidence summary and is written only after all tasks finish.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from color_utils import delta_e2000  # noqa: E402
from paper2_colorimetry import spectrum_to_labels_d65  # noqa: E402
from pipeline_supervisor import atomic_json, file_digest  # noqa: E402
from rcwa_batch import rcwa_spectrum  # noqa: E402
from grcwa.kbloch import Lattice_Reciprocate, Lattice_getG  # noqa: E402


VERSION = "paper2-joint-convergence-v1"
NG_VALUES = (131, 201, 251)
NXY_VALUES = (256, 384)
POLS = ("p", "s")
REFERENCE = (251, 384)
PRODUCTION = (131, 256)
WL_5NM = np.arange(380.0, 785.0, 5.0)
WL_1NM = np.arange(380.0, 781.0, 1.0)
MEAN_DE_LIMIT = 1.15
PER_GEOMETRY_DE_LIMIT = 2.3
CONSERVATION_LIMIT = 1e-6


def retained_order(requested: int, period_nm: float) -> int:
    period_um = period_nm / 1000.0
    reciprocal = Lattice_Reciprocate([period_um, 0.0], [0.0, period_um])
    _, retained = Lattice_getG(int(requested), *reciprocal, method=1)
    return int(retained)


def geometry_key(record: dict) -> tuple[float, float, float, float]:
    return tuple(float(record[name]) for name in ("L", "W", "H", "P"))


def pool_geometries(records: list[dict]) -> list[dict]:
    paired: dict[tuple[float, float, float, float], dict[str, dict]] = {}
    for record in records:
        paired.setdefault(geometry_key(record), {})[str(record["pol"])] = record
    geometries = []
    for key in sorted(paired):
        channels = paired[key]
        if set(channels) != set(POLS):
            continue
        L, W, H, P = key
        fill = float(np.pi * (L / 2.0) * (W / 2.0) / P**2)
        sharpness = max(
            float(np.max(np.abs(np.diff(np.asarray(channels[pol]["R"], dtype=float)))))
            for pol in POLS
        )
        geometries.append({
            "L": L,
            "W": W,
            "H": H,
            "P": P,
            "r": max(L, W) / min(L, W),
            "fill": fill,
            "sharpness_5nm": sharpness,
        })
    return geometries


def _maximin(group: list[dict], count: int) -> list[dict]:
    features = np.array(
        [[g[name] for name in ("L", "W", "H", "P", "r", "fill")] for g in group],
        dtype=float,
    )
    scale = np.ptp(features, axis=0)
    scale[scale == 0] = 1.0
    normalized = (features - np.mean(features, axis=0)) / scale
    first = int(np.argmax(np.linalg.norm(normalized, axis=1)))
    selected = [first]
    while len(selected) < min(count, len(group)):
        distances = np.linalg.norm(
            normalized[:, None, :] - normalized[np.asarray(selected)][None, :, :], axis=2
        )
        minimum = np.min(distances, axis=1)
        minimum[selected] = -1.0
        selected.append(int(np.argmax(minimum)))
    return [group[index] for index in selected]


def select_stratified(geometries: list[dict], count: int = 32) -> list[dict]:
    if len(geometries) < count or count % 4:
        raise ValueError("need at least 32 geometries and a count divisible by four")
    ranked = sorted(
        geometries,
        key=lambda g: (g["sharpness_5nm"], g["L"], g["W"], g["H"], g["P"]),
    )
    groups = np.array_split(np.asarray(ranked, dtype=object), 4)
    selected = []
    for stratum, group_array in enumerate(groups):
        chosen = _maximin(list(group_array), count // 4)
        for geometry in chosen:
            geometry = dict(geometry)
            geometry["sharpness_stratum"] = stratum
            selected.append(geometry)
    return selected


def task_id(index: int, pol: str, nG: int, Nxy: int, step_nm: int) -> str:
    return f"g{index:02d}-{pol}-ng{nG}-nxy{Nxy}-step{step_nm}"


def build_tasks(selected: list[dict], sharp_count: int = 4) -> list[dict]:
    tasks = []
    for index, geometry in enumerate(selected):
        for pol in POLS:
            for nG in NG_VALUES:
                for Nxy in NXY_VALUES:
                    tasks.append({
                        "id": task_id(index, pol, nG, Nxy, 5),
                        "geometry_index": index,
                        "geometry": geometry,
                        "pol": pol,
                        "requested_nG": nG,
                        "retained_nG": retained_order(nG, geometry["P"]),
                        "Nxy": Nxy,
                        "step_nm": 5,
                        "wavelength_nm": WL_5NM,
                    })
    sharp_indices = sorted(
        range(len(selected)),
        key=lambda i: selected[i]["sharpness_5nm"],
        reverse=True,
    )[:sharp_count]
    for index in sharp_indices:
        geometry = selected[index]
        for pol in POLS:
            for nG, Nxy in (PRODUCTION, REFERENCE):
                tasks.append({
                    "id": task_id(index, pol, nG, Nxy, 1),
                    "geometry_index": index,
                    "geometry": geometry,
                    "pol": pol,
                    "requested_nG": nG,
                    "retained_nG": retained_order(nG, geometry["P"]),
                    "Nxy": Nxy,
                    "step_nm": 1,
                    "wavelength_nm": WL_1NM,
                })
    return tasks


def run_task(task: dict) -> dict:
    geometry = task["geometry"]
    started = time.perf_counter()
    base = {
        key: task[key]
        for key in (
            "id", "geometry_index", "pol", "requested_nG", "retained_nG",
            "Nxy", "step_nm",
        )
    }
    base["geometry"] = geometry
    base["wavelength_nm"] = np.asarray(task["wavelength_nm"], dtype=float)
    try:
        R, T = rcwa_spectrum(
            geometry["L"], geometry["H"], geometry["P"], base["wavelength_nm"],
            nG_req=task["requested_nG"], Nxy=task["Nxy"], material="TiO2",
            substrate="SiO2", W_nm=geometry["W"], pol=task["pol"],
            background="air",
        )
        base.update({
            "status": "ok",
            "R": np.asarray(R, dtype=float),
            "T": np.asarray(T, dtype=float),
            "time_s": time.perf_counter() - started,
        })
    except Exception as exc:
        base.update({
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "time_s": time.perf_counter() - started,
        })
    return base


def atomic_pickle(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            # Windows readers and antivirus scanners can briefly hold the old
            # checkpoint without sharing delete access.
            time.sleep(0.25 * (attempt + 1))


def labels_5nm(result: dict) -> np.ndarray:
    spectrum = np.asarray(result["R"], dtype=float)
    if result["step_nm"] == 1:
        spectrum = spectrum[::5]
    return np.asarray(spectrum_to_labels_d65(spectrum)["lab"], dtype=float)


def summarize(checkpoint: dict, checkpoint_path: Path, pool_sha256: str) -> dict:
    results = checkpoint["results"]
    failures = [item for item in results.values() if item["status"] != "ok"]
    conservation_max = max(
        float(np.max(np.abs(item["R"] + item["T"] - 1.0)))
        for item in results.values() if item["status"] == "ok"
    )
    comparisons = {}
    for step_nm in (5, 1):
        geometry_indices = sorted({
            item["geometry_index"] for item in results.values()
            if item["step_nm"] == step_nm and item["status"] == "ok"
        })
        configs = ([(nG, Nxy) for nG in NG_VALUES for Nxy in NXY_VALUES]
                   if step_nm == 5 else [PRODUCTION, REFERENCE])
        for config in configs:
            values = []
            for index in geometry_indices:
                per_pol = []
                for pol in POLS:
                    current = results.get(task_id(index, pol, config[0], config[1], step_nm))
                    reference = results.get(task_id(index, pol, REFERENCE[0], REFERENCE[1], step_nm))
                    if not current or not reference or current["status"] != "ok" or reference["status"] != "ok":
                        per_pol = []
                        break
                    per_pol.append(delta_e2000(labels_5nm(current), labels_5nm(reference)))
                if len(per_pol) == 2:
                    values.append({"geometry_index": index, "p": per_pol[0], "s": per_pol[1], "joint_max": max(per_pol)})
            joint = np.asarray([item["joint_max"] for item in values], dtype=float)
            comparisons[f"step{step_nm}_ng{config[0]}_nxy{config[1]}"] = {
                "reference": config == REFERENCE,
                "count": len(values),
                "mean_joint_dE00": float(np.mean(joint)) if joint.size else None,
                "median_joint_dE00": float(np.median(joint)) if joint.size else None,
                "max_joint_dE00": float(np.max(joint)) if joint.size else None,
                "all_lt_2_3": bool(np.all(joint < PER_GEOMETRY_DE_LIMIT)) if joint.size else False,
                "values": values,
            }
    prod5 = comparisons["step5_ng131_nxy256"]
    prod1 = comparisons["step1_ng131_nxy256"]
    checks = {
        "all_tasks_completed": len(results) == checkpoint["meta"]["expected_tasks"],
        "no_task_failures": not failures,
        "raw_R_T_preserved": all("R" in item and "T" in item for item in results.values() if item["status"] == "ok"),
        "actual_orders_recorded": all(item.get("retained_nG") in (121, 169, 225) for item in results.values()),
        "pointwise_conservation": conservation_max <= CONSERVATION_LIMIT,
        "production_5nm_mean_lt_1_15": prod5["mean_joint_dE00"] is not None and prod5["mean_joint_dE00"] < MEAN_DE_LIMIT,
        "production_5nm_all_lt_2_3": prod5["all_lt_2_3"],
        "sharp_1nm_mean_lt_1_15": prod1["mean_joint_dE00"] is not None and prod1["mean_joint_dE00"] < MEAN_DE_LIMIT,
        "sharp_1nm_all_lt_2_3": prod1["all_lt_2_3"],
    }
    return {
        "schema_version": 1,
        "evidence_version": VERSION,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "passed": all(checks.values()),
        "checks": checks,
        "protocol": checkpoint["meta"]["protocol"],
        "selected_geometries": checkpoint["meta"]["selected_geometries"],
        "requested_to_retained_nG": {"131": 121, "201": 169, "251": 225},
        "comparisons": comparisons,
        "pointwise_conservation_error_max": conservation_max,
        "failures": failures,
        "raw_checkpoint": {
            "path": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": file_digest(checkpoint_path),
            "tasks": len(results),
        },
        "pool_sha256": pool_sha256,
        "implementation": {
            "path": "scripts/run_joint_convergence.py",
            "sha256": file_digest(ROOT / "scripts/run_joint_convergence.py"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="data/rcwa_ellip_TiO2_3000_air.pkl")
    parser.add_argument("--checkpoint", default=".state/joint_convergence_v1_checkpoint.pkl")
    parser.add_argument("--evidence", default=".state/joint_convergence_v1.json")
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    pool_path = ROOT / args.pool
    checkpoint_path = ROOT / args.checkpoint
    evidence_path = ROOT / args.evidence
    pool_sha256 = file_digest(pool_path)
    with pool_path.open("rb") as handle:
        pool_payload = pickle.load(handle)
    selected = select_stratified(pool_geometries(pool_payload["records"]))
    tasks = build_tasks(selected)
    protocol = {
        "geometries": 32,
        "selection": "four pool-sharpness strata; eight maximin geometry samples per stratum",
        "polarizations": list(POLS),
        "nG_requested": list(NG_VALUES),
        "Nxy": list(NXY_VALUES),
        "reference": {"nG_requested": 251, "nG_retained": 225, "Nxy": 384},
        "production": {"nG_requested": 131, "nG_retained": 121, "Nxy": 256},
        "five_nm_tasks": 32 * 2 * 3 * 2,
        "one_nm_sharp_geometries": 4,
        "one_nm_tasks": 4 * 2 * 2,
        "one_nm_color_evaluation": "raw spectra at 1 nm; registered D65 Lab evaluated at exact 5 nm subsamples",
        "mean_joint_dE00_limit": MEAN_DE_LIMIT,
        "per_geometry_joint_dE00_limit": PER_GEOMETRY_DE_LIMIT,
        "pointwise_conservation_limit": CONSERVATION_LIMIT,
    }
    meta = {
        "version": VERSION,
        "pool": args.pool.replace("\\", "/"),
        "pool_sha256": pool_sha256,
        "expected_tasks": len(tasks),
        "selected_geometries": selected,
        "protocol": protocol,
    }
    if args.plan_only:
        print(json.dumps(meta, indent=2))
        return

    if checkpoint_path.exists():
        with checkpoint_path.open("rb") as handle:
            checkpoint = pickle.load(handle)
        if checkpoint.get("meta") != meta:
            raise SystemExit("checkpoint protocol mismatch; use a new versioned path")
    else:
        checkpoint = {"meta": meta, "results": {}}
        atomic_pickle(checkpoint_path, checkpoint)

    pending = [task for task in tasks if task["id"] not in checkpoint["results"]]
    print(f"{VERSION}: {len(tasks)} tasks, {len(pending)} pending", flush=True)
    with Pool(args.n_jobs) as workers:
        for completed, result in enumerate(workers.imap_unordered(run_task, pending, chunksize=1), 1):
            checkpoint["results"][result["id"]] = result
            atomic_pickle(checkpoint_path, checkpoint)
            print(
                f"[{completed}/{len(pending)}] {result['id']} {result['status']} "
                f"{result['time_s']:.1f}s",
                flush=True,
            )

    evidence = summarize(checkpoint, checkpoint_path, pool_sha256)
    if evidence_path.exists():
        existing = json.loads(evidence_path.read_text(encoding="utf-8"))
        if existing != evidence:
            raise SystemExit("existing convergence evidence differs; bump the version")
    else:
        atomic_json(evidence_path, evidence)
    print(json.dumps({"passed": evidence["passed"], "checks": evidence["checks"]}), flush=True)


if __name__ == "__main__":
    main()
