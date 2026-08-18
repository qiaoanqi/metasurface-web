#!/usr/bin/env python3
"""Independently audit the formal eight-geometry v4 budget gate."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_supervisor import atomic_json, file_digest, load_json  # noqa: E402
from scripts import run_reference_resolution_escalation as v1  # noqa: E402

VERSION = "paper2-reference-budget-v4-audit"
PROTOCOL_PATH = ROOT / "protocols/paper2_reference_budget_v4.json"
PLAN_PATH = ROOT / ".state/reference_resolution_budget_v2_plan.json"
PROBE_AUDIT = ROOT / ".state/reference_resolution_budget_v4_probe_audit.json"
PROBE_CHECKPOINT = ROOT / ".state/reference_resolution_budget_v4_probe_checkpoint.pkl"
POLS = ("p", "s")
CONFIGS = ((750, 1024, 0.5), (850, 1024, 1.0), (850, 1024, 0.5))


def task_id(index: int, pol: str, config: tuple[int, int, float]) -> str:
    nG, nxy, step = config
    token = "0p5" if step == 0.5 else "1"
    return f"refbudget-v4-g{index:02d}-{pol}-ng{nG}-nxy{nxy}-step{token}"


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def binding(path: Path) -> dict:
    return {"path": relative(path), "sha256": file_digest(path)}


def load_protocol() -> dict:
    protocol = load_json(PROTOCOL_PATH, {}) or {}
    if protocol.get("evidence_version") != "paper2-reference-budget-v4":
        raise ValueError("formal v4 protocol version differs")
    for item in protocol.get("implementation_hashes", []):
        path = ROOT / item["path"]
        if not path.is_file() or file_digest(path) != item.get("sha256"):
            raise ValueError(f"formal v4 implementation hash mismatch: {item.get('path')}")
    for key, path in (("plan", PLAN_PATH), ("probe_audit", PROBE_AUDIT), ("probe_checkpoint", PROBE_CHECKPOINT)):
        item = protocol.get(key, {})
        if item.get("path") != relative(path) or item.get("sha256") != file_digest(path):
            raise ValueError(f"formal v4 {key} binding mismatch")
    return protocol


def build_tasks(protocol: dict) -> list[dict]:
    plan = load_json(PLAN_PATH, {}) or {}
    selected = plan.get("selection", [])
    if len(selected) != 8:
        raise ValueError("formal v4 selection is not eight geometries")
    tasks = []
    for index, geometry in enumerate(selected):
        for pol in POLS:
            for nG, nxy, step in CONFIGS:
                wavelength = v1.WL_HALF_NM if step == 0.5 else v1.WL_1NM
                tasks.append({"id": task_id(index, pol, (nG, nxy, step)), "geometry_index": index,
                              "geometry": geometry, "pol": pol, "requested_nG": nG,
                              "retained_nG": v1.retained_order(nG, geometry["P"]), "Nxy": nxy,
                              "step_nm": step, "wavelength_nm": wavelength})
    return tasks


def validate_result(result: dict, expected: dict, limit: float) -> None:
    if result.get("status") != "ok":
        raise ValueError(f"task is not successful: {expected['id']}")
    for key in ("id", "geometry_index", "pol", "requested_nG", "retained_nG", "Nxy", "step_nm"):
        if result.get(key) != expected[key]:
            raise ValueError(f"task field mismatch: {expected['id']}:{key}")
    if tuple(float(result["geometry"][k]) for k in ("L", "W", "H", "P")) != tuple(float(expected["geometry"][k]) for k in ("L", "W", "H", "P")):
        raise ValueError(f"geometry mismatch: {expected['id']}")
    wavelength = np.asarray(expected["wavelength_nm"], dtype=float)
    got_wavelength = np.asarray(result.get("wavelength_nm"), dtype=float)
    R = np.asarray(result.get("R"), dtype=float)
    T = np.asarray(result.get("T"), dtype=float)
    if got_wavelength.shape != wavelength.shape or R.shape != wavelength.shape or T.shape != wavelength.shape:
        raise ValueError(f"shape mismatch: {expected['id']}")
    if not (np.isfinite(got_wavelength).all() and np.isfinite(R).all() and np.isfinite(T).all()):
        raise ValueError(f"non-finite spectrum: {expected['id']}")
    if not np.array_equal(got_wavelength, wavelength):
        raise ValueError(f"wavelength grid mismatch: {expected['id']}")
    if np.max(np.abs(R + T - 1.0)) > limit:
        raise ValueError(f"conservation failure: {expected['id']}")


def labels(item: dict) -> np.ndarray:
    return v1.labels_on_grid(np.asarray(item["wavelength_nm"], dtype=float), np.asarray(item["R"], dtype=float))


def compare(left: dict[str, dict], right: dict[str, dict], thresholds: dict) -> dict:
    rows = [{"geometry_index": i, "dE00": float(v1.delta_e2000(labels(left[i]), labels(right[i])))} for i in sorted(left)]
    values = np.asarray([row["dE00"] for row in rows], dtype=float)
    mean_ok = bool(np.mean(values) < float(thresholds["mean_joint_dE00_lt"]))
    all_ok = bool(np.all(values < float(thresholds["all_joint_dE00_lt"])))
    return {"count": int(values.size), "mean": float(np.mean(values)), "max": float(np.max(values)),
            "mean_lt_1_15": mean_ok, "all_lt_2_3": all_ok, "passed": bool(mean_ok and all_ok), "rows": rows}


def audit(checkpoint_path: Path, evidence_path: Path) -> dict:
    protocol = load_protocol()
    tasks = build_tasks(protocol)
    expected = {task["id"]: task for task in tasks}
    with checkpoint_path.open("rb") as handle:
        state = pickle.load(handle)
    results = state.get("results", {})
    if state.get("version") != "paper2-reference-budget-v4" or state.get("protocol_sha256") != file_digest(PROTOCOL_PATH):
        raise ValueError("checkpoint provenance mismatch")
    if set(results) != set(expected):
        raise ValueError(f"checkpoint task set mismatch: {len(results)}/{len(expected)}")
    limit = float(protocol["thresholds"]["pointwise_conservation_lte"])
    for key, result in results.items():
        validate_result(result, expected[key], limit)
    groups: dict[tuple[int, str, tuple[int, int, float]], dict] = {}
    for key, result in results.items():
        task = expected[key]
        groups[(task["geometry_index"], task["pol"], (task["requested_nG"], task["Nxy"], task["step_nm"]))] = result
    order_left, order_right, spectral_left, spectral_right = {}, {}, {}, {}
    for index in range(8):
        for pol in POLS:
            order_left[index] = groups[(index, pol, CONFIGS[0])]
            order_right[index] = groups[(index, pol, CONFIGS[2])]
    for index in range(8):
        for pol in POLS:
            spectral_left[index] = groups[(index, pol, CONFIGS[1])]
            spectral_right[index] = groups[(index, pol, CONFIGS[2])]
    # Compare p/s as a paired per-geometry maximum, then apply frozen thresholds.
    def paired(left_cfg: tuple, right_cfg: tuple) -> dict:
        rows = []
        for index in range(8):
            vals = []
            for pol in POLS:
                vals.append(float(v1.delta_e2000(labels(groups[(index, pol, left_cfg)]), labels(groups[(index, pol, right_cfg)]))))
            rows.append({"geometry_index": index, "p_dE00": vals[0], "s_dE00": vals[1], "joint_dE00": max(vals)})
        values = np.asarray([row["joint_dE00"] for row in rows], dtype=float)
        mean_ok = bool(np.mean(values) < float(protocol["thresholds"]["mean_joint_dE00_lt"]))
        all_ok = bool(np.all(values < float(protocol["thresholds"]["all_joint_dE00_lt"])))
        return {"count": 8, "mean": float(np.mean(values)), "max": float(np.max(values)), "mean_lt_1_15": mean_ok,
                "all_lt_2_3": all_ok, "passed": bool(mean_ok and all_ok), "rows": rows}
    comparisons = {"order_750_to_850_0p5nm": paired(CONFIGS[0], CONFIGS[2]),
                   "spectral_850_1p0_to_0p5nm": paired(CONFIGS[1], CONFIGS[2])}
    checks = {"all_tasks_completed": True, "no_task_failures": True, "spectra_and_conservation_valid": True,
              "p_s_pairing_complete": all((i, pol, cfg) in groups for i in range(8) for pol in POLS for cfg in CONFIGS),
              "order_axis_converged": comparisons["order_750_to_850_0p5nm"]["passed"],
              "spectral_axis_converged": comparisons["spectral_850_1p0_to_0p5nm"]["passed"],
              "runtime_hashes_verified": state.get("runtime_hashes") == protocol.get("runtime_hashes")}
    passed = all(checks.values())
    return {"schema_version": 1, "evidence_version": VERSION, "passed": passed,
            "classification": "reference_resolution_budget_v4_passed" if passed else "reference_resolution_budget_v4_failed",
            "protocol": binding(PROTOCOL_PATH), "plan": binding(PLAN_PATH), "checkpoint": binding(checkpoint_path) | {"tasks": len(results)},
            "producer": binding(evidence_path), "comparisons": comparisons, "checks": checks,
            "thresholds": protocol["thresholds"], "training_allowed": False, "gate_registration_allowed": passed}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=".state/reference_resolution_budget_v4_checkpoint.pkl")
    parser.add_argument("--evidence", default=".state/reference_resolution_budget_v4.json")
    parser.add_argument("--output", default=".state/reference_resolution_budget_v4_audit.json")
    args = parser.parse_args()
    try:
        result = audit(ROOT / args.checkpoint, ROOT / args.evidence)
    except Exception as exc:
        result = {"schema_version": 1, "evidence_version": VERSION, "passed": False,
                  "classification": "execution_integrity_failure", "error": f"{type(exc).__name__}: {exc}",
                  "training_allowed": False, "gate_registration_allowed": False}
    atomic_json(ROOT / args.output, result)
    print(json.dumps({"passed": result["passed"], "classification": result["classification"]}, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
