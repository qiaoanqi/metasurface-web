#!/usr/bin/env python3
"""Audit the completed v4 checkpoint against its frozen producer runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_supervisor import atomic_json, file_digest, load_json  # noqa: E402

VERSION = "paper2-reference-budget-v4-audit-recovery-v1"
RECOVERY_PROTOCOL = ROOT / "protocols/paper2_reference_budget_v4_audit_recovery_v1.json"
FORMAL_PROTOCOL = ROOT / "protocols/paper2_reference_budget_v4.json"
PLAN = ROOT / ".state/reference_resolution_budget_v2_plan.json"
POLS = ("p", "s")
CONFIGS = ((750, 1024, 0.5), (850, 1024, 1.0), (850, 1024, 0.5))


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def binding(path: Path) -> dict:
    return {"path": relative(path), "sha256": file_digest(path)}


def frozen_bytes(commit: str, path: str, expected: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"], cwd=ROOT, capture_output=True, check=False
    )
    if completed.returncode != 0:
        raise ValueError(f"frozen runtime is unavailable: {path}")
    raw = completed.stdout
    candidates = (raw, raw.replace(b"\n", b"\r\n"))
    for candidate in candidates:
        if hashlib.sha256(candidate).hexdigest().upper() == expected:
            return candidate
    raise ValueError(f"frozen runtime hash mismatch: {path}")


def frozen_colorimetry(protocol: dict):
    commit = protocol["producer_commit"]
    expected = protocol["producer_runtime_hashes"]
    sources = {
        path: frozen_bytes(commit, path, digest) for path, digest in expected.items()
    }
    color = types.ModuleType("color_utils")
    exec(compile(sources["color_utils.py"], "color_utils.py", "exec"), color.__dict__)
    previous = sys.modules.get("color_utils")
    sys.modules["color_utils"] = color
    try:
        paper2 = types.ModuleType("paper2_colorimetry")
        exec(
            compile(sources["paper2_colorimetry.py"], "paper2_colorimetry.py", "exec"),
            paper2.__dict__,
        )
    finally:
        if previous is None:
            sys.modules.pop("color_utils", None)
        else:
            sys.modules["color_utils"] = previous
    return color.CIE_X, color.CIE_Y, color.CIE_Z, paper2.D65_SPD, color.delta_e2000


def load_recovery_protocol() -> dict:
    protocol = load_json(RECOVERY_PROTOCOL, {}) or {}
    if protocol.get("evidence_version") != "paper2-reference-budget-v4-audit-recovery-protocol-v1":
        raise ValueError("unexpected recovery protocol")
    for key, path in (("formal_protocol", FORMAL_PROTOCOL), ("plan", PLAN)):
        item = protocol.get(key, {})
        if item.get("path") != relative(path) or item.get("sha256") != file_digest(path):
            raise ValueError(f"recovery {key} binding mismatch")
    for item in protocol.get("implementation_hashes", []):
        path = ROOT / item["path"]
        if not path.is_file() or file_digest(path) != item.get("sha256"):
            raise ValueError(f"recovery implementation hash mismatch: {item.get('path')}")
    if protocol.get("training_allowed") is not False:
        raise ValueError("recovery protocol cannot authorize training")
    return protocol


def task_id(index: int, pol: str, config: tuple[int, int, float]) -> str:
    nG, nxy, step = config
    return f"refbudget-v4-g{index:02d}-{pol}-ng{nG}-nxy{nxy}-step{'0p5' if step == 0.5 else '1'}"


def labels(item: dict, colorimetry: tuple) -> np.ndarray:
    cie_x, cie_y, cie_z, d65, _delta_e = colorimetry
    wavelength = np.asarray(item["wavelength_nm"], dtype=float)
    reflectance = np.asarray(item["R"], dtype=float)
    grid = np.arange(380.0, 785.0, 5.0)
    cmf = np.column_stack([np.interp(wavelength, grid, x) for x in (cie_x, cie_y, cie_z)])
    spd = np.interp(wavelength, grid, d65)
    norm = float(np.trapezoid(spd * cmf[:, 1], wavelength))
    white = np.trapezoid(spd[:, None] * cmf, wavelength, axis=0) / norm
    xyz = np.trapezoid(reflectance[:, None] * spd[:, None] * cmf, wavelength, axis=0) / norm
    ratio = xyz / white
    epsilon, kappa = 216.0 / 24389.0, 24389.0 / 27.0
    f = np.where(ratio > epsilon, np.cbrt(ratio), (kappa * ratio + 16.0) / 116.0)
    return np.array([116.0 * f[1] - 16.0, 500.0 * (f[0] - f[1]), 200.0 * (f[1] - f[2])])


def audit(checkpoint: Path, evidence: Path, dispatch_path: Path) -> dict:
    recovery = load_recovery_protocol()
    formal = load_json(FORMAL_PROTOCOL, {}) or {}
    dispatch = load_json(dispatch_path, {}) or {}
    authorization = {"request_id": dispatch.get("request_id"), "attempt": int(dispatch.get("attempt", 0))}
    if dispatch.get("action") != "reference_resolution_budget_v4" or dispatch.get("status") != "in_progress":
        raise ValueError("audit recovery requires the active v4 request")
    if dispatch.get("strategy_based_on") != recovery.get("source_request_id"):
        raise ValueError("audit recovery request lineage mismatch")
    if file_digest(checkpoint) != recovery["checkpoint"]["sha256"]:
        raise ValueError("completed checkpoint bytes changed")
    if file_digest(evidence) != recovery["worker_evidence"]["sha256"]:
        raise ValueError("completed worker evidence bytes changed")
    with checkpoint.open("rb") as handle:
        state = pickle.load(handle)
    if state.get("version") != "paper2-reference-budget-v4":
        raise ValueError("checkpoint version mismatch")
    if state.get("protocol_sha256") != file_digest(FORMAL_PROTOCOL):
        raise ValueError("checkpoint formal protocol mismatch")
    if state.get("runtime_hashes") != recovery.get("producer_runtime_hashes"):
        raise ValueError("checkpoint runtime hashes mismatch")
    colorimetry = frozen_colorimetry(recovery)
    plan = load_json(PLAN, {}) or {}
    geometries = plan.get("selection", [])
    if len(geometries) != 8:
        raise ValueError("frozen geometry plan is incomplete")
    expected_ids = {task_id(i, pol, cfg) for i in range(8) for pol in POLS for cfg in CONFIGS}
    results = state.get("results", {})
    if set(results) != expected_ids:
        raise ValueError(f"checkpoint task identity mismatch: {len(results)}/48")
    groups = {}
    maximum_conservation = 0.0
    for identifier, item in results.items():
        if item.get("status") != "ok" or item.get("id") != identifier:
            raise ValueError(f"task is not complete: {identifier}")
        index, pol = int(item["geometry_index"]), item["pol"]
        geometry = geometries[index]
        if any(float(item["geometry"][key]) != float(geometry[key]) for key in ("L", "W", "H", "P")):
            raise ValueError(f"task geometry mismatch: {identifier}")
        config = (int(item["requested_nG"]), int(item["Nxy"]), float(item["step_nm"]))
        if pol not in POLS or config not in CONFIGS or identifier != task_id(index, pol, config):
            raise ValueError(f"task metadata mismatch: {identifier}")
        wavelength = np.asarray(item.get("wavelength_nm"), dtype=float)
        r = np.asarray(item.get("R"), dtype=float)
        t = np.asarray(item.get("T"), dtype=float)
        expected_size = 801 if config[2] == 0.5 else 401
        if wavelength.shape != (expected_size,) or r.shape != wavelength.shape or t.shape != wavelength.shape:
            raise ValueError(f"task shape mismatch: {identifier}")
        if not (np.isfinite(wavelength).all() and np.isfinite(r).all() and np.isfinite(t).all()):
            raise ValueError(f"task contains non-finite values: {identifier}")
        if not np.array_equal(wavelength, np.linspace(380.0, 780.0, expected_size)):
            raise ValueError(f"task wavelength mismatch: {identifier}")
        maximum_conservation = max(maximum_conservation, float(np.max(np.abs(r + t - 1.0))))
        groups[(index, pol, config)] = item
    limit = float(formal["thresholds"]["pointwise_conservation_lte"])
    if maximum_conservation > limit:
        raise ValueError("pointwise conservation threshold failed")

    def comparison(left: tuple, right: tuple) -> dict:
        rows = []
        for index in range(8):
            values = [float(colorimetry[4](labels(groups[(index, pol, left)], colorimetry), labels(groups[(index, pol, right)], colorimetry))) for pol in POLS]
            rows.append({"geometry_index": index, "p_dE00": values[0], "s_dE00": values[1], "joint_dE00": max(values)})
        joint = np.asarray([row["joint_dE00"] for row in rows])
        mean_ok = bool(np.mean(joint) < float(formal["thresholds"]["mean_joint_dE00_lt"]))
        all_ok = bool(np.all(joint < float(formal["thresholds"]["all_joint_dE00_lt"])))
        return {"count": 8, "mean": float(np.mean(joint)), "max": float(np.max(joint)), "mean_lt_1_15": mean_ok, "all_lt_2_3": all_ok, "passed": mean_ok and all_ok, "rows": rows}

    comparisons = {
        "order_750_to_850_0p5nm": comparison(CONFIGS[0], CONFIGS[2]),
        "spectral_850_1p0_to_0p5nm": comparison(CONFIGS[1], CONFIGS[2]),
    }
    checks = {
        "all_tasks_completed": True,
        "no_task_failures": True,
        "spectra_and_conservation_valid": True,
        "p_s_pairing_complete": len(groups) == 48,
        "order_axis_converged": comparisons["order_750_to_850_0p5nm"]["passed"],
        "spectral_axis_converged": comparisons["spectral_850_1p0_to_0p5nm"]["passed"],
        "runtime_hashes_verified": True,
        "frozen_runtime_recovered_from_commit": True,
        "checkpoint_bytes_unchanged": True,
        "worker_evidence_bytes_unchanged": True,
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "evidence_version": VERSION,
        "authorization_request": authorization,
        "producer_request": {"request_id": recovery["source_request_id"], "attempt": recovery["source_attempt"]},
        "passed": passed,
        "classification": "reference_resolution_budget_v4_passed" if passed else "reference_resolution_budget_v4_failed",
        "recovery_protocol": binding(RECOVERY_PROTOCOL),
        "protocol": binding(FORMAL_PROTOCOL),
        "plan": binding(PLAN),
        "checkpoint": binding(checkpoint) | {"tasks": len(results)},
        "producer": binding(evidence),
        "producer_commit": recovery["producer_commit"],
        "producer_runtime_hashes": recovery["producer_runtime_hashes"],
        "comparisons": comparisons,
        "checks": checks,
        "maximum_pointwise_conservation_error": maximum_conservation,
        "thresholds": formal["thresholds"],
        "training_allowed": False,
        "gate_registration_allowed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=".state/reference_resolution_budget_v4_checkpoint.pkl")
    parser.add_argument("--evidence", default=".state/reference_resolution_budget_v4.json")
    parser.add_argument("--output", default=".state/reference_resolution_budget_v4_audit_recovery_v1.json")
    parser.add_argument("--dispatch", default=".state/dispatch_request.json")
    args = parser.parse_args()
    try:
        result = audit(ROOT / args.checkpoint, ROOT / args.evidence, ROOT / args.dispatch)
    except Exception as exc:
        result = {"schema_version": 1, "evidence_version": VERSION, "passed": False, "classification": "execution_integrity_failure", "error": f"{type(exc).__name__}: {exc}", "training_allowed": False, "gate_registration_allowed": False}
    atomic_json(ROOT / args.output, result)
    print(json.dumps({"passed": result["passed"], "classification": result["classification"]}, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
