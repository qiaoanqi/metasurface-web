#!/usr/bin/env python3
"""Independently audit the completed v4 engineering probe."""
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


VERSION = "paper2-reference-budget-v4-probe-audit-v1"
PROTOCOL_PATH = ROOT / "protocols/paper2_reference_budget_v4_probe_v1.json"
PRODUCER_PATH = ROOT / ".state/reference_resolution_budget_v4_probe.json"
CHECKPOINT_PATH = ROOT / ".state/reference_resolution_budget_v4_probe_checkpoint.pkl"
SOURCE_CHECKPOINT = ROOT / ".state/reference_resolution_budget_v3_probe_checkpoint.pkl"
POLARIZATIONS = ("p", "s")


def task_id(pol: str, nG: int, step: float) -> str:
    token = "0p5" if step == 0.5 else "1"
    return f"probe-v4-g02-{pol}-ng{nG}-nxy1024-step{token}"


def source_task_id(pol: str) -> str:
    return f"probe-v3-g02-{pol}-ng650-nxy1024-step0p5"


def comparison(left: dict[str, dict], right: dict[str, dict], thresholds: dict) -> dict:
    rows = []
    for pol in POLARIZATIONS:
        value = float(v1.delta_e2000(
            v1.labels_on_grid(left[pol]["wavelength_nm"], left[pol]["R"]),
            v1.labels_on_grid(right[pol]["wavelength_nm"], right[pol]["R"]),
        ))
        rows.append({"pol": pol, "dE00": value})
    values = np.asarray([row["dE00"] for row in rows], dtype=float)
    mean_ok = bool(np.mean(values) < float(thresholds["mean_joint_dE00_lt"]))
    all_ok = bool(np.all(values < float(thresholds["all_joint_dE00_lt"])))
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
        "mean_lt_1_15": mean_ok,
        "all_lt_2_3": all_ok,
        "passed": bool(mean_ok and all_ok),
        "rows": rows,
    }


def validate_spectrum(
    item: dict,
    task_id_value: str,
    conservation_limit: float,
    *,
    pol: str,
    requested_nG: int,
    retained_nG: int,
    Nxy: int,
    step_nm: float,
) -> None:
    if not isinstance(item, dict) or item.get("id") != task_id_value or item.get("status") != "ok":
        raise ValueError(f"v4 audit invalid task: {task_id_value}")
    expected_fields = {
        "geometry_index": 2,
        "pol": pol,
        "requested_nG": requested_nG,
        "retained_nG": retained_nG,
        "Nxy": Nxy,
        "step_nm": step_nm,
    }
    for key, value in expected_fields.items():
        if item.get(key) != value:
            raise ValueError(f"v4 audit task field differs: {task_id_value}:{key}")
    wavelength = np.asarray(item.get("wavelength_nm"), dtype=float)
    reflection = np.asarray(item.get("R"), dtype=float)
    transmission = np.asarray(item.get("T"), dtype=float)
    if wavelength.ndim != 1 or reflection.shape != wavelength.shape or transmission.shape != wavelength.shape:
        raise ValueError(f"v4 audit shape mismatch: {task_id_value}")
    expected_wavelength = v1.WL_HALF_NM if step_nm == 0.5 else v1.WL_1NM
    if not np.array_equal(wavelength, expected_wavelength):
        raise ValueError(f"v4 audit wavelength grid differs: {task_id_value}")
    if not all(np.isfinite(value).all() for value in (wavelength, reflection, transmission)):
        raise ValueError(f"v4 audit non-finite spectrum: {task_id_value}")
    if np.any(reflection < -1e-8) or np.any(reflection > 1.0 + 1e-8):
        raise ValueError(f"v4 audit reflectance range: {task_id_value}")
    if np.max(np.abs(reflection + transmission - 1.0)) > conservation_limit:
        raise ValueError(f"v4 audit conservation: {task_id_value}")


def close_enough(actual: dict, expected: dict) -> bool:
    if actual.keys() != expected.keys():
        return False
    for key in actual:
        if key == "rows":
            if [row["pol"] for row in actual[key]] != [row["pol"] for row in expected[key]]:
                return False
            if not np.allclose(
                [row["dE00"] for row in actual[key]],
                [row["dE00"] for row in expected[key]],
                rtol=0.0,
                atol=1e-12,
            ):
                return False
        elif isinstance(actual[key], float):
            if not np.isclose(actual[key], expected[key], rtol=0.0, atol=1e-12):
                return False
        elif actual[key] != expected[key]:
            return False
    return True


def audit() -> dict:
    protocol = load_json(PROTOCOL_PATH, {}) or {}
    producer = load_json(PRODUCER_PATH, {}) or {}
    if protocol.get("evidence_version") != "paper2-reference-budget-v4-probe-v1":
        raise ValueError("v4 audit protocol version differs")
    implementations = protocol.get("implementation_hashes", [])
    if {item.get("path") for item in implementations} != {
        "scripts/probe_reference_budget_v4.py",
        "scripts/audit_reference_budget_v4_probe.py",
    }:
        raise ValueError("v4 audit implementation manifest differs")
    for item in implementations:
        if file_digest(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"v4 audit implementation differs: {item['path']}")
    expected_protocol = {
        "path": "protocols/paper2_reference_budget_v4_probe_v1.json",
        "sha256": file_digest(PROTOCOL_PATH),
    }
    if producer.get("protocol") != expected_protocol:
        raise ValueError("v4 audit producer protocol binding differs")
    for key in ("source_v3_probe", "source_v3_checkpoint"):
        item = protocol[key]
        path = ROOT / item["path"]
        if file_digest(path) != item["sha256"] or producer.get(key) != item:
            raise ValueError(f"v4 audit source binding differs: {key}")
    with CHECKPOINT_PATH.open("rb") as handle:
        state = pickle.load(handle)
    with SOURCE_CHECKPOINT.open("rb") as handle:
        source_state = pickle.load(handle)
    results = state.get("results", {})
    expected_ids = {
        task_id(pol, nG, step)
        for pol in POLARIZATIONS
        for nG, steps in ((750, (0.5,)), (850, (1.0, 0.5)))
        for step in steps
    }
    if set(results) != expected_ids or state.get("task_ids") != [
        task_id(pol, nG, step)
        for pol in POLARIZATIONS
        for nG, steps in ((750, (0.5,)), (850, (1.0, 0.5)))
        for step in steps
    ]:
        raise ValueError("v4 audit task set differs")
    if state.get("protocol_sha256") != file_digest(PROTOCOL_PATH):
        raise ValueError("v4 audit checkpoint protocol binding differs")
    if state.get("version") != "paper2-reference-budget-v4-probe-v1":
        raise ValueError("v4 audit checkpoint version differs")
    limit = float(protocol["thresholds"]["pointwise_conservation_lte"])
    for pol in POLARIZATIONS:
        for nG, retained, steps in ((750, 729, (0.5,)), (850, 841, (1.0, 0.5))):
            for step in steps:
                key = task_id(pol, nG, step)
                validate_spectrum(
                    results[key], key, limit, pol=pol, requested_nG=nG,
                    retained_nG=retained, Nxy=1024, step_nm=step,
                )
    source_results = source_state.get("results", {})
    for pol in POLARIZATIONS:
        validate_spectrum(
            source_results[source_task_id(pol)], source_task_id(pol), limit,
            pol=pol, requested_nG=650, retained_nG=625, Nxy=1024, step_nm=0.5,
        )

    def group(nG: int, step: float) -> dict[str, dict]:
        return {pol: results[task_id(pol, nG, step)] for pol in POLARIZATIONS}

    anchor = {pol: source_results[source_task_id(pol)] for pol in POLARIZATIONS}
    thresholds = protocol["thresholds"]
    recomputed = {
        "order_650_to_750_0p5nm_diagnostic": comparison(anchor, group(750, 0.5), thresholds),
        "order_750_to_850_0p5nm": comparison(group(750, 0.5), group(850, 0.5), thresholds),
        "spectral_850": comparison(group(850, 1.0), group(850, 0.5), thresholds),
    }
    if set(producer.get("comparisons", {})) != set(recomputed):
        raise ValueError("v4 audit comparison set differs")
    if not all(close_enough(producer["comparisons"][key], value) for key, value in recomputed.items()):
        raise ValueError("v4 audit producer comparison differs")
    passed = recomputed["order_750_to_850_0p5nm"]["passed"] and recomputed["spectral_850"]["passed"]
    expected_classification = "candidate_budget_supported" if passed else "candidate_budget_insufficient"
    if producer.get("passed") is not bool(passed) or producer.get("classification") != expected_classification:
        raise ValueError("v4 audit producer verdict differs")
    expected_checkpoint = {
        "path": ".state/reference_resolution_budget_v4_probe_checkpoint.pkl",
        "sha256": file_digest(CHECKPOINT_PATH),
        "tasks": 6,
    }
    if producer.get("checkpoint") != expected_checkpoint:
        raise ValueError("v4 audit producer checkpoint binding differs")
    runtime_hashes = producer.get("runtime_hashes", {})
    for path in ("rcwa_batch.py", "paper2_colorimetry.py", "color_utils.py"):
        if runtime_hashes.get(path) != file_digest(ROOT / path):
            raise ValueError(f"v4 audit runtime differs: {path}")
    if producer.get("training_allowed") is not False or producer.get("gate_registration_allowed") is not False:
        raise ValueError("v4 audit producer safety flags differ")
    return {
        "schema_version": 1,
        "evidence_version": VERSION,
        "passed": bool(passed),
        "classification": expected_classification,
        "checks": {
            "protocol_and_sources_verified": True,
            "exact_task_set": True,
            "spectra_valid_and_conservative": True,
            "producer_reproduction_matches": True,
            "training_allowed": False,
            "gate_registration_allowed": False,
        },
        "protocol": expected_protocol,
        "producer": {"path": str(PRODUCER_PATH.relative_to(ROOT)).replace("\\", "/"), "sha256": file_digest(PRODUCER_PATH)},
        "checkpoint": expected_checkpoint,
        "comparisons": recomputed,
        "candidate_reference": protocol["candidate_reference"],
        "training_allowed": False,
        "gate_registration_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=".state/reference_resolution_budget_v4_probe_audit.json")
    args = parser.parse_args()
    try:
        evidence = audit()
    except Exception as exc:
        evidence = {
            "schema_version": 1,
            "evidence_version": VERSION,
            "passed": False,
            "classification": "execution_integrity_failure",
            "error": f"{type(exc).__name__}: {exc}",
            "training_allowed": False,
            "gate_registration_allowed": False,
        }
    atomic_json(ROOT / args.output, evidence)
    print(json.dumps({"passed": evidence["passed"], "classification": evidence["classification"]}, sort_keys=True))
    return 0 if evidence["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
