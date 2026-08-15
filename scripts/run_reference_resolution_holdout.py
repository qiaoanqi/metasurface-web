#!/usr/bin/env python3
"""Extend the candidate reference from 8 to 32 pre-frozen geometries."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_supervisor import atomic_json, file_digest  # noqa: E402
from scripts import run_reference_resolution_escalation as base  # noqa: E402
from scripts.run_joint_convergence import retained_order  # noqa: E402
from scripts.run_replacement_pool import RunLock  # noqa: E402
from scripts.reference_protocol_selection import evaluate_protocols  # noqa: E402


VERSION = "paper2-reference-holdout-v1"
PLAN_VERSION = "paper2-reference-holdout-v1-plan"


def holdout_task_id(index: int, pol: str, config: tuple[int, int], step_nm: float) -> str:
    return (
        f"refhold-g{index:02d}-{pol}-ng{config[0]}-nxy{config[1]}"
        f"-step{base.step_token(step_nm)}"
    )


def build_new_tasks(plan: dict) -> list[dict]:
    cases = plan["new_cases"]
    if len(cases) != 24:
        raise ValueError("holdout plan must contain exactly 24 new cases")
    tasks = []
    for index, geometry in enumerate(cases, start=8):
        for pol in base.POLS:
            for config in base.CONFIGS_1NM:
                tasks.append(
                    {
                        "id": holdout_task_id(index, pol, config, 1.0),
                        "geometry_index": index,
                        "geometry": geometry,
                        "pol": pol,
                        "requested_nG": config[0],
                        "retained_nG": retained_order(config[0], geometry["P"]),
                        "Nxy": config[1],
                        "step_nm": 1.0,
                        "wavelength_nm": base.WL_1NM,
                    }
                )
            tasks.append(
                {
                    "id": holdout_task_id(index, pol, base.FINE_CONFIG, 0.5),
                    "geometry_index": index,
                    "geometry": geometry,
                    "pol": pol,
                    "requested_nG": base.FINE_CONFIG[0],
                    "retained_nG": retained_order(base.FINE_CONFIG[0], geometry["P"]),
                    "Nxy": base.FINE_CONFIG[1],
                    "step_nm": 0.5,
                    "wavelength_nm": base.WL_HALF_NM,
                }
            )
    return tasks


def load_plan(path: Path) -> dict:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != 1 or plan.get("evidence_version") != PLAN_VERSION:
        raise ValueError("unexpected holdout plan version")
    if plan.get("plan_valid") is not True or plan.get("created_before_candidate_result") is not True:
        raise ValueError("holdout plan was not pre-frozen")
    if len(plan.get("combined_cases", [])) != 32 or len(plan.get("new_cases", [])) != 24:
        raise ValueError("holdout plan case counts are invalid")
    if int(plan.get("expected_new_tasks", -1)) != 240:
        raise ValueError("holdout plan must contain exactly 240 new tasks")
    for runtime_path, expected in plan.get("runtime_hashes", {}).items():
        if file_digest(ROOT / runtime_path) != expected:
            raise ValueError(f"holdout runtime hash mismatch: {runtime_path}")
    return plan


def load_candidate(evidence_path: Path, checkpoint_path: Path, plan: dict) -> dict:
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("evidence_version") != base.VERSION or evidence.get("passed") is not True:
        raise ValueError("eight-case candidate reference is not passed")
    if evidence.get("pool_sha256") != plan["pool"]["sha256"]:
        raise ValueError("candidate reference and holdout plan pool hashes differ")
    with checkpoint_path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    if len(checkpoint.get("results", {})) != 80:
        raise ValueError("candidate reference checkpoint is not complete 80/80")
    if file_digest(checkpoint_path) != evidence.get("checkpoint", {}).get("sha256"):
        raise ValueError("candidate reference checkpoint hash mismatch")
    return checkpoint


def result_id(index: int, pol: str, config: tuple[int, int], step: float) -> str:
    if index < 8:
        return base.task_id(index, pol, config, step)
    return holdout_task_id(index, pol, config, step)


def validate_results(plan: dict, results: dict) -> dict:
    expected_tasks = []
    for index, geometry in enumerate(plan["combined_cases"]):
        for pol in base.POLS:
            for config in base.CONFIGS_1NM:
                expected_tasks.append(
                    (result_id(index, pol, config, 1.0), config, 1.0, geometry)
                )
            expected_tasks.append(
                (result_id(index, pol, base.FINE_CONFIG, 0.5), base.FINE_CONFIG, 0.5, geometry)
            )
    expected_ids = {item[0] for item in expected_tasks}
    valid = set(results) == expected_ids
    maximum_error = 0.0
    failures = []
    for identifier, config, step, geometry in expected_tasks:
        result = results.get(identifier, {})
        wavelength = base.WL_HALF_NM if step == 0.5 else base.WL_1NM
        if result.get("status") != "ok":
            failures.append({"id": identifier, "error": result.get("error", "missing")})
            valid = False
            continue
        R = np.asarray(result.get("R"), dtype=float)
        T = np.asarray(result.get("T"), dtype=float)
        stored_wavelength = np.asarray(result.get("wavelength_nm"), dtype=float)
        if (
            R.shape != wavelength.shape
            or T.shape != wavelength.shape
            or not np.array_equal(stored_wavelength, wavelength)
            or not np.isfinite(R).all()
            or not np.isfinite(T).all()
        ):
            valid = False
            failures.append({"id": identifier, "error": "invalid spectrum"})
            continue
        if int(result.get("requested_nG", -1)) != config[0] or int(result.get("Nxy", -1)) != config[1]:
            valid = False
        if int(result.get("retained_nG", -1)) != retained_order(config[0], geometry["P"]):
            valid = False
        maximum_error = max(maximum_error, float(np.max(np.abs(R + T - 1.0))))
    return {
        "passed": valid and maximum_error <= base.CONSERVATION_LIMIT,
        "records": len(results),
        "expected_records": 320,
        "failures": failures,
        "pointwise_conservation_error_max": maximum_error,
    }


def summarize(
    plan: dict,
    combined_results: dict,
    plan_path: Path,
    candidate_evidence_path: Path,
    candidate_checkpoint_path: Path,
    holdout_checkpoint_path: Path,
) -> dict:
    validation = validate_results(plan, combined_results)

    def identifier(config: tuple[int, int], step: float):
        return lambda index, pol: result_id(index, pol, config, step)

    comparisons = {}
    if validation["passed"]:
        comparisons = {
            "order_17x17_to_19x19": base.joint_comparison(
                plan["combined_cases"], combined_results, combined_results,
                identifier(base.ORDER_17, 1.0), identifier(base.ORDER_19, 1.0),
            ),
            "grid_Nxy384_to_512_at_19x19": base.joint_comparison(
                plan["combined_cases"], combined_results, combined_results,
                identifier(base.GRID_384, 1.0), identifier(base.ORDER_19, 1.0),
            ),
            "spectral_1nm_to_0p5nm_at_19x19_Nxy512": base.joint_comparison(
                plan["combined_cases"], combined_results, combined_results,
                identifier(base.FINE_CONFIG, 1.0), identifier(base.FINE_CONFIG, 0.5),
            ),
        }
    stable = all(
        item.get("mean_lt_1_15") and item.get("all_lt_2_3")
        for item in comparisons.values()
    ) and len(comparisons) == 3
    protocol_selection = (
        evaluate_protocols(plan["combined_cases"], combined_results, result_id)
        if validation["passed"]
        else {"any_protocol_passed": False, "lowest_cost_passing_protocol": None, "evaluations": []}
    )
    checks = {
        "pre_frozen_32_case_selection": plan.get("created_before_candidate_result") is True,
        "exact_320_tasks": validation["records"] == 320,
        "raw_spectra_and_conservation_valid": validation["passed"],
        "order_converged": bool(comparisons.get("order_17x17_to_19x19", {}).get("mean_lt_1_15") and comparisons.get("order_17x17_to_19x19", {}).get("all_lt_2_3")),
        "grid_converged": bool(comparisons.get("grid_Nxy384_to_512_at_19x19", {}).get("mean_lt_1_15") and comparisons.get("grid_Nxy384_to_512_at_19x19", {}).get("all_lt_2_3")),
        "spectral_grid_converged": bool(comparisons.get("spectral_1nm_to_0p5nm_at_19x19_Nxy512", {}).get("mean_lt_1_15") and comparisons.get("spectral_1nm_to_0p5nm_at_19x19_Nxy512", {}).get("all_lt_2_3")),
        "direct_candidate_to_reference_protocol_passed": protocol_selection["any_protocol_passed"],
    }
    passed = all(checks.values()) and stable
    return {
        "schema_version": 1,
        "evidence_version": VERSION,
        "passed": passed,
        "production_reference_approved": passed,
        "classification": "production_reference_converged" if passed else "reference_requires_further_escalation",
        "pool_sha256": plan["pool"]["sha256"],
        "selection": {
            "combined_case_count": 32,
            "new_case_count": 24,
            "combined_selection_sha256": plan["combined_selection_sha256"],
        },
        "thresholds": plan["thresholds"],
        "validation": validation,
        "checks": checks,
        "comparisons": comparisons,
        "protocol_selection": protocol_selection,
        "plan": {"path": str(plan_path.relative_to(ROOT)).replace("\\", "/"), "sha256": file_digest(plan_path)},
        "candidate_evidence": {"path": str(candidate_evidence_path.relative_to(ROOT)).replace("\\", "/"), "sha256": file_digest(candidate_evidence_path)},
        "candidate_checkpoint": {"path": str(candidate_checkpoint_path.relative_to(ROOT)).replace("\\", "/"), "sha256": file_digest(candidate_checkpoint_path)},
        "holdout_checkpoint": {"path": str(holdout_checkpoint_path.relative_to(ROOT)).replace("\\", "/"), "sha256": file_digest(holdout_checkpoint_path)},
        "training_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default=".state/reference_resolution_holdout_v1_plan.json")
    parser.add_argument("--candidate-evidence", default=".state/reference_resolution_v1.json")
    parser.add_argument("--candidate-checkpoint", default=".state/reference_resolution_v1_checkpoint.pkl")
    parser.add_argument("--checkpoint", default=".state/reference_resolution_holdout_v1_checkpoint.pkl")
    parser.add_argument("--evidence", default=".state/reference_resolution_holdout_v1.json")
    parser.add_argument("--lock", default=".state/reference_resolution_holdout_v1.lock")
    parser.add_argument("--n-jobs", type=int, default=16)
    args = parser.parse_args()

    plan_path = ROOT / args.plan
    candidate_evidence_path = ROOT / args.candidate_evidence
    candidate_checkpoint_path = ROOT / args.candidate_checkpoint
    checkpoint_path = ROOT / args.checkpoint
    evidence_path = ROOT / args.evidence
    plan = load_plan(plan_path)
    candidate = load_candidate(candidate_evidence_path, candidate_checkpoint_path, plan)
    meta = {
        "version": VERSION,
        "plan": {"path": args.plan.replace("\\", "/"), "sha256": file_digest(plan_path)},
        "candidate_checkpoint_sha256": file_digest(candidate_checkpoint_path),
        "expected_tasks": 240,
    }
    with RunLock(ROOT / args.lock, file_digest(plan_path)):
        if checkpoint_path.exists():
            with checkpoint_path.open("rb") as handle:
                checkpoint = pickle.load(handle)
            if checkpoint.get("meta") != meta:
                raise SystemExit("holdout checkpoint protocol mismatch")
        else:
            checkpoint = {"meta": meta, "results": {}}
            base.atomic_pickle(checkpoint_path, checkpoint)
        tasks = [task for task in build_new_tasks(plan) if task["id"] not in checkpoint["results"]]
        with Pool(max(1, args.n_jobs)) as workers:
            for result in workers.imap_unordered(base.run_task, tasks, chunksize=1):
                checkpoint["results"][result["id"]] = result
                base.atomic_pickle(checkpoint_path, checkpoint)

        combined_results = dict(candidate["results"])
        combined_results.update(checkpoint["results"])
        evidence = summarize(
            plan, combined_results, plan_path, candidate_evidence_path,
            candidate_checkpoint_path, checkpoint_path,
        )
        if evidence_path.exists():
            existing = json.loads(evidence_path.read_text(encoding="utf-8"))
            if existing != evidence:
                raise SystemExit("existing holdout evidence differs")
        else:
            atomic_json(evidence_path, evidence)
    print(json.dumps({"passed": evidence["passed"], "checks": evidence["checks"]}, sort_keys=True))
    return 0 if evidence["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
