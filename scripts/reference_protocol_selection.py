"""Select a production candidate on the initial eight cases only."""

from __future__ import annotations

from typing import Callable

import numpy as np

from color_utils import delta_e2000
from scripts import run_reference_resolution_escalation as base


CONFIGS = {
    "BASE": (365, 512),
    "ORDER": (450, 512),
    "GRID": (365, 768),
    "CORNER": (450, 768),
}
STEPS = (1.0, 0.5)
REFERENCE_CONFIG_NAME = "CORNER"
REFERENCE_STEP_NM = 0.5


def evaluate_protocols(
    selected: list[dict],
    results: dict,
    identifier: Callable[[int, str, tuple[int, int], float], str],
) -> dict:
    """Compare the eight measured v2 protocols to the final reference.

    This function is for candidate freezing on the original eight cases.  The
    independent 24-case holdout must never call it to reselect a candidate.
    """
    if len(selected) != 8:
        raise ValueError("protocol selection requires exactly the initial eight cases")
    reference_config = CONFIGS[REFERENCE_CONFIG_NAME]
    evaluations = []
    for name, config in CONFIGS.items():
        for step in STEPS:
            joint = []
            source_times = []
            for index, _geometry in enumerate(selected):
                channels = []
                for pol in base.POLS:
                    source = results[identifier(index, pol, config, step)]
                    reference = results[
                        identifier(index, pol, reference_config, REFERENCE_STEP_NM)
                    ]
                    source_wavelength = np.asarray(source["wavelength_nm"], dtype=float)
                    reference_wavelength = np.asarray(reference["wavelength_nm"], dtype=float)
                    source_R = np.asarray(source["R"], dtype=float)
                    reference_R = np.asarray(reference["R"], dtype=float)
                    channels.append(
                        float(
                            delta_e2000(
                                base.labels_on_grid(source_wavelength, source_R),
                                base.labels_on_grid(reference_wavelength, reference_R),
                            )
                        )
                    )
                    runtime = float(source["time_s"])
                    if not np.isfinite(runtime) or runtime < 0.0:
                        raise ValueError("protocol source runtime is invalid")
                    source_times.append(runtime)
                joint.append(max(channels))
            summary = base.threshold_summary(joint)
            mean_task_seconds = float(np.mean(source_times))
            summary.update(
                {
                    "config_name": name,
                    "requested_nG": config[0],
                    "Nxy": config[1],
                    "wavelength_step_nm": step,
                    "mean_task_seconds_estimate": mean_task_seconds,
                    "estimated_wall_hours_16_workers_6000": (
                        mean_task_seconds * 6000 / 16 / 3600
                    ),
                    "passed": summary["mean_lt_1_15"] and summary["all_lt_2_3"],
                }
            )
            evaluations.append(summary)
    passed = [item for item in evaluations if item["passed"]]
    candidate = (
        min(
            passed,
            key=lambda item: (
                item["estimated_wall_hours_16_workers_6000"],
                item["requested_nG"],
                item["Nxy"],
                item["wavelength_step_nm"],
            ),
        )
        if passed
        else None
    )
    return {
        "selection_population": "initial_eight_cases_only",
        "holdout_used_for_selection": False,
        "reference": {
            "config_name": REFERENCE_CONFIG_NAME,
            "requested_nG": reference_config[0],
            "Nxy": reference_config[1],
            "wavelength_step_nm": REFERENCE_STEP_NM,
        },
        "thresholds": {
            "mean_joint_dE00_lt": base.MEAN_DE_LIMIT,
            "all_joint_dE00_lt": base.PER_GEOMETRY_DE_LIMIT,
        },
        "evaluations": evaluations,
        "lowest_cost_passing_protocol": candidate,
        "any_protocol_passed": bool(passed),
    }
