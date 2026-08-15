"""Direct, cost-aware comparison of production candidates to the 0.5 nm reference."""

from __future__ import annotations

from typing import Callable

import numpy as np

from color_utils import delta_e2000
from scripts import run_reference_resolution_escalation as base


CONFIGS = {
    "A": (290, 384),
    "B": (290, 512),
    "C": (365, 384),
    "D": (365, 512),
}
STEPS_FROM_1NM = (1.0, 2.0, 5.0)
STEPS_FROM_HALF_NM_D_ONLY = (0.5, 2.5)


def subsample(result: dict, step_nm: float) -> tuple[np.ndarray, np.ndarray]:
    wavelength = np.asarray(result["wavelength_nm"], dtype=float)
    reflectance = np.asarray(result["R"], dtype=float)
    source_step = float(result["step_nm"])
    ratio = step_nm / source_step
    stride = int(round(ratio))
    if stride < 1 or abs(ratio - stride) > 1e-12:
        raise ValueError(f"{step_nm} nm is not an exact subset of {source_step} nm")
    selected_wavelength = wavelength[::stride]
    selected_reflectance = reflectance[::stride]
    if selected_wavelength[0] != 380.0 or selected_wavelength[-1] != 780.0:
        raise ValueError("subsampled spectrum does not preserve both endpoints")
    return selected_wavelength, selected_reflectance


def evaluate_protocols(
    selected: list[dict],
    results: dict,
    identifier: Callable[[int, str, tuple[int, int], float], str],
) -> dict:
    evaluations = []
    combinations = [
        (name, config, step, 1.0)
        for name, config in CONFIGS.items()
        for step in STEPS_FROM_1NM
    ] + [
        ("D", CONFIGS["D"], step, 0.5) for step in STEPS_FROM_HALF_NM_D_ONLY
    ]
    for name, config, step, source_step in combinations:
        joint = []
        source_times = []
        source_samples = None
        target_samples = None
        for index, _geometry in enumerate(selected):
            channel_values = []
            for pol in base.POLS:
                source = results[identifier(index, pol, config, source_step)]
                reference = results[identifier(index, pol, base.FINE_CONFIG, 0.5)]
                candidate_wavelength, candidate_R = subsample(source, step)
                candidate_lab = base.labels_on_grid(candidate_wavelength, candidate_R)
                reference_lab = base.labels_on_grid(reference["wavelength_nm"], reference["R"])
                channel_values.append(float(delta_e2000(candidate_lab, reference_lab)))
                source_times.append(float(source["time_s"]))
                source_samples = len(source["wavelength_nm"])
                target_samples = len(candidate_wavelength)
            joint.append(max(channel_values))
        summary = base.threshold_summary(joint)
        mean_task_seconds = float(np.mean(source_times)) * target_samples / source_samples
        summary.update(
            {
                "config_name": name,
                "requested_nG": config[0],
                "Nxy": config[1],
                "wavelength_step_nm": step,
                "source_grid_nm": source_step,
                "mean_task_seconds_estimate": mean_task_seconds,
                "estimated_wall_hours_16_workers_6000": mean_task_seconds * 6000 / 16 / 3600,
                "passed": summary["mean_lt_1_15"] and summary["all_lt_2_3"],
            }
        )
        evaluations.append(summary)
    passed = [item for item in evaluations if item["passed"]]
    selected_protocol = min(
        passed,
        key=lambda item: (
            item["estimated_wall_hours_16_workers_6000"],
            item["requested_nG"],
            item["Nxy"],
            item["wavelength_step_nm"],
        ),
    ) if passed else None
    return {
        "reference": {"requested_nG": 365, "Nxy": 512, "wavelength_step_nm": 0.5},
        "thresholds": {
            "mean_joint_dE00_lt": base.MEAN_DE_LIMIT,
            "all_joint_dE00_lt": base.PER_GEOMETRY_DE_LIMIT,
        },
        "evaluations": evaluations,
        "lowest_cost_passing_protocol": selected_protocol,
        "any_protocol_passed": bool(passed),
    }
