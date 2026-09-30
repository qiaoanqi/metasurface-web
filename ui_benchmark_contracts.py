"""Pure contracts for honest inverse-method benchmark results."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from color_utils import delta_e2000, rgb_to_lab


_STATUSES = {"available", "unavailable", "error"}


def _required_text(value: object, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} is required")
    return text


@dataclass(frozen=True)
class BenchmarkRow:
    method_id: str
    label: str
    status: str
    elapsed_s: float | None
    delta_e2000: float | None
    route: str
    timing_source: str
    metric_source: str
    detail: str

    def __post_init__(self) -> None:
        for field in (
            "method_id", "label", "route", "timing_source", "metric_source",
            "detail",
        ):
            object.__setattr__(self, field, _required_text(getattr(self, field), field))
        status = str(self.status).strip()
        if status not in _STATUSES:
            raise ValueError(f"unsupported benchmark status: {status}")
        object.__setattr__(self, "status", status)

        if status == "available":
            if self.elapsed_s is None or self.delta_e2000 is None:
                raise ValueError("available benchmark rows require timing and delta_e2000")
            elapsed = float(self.elapsed_s)
            metric = float(self.delta_e2000)
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("benchmark elapsed_s must be finite and non-negative")
            if not math.isfinite(metric) or metric < 0:
                raise ValueError("benchmark delta_e2000 must be finite and non-negative")
            object.__setattr__(self, "elapsed_s", elapsed)
            object.__setattr__(self, "delta_e2000", metric)
        elif self.elapsed_s is not None or self.delta_e2000 is not None:
            raise ValueError("unavailable/error rows cannot carry numeric benchmark results")

    @classmethod
    def unavailable(
        cls, method_id: str, label: str, *, route: str, detail: str,
    ) -> "BenchmarkRow":
        return cls(
            method_id, label, "unavailable", None, None, route,
            "未执行；无耗时数值", "未生成候选颜色；无 ΔE00 数值", detail,
        )

    @classmethod
    def error(
        cls, method_id: str, label: str, *, route: str, detail: str,
    ) -> "BenchmarkRow":
        return cls(
            method_id, label, "error", None, None, route,
            "执行未完成；不报告耗时数值",
            "未生成可验收的有限 sRGB；无 ΔE00 数值", detail,
        )


def validate_benchmark_cache(value: object) -> tuple[BenchmarkRow, ...] | None:
    """Return an immutable typed cache, or ``None`` for stale/invalid data."""
    if not isinstance(value, (tuple, list)):
        return None
    if not all(isinstance(row, BenchmarkRow) for row in value):
        return None
    return tuple(value)


def benchmark_row_from_rgb(
    method_id: str,
    label: str,
    *,
    elapsed_s: float,
    predicted_rgb: Iterable[float],
    target_rgb: Iterable[float],
    route: str,
    detail: str,
) -> BenchmarkRow:
    """Create one available row from a finite sRGB result using CIEDE2000."""
    predicted = np.asarray(tuple(predicted_rgb), dtype=float)
    target = np.asarray(tuple(target_rgb), dtype=float)
    for name, rgb in (("predicted_rgb", predicted), ("target_rgb", target)):
        if rgb.shape != (3,):
            raise ValueError(f"{name} must have shape (3,)")
        if not np.all(np.isfinite(rgb)):
            raise ValueError(f"{name} must be finite")
        if np.any(rgb < 0.0) or np.any(rgb > 1.0):
            raise ValueError(f"{name} must be normalized sRGB in [0, 1]")
    metric = delta_e2000(rgb_to_lab(target), rgb_to_lab(predicted))
    return BenchmarkRow(
        method_id=method_id,
        label=label,
        status="available",
        elapsed_s=elapsed_s,
        delta_e2000=metric,
        route=route,
        timing_source="time.perf_counter；仅计当次方法调用",
        metric_source=(
            "方法返回的有限归一化 sRGB → color_utils.rgb_to_lab "
            "→ color_utils.delta_e2000（相对固定目标色）"
        ),
        detail=detail,
    )
