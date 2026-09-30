"""Fail-closed contracts for CIE plots and model-gamut presentation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from color_utils import spectrum_to_xyz, xyz_to_xy
from ui_forward_routes import ForwardResult


SRGB_TRIANGLE_XY = np.array(
    [[0.64, 0.33], [0.30, 0.60], [0.15, 0.06], [0.64, 0.33]],
    dtype=float,
)


@dataclass(frozen=True)
class CurrentCIEPoint:
    """One chromaticity point derived from an available canonical spectrum."""

    available: bool
    xy: tuple[float, float] | None
    rgb: np.ndarray | None
    reason: str
    provenance_caption: str


@dataclass(frozen=True)
class CIEPlotData:
    """Validated standard plot data plus an optional canonical current point."""

    spectral_locus_xy: np.ndarray
    srgb_triangle_xy: np.ndarray
    current: CurrentCIEPoint


@dataclass(frozen=True)
class GamutResult:
    """A named model-route gamut result with explicit partial/failure evidence."""

    route_label: str
    system_label: str
    points_xy: np.ndarray
    available: bool
    warning: str = ""
    error: str = ""


@dataclass(frozen=True)
class GamutSamples:
    """Raw xy samples plus an explicit count of failed sampling calls."""

    points_xy: Any
    failed_samples: int = 0


def forward_provenance_caption(provenance: Mapping[str, Any]) -> str:
    """Render result identity only from the canonical provenance mapping."""
    material = str(provenance.get("material") or "未记录")
    substrate = str(provenance.get("substrate") or "未记录")
    polarization = str(provenance.get("polarization") or "未记录")
    route = str(provenance.get("route_label") or "未知路线")
    model = str(provenance.get("model_version") or "未记录")
    structure = str(provenance.get("structure_label") or "unknown")
    geometry_summary = str(provenance.get("geometry_summary") or "unknown")
    angle = provenance.get("angle_deg", "未记录")
    na = provenance.get("na", "未启用")
    theta_obs = provenance.get("theta_obs_deg", 0.0)
    try:
        angle_text = f"{float(angle):g} deg"
    except (TypeError, ValueError):
        angle_text = str(angle)
    try:
        theta_text = f"{float(theta_obs):g} deg"
    except (TypeError, ValueError):
        theta_text = str(theta_obs)
    return (
        f"结构：{structure}；几何：{geometry_summary}；"
        f"材料：{material}；衬底：{substrate}；偏振：{polarization}；"
        f"入射角：{angle_text}；NA：{na}；观察角：{theta_text}；"
        f"路线：{route}；模型：{model}"
    )


def current_cie_point(forward: ForwardResult) -> CurrentCIEPoint:
    """Derive xy from the canonical spectrum; unavailable results stay empty."""
    caption = forward_provenance_caption(getattr(forward, "provenance", {}))
    if not isinstance(forward, ForwardResult):
        return CurrentCIEPoint(False, None, None, "当前结果不是 ForwardResult", caption)
    if not forward.spectrum_available:
        return CurrentCIEPoint(
            False, None, None, forward.error or "当前路线没有可用光谱", caption)
    if forward.wavelengths_nm is None or forward.reflectance is None or forward.rgb is None:
        return CurrentCIEPoint(False, None, None, "当前可用结果缺少光谱或 RGB", caption)
    try:
        xyz = np.asarray(
            spectrum_to_xyz(forward.wavelengths_nm, forward.reflectance), dtype=float,
        ).reshape(-1)
        xy = np.asarray(xyz_to_xy(xyz), dtype=float).reshape(-1)
        rgb = np.asarray(forward.rgb, dtype=float).reshape(-1)
    except Exception as exc:
        return CurrentCIEPoint(
            False, None, None, f"当前光谱的 CIE 坐标计算失败：{type(exc).__name__}", caption)
    if xy.shape != (2,) or rgb.shape != (3,) or not np.all(np.isfinite(xy)) or not np.all(np.isfinite(rgb)):
        return CurrentCIEPoint(False, None, None, "当前 CIE 点未通过形状或有限值校验", caption)
    return CurrentCIEPoint(True, (float(xy[0]), float(xy[1])), rgb.copy(), "", caption)


def build_cie_plot_data(
    forward: ForwardResult,
    cie_x: Any,
    cie_y: Any,
    cie_z: Any,
) -> CIEPlotData:
    """Build standard locus data and the optional current point."""
    x_bar = np.asarray(cie_x, dtype=float).reshape(-1)
    y_bar = np.asarray(cie_y, dtype=float).reshape(-1)
    z_bar = np.asarray(cie_z, dtype=float).reshape(-1)
    if x_bar.shape != y_bar.shape or x_bar.shape != z_bar.shape or x_bar.size < 3:
        raise ValueError("CIE CMF arrays must have one matching non-empty shape")
    denominator = x_bar + y_bar + z_bar
    valid = np.isfinite(denominator) & (denominator > 0)
    if np.count_nonzero(valid) < 3:
        raise ValueError("CIE CMF arrays do not contain a valid spectral locus")
    locus = np.column_stack((x_bar[valid] / denominator[valid], y_bar[valid] / denominator[valid]))
    if not np.all(np.isfinite(locus)):
        raise ValueError("CIE spectral locus contains non-finite coordinates")
    return CIEPlotData(locus, SRGB_TRIANGLE_XY.copy(), current_cie_point(forward))


def evaluate_gamut(
    route_label: str,
    system_label: str,
    evaluator: Callable[[], Any],
    *,
    sample_failures: int = 0,
) -> GamutResult:
    """Evaluate and validate one declared route without inventing fallback points."""
    try:
        evaluated = evaluator()
    except Exception as exc:
        return GamutResult(
            route_label, system_label, np.zeros((0, 2), dtype=float), False,
            error=f"{system_label} 色域计算失败：{type(exc).__name__}: {exc}",
        )
    try:
        if isinstance(evaluated, GamutSamples):
            raw_points = evaluated.points_xy
            sample_failures += max(0, int(evaluated.failed_samples))
        else:
            raw_points = evaluated
        points = np.asarray(raw_points, dtype=float)
    except (TypeError, ValueError) as exc:
        return GamutResult(
            route_label, system_label, np.zeros((0, 2), dtype=float), False,
            error=f"{system_label} 色域结果无法转换：{type(exc).__name__}",
        )
    if points.ndim != 2 or points.shape[1:] != (2,) or points.shape[0] < 3:
        return GamutResult(
            route_label, system_label, np.zeros((0, 2), dtype=float), False,
            error=f"{system_label} 色域点不足或形状错误",
        )
    finite_rows = np.all(np.isfinite(points), axis=1)
    if not np.all(finite_rows):
        return GamutResult(
            route_label, system_label, np.zeros((0, 2), dtype=float), False,
            error=f"{system_label} 色域含 NaN/Inf",
        )
    warning = (
        f"{system_label} 有 {int(sample_failures)} 个采样点计算失败；"
        "图中仅显示已成功样本。"
        if sample_failures else ""
    )
    return GamutResult(route_label, system_label, points, True, warning=warning)
