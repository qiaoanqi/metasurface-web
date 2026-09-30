"""Small route contracts shared by deterministic UI analyses."""
from __future__ import annotations

from dataclasses import dataclass
import csv
import io
import json
import numpy as np
import re
from typing import Any, Callable, Iterable, Mapping

from color_utils import clamp01, spectrum_to_srgb


@dataclass(frozen=True)
class ForwardResult:
    """The single forward result consumed by preview, plots, CIE and exports."""

    wavelengths_nm: Any
    reflectance: Any
    rgb: Any
    provenance: dict[str, Any]
    spectrum_available: bool = True
    error: str = ""


def normalize_forward_result(
    wavelengths: Any,
    reflectance: Any,
    provenance: Mapping[str, Any],
    *,
    rgb: Any = None,
    error: str = "",
) -> ForwardResult:
    """Validate one route output and derive its only UI color from its spectrum."""
    provenance_dict = dict(provenance)
    if wavelengths is None or reflectance is None:
        return ForwardResult(
            None, None, None if rgb is None else np.asarray(rgb, dtype=float),
            provenance_dict, False, error or "当前路由未提供可用光谱")
    wls = np.asarray(wavelengths, dtype=float).reshape(-1)
    refl = np.asarray(reflectance, dtype=float).reshape(-1)
    if (wls.shape != refl.shape or wls.size == 0
            or not np.all(np.isfinite(wls)) or not np.all(np.isfinite(refl))):
        return ForwardResult(None, None, None, provenance_dict, False, "光谱形状或有限值校验失败")
    refl = np.clip(refl, 0.0, None)
    try:
        computed_rgb = np.asarray(spectrum_to_srgb(wls, refl), dtype=float).reshape(-1)
    except Exception as exc:
        return ForwardResult(None, None, None, provenance_dict, False, f"光谱颜色转换失败: {exc}")
    if computed_rgb.size != 3 or not np.all(np.isfinite(computed_rgb)):
        return ForwardResult(None, None, None, provenance_dict, False, "光谱颜色转换失败")
    return ForwardResult(wls, refl, clamp01(computed_rgb), provenance_dict, True, "")


def sync_forward_status(result: ForwardResult) -> ForwardResult:
    """Expose availability/error in the same provenance object as the result."""
    result.provenance["spectrum_available"] = bool(result.spectrum_available)
    result.provenance["spectrum_error"] = str(result.error)
    return result


def inverse_candidates_available(result: Any) -> bool:
    """Fail closed when an inverse model returns None or an empty collection."""
    if result is None:
        return False
    try:
        return len(result) > 0
    except TypeError:
        return False


def single_geometry_is_valid(diameter_nm: float, period_nm: float) -> bool:
    """Return whether a single pillar stays inside its unit-cell period."""
    try:
        return bool(np.isfinite(diameter_nm) and np.isfinite(period_nm)
                    and float(diameter_nm) <= float(period_nm))
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class FrozenSpectrumRoute:
    """One immutable evaluator choice for an entire UI analysis."""

    route_id: str
    route_label: str
    evaluator: Callable[..., Any]

    def evaluate(self, request: Mapping[str, Any]) -> Any:
        return self.evaluator(**dict(request))


@dataclass(frozen=True)
class PerturbationResolution:
    """A requested process perturbation and its explicit domain decision."""

    requested_value: float
    evaluated_value: float | None
    status: str
    reason: str = ""
    actual_delta_nm: float | None = None


@dataclass(frozen=True)
class ForwardExports:
    """Deterministic export representations of one validated forward result."""

    csv_text: str
    json_text: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class MappingDomainContract:
    """A material/route-specific domain explicitly allowed for a UI map."""

    available: bool
    domains: dict[str, tuple[float, float]]
    boundary: str
    reason: str = ""


@dataclass(frozen=True)
class MappingCellResult:
    """One map cell; unavailable cells intentionally carry no RGB value."""

    status: str
    rgb: Any = None
    reason: str = ""
    forward: ForwardResult | None = None


_ANALYTICAL_MAPPING_DOMAIN = {
    "d": (50.0, 350.0), "h": (80.0, 600.0), "p": (200.0, 600.0),
}

# Every entry is independently evidenced by the exact route implementation:
# - Analytical/far-field bounds: engine.MetaSurfaceColorEngine; materials are
#   limited to exact torch_model.CAUCHY and ccm.CCM_COEFF_TABLE pair keys so
#   neither backend can silently use a fuzzy or default material/substrate.
# - rcwa_surrogate and ml_surrogate are deliberately absent: their runtime
#   ONNX registries expose no per-model training-domain manifest. Optimizer
#   clamps and input normalization are not accepted as training-domain proof.
_MAPPING_DOMAIN_REGISTRY: dict[
    tuple[str, str, str], tuple[dict[str, tuple[float, float]], str]
] = {
    ("TiO2 (anatase)", "SiO2 (fused silica)", "lorentz_fano_fallback"): (_ANALYTICAL_MAPPING_DOMAIN, "TiO2/SiO2 解析响应控制域：D 50-350 nm，H 80-600 nm，P 200-600 nm"),
    ("TiO2 (anatase)", "SiO2 (fused silica)", "far_field_postprocessing"): (_ANALYTICAL_MAPPING_DOMAIN, "TiO2/SiO2 解析响应及远场后处理控制域：D 50-350 nm，H 80-600 nm，P 200-600 nm"),
}


def mapping_domain_contract(
    material: str,
    substrate: str,
    route_id: str,
) -> MappingDomainContract:
    """Return only an evidenced (material, substrate, route) mapping domain."""
    registered = _MAPPING_DOMAIN_REGISTRY.get(
        (str(material), str(substrate), str(route_id)))
    if registered is not None:
        domains, boundary = registered
        return MappingDomainContract(True, dict(domains), boundary)
    return MappingDomainContract(
        False, {}, "未注册",
        f"未注册 D-H 映射组合：material={material}，substrate={substrate}，"
        f"route={route_id or 'unknown'}；"
        "未借用其他材料或路线边界",
    )


def evaluate_mapping_cell(
    geometry: Mapping[str, float],
    domains: Mapping[str, tuple[float, float]],
    evaluator: Callable[[Mapping[str, float]], ForwardResult],
) -> MappingCellResult:
    """Evaluate one cell only after every geometry field passes its domain."""
    resolved = {key: float(value) for key, value in geometry.items()}
    for key in domains:
        if key not in resolved:
            continue
        resolved, resolution = resolve_perturbation(
            resolved, key, 0.0, domains=domains)
        if not resolved:
            return MappingCellResult(resolution.status, None, resolution.reason, None)
    try:
        forward = evaluator(resolved)
    except Exception as exc:
        # Resource identity failures invalidate the entire frozen analysis;
        # they must reach the snapshot transaction instead of becoming a cell.
        from ui_model_resources import ModelResourceUnavailable
        if isinstance(exc, ModelResourceUnavailable):
            raise
        return MappingCellResult(
            "unavailable_execution", None, f"当前路线求值失败: {type(exc).__name__}", None)
    if not isinstance(forward, ForwardResult) or not forward.spectrum_available or forward.rgb is None:
        reason = forward.error if isinstance(forward, ForwardResult) else "当前路线未返回 ForwardResult"
        return MappingCellResult("unavailable_execution", None, reason, forward if isinstance(forward, ForwardResult) else None)
    rgb = np.asarray(forward.rgb, dtype=float).reshape(-1)
    if rgb.shape != (3,) or not np.all(np.isfinite(rgb)):
        return MappingCellResult("unavailable_execution", None, "当前路线 RGB 未通过校验", forward)
    return MappingCellResult("available", clamp01(rgb), "", forward)


def build_mapping_cells(
    d_values: Iterable[float],
    h_values: Iterable[float],
    period_nm: float,
    contract: MappingDomainContract,
    evaluator: Callable[[Mapping[str, float]], ForwardResult],
) -> dict[tuple[int, int], MappingCellResult | None]:
    """Evaluate each requested mapping coordinate at most once."""
    d_samples = list(d_values)
    h_samples = list(h_values)
    cells: dict[tuple[int, int], MappingCellResult | None] = {}
    for hi, h_nm in enumerate(h_samples):
        for di, d_nm in enumerate(d_samples):
            cells[(hi, di)] = (
                evaluate_mapping_cell(
                    {"d": float(d_nm), "h": float(h_nm), "p": float(period_nm)},
                    contract.domains, evaluator)
                if contract.available else None
            )
    return cells


def nearest_available_mapping_index(
    current_d_nm: float,
    current_h_nm: float,
    d_values: Iterable[float],
    h_values: Iterable[float],
    cells: Mapping[tuple[int, int], MappingCellResult | None],
) -> tuple[int, int] | None:
    """Choose the nearest cell only from results that are actually available."""
    d_samples = np.asarray(list(d_values), dtype=float)
    h_samples = np.asarray(list(h_values), dtype=float)
    if not np.isfinite(current_d_nm) or not np.isfinite(current_h_nm):
        return None
    candidates: list[tuple[float, tuple[int, int]]] = []
    for index, cell in cells.items():
        hi, di = index
        if (cell is None or cell.status != "available"
                or hi < 0 or di < 0 or hi >= h_samples.size or di >= d_samples.size):
            continue
        distance_sq = ((d_samples[di] - float(current_d_nm)) ** 2
                       + (h_samples[hi] - float(current_h_nm)) ** 2)
        candidates.append((float(distance_sq), (hi, di)))
    return min(candidates, key=lambda item: (item[0], item[1]))[1] if candidates else None


def make_result_provenance(
    route_id: str,
    route_label: str,
    chain: Iterable[str],
    boundary: str,
    fallback_reason: str,
    model_version: str,
    material: str,
    substrate: str,
    polarization: str,
    angle_deg: float,
    na: float | str,
    theta_obs_deg: float,
    *,
    structure_type: str = "unknown",
    structure_label: str = "unknown",
    geometry: Mapping[str, Any] | None = None,
    geometry_summary: str = "unknown",
    mirror_type: str = "",
    stack_identity: str = "",
    spectrum_available: bool = False,
    spectrum_error: str = "",
) -> dict[str, Any]:
    """Build the auditable provenance shape consumed by every result surface."""
    geometry_payload: dict[str, float] = {}
    geometry_error = ""
    if geometry is None:
        geometry_error = "geometry provenance not provided"
    elif not isinstance(geometry, Mapping):
        geometry_error = "geometry provenance is not a mapping"
    else:
        try:
            for key, value in geometry.items():
                if isinstance(value, (bool, np.bool_)):
                    raise ValueError(f"boolean geometry field: {key}")
                numeric = float(value)
                if not np.isfinite(numeric):
                    raise ValueError(f"non-finite geometry field: {key}")
                geometry_payload[str(key)] = numeric
        except (TypeError, ValueError) as exc:
            geometry_payload = {}
            geometry_error = str(exc)
    normalized_structure_type = str(structure_type or "unknown")
    normalized_structure_label = str(structure_label or "unknown")
    normalized_summary = str(geometry_summary or "unknown")
    if not geometry_payload:
        normalized_summary = "unknown"
    return {
        "route_id": str(route_id),
        "route_label": str(route_label),
        "chain": list(chain),
        "boundary": str(boundary),
        "fallback_reason": str(fallback_reason),
        "model_version": str(model_version),
        "model_state": "not_recorded",
        "material": str(material),
        "substrate": str(substrate),
        "polarization": str(polarization),
        "angle_deg": float(angle_deg),
        "na": na,
        "theta_obs_deg": float(theta_obs_deg),
        "structure_type": normalized_structure_type,
        "structure_label": normalized_structure_label,
        "geometry": geometry_payload,
        "geometry_summary": normalized_summary,
        "geometry_provenance_error": geometry_error,
        "mirror_type": str(mirror_type or ""),
        "stack_identity": str(stack_identity or ""),
        "spectrum_available": bool(spectrum_available),
        "spectrum_error": str(spectrum_error),
    }


def build_forward_exports(
    wavelengths_nm: Any,
    reflectance: Any,
    rgb: Any,
    provenance: Mapping[str, Any],
) -> ForwardExports:
    """Build CSV/JSON from exactly the arrays and provenance shown in the UI."""
    wavelengths = np.asarray(wavelengths_nm, dtype=float).reshape(-1)
    spectrum = np.asarray(reflectance, dtype=float).reshape(-1)
    color = np.asarray(rgb, dtype=float).reshape(-1)
    if wavelengths.size == 0 or wavelengths.shape != spectrum.shape:
        raise ValueError("export requires matching non-empty wavelength/reflectance arrays")
    if not np.all(np.isfinite(wavelengths)) or not np.all(np.isfinite(spectrum)):
        raise ValueError("export requires finite wavelength/reflectance arrays")
    if color.shape != (3,) or not np.all(np.isfinite(color)):
        raise ValueError("export requires finite RGB shape (3,)")
    payload = {
        "wavelengths_nm": wavelengths.tolist(),
        "reflectance": spectrum.tolist(),
        "rgb": color.tolist(),
        "provenance": dict(provenance),
    }
    provenance_json = json.dumps(
        dict(provenance), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    )
    csv_buffer = io.StringIO(newline="")
    writer = csv.writer(csv_buffer, lineterminator="\n")
    writer.writerow(("Wavelength_nm", "Reflectance", "provenance_json"))
    for wavelength, value in zip(wavelengths, spectrum):
        writer.writerow((f"{wavelength:.6g}", f"{value:.9g}", provenance_json))
    return ForwardExports(
        csv_text=csv_buffer.getvalue(),
        json_text=json.dumps(payload, ensure_ascii=False, indent=2),
        payload=payload,
    )


def forward_export_basename(
    structure_kind: str,
    parameters: Mapping[str, Any],
) -> str:
    """Build a filesystem-safe basename that describes the actual structure."""
    kind = str(structure_kind).strip().lower()

    def nm(key: str) -> str:
        value = float(parameters[key])
        if not np.isfinite(value):
            raise ValueError(f"non-finite export parameter: {key}")
        text = f"{value:.1f}".rstrip("0").rstrip(".")
        return text.replace("-", "m").replace(".", "p")

    if kind == "single":
        return f"single_D-{nm('d')}_H-{nm('h')}_P-{nm('p')}"
    if kind == "dual":
        return (
            f"dual_D1-{nm('d1')}_H1-{nm('h1')}_D2-{nm('d2')}_H2-{nm('h2')}_P-{nm('p')}"
        )
    if kind == "fp":
        mirror_label = str(parameters["mirror"])
        if "DBR" in mirror_label.upper() or "介质" in mirror_label:
            mirror_token = "dbr"
        elif "AG" in mirror_label.upper() or "银" in mirror_label:
            mirror_token = "ag"
        else:
            mirror_token = re.sub(r"[^a-z0-9]+", "-", mirror_label.lower()).strip("-") or "mirror"
        basename = f"fp_{mirror_token}_T-{nm('t')}"
        if parameters.get("center_wavelength") is not None:
            basename += f"_C-{nm('center_wavelength')}"
        return basename
    raise ValueError(f"unsupported structure kind for export: {structure_kind}")


def resolve_perturbation(
    geometry: Mapping[str, float],
    key: str,
    delta_nm: float,
    *,
    domains: Mapping[str, tuple[float, float]] | None = None,
) -> tuple[dict[str, float], PerturbationResolution]:
    """Resolve +/- tolerance without clipping or silently changing geometry.

    ``evaluated_geometry`` is empty when the requested point is outside the
    registered model/physical domain.  Diameter/period feasibility follows the
    engine's D <= P boundary; no additional 0.8P rule is introduced here.
    """
    varied = {name: float(value) for name, value in geometry.items()}
    requested = float(varied[key]) + float(delta_nm)
    bounds = dict(domains or {
        "d": (50.0, 350.0), "d1": (50.0, 350.0), "d2": (50.0, 350.0),
        "h": (80.0, 600.0), "h1": (80.0, 600.0), "h2": (80.0, 600.0),
        "p": (200.0, 600.0),
    })
    lo, hi = bounds.get(key, (-float("inf"), float("inf")))
    reason = ""
    if requested < lo or requested > hi:
        reason = f"{key}={requested:g} 超出注册模型/物理域 [{lo:g}, {hi:g}]"
    else:
        varied[key] = requested
        diameters = [value for name, value in varied.items() if name.startswith("d")]
        period = varied.get("p")
        if period is not None and any(diameter > period for diameter in diameters):
            reason = f"几何不可行：D={max(diameters):g} 大于 P={period:g}"
    if reason:
        return {}, PerturbationResolution(requested, None, "unavailable_out_of_domain", reason)
    return varied, PerturbationResolution(
        requested, requested, "available", actual_delta_nm=requested - float(geometry[key]))


def evaluate_frozen_series(
    route: FrozenSpectrumRoute,
    requests: Iterable[Mapping[str, Any]],
) -> list[tuple[str, Any]]:
    """Evaluate every request with exactly one route; never substitute failures."""

    return [(route.route_id, route.evaluate(request)) for request in requests]


def route_results_consistent(expected_route_id: str, results: Iterable[tuple[str, Any]]) -> bool:
    """Explicit runtime check used by UI fail-closed guards."""
    materialized = list(results)
    return bool(materialized) and all(
        route_id == expected_route_id for route_id, _ in materialized)
