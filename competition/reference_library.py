"""Read-only TiO2/SiO2/air RCWA reference library for the competition UI.

The JSONL file is an audited display/reference asset.  This module binds the
asset to its audit hash before exposing any record and deliberately supports
only the conditions represented by that audit: TiO2 on SiO2, air background,
p polarization, normal incidence, nG=151 and Nxy=256.

This library is not a training source and does not infer or interpolate a
reference result for a geometry that is absent from the JSONL file.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


class ReferenceLibraryError(RuntimeError):
    """Raised when the bound reference asset cannot be trusted or loaded."""


_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORDS_PATH = _ROOT / "competition" / "tio2_air_reference_records_v1.jsonl"
DEFAULT_AUDIT_PATH = _ROOT / "competition" / "tio2_air_day_audit_20260930.json"
DEFAULT_COLOR_AUDIT_PATH = _ROOT / "competition" / "tio2_air_day_color_audit_20260930.json"

# Bound to the independently audited 21,088-record JSONL file.
EXPECTED_RECORDS_SHA256 = (
    "01BD65E80DF2A3C55BD92316CCD7882CD050969057278DB885BD0EAE83BB8403"
)
EXPECTED_RECORD_COUNT = 21088
EXPECTED_UNIQUE_GEOMETRIES = 21083
REFERENCE_CONDITIONS = {
    "material": "TiO2",
    "substrate": "SiO2",
    "background": "air",
    "polarization": "p",
    "incidence_angle_deg": 0.0,
    "nG_requested": 151,
    "Nxy": 256,
    "wavelength_start_nm": 380.0,
    "wavelength_stop_nm": 780.0,
    "wavelength_step_nm": 5.0,
    "wavelength_count": 81,
}

_DISPLAY_MATERIAL = {
    "TiO2": "TiO2 (anatase)",
    "TiO2 (anatase)": "TiO2",
}
_DISPLAY_SUBSTRATE = {
    "SiO2": "SiO2 (fused silica)",
    "SiO2 (fused silica)": "SiO2",
}
_DISPLAY_POLARIZATION = {
    "p": "TM (p-pol)",
    "TM (p-pol)": "p",
}
_WAVELENGTHS_NM = np.arange(380.0, 781.0, 5.0, dtype=np.float64)
_SCALAR_RECORD_FIELDS = (
    "type", "index", "batch", "D_nm", "H_nm", "P_nm", "material",
    "substrate", "background", "polarization", "incidence_angle_deg",
    "nG_requested", "Nxy", "finite", "max_abs_rt_error",
    "conservation_pass", "elapsed_s",
)


def sha256_file(path: Path) -> str:
    """Return an uppercase SHA256 without loading the 95 MB asset at once."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def geometry_key(d_nm: Any, h_nm: Any, p_nm: Any) -> tuple[int, int, int] | None:
    """Normalize an exact integer geometry; reject silently rounded inputs."""
    try:
        values = tuple(float(value) for value in (d_nm, h_nm, p_nm))
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(value) for value in values):
        return None
    if not all(abs(value - round(value)) <= 1e-9 for value in values):
        return None
    return tuple(int(round(value)) for value in values)


def _validate_audit_report(path: Path, records_hash: str) -> dict[str, Any]:
    if not path.is_file():
        raise ReferenceLibraryError(f"审核报告不存在: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReferenceLibraryError(f"审核报告无法读取: {type(exc).__name__}") from exc
    checks = report.get("checks")
    if not isinstance(checks, Mapping) or not checks or not all(checks.values()):
        raise ReferenceLibraryError("结构/守恒审核报告未全部通过")
    if str(report.get("records_sha256", "")).upper() != records_hash:
        raise ReferenceLibraryError("记录文件与结构审核报告哈希不一致")
    if int(report.get("records", -1)) != EXPECTED_RECORD_COUNT:
        raise ReferenceLibraryError("结构审核报告记录数不符合绑定版本")
    return report


def _validate_color_audit(path: Path, records_hash: str) -> dict[str, Any]:
    if not path.is_file():
        raise ReferenceLibraryError(f"颜色审核报告不存在: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReferenceLibraryError(f"颜色审核报告无法读取: {type(exc).__name__}") from exc
    checks = report.get("checks")
    if not isinstance(checks, Mapping) or not checks or not all(checks.values()):
        raise ReferenceLibraryError("颜色复算审核报告未全部通过")
    if str(report.get("records_sha256", "")).upper() != records_hash:
        raise ReferenceLibraryError("记录文件与颜色审核报告哈希不一致")
    return report


def _validate_record(row: Mapping[str, Any]) -> None:
    required = {
        "type", "index", "batch", "D_nm", "H_nm", "P_nm", "material",
        "substrate", "background", "polarization", "incidence_angle_deg",
        "nG_requested", "Nxy", "wavelength_nm", "R", "T", "finite",
        "max_abs_rt_error", "conservation_pass", "xyz_d65", "lab_d65",
        "srgb_display",
    }
    missing = required.difference(row)
    if missing:
        raise ReferenceLibraryError(f"参考记录缺少字段: {sorted(missing)}")
    if row["type"] != "record" or not bool(row["finite"]):
        raise ReferenceLibraryError("参考记录类型或 finite 标志无效")
    for field, expected in (
        ("material", "TiO2"), ("substrate", "SiO2"),
        ("background", "air"), ("polarization", "p"),
        ("nG_requested", 151), ("Nxy", 256),
    ):
        if row[field] != expected:
            raise ReferenceLibraryError(f"参考记录条件漂移: {field}={row[field]!r}")
    if abs(float(row["incidence_angle_deg"])) > 1e-9:
        raise ReferenceLibraryError("参考记录不是法向入射")
    wavelengths = np.asarray(row["wavelength_nm"], dtype=float)
    reflectance = np.asarray(row["R"], dtype=float)
    transmittance = np.asarray(row["T"], dtype=float)
    if wavelengths.shape != (81,) or reflectance.shape != (81,) or transmittance.shape != (81,):
        raise ReferenceLibraryError("参考记录波长/R/T 不是 81 点")
    if not np.all(np.isfinite(wavelengths)) or not np.all(np.isfinite(reflectance)) or not np.all(np.isfinite(transmittance)):
        raise ReferenceLibraryError("参考记录含非有限数")
    if not np.allclose(wavelengths, np.arange(380.0, 781.0, 5.0), atol=1e-9, rtol=0.0):
        raise ReferenceLibraryError("参考记录波长网格漂移")
    if not bool(row["conservation_pass"]):
        raise ReferenceLibraryError("参考记录守恒标志未通过")


@dataclass(frozen=True)
class ReferenceMatch:
    """One canonical exact geometry record from the audited JSONL."""

    _metadata: Mapping[str, Any]
    _reflectance: np.ndarray
    _transmittance: np.ndarray
    _xyz: np.ndarray
    _lab: np.ndarray
    _srgb: np.ndarray
    source_sha256: str

    @classmethod
    def from_row(cls, row: Mapping[str, Any], source_sha256: str) -> "ReferenceMatch":
        return cls(
            _metadata={field: row[field] for field in _SCALAR_RECORD_FIELDS},
            _reflectance=np.asarray(row["R"], dtype=np.float64),
            _transmittance=np.asarray(row["T"], dtype=np.float64),
            _xyz=np.asarray(row["xyz_d65"], dtype=np.float64),
            _lab=np.asarray(row["lab_d65"], dtype=np.float64),
            _srgb=np.asarray(row["srgb_display"], dtype=np.float64),
            source_sha256=source_sha256,
        )

    @property
    def record(self) -> dict[str, Any]:
        """Rebuild one export-compatible row on demand, not for every record."""
        row = dict(self._metadata)
        row.update({
            "wavelength_nm": _WAVELENGTHS_NM.tolist(),
            "R": self._reflectance.tolist(),
            "T": self._transmittance.tolist(),
            "xyz_d65": self._xyz.tolist(),
            "lab_d65": self._lab.tolist(),
            "srgb_display": self._srgb.tolist(),
        })
        return row

    @property
    def geometry(self) -> tuple[int, int, int]:
        return (
            int(self._metadata["D_nm"]),
            int(self._metadata["H_nm"]),
            int(self._metadata["P_nm"]),
        )

    @property
    def wavelengths_nm(self) -> np.ndarray:
        return _WAVELENGTHS_NM

    @property
    def reflectance(self) -> np.ndarray:
        return self._reflectance

    @property
    def transmittance(self) -> np.ndarray:
        return self._transmittance

    @property
    def xyz_d65(self) -> np.ndarray:
        return self._xyz

    @property
    def lab_d65(self) -> np.ndarray:
        return self._lab

    @property
    def srgb_display(self) -> np.ndarray:
        return self._srgb

    @property
    def max_abs_rt_error(self) -> float:
        return float(self._metadata["max_abs_rt_error"])

    def export_payload(self) -> dict[str, Any]:
        return {
            "schema": "competition-tio2-air-reference-match-v1",
            "conditions": dict(REFERENCE_CONDITIONS),
            "records_sha256": self.source_sha256,
            "record": dict(self.record),
            "boundary": (
                "仅为已审核 JSONL 中存在的 TiO2/SiO2/air、p 偏振、法向入射、"
                "nG=151、Nxy=256 单柱结构；不作插值、训练或收敛声明。"
            ),
        }


@dataclass(frozen=True)
class ReferenceLibrary:
    """In-memory index built once from the hash-bound JSONL asset."""

    records_path: Path
    records_sha256: str
    by_geometry: Mapping[tuple[int, int, int], ReferenceMatch]
    record_count: int
    unique_geometry_count: int
    audit_report: Mapping[str, Any]
    color_audit_report: Mapping[str, Any]

    def lookup(self, d_nm: Any, h_nm: Any, p_nm: Any) -> ReferenceMatch | None:
        key = geometry_key(d_nm, h_nm, p_nm)
        return None if key is None else self.by_geometry.get(key)

    def supports_display_conditions(
        self,
        material: str,
        substrate: str,
        polarization: str,
        angle_deg: Any,
        *,
        structure_type: str = "single",
        far_field_enabled: bool = False,
    ) -> tuple[bool, str]:
        if structure_type != "single":
            return False, "当前参考库只覆盖单柱结构，双柱和 FP 腔暂不匹配。"
        if _DISPLAY_MATERIAL.get(str(material)) != "TiO2":
            return False, "参考库只覆盖 TiO2 (anatase) 柱材料。"
        if _DISPLAY_SUBSTRATE.get(str(substrate)) != "SiO2":
            return False, "参考库只覆盖 SiO2 (fused silica) 衬底。"
        if _DISPLAY_POLARIZATION.get(str(polarization)) != "p":
            return False, "参考库只绑定 TM (p-pol)；当前偏振不是 p。"
        try:
            if abs(float(angle_deg)) > 1e-9:
                return False, "参考库只覆盖 0° 法向入射。"
        except (TypeError, ValueError):
            return False, "入射角无效，无法查询参考库。"
        if far_field_enabled:
            return False, "当前启用了远场后处理，结果与法向入射参考不在同一路线。"
        return True, ""

    def metadata(self) -> dict[str, Any]:
        return {
            "records_path": str(self.records_path),
            "records_sha256": self.records_sha256,
            "record_count": self.record_count,
            "unique_geometry_count": self.unique_geometry_count,
            "conditions": dict(REFERENCE_CONDITIONS),
            "audit_schema": self.audit_report.get("schema"),
            "color_audit_schema": self.color_audit_report.get("schema"),
            "interpretation": self.audit_report.get("interpretation"),
            "color_interpretation": self.color_audit_report.get("interpretation"),
        }


def _load_reference_library_uncached(
    records_path: str,
    audit_path: str,
    color_audit_path: str,
) -> ReferenceLibrary:
    records = Path(records_path)
    if not records.is_file():
        raise ReferenceLibraryError(f"参考记录文件不存在: {records}")
    records_hash = sha256_file(records)
    if records_hash != EXPECTED_RECORDS_SHA256:
        raise ReferenceLibraryError(
            "参考记录 SHA256 与审核绑定值不一致: "
            f"{records_hash} != {EXPECTED_RECORDS_SHA256}"
        )
    audit_report = _validate_audit_report(Path(audit_path), records_hash)
    color_audit_report = _validate_color_audit(Path(color_audit_path), records_hash)

    by_geometry: dict[tuple[int, int, int], ReferenceMatch] = {}
    record_count = 0
    with records.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReferenceLibraryError(f"第 {line_no} 行 JSON 无法解析") from exc
            _validate_record(row)
            key = geometry_key(row["D_nm"], row["H_nm"], row["P_nm"])
            if key is None:
                raise ReferenceLibraryError(f"第 {line_no} 行几何不是整数")
            record_count += 1
            if key not in by_geometry:
                by_geometry[key] = ReferenceMatch.from_row(row, records_hash)
            else:
                # The independent color audit already checks duplicate outputs;
                # keep the first canonical row for deterministic UI lookups.
                previous = by_geometry[key]
                if (
                    not np.array_equal(
                        previous.reflectance, np.asarray(row["R"], dtype=float)
                    )
                    or not np.array_equal(
                        previous.transmittance, np.asarray(row["T"], dtype=float)
                    )
                ):
                    raise ReferenceLibraryError(f"几何 {key} 的重复记录输出不一致")

    if record_count != EXPECTED_RECORD_COUNT:
        raise ReferenceLibraryError(
            f"参考记录数不符合绑定版本: {record_count} != {EXPECTED_RECORD_COUNT}"
        )
    if len(by_geometry) != EXPECTED_UNIQUE_GEOMETRIES:
        raise ReferenceLibraryError(
            f"唯一几何数不符合绑定版本: {len(by_geometry)} != {EXPECTED_UNIQUE_GEOMETRIES}"
        )
    return ReferenceLibrary(
        records_path=records,
        records_sha256=records_hash,
        by_geometry=by_geometry,
        record_count=record_count,
        unique_geometry_count=len(by_geometry),
        audit_report=audit_report,
        color_audit_report=color_audit_report,
    )


@lru_cache(maxsize=1)
def load_reference_library(
    records_path: str | Path = DEFAULT_RECORDS_PATH,
    audit_path: str | Path = DEFAULT_AUDIT_PATH,
    color_audit_path: str | Path = DEFAULT_COLOR_AUDIT_PATH,
) -> ReferenceLibrary:
    """Load and hash-check the reference asset once per Python process."""
    return _load_reference_library_uncached(
        str(Path(records_path).resolve()),
        str(Path(audit_path).resolve()),
        str(Path(color_audit_path).resolve()),
    )


def clear_reference_library_cache() -> None:
    """Clear the process cache for an explicit local asset refresh."""
    load_reference_library.cache_clear()


def reference_condition_summary() -> str:
    return (
        "TiO2 / SiO2 / air · TM (p-pol) · 0° · nG=151 · Nxy=256 · "
        "380–780 nm / 5 nm / 81 点"
    )


__all__ = [
    "DEFAULT_RECORDS_PATH", "EXPECTED_RECORDS_SHA256", "EXPECTED_RECORD_COUNT",
    "EXPECTED_UNIQUE_GEOMETRIES", "REFERENCE_CONDITIONS", "ReferenceLibrary",
    "ReferenceLibraryError", "ReferenceMatch", "clear_reference_library_cache",
    "geometry_key", "load_reference_library", "reference_condition_summary",
    "sha256_file",
]
