"""Deterministic, session-local contracts for expensive UI analyses."""
from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, MutableMapping

import numpy as np


ANALYSIS_SNAPSHOT_SCHEMA_VERSION = 1
ANALYSIS_PROTOCOL_VERSION = "ui-analysis-snapshot-v1"
ANALYSIS_SESSION_KEYS = {
    "sensitivity": "_ui_analysis_sensitivity_v1",
    "mapping": "_ui_analysis_mapping_v1",
    "angle": "_ui_analysis_angle_v1",
    "difference": "_ui_analysis_difference_v1",
}
SOURCE_ARTIFACT_PROTOCOL_VERSION = "ui-source-artifact-v1"
SOURCE_ARTIFACT_MAX_BYTES = 2 * 1024 * 1024


def _json_value(value: Any) -> Any:
    """Return a strict JSON value; reject non-finite numbers and odd objects."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON does not allow NaN/Inf")
        return float(value)
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            text = str(key)
            if not text:
                raise ValueError("canonical JSON keys must be non-empty")
            normalized[text] = _json_value(item)
        return normalized
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        _json_value(value), ensure_ascii=True, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SourceDependencyIdentity:
    relative_path: str
    status: str
    sha256: str | None
    byte_count: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "status": self.status,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
        }


@dataclass(frozen=True)
class SourceArtifactIdentity:
    available: bool
    version: str
    reason: str
    dependencies: tuple[SourceDependencyIdentity, ...]


def source_artifact_identity(
    project_root: str | Path,
    relative_paths: Any,
    *,
    max_bytes: int = SOURCE_ARTIFACT_MAX_BYTES,
    expected_sha256: Mapping[str, str] | None = None,
) -> SourceArtifactIdentity:
    """Hash current small source bytes without a cache or model-file access."""
    limit = int(max_bytes)
    if limit < 1:
        raise ValueError("source artifact max_bytes must be positive")
    try:
        root = Path(project_root).resolve()
    except (OSError, RuntimeError):
        dependency = SourceDependencyIdentity(
            relative_path="<project_root>", status="unresolvable",
            sha256=None, byte_count=None,
        )
        digest = canonical_sha256({
            "protocol_version": SOURCE_ARTIFACT_PROTOCOL_VERSION,
            "dependencies": [dependency.to_dict()],
        })
        return SourceArtifactIdentity(
            False, f"unavailable:{digest}", "project root:unresolvable", (dependency,),
        )
    dependencies = []
    for raw_relative_path in tuple(relative_paths):
        relative_path = str(raw_relative_path).replace("\\", "/")
        status, digest, byte_count = "available", None, None
        try:
            unresolved = Path(raw_relative_path)
            if unresolved.is_absolute():
                raise ValueError("path_escape")
            resolved = (root / unresolved).resolve()
            resolved.relative_to(root)
            stat = resolved.stat()
            if stat.st_size > limit:
                status, byte_count = "oversize", int(stat.st_size)
            else:
                data = resolved.read_bytes()
                if not isinstance(data, (bytes, bytearray)):
                    status = "invalid_bytes"
                    data = b""
                byte_count = len(data)
                if status == "invalid_bytes":
                    pass
                elif byte_count > limit:
                    status = "oversize"
                else:
                    digest = hashlib.sha256(data).hexdigest()
                    expected = (expected_sha256 or {}).get(relative_path)
                    if expected is not None and digest.lower() != str(expected).lower():
                        status = "hash_mismatch"
        except FileNotFoundError:
            status = "missing"
        except ValueError:
            status = "path_escape"
        except RuntimeError:
            status = "unresolvable"
        except OSError:
            status = "unreadable"
        dependencies.append(SourceDependencyIdentity(
            relative_path=relative_path,
            status=status,
            sha256=digest,
            byte_count=byte_count,
        ))
    records = tuple(dependencies)
    available = bool(records) and all(item.status == "available" for item in records)
    identity_payload = {
        "protocol_version": SOURCE_ARTIFACT_PROTOCOL_VERSION,
        "dependencies": [item.to_dict() for item in records],
    }
    digest = canonical_sha256(identity_payload)
    version = f"sha256:{digest}" if available else f"unavailable:{digest}"
    reason = "" if available else ", ".join(
        f"{item.relative_path}:{item.status}"
        for item in records if item.status != "available"
    ) or "no source dependencies registered"
    return SourceArtifactIdentity(available, version, reason, records)


@dataclass(frozen=True)
class AnalysisContext:
    analysis_type: str
    structure_type: str
    geometry_json: str
    material: str
    substrate: str
    polarization: str
    angle_deg: float
    route_id: str
    model_version: str
    artifact_version: str
    registry_version: str
    far_field_enabled: bool
    na: float
    theta_obs_deg: float
    sampling_json: str
    schema_version: int = ANALYSIS_SNAPSHOT_SCHEMA_VERSION
    protocol_version: str = ANALYSIS_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.analysis_type not in ANALYSIS_SESSION_KEYS:
            raise ValueError(f"unsupported analysis type: {self.analysis_type}")
        for name in (
            "structure_type", "material", "substrate", "polarization",
            "route_id", "model_version", "artifact_version",
            "registry_version", "protocol_version",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"empty analysis context field: {name}")
        if int(self.schema_version) != ANALYSIS_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError("unsupported analysis context schema")
        angle = float(self.angle_deg)
        if not math.isfinite(angle) or not 0.0 <= angle <= 80.0:
            raise ValueError("angle_deg must be finite within [0, 80]")
        geometry = json.loads(self.geometry_json)
        sampling = json.loads(self.sampling_json)
        if not isinstance(geometry, dict) or not geometry:
            raise ValueError("analysis geometry must be a non-empty object")
        if not isinstance(sampling, dict) or not sampling:
            raise ValueError("analysis sampling must be a non-empty object")
        geometry_json = canonical_json(geometry)
        sampling_json = canonical_json(sampling)
        far_field = bool(self.far_field_enabled)
        if far_field:
            na = float(self.na)
            theta = float(self.theta_obs_deg)
            if not math.isfinite(na) or not 0.05 <= na <= 0.95:
                raise ValueError("enabled far-field NA must be within [0.05, 0.95]")
            if not math.isfinite(theta) or not 0.0 <= theta <= 80.0:
                raise ValueError("enabled far-field theta must be within [0, 80]")
        else:
            na, theta = 0.1, 0.0
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "angle_deg", angle)
        object.__setattr__(self, "geometry_json", geometry_json)
        object.__setattr__(self, "sampling_json", sampling_json)
        object.__setattr__(self, "far_field_enabled", far_field)
        object.__setattr__(self, "na", na)
        object.__setattr__(self, "theta_obs_deg", theta)

    @classmethod
    def create(
        cls, analysis_type: str, *, structure_type: str,
        geometry: Mapping[str, Any], material: str, substrate: str,
        polarization: str, angle_deg: float, route_id: str,
        model_version: str, artifact_version: str, registry_version: str,
        far_field_enabled: bool, na: float, theta_obs_deg: float,
        sampling: Mapping[str, Any],
    ) -> "AnalysisContext":
        return cls(
            analysis_type=str(analysis_type), structure_type=str(structure_type),
            geometry_json=canonical_json(dict(geometry)), material=str(material),
            substrate=str(substrate), polarization=str(polarization),
            angle_deg=float(angle_deg), route_id=str(route_id),
            model_version=str(model_version), artifact_version=str(artifact_version),
            registry_version=str(registry_version),
            far_field_enabled=bool(far_field_enabled), na=float(na),
            theta_obs_deg=float(theta_obs_deg),
            sampling_json=canonical_json(dict(sampling)),
        )

    @property
    def geometry(self) -> dict[str, Any]:
        return json.loads(self.geometry_json)

    @property
    def sampling(self) -> dict[str, Any]:
        return json.loads(self.sampling_json)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol_version": self.protocol_version,
            "analysis_type": self.analysis_type,
            "structure_type": self.structure_type,
            "geometry": self.geometry,
            "material": self.material,
            "substrate": self.substrate,
            "polarization": self.polarization,
            "angle_deg": self.angle_deg,
            "route_id": self.route_id,
            "model_version": self.model_version,
            "artifact_version": self.artifact_version,
            "registry_version": self.registry_version,
            "far_field": {
                "enabled": self.far_field_enabled,
                "na": self.na,
                "theta_obs_deg": self.theta_obs_deg,
            },
            "sampling": self.sampling,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AnalysisContext":
        expected = {
            "schema_version", "protocol_version", "analysis_type",
            "structure_type", "geometry", "material", "substrate",
            "polarization", "angle_deg", "route_id", "model_version",
            "artifact_version", "registry_version", "far_field", "sampling",
        }
        if set(value) != expected or not isinstance(value.get("far_field"), Mapping):
            raise ValueError("analysis context fields are incomplete or unexpected")
        far_field = value["far_field"]
        if set(far_field) != {"enabled", "na", "theta_obs_deg"}:
            raise ValueError("far-field context fields are incomplete or unexpected")
        return cls(
            analysis_type=value["analysis_type"],
            structure_type=value["structure_type"],
            geometry_json=canonical_json(value["geometry"]),
            material=value["material"], substrate=value["substrate"],
            polarization=value["polarization"], angle_deg=value["angle_deg"],
            route_id=value["route_id"], model_version=value["model_version"],
            artifact_version=value["artifact_version"],
            registry_version=value["registry_version"],
            far_field_enabled=far_field["enabled"], na=far_field["na"],
            theta_obs_deg=far_field["theta_obs_deg"],
            sampling_json=canonical_json(value["sampling"]),
            schema_version=value["schema_version"],
            protocol_version=value["protocol_version"],
        )

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.to_dict())

    @property
    def session_key(self) -> str:
        return ANALYSIS_SESSION_KEYS[self.analysis_type]


def _finite_rgb(value: Any) -> list[float]:
    rgb = np.asarray(value, dtype=float)
    if rgb.shape != (3,) or not np.all(np.isfinite(rgb)):
        raise ValueError("RGB must be finite shape (3,)")
    if np.any(rgb < 0.0) or np.any(rgb > 1.0):
        raise ValueError("RGB must stay within [0, 1]")
    return [float(channel) for channel in rgb]


def _analysis_artifact_fields(payload: Mapping[str, Any]) -> dict[str, str]:
    analysis_value = str(payload.get("analysis_artifact_version", ""))
    if not analysis_value.startswith(("sha256:", "unavailable:")):
        raise ValueError("invalid analysis artifact version")
    if "model_artifact_version" not in payload:
        return {"analysis_artifact_version": analysis_value}
    value = str(payload["model_artifact_version"])
    if value != "not_applicable" and not value.startswith("sha256:"):
        raise ValueError("invalid model artifact version")
    return {
        "analysis_artifact_version": analysis_value,
        "model_artifact_version": value,
    }


def _validate_sensitivity(payload: Mapping[str, Any]) -> dict[str, Any]:
    status = str(payload.get("status", ""))
    if status == "unavailable":
        return {
            "status": status, "reason": str(payload.get("reason", "")), "rows": [],
            **_analysis_artifact_fields(payload),
        }
    if status != "available":
        raise ValueError("unsupported sensitivity status")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("available sensitivity requires rows")
    normalized_rows = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"label", "field", "lower", "upper"}:
            raise ValueError("invalid sensitivity row")
        sides = {}
        for side in ("lower", "upper"):
            entry = row[side]
            if not isinstance(entry, Mapping):
                raise ValueError("invalid sensitivity side")
            entry_status = str(entry.get("status", ""))
            requested = float(entry.get("requested_value"))
            if not math.isfinite(requested):
                raise ValueError("non-finite sensitivity request")
            if entry_status == "available":
                evaluated = float(entry.get("evaluated_value"))
                actual_delta = float(entry.get("actual_delta_nm"))
                delta_e = float(entry.get("delta_e2000"))
                if not all(math.isfinite(v) for v in (evaluated, actual_delta, delta_e)) or delta_e < 0:
                    raise ValueError("invalid available sensitivity numbers")
                sides[side] = {
                    "status": entry_status, "requested_value": requested,
                    "evaluated_value": evaluated, "actual_delta_nm": actual_delta,
                    "rgb": _finite_rgb(entry.get("rgb")), "delta_e2000": delta_e,
                    "reason": "",
                }
            elif entry_status == "unavailable":
                if entry.get("rgb") is not None or entry.get("delta_e2000") is not None:
                    raise ValueError("unavailable sensitivity side cannot carry numbers")
                sides[side] = {
                    "status": entry_status, "requested_value": requested,
                    "evaluated_value": None, "actual_delta_nm": None,
                    "rgb": None, "delta_e2000": None,
                    "reason": str(entry.get("reason", "")),
                }
            else:
                raise ValueError("unsupported sensitivity side status")
        normalized_rows.append({
            "label": str(row["label"]), "field": str(row["field"]), **sides,
        })
    return {
        "status": status, "reason": "", "base_rgb": _finite_rgb(payload.get("base_rgb")),
        "rows": normalized_rows, "route_id": str(payload.get("route_id", "")),
        **_analysis_artifact_fields(payload),
    }


def _validate_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    status = str(payload.get("status", ""))
    if status == "unavailable":
        return {
            "status": status, "reason": str(payload.get("reason", "")), "cells": [],
            **_analysis_artifact_fields(payload),
        }
    if status != "available":
        raise ValueError("unsupported mapping status")
    d_values = np.asarray(payload.get("d_values"), dtype=float)
    h_values = np.asarray(payload.get("h_values"), dtype=float)
    if d_values.ndim != 1 or h_values.ndim != 1 or d_values.size < 1 or h_values.size < 1:
        raise ValueError("mapping samples must be non-empty 1-D arrays")
    if not np.all(np.isfinite(d_values)) or not np.all(np.isfinite(h_values)):
        raise ValueError("mapping samples contain NaN/Inf")
    cells = payload.get("cells")
    if not isinstance(cells, list) or len(cells) != d_values.size * h_values.size:
        raise ValueError("mapping cell count does not match samples")
    normalized_cells = []
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise ValueError("invalid mapping cell")
        cell_status = str(cell.get("status", ""))
        hi, di = int(cell.get("hi", -1)), int(cell.get("di", -1))
        if not (0 <= hi < h_values.size and 0 <= di < d_values.size):
            raise ValueError("mapping cell index is out of range")
        if cell_status == "available":
            rgb, reason = _finite_rgb(cell.get("rgb")), ""
        elif cell_status == "unavailable":
            if cell.get("rgb") is not None:
                raise ValueError("unavailable mapping cell cannot carry RGB")
            rgb, reason = None, str(cell.get("reason", ""))
        else:
            raise ValueError("unsupported mapping cell status")
        normalized_cells.append({"hi": hi, "di": di, "status": cell_status, "rgb": rgb, "reason": reason})
    if len({(cell["hi"], cell["di"]) for cell in normalized_cells}) != len(cells):
        raise ValueError("mapping cell indices are duplicated")
    return {
        "status": status, "reason": "", "d_values": d_values.tolist(),
        "h_values": h_values.tolist(), "cells": normalized_cells,
        "route_id": str(payload.get("route_id", "")),
        "boundary": str(payload.get("boundary", "")),
        **_analysis_artifact_fields(payload),
    }


def build_angle_payload(
    angles_deg: Any, raw_rgb: Any, available_mask: Any, route_ids: Any,
) -> dict[str, Any]:
    angles = np.asarray(angles_deg, dtype=float)
    rgb = np.asarray(raw_rgb, dtype=float)
    mask = np.asarray(available_mask, dtype=bool)
    routes = [str(value) for value in route_ids]
    if angles.ndim != 1 or angles.size < 1 or not np.all(np.isfinite(angles)):
        raise ValueError("angle samples must be a finite non-empty 1-D array")
    if rgb.shape != (angles.size, 3) or mask.shape != (angles.size,) or len(routes) != angles.size:
        raise ValueError("angle scan shapes do not match")
    if np.any(mask & ~np.all(np.isfinite(rgb), axis=1)):
        raise ValueError("available angle rows must have finite RGB")
    if np.any(mask & (np.any(rgb < 0.0, axis=1) | np.any(rgb > 1.0, axis=1))):
        raise ValueError("available angle RGB must stay within [0, 1]")
    if np.any(~mask & ~np.all(np.isnan(rgb), axis=1)):
        raise ValueError("unavailable angle rows must be raw NaN")
    serialized = [
        [float(channel) for channel in row] if available else None
        for row, available in zip(rgb, mask)
    ]
    return {
        "status": "available", "reason": "", "angles_deg": angles.tolist(),
        "rgb": serialized, "available_mask": mask.tolist(), "route_ids": routes,
    }


def angle_payload_arrays(payload: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    if str(payload.get("status", "")) != "available":
        raise ValueError("angle payload is unavailable")
    angles = np.asarray(payload.get("angles_deg"), dtype=float)
    mask = np.asarray(payload.get("available_mask"), dtype=bool)
    rows = payload.get("rgb")
    routes = tuple(str(value) for value in payload.get("route_ids", []))
    if angles.ndim != 1 or not isinstance(rows, list) or len(rows) != angles.size:
        raise ValueError("angle payload shape mismatch")
    rgb = np.full((angles.size, 3), np.nan, dtype=float)
    if mask.shape != (angles.size,) or len(routes) != angles.size:
        raise ValueError("angle payload mask/route shape mismatch")
    for index, (row, available) in enumerate(zip(rows, mask)):
        if available:
            if row is None:
                raise ValueError("available angle row is null")
            rgb[index] = _finite_rgb(row)
        elif row is not None:
            raise ValueError("unavailable angle row must be null")
    # Re-run the raw contract so bool coercion or malformed values cannot pass.
    build_angle_payload(angles, rgb, mask, routes)
    return angles, rgb, mask, routes


def _validate_difference(payload: Mapping[str, Any]) -> dict[str, Any]:
    status = str(payload.get("status", ""))
    if status == "unavailable":
        return {
            "status": status, "reason": str(payload.get("reason", "")),
            **_analysis_artifact_fields(payload),
        }
    if status != "available":
        raise ValueError("unsupported difference status")
    metrics = np.asarray(payload.get("delta_e2000"), dtype=float)
    left = np.asarray(payload.get("fano_rgb"), dtype=float)
    right = np.asarray(payload.get("generic_rgb"), dtype=float)
    if metrics.ndim != 1 or metrics.size < 1:
        raise ValueError("difference metrics must be non-empty")
    if left.shape != (metrics.size, 3) or right.shape != (metrics.size, 3):
        raise ValueError("difference RGB shapes do not match")
    if not np.all(np.isfinite(metrics)) or np.any(metrics < 0):
        raise ValueError("difference metrics are invalid")
    normalized = {
        "status": status, "reason": "", "delta_e2000": metrics.tolist(),
        "fano_rgb": left.tolist(), "generic_rgb": right.tolist(),
        **_analysis_artifact_fields(payload),
    }
    for row in normalized["fano_rgb"] + normalized["generic_rgb"]:
        _finite_rgb(row)
    evidence = payload.get("evidence")
    if evidence is not None:
        if not isinstance(evidence, Mapping):
            raise ValueError("difference evidence must be a mapping")
        required = {
            "protocol_sha256", "result_sha256", "source_pt_sha256",
            "sample_count", "max_abs_diff", "tolerance",
        }
        if set(evidence) != required:
            raise ValueError("difference evidence fields are incomplete or unexpected")
        sample_count = int(evidence["sample_count"])
        max_abs_diff = float(evidence["max_abs_diff"])
        tolerance = float(evidence["tolerance"])
        if sample_count < 1 or not all(math.isfinite(v) and v >= 0 for v in (max_abs_diff, tolerance)):
            raise ValueError("difference evidence numbers are invalid")
        normalized["evidence"] = {
            "protocol_sha256": str(evidence["protocol_sha256"]),
            "result_sha256": str(evidence["result_sha256"]),
            "source_pt_sha256": str(evidence["source_pt_sha256"]),
            "sample_count": sample_count,
            "max_abs_diff": max_abs_diff,
            "tolerance": tolerance,
        }
    return normalized


def validate_analysis_payload(analysis_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("analysis payload must be a mapping")
    if analysis_type == "sensitivity":
        return _validate_sensitivity(payload)
    if analysis_type == "mapping":
        return _validate_mapping(payload)
    if analysis_type == "angle":
        if str(payload.get("status", "")) == "unavailable":
            return {
                "status": "unavailable", "reason": str(payload.get("reason", "")),
                **_analysis_artifact_fields(payload),
            }
        angles, rgb, mask, routes = angle_payload_arrays(payload)
        normalized = build_angle_payload(angles, rgb, mask, routes)
        normalized.update(_analysis_artifact_fields(payload))
        return normalized
    if analysis_type == "difference":
        return _validate_difference(payload)
    raise ValueError(f"unsupported analysis payload: {analysis_type}")


@dataclass(frozen=True)
class AnalysisSnapshot:
    context: AnalysisContext
    payload: dict[str, Any]
    payload_sha256: str

    @classmethod
    def create(cls, context: AnalysisContext, payload: Mapping[str, Any]) -> "AnalysisSnapshot":
        payload_with_context = dict(payload)
        payload_with_context.setdefault(
            "analysis_artifact_version", context.artifact_version)
        normalized = validate_analysis_payload(
            context.analysis_type, payload_with_context)
        if normalized["analysis_artifact_version"] != context.artifact_version:
            raise ValueError("payload analysis artifact does not match its context")
        return cls(context, normalized, canonical_sha256(normalized))

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": ANALYSIS_SNAPSHOT_SCHEMA_VERSION,
            "protocol_version": ANALYSIS_PROTOCOL_VERSION,
            "analysis_type": self.context.analysis_type,
            "context": self.context.to_dict(),
            "context_fingerprint": self.context.fingerprint,
            "payload": _json_value(self.payload),
            "payload_sha256": self.payload_sha256,
        }


@dataclass(frozen=True)
class SnapshotLoad:
    state: str
    snapshot: AnalysisSnapshot | None = None
    reason: str = ""


def validate_snapshot_record(value: Any) -> AnalysisSnapshot:
    if not isinstance(value, Mapping):
        raise TypeError("analysis snapshot record must be a mapping")
    expected = {
        "schema_version", "protocol_version", "analysis_type", "context",
        "context_fingerprint", "payload", "payload_sha256",
    }
    if set(value) != expected:
        raise ValueError("analysis snapshot fields are incomplete or unexpected")
    if value["schema_version"] != ANALYSIS_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("unsupported analysis snapshot schema")
    if value["protocol_version"] != ANALYSIS_PROTOCOL_VERSION:
        raise ValueError("unsupported analysis snapshot protocol")
    context = AnalysisContext.from_dict(value["context"])
    if value["analysis_type"] != context.analysis_type:
        raise ValueError("analysis snapshot type does not match context")
    if value["context_fingerprint"] != context.fingerprint:
        raise ValueError("analysis context fingerprint mismatch")
    payload = validate_analysis_payload(context.analysis_type, value["payload"])
    payload_hash = canonical_sha256(payload)
    if value["payload_sha256"] != payload_hash:
        raise ValueError("analysis payload hash mismatch")
    return AnalysisSnapshot(context, payload, payload_hash)


def load_analysis_snapshot(
    session_state: Mapping[str, Any], context: AnalysisContext,
) -> SnapshotLoad:
    value = session_state.get(context.session_key)
    if value is None:
        return SnapshotLoad("missing")
    try:
        snapshot = validate_snapshot_record(value)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return SnapshotLoad("invalid", None, str(exc))
    if snapshot.context.analysis_type != context.analysis_type:
        return SnapshotLoad("invalid", None, "snapshot type does not match fixed session key")
    if snapshot.context.fingerprint != context.fingerprint:
        return SnapshotLoad("stale", None, "分析上下文已变化")
    return SnapshotLoad("fresh", snapshot)


def store_analysis_snapshot(
    session_state: MutableMapping[str, Any], snapshot: AnalysisSnapshot,
) -> None:
    session_state[snapshot.context.session_key] = snapshot.to_record()


ENGINE_TRANSACTION_FIELDS = (
    "_ui_session_contract_version", "_ui_session_owner", "_ui_library_identity",
    "_last_material", "_last_substrate", "_last_polarization", "_last_angle",
    "_enable_far_field", "_na", "_theta_obs_deg",
    "_cache", "_coarse_grid_cache",
    "grid_params", "grid_rgb", "grid_lab", "grid_xy",
)
ENGINE_SESSION_KEY = "_ui_engine_instance_v1"


class EngineTransactionError(RuntimeError):
    """Base class for analysis transaction integrity failures."""


class EngineStateMutationError(EngineTransactionError):
    """The analysis mutated session state, but the before-state was restored."""


class EngineStateRestoreError(EngineTransactionError):
    """The transaction could not prove exact restoration of the before-state."""


@dataclass(frozen=True)
class EngineAttributeSnapshot:
    name: str
    present: bool
    value: Any


@dataclass(frozen=True)
class EngineStateSnapshot:
    object_id: int
    attributes: tuple[EngineAttributeSnapshot, ...]

    def value(self, name: str) -> Any:
        field = next(item for item in self.attributes if item.name == name)
        if not field.present:
            raise AttributeError(name)
        return field.value


def capture_engine_state(engine: Any) -> EngineStateSnapshot:
    """Deep-copy every mutable session-engine field touched by UI analyses."""
    attributes = []
    for name in ENGINE_TRANSACTION_FIELDS:
        present = hasattr(engine, name)
        value = copy.deepcopy(getattr(engine, name)) if present else None
        attributes.append(EngineAttributeSnapshot(name, present, value))
    return EngineStateSnapshot(id(engine), tuple(attributes))


def _state_values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            return False
        return bool(
            left.shape == right.shape and left.dtype == right.dtype
            and np.array_equal(left, right, equal_nan=True)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        if set(left) != set(right):
            return False
        return all(_state_values_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            return False
        return all(_state_values_equal(a, b) for a, b in zip(left, right))
    try:
        result = left == right
        return bool(result) if not isinstance(result, np.ndarray) else bool(np.all(result))
    except Exception:
        return False


def _engine_matches_snapshot(engine: Any, before: EngineStateSnapshot) -> bool:
    if id(engine) != before.object_id:
        return False
    try:
        for field in before.attributes:
            if hasattr(engine, field.name) != field.present:
                return False
            if field.present and not _state_values_equal(getattr(engine, field.name), field.value):
                return False
    except Exception:
        return False
    return True


def require_engine_state_unchanged(engine: Any, before: EngineStateSnapshot) -> None:
    if not _engine_matches_snapshot(engine, before):
        raise RuntimeError("analysis mutated the session engine identity or grid")


def restore_engine_state(engine: Any, before: EngineStateSnapshot) -> None:
    """Restore a deep before-state and prove shape, dtype, values and identity."""
    if id(engine) != before.object_id:
        raise EngineStateRestoreError("session engine object identity changed")
    failures = []
    for field in before.attributes:
        try:
            if field.present:
                setattr(engine, field.name, copy.deepcopy(field.value))
            elif hasattr(engine, field.name):
                delattr(engine, field.name)
        except Exception as exc:
            failures.append(f"{field.name}: {type(exc).__name__}")
    if failures:
        raise EngineStateRestoreError(
            "session engine restore assignment failed: " + ", ".join(failures))
    try:
        require_engine_state_unchanged(engine, before)
    except Exception as exc:
        raise EngineStateRestoreError(
            "session engine restore verification failed") from exc


@contextmanager
def analysis_engine_transaction(
    engine: Any,
    session_state: MutableMapping[str, Any] | None = None,
    engine_session_key: str = ENGINE_SESSION_KEY,
):
    """Restore the session engine in all exits and reject mutated results."""
    before = capture_engine_state(engine)
    mapped_before = None
    if session_state is not None:
        mapped_before = session_state.get(engine_session_key)
        if mapped_before is not engine:
            raise EngineTransactionError(
                "session engine mapping does not match transaction engine")

    error_info = None
    try:
        yield before
    except BaseException:
        error_info = sys.exc_info()

    engine_mutated = not _engine_matches_snapshot(engine, before)
    mapping_mutated = bool(
        session_state is not None
        and session_state.get(engine_session_key) is not mapped_before
    )
    restore_error = None
    try:
        restore_engine_state(engine, before)
    except Exception as exc:
        restore_error = exc
    if session_state is not None:
        try:
            session_state[engine_session_key] = mapped_before
            if session_state.get(engine_session_key) is not engine:
                raise EngineStateRestoreError(
                    "session engine mapping restore verification failed")
        except Exception as exc:
            restore_error = restore_error or exc
    if restore_error is not None:
        cause = error_info[1] if error_info is not None else restore_error
        raise EngineStateRestoreError(
            "analysis transaction could not restore the session engine") from cause

    if engine_mutated or mapping_mutated:
        cause = error_info[1] if error_info is not None else None
        error = EngineStateMutationError(
            "analysis mutated session engine state; before-state restored")
        if cause is not None:
            raise error from cause
        raise error
    if error_info is not None:
        raise error_info[1].with_traceback(error_info[2])
