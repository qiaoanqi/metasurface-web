"""Fail-closed contracts for the isolated single-pillar pattern tool."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, MutableMapping

import numpy as np
from PIL import Image

from ccm import CCM_COEFF_TABLE
from ui_analysis_snapshots import canonical_json, canonical_sha256


PATTERN_SNAPSHOT_SCHEMA_VERSION = 2
PATTERN_PROTOCOL_VERSION = "ui-pattern-snapshot-v2"
PATTERN_CONTRACT_VERSION = "single-scalar-analytical-pattern-v2"
PATTERN_SESSION_KEY = "_ui_pattern_snapshot_v1"
PATTERN_MODEL_IDENTITY = (
    "engine.MetaSurfaceColorEngine._build_library_vectorised+"
    "image_to_metasurface_map"
)
PATTERN_MODEL_VERSION = "scalar-analytical-ccm-lab-nearest-v1"
PATTERN_ASSUMPTIONS = {
    "structure_type": "single",
    "polarization": "TE (s-pol)",
    "angle_deg": 0.0,
    "far_field_enabled": False,
    "response": "scalar analytical",
    "preview_ml_inherited": False,
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _source_sha256(filename: str) -> str:
    path = Path(__file__).resolve().with_name(filename)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_ccm_registry() -> tuple[tuple[str, str], ...]:
    """Return only literal disk-registered keys; never fuzzy/default matches."""
    return tuple(sorted((str(material), str(substrate)) for material, substrate in CCM_COEFF_TABLE))


def _registry_payload() -> dict[str, Any]:
    entries = []
    for material, substrate in exact_ccm_registry():
        entries.append({
            "material": material,
            "substrate": substrate,
            "coefficients": asdict(CCM_COEFF_TABLE[(material, substrate)]),
        })
    return {"policy": "exact-only-no-fuzzy-no-default", "entries": entries}


def pattern_registry_version() -> str:
    return "ccm-exact-v1:" + canonical_sha256(_registry_payload())


def pattern_dependency_hashes() -> dict[str, str]:
    """Hash the project-source closure that determines pattern mapping."""
    return {
        filename: _source_sha256(filename)
        for filename in ("ccm.py", "color_utils.py", "engine.py")
    }


def pattern_artifact_identity(
    dependencies: Mapping[str, str] | None = None,
    *,
    registry_version: str | None = None,
    model_identity: str | None = None,
    model_version: str | None = None,
    protocol_version: str | None = None,
    contract_version: str | None = None,
) -> str:
    return canonical_sha256({
        "dependency_sha256": dict(
            dependencies if dependencies is not None else pattern_dependency_hashes()),
        "registry_version": registry_version or pattern_registry_version(),
        "model_identity": model_identity or PATTERN_MODEL_IDENTITY,
        "model_version": model_version or PATTERN_MODEL_VERSION,
        "protocol_version": protocol_version or PATTERN_PROTOCOL_VERSION,
        "contract_version": contract_version or PATTERN_CONTRACT_VERSION,
    })


@dataclass(frozen=True)
class PatternMappingContract:
    available: bool
    reason: str
    structure_type: str
    structure_identity: str
    material: str
    substrate: str
    assumptions_json: str
    exact_registry_json: str
    contract_version: str
    registry_version: str
    model_identity: str
    model_version: str
    artifact_dependencies_json: str
    artifact_identity: str

    def __post_init__(self) -> None:
        if not all(str(getattr(self, name)).strip() for name in (
            "structure_type", "structure_identity", "material", "substrate",
            "contract_version", "registry_version", "model_identity",
            "model_version", "artifact_identity",
        )):
            raise ValueError("pattern contract has empty identity fields")
        assumptions = json.loads(self.assumptions_json)
        registry = json.loads(self.exact_registry_json)
        dependencies = json.loads(self.artifact_dependencies_json)
        if assumptions != PATTERN_ASSUMPTIONS:
            raise ValueError("pattern assumptions are not canonical")
        if not isinstance(registry, list) or not registry:
            raise ValueError("pattern exact registry must be non-empty")
        if any(
            not isinstance(entry, list) or len(entry) != 2
            or not all(isinstance(value, str) and value for value in entry)
            for entry in registry
        ):
            raise ValueError("pattern exact registry entries are invalid")
        normalized_registry = canonical_json(registry)
        object.__setattr__(self, "assumptions_json", canonical_json(assumptions))
        object.__setattr__(self, "exact_registry_json", normalized_registry)
        if (
            not isinstance(dependencies, dict)
            or set(dependencies) != {"ccm.py", "color_utils.py", "engine.py"}
            or any(not _SHA256_RE.fullmatch(str(value)) for value in dependencies.values())
        ):
            raise ValueError("pattern artifact dependency hashes are invalid")
        object.__setattr__(
            self, "artifact_dependencies_json", canonical_json(dependencies))
        if not _SHA256_RE.fullmatch(str(self.artifact_identity)):
            raise ValueError("pattern artifact identity must be SHA-256")
        expected_artifact = pattern_artifact_identity(
            dependencies,
            registry_version=self.registry_version,
            model_identity=self.model_identity,
            model_version=self.model_version,
            protocol_version=PATTERN_PROTOCOL_VERSION,
            contract_version=self.contract_version,
        )
        if self.artifact_identity != expected_artifact:
            raise ValueError("pattern artifact identity does not match dependencies")
        expected_available = bool(
            self.structure_type == "single"
            and (self.material, self.substrate)
            in {(entry[0], entry[1]) for entry in registry}
        )
        if self.available and not expected_available:
            raise ValueError("pattern contract cannot mark unsupported context available")
        if self.available and self.reason:
            raise ValueError("available pattern contract cannot carry rejection reason")
        if not self.available and not self.reason:
            raise ValueError("unavailable pattern contract requires reason")

    @property
    def assumptions(self) -> dict[str, Any]:
        return json.loads(self.assumptions_json)

    @property
    def exact_registry(self) -> tuple[tuple[str, str], ...]:
        return tuple(tuple(item) for item in json.loads(self.exact_registry_json))

    @property
    def artifact_dependencies(self) -> dict[str, str]:
        return json.loads(self.artifact_dependencies_json)

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "structure_type": self.structure_type,
            "structure_identity": self.structure_identity,
            "material": self.material,
            "substrate": self.substrate,
            "assumptions": self.assumptions,
            "exact_registry": [list(item) for item in self.exact_registry],
            "contract_version": self.contract_version,
            "registry_version": self.registry_version,
            "model_identity": self.model_identity,
            "model_version": self.model_version,
            "artifact_dependencies": self.artifact_dependencies,
            "artifact_identity": self.artifact_identity,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.to_dict())


def make_pattern_contract(
    *, structure_type: str, structure_identity: str, material: str,
    substrate: str, angle_deg: float, far_field_enabled: bool,
) -> PatternMappingContract:
    registry = exact_ccm_registry()
    selected = (str(material), str(substrate))
    structure = str(structure_type)
    angle = float(angle_deg)
    far_field = bool(far_field_enabled)
    if structure != "single":
        reason = f"当前结构 {structure_identity} 不可用；图案工具只支持独立单柱。"
    elif far_field:
        reason = "远场已启用；图案工具固定 no-far-field，不能继承当前预览状态。"
    elif not math.isfinite(angle) or abs(angle) > 1e-12:
        reason = "当前入射角非 0°；图案工具只注册固定 0° 解析近似合同。"
    elif selected not in registry:
        reason = (
            f"材料/衬底精确组合 {material} / {substrate} 未注册；"
            "禁止 fuzzy、材料单项或默认系数替代。"
        )
    else:
        reason = ""
    registry_version = pattern_registry_version()
    dependencies = pattern_dependency_hashes()
    return PatternMappingContract(
        available=not reason,
        reason=reason,
        structure_type=structure,
        structure_identity=str(structure_identity),
        material=str(material),
        substrate=str(substrate),
        assumptions_json=canonical_json(PATTERN_ASSUMPTIONS),
        exact_registry_json=canonical_json([list(item) for item in registry]),
        contract_version=PATTERN_CONTRACT_VERSION,
        registry_version=registry_version,
        model_identity=PATTERN_MODEL_IDENTITY,
        model_version=PATTERN_MODEL_VERSION,
        artifact_dependencies_json=canonical_json(dependencies),
        artifact_identity=pattern_artifact_identity(
            dependencies,
            registry_version=registry_version,
            model_identity=PATTERN_MODEL_IDENTITY,
            model_version=PATTERN_MODEL_VERSION,
            protocol_version=PATTERN_PROTOCOL_VERSION,
            contract_version=PATTERN_CONTRACT_VERSION,
        ),
    )


def pattern_context(
    contract: PatternMappingContract, upload_sha256: str, max_size: int,
) -> dict[str, Any]:
    upload_hash = str(upload_sha256).lower()
    size = int(max_size)
    if not contract.available:
        raise ValueError("unavailable pattern contract cannot create a context")
    if not _SHA256_RE.fullmatch(upload_hash):
        raise ValueError("pattern upload SHA-256 is invalid")
    if not 20 <= size <= 64:
        raise ValueError("pattern max_size must be within [20, 64]")
    return {
        "schema_version": PATTERN_SNAPSHOT_SCHEMA_VERSION,
        "protocol_version": PATTERN_PROTOCOL_VERSION,
        "contract": contract.to_dict(),
        "contract_fingerprint": contract.fingerprint,
        "upload_sha256": upload_hash,
        "max_size": size,
    }


def pattern_context_fingerprint(
    contract: PatternMappingContract, upload_sha256: str, max_size: int,
) -> str:
    return canonical_sha256(pattern_context(contract, upload_sha256, max_size))


def pattern_contract_from_dict(value: Mapping[str, Any]) -> PatternMappingContract:
    expected = {
        "available", "reason", "structure_type", "structure_identity",
        "material", "substrate", "assumptions", "exact_registry",
        "contract_version", "registry_version", "model_identity",
        "model_version", "artifact_dependencies", "artifact_identity",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("pattern contract fields are incomplete or unexpected")
    return PatternMappingContract(
        available=value["available"], reason=value["reason"],
        structure_type=value["structure_type"],
        structure_identity=value["structure_identity"],
        material=value["material"], substrate=value["substrate"],
        assumptions_json=canonical_json(value["assumptions"]),
        exact_registry_json=canonical_json(value["exact_registry"]),
        contract_version=value["contract_version"],
        registry_version=value["registry_version"],
        model_identity=value["model_identity"], model_version=value["model_version"],
        artifact_dependencies_json=canonical_json(value["artifact_dependencies"]),
        artifact_identity=value["artifact_identity"],
    )


def _finite_rgb_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 3 or array.shape[2] != 3 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite HxWx3")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} must stay within [0, 1]")
    return array


def build_pattern_payload(
    contract: PatternMappingContract, upload_sha256: str, max_size: int,
    original_rgb: Any, mapped_rgb: Any, parameters_nm: Any,
) -> dict[str, Any]:
    context_fp = pattern_context_fingerprint(contract, upload_sha256, max_size)
    original = _finite_rgb_array(original_rgb, "original_rgb")
    mapped = _finite_rgb_array(mapped_rgb, "mapped_rgb")
    params = np.asarray(parameters_nm, dtype=float)
    if original.shape != mapped.shape or params.shape != original.shape:
        raise ValueError("pattern arrays must share HxWx3 shape")
    if not np.all(np.isfinite(params)):
        raise ValueError("pattern parameters contain NaN/Inf")
    d, h, p = params[..., 0], params[..., 1], params[..., 2]
    if (
        np.any((d < 50.0) | (d > 350.0))
        or np.any((h < 80.0) | (h > 600.0))
        or np.any((p < 200.0) | (p > 600.0))
        or np.any(d >= p)
    ):
        raise ValueError("pattern parameters are outside the registered geometry domain")
    arrays = {
        "original_rgb": original.tolist(),
        "mapped_rgb": mapped.tolist(),
        "parameters_nm": params.tolist(),
    }
    hashes = {name: canonical_sha256(value) for name, value in arrays.items()}
    return {
        "status": "available",
        "context_fingerprint": context_fp,
        "source_upload_sha256": str(upload_sha256).lower(),
        "shape": [int(original.shape[0]), int(original.shape[1])],
        **arrays,
        "array_sha256": hashes,
        "mean_srgb_error": float(np.mean(np.linalg.norm(original - mapped, axis=2))),
    }


def validate_pattern_payload(
    payload: Mapping[str, Any], context_value: Mapping[str, Any],
) -> dict[str, Any]:
    expected = {
        "status", "context_fingerprint", "source_upload_sha256", "shape",
        "original_rgb", "mapped_rgb", "parameters_nm", "array_sha256",
        "mean_srgb_error",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError("pattern payload fields are incomplete or unexpected")
    if payload["status"] != "available":
        raise ValueError("stored pattern payload must be available")
    context_fp = canonical_sha256(context_value)
    if payload["context_fingerprint"] != context_fp:
        raise ValueError("pattern payload context fingerprint mismatch")
    if payload["source_upload_sha256"] != context_value["upload_sha256"]:
        raise ValueError("pattern payload upload identity mismatch")
    original = _finite_rgb_array(payload["original_rgb"], "original_rgb")
    mapped = _finite_rgb_array(payload["mapped_rgb"], "mapped_rgb")
    params = np.asarray(payload["parameters_nm"], dtype=float)
    shape = [int(value) for value in payload["shape"]]
    if shape != list(original.shape[:2]) or mapped.shape != original.shape or params.shape != original.shape:
        raise ValueError("pattern payload shape mismatch")
    rebuilt = build_pattern_payload(
        pattern_contract_from_dict(context_value["contract"]),
        context_value["upload_sha256"], context_value["max_size"],
        original, mapped, params,
    )
    if payload["array_sha256"] != rebuilt["array_sha256"]:
        raise ValueError("pattern array hash mismatch")
    error = float(payload["mean_srgb_error"])
    if not math.isfinite(error) or abs(error - rebuilt["mean_srgb_error"]) > 1e-12:
        raise ValueError("pattern error summary mismatch")
    return rebuilt


@dataclass(frozen=True)
class PatternSnapshot:
    context: dict[str, Any]
    payload: dict[str, Any]
    payload_sha256: str

    @classmethod
    def create(
        cls, contract: PatternMappingContract, upload_sha256: str, max_size: int,
        payload: Mapping[str, Any],
    ) -> "PatternSnapshot":
        context_value = pattern_context(contract, upload_sha256, max_size)
        normalized = validate_pattern_payload(payload, context_value)
        return cls(context_value, normalized, canonical_sha256(normalized))

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.context)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": PATTERN_SNAPSHOT_SCHEMA_VERSION,
            "protocol_version": PATTERN_PROTOCOL_VERSION,
            "context": self.context,
            "context_fingerprint": self.fingerprint,
            "payload": self.payload,
            "payload_sha256": self.payload_sha256,
        }


@dataclass(frozen=True)
class PatternSnapshotLoad:
    state: str
    snapshot: PatternSnapshot | None = None
    reason: str = ""


def validate_pattern_snapshot_record(value: Any) -> PatternSnapshot:
    expected = {
        "schema_version", "protocol_version", "context", "context_fingerprint",
        "payload", "payload_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("pattern snapshot fields are incomplete or unexpected")
    if value["schema_version"] != PATTERN_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("unsupported pattern snapshot schema")
    if value["protocol_version"] != PATTERN_PROTOCOL_VERSION:
        raise ValueError("unsupported pattern snapshot protocol")
    context_value = value["context"]
    if not isinstance(context_value, Mapping):
        raise ValueError("pattern snapshot context is invalid")
    expected_context = {
        "schema_version", "protocol_version", "contract", "contract_fingerprint",
        "upload_sha256", "max_size",
    }
    if set(context_value) != expected_context:
        raise ValueError("pattern snapshot context fields are incomplete or unexpected")
    if context_value["schema_version"] != PATTERN_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("unsupported pattern context schema")
    if context_value["protocol_version"] != PATTERN_PROTOCOL_VERSION:
        raise ValueError("unsupported pattern context protocol")
    contract = pattern_contract_from_dict(context_value["contract"])
    if contract.fingerprint != context_value["contract_fingerprint"]:
        raise ValueError("pattern contract fingerprint mismatch")
    expected_context_value = pattern_context(
        contract, context_value["upload_sha256"], context_value["max_size"])
    if canonical_json(context_value) != canonical_json(expected_context_value):
        raise ValueError("pattern snapshot context is not canonical")
    if canonical_sha256(context_value) != value["context_fingerprint"]:
        raise ValueError("pattern snapshot context hash mismatch")
    payload = validate_pattern_payload(value["payload"], context_value)
    payload_hash = canonical_sha256(payload)
    if payload_hash != value["payload_sha256"]:
        raise ValueError("pattern snapshot payload hash mismatch")
    return PatternSnapshot(dict(context_value), payload, payload_hash)


def load_pattern_snapshot(
    session_state: Mapping[str, Any], contract: PatternMappingContract,
    upload_sha256: str, max_size: int,
) -> PatternSnapshotLoad:
    value = session_state.get(PATTERN_SESSION_KEY)
    if value is None:
        return PatternSnapshotLoad("missing")
    try:
        snapshot = validate_pattern_snapshot_record(value)
        expected_fp = pattern_context_fingerprint(contract, upload_sha256, max_size)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return PatternSnapshotLoad("invalid", None, str(exc))
    if snapshot.fingerprint != expected_fp:
        return PatternSnapshotLoad("stale", None, "图案上下文已变化")
    return PatternSnapshotLoad("fresh", snapshot)


def store_pattern_snapshot(
    session_state: MutableMapping[str, Any], snapshot: PatternSnapshot,
) -> None:
    session_state[PATTERN_SESSION_KEY] = snapshot.to_record()


@dataclass(frozen=True)
class PatternExports:
    mapped_png: bytes
    csv_bytes: bytes
    metadata_json: str


def _mapped_png_bytes(payload: Mapping[str, Any]) -> bytes:
    mapped = np.asarray(payload["mapped_rgb"], dtype=float)
    mapped_u8 = np.rint(mapped * 255.0).astype(np.uint8)
    png_buffer = io.BytesIO()
    Image.fromarray(mapped_u8).save(png_buffer, format="PNG")
    return png_buffer.getvalue()


def _pixel_csv_bytes(payload: Mapping[str, Any]) -> bytes:
    mapped = np.asarray(payload["mapped_rgb"], dtype=float)
    params = np.asarray(payload["parameters_nm"], dtype=float)
    csv_buffer = io.StringIO(newline="")
    writer = csv.writer(csv_buffer, lineterminator="\n")
    writer.writerow(["row", "col", "R", "G", "B", "D", "H", "P"])
    for row in range(mapped.shape[0]):
        for col in range(mapped.shape[1]):
            writer.writerow([
                row, col,
                *(f"{value:.12g}" for value in mapped[row, col]),
                *(f"{value:.12g}" for value in params[row, col]),
            ])
    return csv_buffer.getvalue().encode("utf-8")


def _pattern_export_metadata(
    snapshot: PatternSnapshot, png_bytes: bytes, csv_bytes: bytes,
) -> dict[str, Any]:
    payload = snapshot.payload
    mapped = np.asarray(payload["mapped_rgb"], dtype=float)
    return {
        "schema_version": PATTERN_SNAPSHOT_SCHEMA_VERSION,
        "protocol_version": PATTERN_PROTOCOL_VERSION,
        "context_fingerprint": snapshot.fingerprint,
        "context": snapshot.context,
        "shape": payload["shape"],
        "array_sha256": payload["array_sha256"],
        "payload_sha256": snapshot.payload_sha256,
        "mapped_png_sha256": hashlib.sha256(png_bytes).hexdigest(),
        "pixel_csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "pixel_csv_byte_count": len(csv_bytes),
        "csv_row_count": int(mapped.shape[0] * mapped.shape[1]),
        "mean_srgb_error": payload["mean_srgb_error"],
    }


def validate_pattern_exports(
    snapshot: PatternSnapshot, exports: PatternExports,
) -> PatternExports:
    validated = validate_pattern_snapshot_record(snapshot.to_record())
    if not isinstance(exports.mapped_png, bytes) or not isinstance(exports.csv_bytes, bytes):
        raise ValueError("pattern PNG and CSV exports must be bytes")
    if not isinstance(exports.metadata_json, str):
        raise ValueError("pattern metadata export must be JSON text")
    expected_png = _mapped_png_bytes(validated.payload)
    expected_csv = _pixel_csv_bytes(validated.payload)
    if exports.mapped_png != expected_png:
        raise ValueError("pattern mapped PNG bytes do not match snapshot")
    if exports.csv_bytes != expected_csv:
        raise ValueError("pattern pixel CSV bytes do not match snapshot")
    try:
        exports.csv_bytes.decode("utf-8")
        metadata = json.loads(exports.metadata_json)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("pattern export encoding or metadata JSON is invalid") from exc
    expected_metadata = _pattern_export_metadata(
        validated, expected_png, expected_csv)
    if metadata != expected_metadata:
        raise ValueError("pattern export metadata does not match bytes or snapshot")
    return exports


def build_pattern_exports(snapshot: PatternSnapshot) -> PatternExports:
    validated = validate_pattern_snapshot_record(snapshot.to_record())
    png_bytes = _mapped_png_bytes(validated.payload)
    csv_bytes = _pixel_csv_bytes(validated.payload)
    metadata = _pattern_export_metadata(validated, png_bytes, csv_bytes)
    exports = PatternExports(
        png_bytes, csv_bytes,
        json.dumps(
            metadata, ensure_ascii=False, sort_keys=True, indent=2,
            allow_nan=False,
        ),
    )
    return validate_pattern_exports(validated, exports)
