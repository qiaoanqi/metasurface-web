"""Fail-closed contract for the project-local historical FDTD figure."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
from typing import Literal

from PIL import Image


FDTD_ASSET_PROTOCOL_VERSION = "ui-fdtd-asset-v1"
FDTD_ASSET_RELATIVE_PATH = "data/fano_vs_fdtd_smallD.png"
FDTD_ASSET_SHA256 = (
    "7926A398859FAFAC594CBAAFAF801BE2A778BAA88344BEF013B3757108DC54F6"
)
FDTD_ASSET_BYTE_COUNT = 409_126
FDTD_ASSET_MAX_BYTES = 512 * 1024
FDTD_ASSET_SIZE = (2100, 2400)
FDTD_ASSET_MODE = "RGBA"
FDTD_ASSET_FORMAT = "PNG"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class FDTDAssetSpec:
    protocol_version: str
    relative_path: str
    expected_sha256: str
    expected_byte_count: int
    max_bytes: int
    expected_size: tuple[int, int]
    expected_mode: str
    expected_format: str


FDTD_ASSET_SPEC = FDTDAssetSpec(
    protocol_version=FDTD_ASSET_PROTOCOL_VERSION,
    relative_path=FDTD_ASSET_RELATIVE_PATH,
    expected_sha256=FDTD_ASSET_SHA256,
    expected_byte_count=FDTD_ASSET_BYTE_COUNT,
    max_bytes=FDTD_ASSET_MAX_BYTES,
    expected_size=FDTD_ASSET_SIZE,
    expected_mode=FDTD_ASSET_MODE,
    expected_format=FDTD_ASSET_FORMAT,
)


@dataclass(frozen=True)
class FDTDAssetResult:
    status: Literal["available", "unavailable"]
    reason_code: str
    detail: str
    relative_path: str = FDTD_ASSET_RELATIVE_PATH
    image_bytes: bytes | None = None
    sha256: str = ""
    byte_count: int = 0
    size: tuple[int, int] | None = None
    mode: str = ""
    image_format: str = ""

    @property
    def available(self) -> bool:
        return self.status == "available" and self.image_bytes is not None


@dataclass(frozen=True)
class FDTDMetric:
    key: str
    label: str
    minimum: float
    maximum: float
    unit: str
    interpretation: str


@dataclass(frozen=True)
class FDTDComparisonManifest:
    schema_version: str
    revision: str
    quantity: str
    geometry: str
    metrics: tuple[FDTDMetric, ...]
    source: str
    asset_sha256: str

    def metric(self, key: str) -> FDTDMetric:
        matches = [metric for metric in self.metrics if metric.key == key]
        if len(matches) != 1:
            raise KeyError(key)
        return matches[0]


FDTD_COMPARISON_MANIFEST = FDTDComparisonManifest(
    schema_version="historical-fdtd-comparison-v1",
    quantity=(
        "历史 FDTD 透射振幅与当前 Fano 近似反射响应的有限趋势对照"
    ),
    geometry="P=200 nm，D=30-76 nm（历史小直径图示范围）",
    metrics=(
        FDTDMetric(
            key="resonance_peak_offset_nm",
            label="共振峰位偏差",
            minimum=16.0,
            maximum=24.0,
            unit="nm",
            interpretation="仅用于历史图所示小直径范围的峰位趋势对照",
        ),
        FDTDMetric(
            key="spectral_shape_correlation",
            label="光谱形状相关系数",
            minimum=0.48,
            maximum=0.51,
            unit="",
            interpretation="只表示有限形状一致性，不是定量精度或模型验证",
        ),
    ),
    source=FDTD_ASSET_RELATIVE_PATH,
    revision="fano-fdtd-small-d-r1",
    asset_sha256=FDTD_ASSET_SHA256,
)


@dataclass(frozen=True)
class FDTDEvidence:
    asset: FDTDAssetResult
    manifest: FDTDComparisonManifest | None


def _unavailable(reason_code: str, detail: str) -> FDTDAssetResult:
    return FDTDAssetResult(
        status="unavailable", reason_code=str(reason_code), detail=str(detail)
    )


def validate_fdtd_asset_bytes(
    image_bytes: bytes,
    spec: FDTDAssetSpec = FDTD_ASSET_SPEC,
) -> FDTDAssetResult:
    """Validate immutable bytes without filesystem or network side effects."""
    try:
        if not isinstance(image_bytes, bytes):
            return _unavailable("invalid_bytes", "资产内容不是 bytes")
        byte_count = len(image_bytes)
        if byte_count > int(spec.max_bytes):
            return _unavailable("oversize", "资产超过本地证据大小上限")
        if not image_bytes.startswith(PNG_SIGNATURE):
            return _unavailable("wrong_signature", "资产不是 PNG 文件")

        with Image.open(io.BytesIO(image_bytes)) as image:
            image_format = str(image.format or "")
            mode = str(image.mode or "")
            size = tuple(int(value) for value in image.size)
            if image_format != spec.expected_format:
                return _unavailable("wrong_format", "PNG 格式标识不匹配")
            if mode != spec.expected_mode:
                return _unavailable("wrong_mode", "PNG 色彩模式不匹配")
            if size != tuple(spec.expected_size):
                return _unavailable("wrong_dimensions", "PNG 尺寸不匹配")
            image.verify()

        if byte_count != int(spec.expected_byte_count):
            return _unavailable("wrong_byte_count", "资产字节数不匹配")
        digest = hashlib.sha256(image_bytes).hexdigest()
        if digest != str(spec.expected_sha256).lower():
            return _unavailable("hash_mismatch", "资产 SHA256 不匹配")
        return FDTDAssetResult(
            status="available",
            reason_code="verified",
            detail="项目内历史 FDTD 图已通过完整性校验",
            relative_path=spec.relative_path,
            image_bytes=image_bytes,
            sha256=digest,
            byte_count=byte_count,
            size=size,
            mode=mode,
            image_format=image_format,
        )
    except Exception as exc:
        return _unavailable(
            "invalid_png", f"PNG 完整性校验失败：{type(exc).__name__}"
        )


def resolve_fdtd_asset(
    project_root: str | Path | None = None,
    *,
    relative_path: str = FDTD_ASSET_RELATIVE_PATH,
) -> FDTDAssetResult:
    """Resolve only the registered project-local asset and never raise to UI."""
    try:
        if str(relative_path).replace("\\", "/") != FDTD_ASSET_RELATIVE_PATH:
            return _unavailable("path_not_registered", "资产路径未注册")
        root = (
            Path(__file__).resolve().parent
            if project_root is None
            else Path(project_root).resolve()
        )
        data_dir = (root / "data").resolve()
        candidate = (root / relative_path).resolve()
        try:
            candidate.relative_to(data_dir)
        except ValueError:
            return _unavailable("path_escape", "资产路径离开 data 目录")
        if not candidate.is_file():
            return _unavailable("missing", "项目内历史 FDTD 图不存在")
        if candidate.stat().st_size > FDTD_ASSET_MAX_BYTES:
            return _unavailable("oversize", "资产超过本地证据大小上限")
        return validate_fdtd_asset_bytes(candidate.read_bytes(), FDTD_ASSET_SPEC)
    except Exception as exc:
        return _unavailable(
            "read_error", f"项目内历史 FDTD 图读取失败：{type(exc).__name__}"
        )


def manifest_for_asset(
    asset: FDTDAssetResult,
    manifest: FDTDComparisonManifest = FDTD_COMPARISON_MANIFEST,
) -> FDTDComparisonManifest | None:
    """Return metrics only when the versioned manifest binds to verified bytes."""
    try:
        if not isinstance(asset, FDTDAssetResult) or not asset.available:
            return None
        if asset.status != "available" or asset.reason_code != "verified":
            return None
        if asset.relative_path != FDTD_ASSET_SPEC.relative_path:
            return None
        if asset.byte_count != FDTD_ASSET_SPEC.expected_byte_count:
            return None
        if asset.size != FDTD_ASSET_SPEC.expected_size:
            return None
        if asset.mode != FDTD_ASSET_SPEC.expected_mode:
            return None
        if asset.image_format != FDTD_ASSET_SPEC.expected_format:
            return None
        if not isinstance(asset.image_bytes, bytes):
            return None

        revalidated = validate_fdtd_asset_bytes(
            asset.image_bytes, FDTD_ASSET_SPEC
        )
        if not revalidated.available:
            return None
        if revalidated != asset:
            return None

        if not isinstance(manifest, FDTDComparisonManifest):
            return None
        if len(manifest.metrics) != 2:
            return None
        metric_keys = [metric.key for metric in manifest.metrics]
        if len(set(metric_keys)) != len(metric_keys):
            return None
        if manifest != FDTD_COMPARISON_MANIFEST:
            return None
        return manifest
    except Exception:
        return None


def resolve_fdtd_evidence(
    project_root: str | Path | None = None,
) -> FDTDEvidence:
    asset = resolve_fdtd_asset(project_root)
    return FDTDEvidence(asset=asset, manifest=manifest_for_asset(asset))
