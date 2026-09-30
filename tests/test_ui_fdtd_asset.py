from dataclasses import replace
import hashlib
import io
from pathlib import Path

from PIL import Image

from ui_fdtd_asset import (
    FDTD_ASSET_BYTE_COUNT,
    FDTD_ASSET_MAX_BYTES,
    FDTD_ASSET_PROTOCOL_VERSION,
    FDTD_ASSET_SHA256,
    FDTD_ASSET_SIZE,
    FDTD_ASSET_SPEC,
    FDTDAssetResult,
    FDTD_COMPARISON_MANIFEST,
    FDTDMetric,
    manifest_for_asset,
    resolve_fdtd_asset,
    resolve_fdtd_evidence,
    validate_fdtd_asset_bytes,
)


def _asset_bytes() -> bytes:
    return Path("data/fano_vs_fdtd_smallD.png").read_bytes()


def _png_bytes(mode: str, size: tuple[int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, 0).save(buffer, format="PNG")
    return buffer.getvalue()


def _spec_for_bytes(data: bytes, **updates):
    values = {
        "expected_sha256": hashlib.sha256(data).hexdigest(),
        "expected_byte_count": len(data),
        "max_bytes": max(len(data) + 1, 1024),
    }
    values.update(updates)
    return replace(FDTD_ASSET_SPEC, **values)


def test_registered_fdtd_asset_matches_exact_disk_contract():
    result = resolve_fdtd_asset(Path.cwd())
    evidence = resolve_fdtd_evidence(Path.cwd())

    assert result.available
    assert result.status == "available"
    assert result.reason_code == "verified"
    assert result.sha256.upper() == FDTD_ASSET_SHA256
    assert result.byte_count == FDTD_ASSET_BYTE_COUNT == 409_126
    assert result.byte_count <= FDTD_ASSET_MAX_BYTES
    assert result.size == FDTD_ASSET_SIZE == (2100, 2400)
    assert result.mode == "RGBA"
    assert result.image_format == "PNG"
    assert evidence.asset == result
    assert evidence.manifest == FDTD_COMPARISON_MANIFEST
    assert FDTD_ASSET_PROTOCOL_VERSION == "ui-fdtd-asset-v1"


def test_missing_asset_returns_typed_unavailable_without_exception(tmp_path):
    result = resolve_fdtd_asset(tmp_path)

    assert not result.available
    assert result.status == "unavailable"
    assert result.reason_code == "missing"
    assert result.image_bytes is None
    assert manifest_for_asset(result) is None


def test_tampered_or_corrupt_asset_fails_closed(tmp_path):
    target = tmp_path / "data" / "fano_vs_fdtd_smallD.png"
    target.parent.mkdir()
    data = bytearray(_asset_bytes())
    data[-16] ^= 0x01
    target.write_bytes(bytes(data))

    result = resolve_fdtd_asset(tmp_path)

    assert not result.available
    assert result.status == "unavailable"
    assert result.reason_code in {"invalid_png", "hash_mismatch"}
    assert result.image_bytes is None


def test_wrong_hash_is_rejected_after_png_verification():
    result = validate_fdtd_asset_bytes(
        _asset_bytes(), replace(FDTD_ASSET_SPEC, expected_sha256="0" * 64)
    )

    assert not result.available
    assert result.reason_code == "hash_mismatch"


def test_wrong_format_and_dimensions_are_distinguished():
    wrong_format = b"not-png" + b"\0" * 64
    format_result = validate_fdtd_asset_bytes(
        wrong_format, _spec_for_bytes(wrong_format)
    )
    wrong_size = _png_bytes("RGBA", (7, 9))
    size_result = validate_fdtd_asset_bytes(
        wrong_size, _spec_for_bytes(wrong_size)
    )

    assert format_result.reason_code == "wrong_signature"
    assert size_result.reason_code == "wrong_dimensions"
    assert not format_result.available
    assert not size_result.available


def test_wrong_mode_and_oversize_are_rejected():
    wrong_mode = _png_bytes("RGB", FDTD_ASSET_SIZE)
    mode_result = validate_fdtd_asset_bytes(
        wrong_mode, _spec_for_bytes(wrong_mode)
    )
    oversize_result = validate_fdtd_asset_bytes(
        _asset_bytes(), replace(FDTD_ASSET_SPEC, max_bytes=128)
    )

    assert mode_result.reason_code == "wrong_mode"
    assert oversize_result.reason_code == "oversize"
    assert not mode_result.available
    assert not oversize_result.available


def test_unregistered_or_escaping_path_is_rejected_before_read(tmp_path):
    outside = tmp_path / "outside.png"
    outside.write_bytes(_asset_bytes())

    result = resolve_fdtd_asset(
        tmp_path, relative_path="data/../outside.png"
    )

    assert not result.available
    assert result.reason_code == "path_not_registered"
    assert result.image_bytes is None


def test_manifest_metrics_require_exact_verified_asset_hash_and_revision():
    asset = resolve_fdtd_asset(Path.cwd())
    wrong_hash = replace(FDTD_COMPARISON_MANIFEST, asset_sha256="0" * 64)
    wrong_revision = replace(FDTD_COMPARISON_MANIFEST, revision="")
    wrong_source = replace(FDTD_COMPARISON_MANIFEST, source="data/other.png")

    assert manifest_for_asset(asset) is FDTD_COMPARISON_MANIFEST
    assert manifest_for_asset(asset, wrong_hash) is None
    assert manifest_for_asset(asset, wrong_revision) is None
    assert manifest_for_asset(asset, wrong_source) is None
    peak = FDTD_COMPARISON_MANIFEST.metric("resonance_peak_offset_nm")
    correlation = FDTD_COMPARISON_MANIFEST.metric(
        "spectral_shape_correlation"
    )
    assert (peak.minimum, peak.maximum, peak.unit) == (16.0, 24.0, "nm")
    assert (correlation.minimum, correlation.maximum) == (0.48, 0.51)


def test_manifest_rejects_duplicate_or_changed_metric_fields():
    asset = resolve_fdtd_asset(Path.cwd())
    peak, correlation = FDTD_COMPARISON_MANIFEST.metrics
    changed_metrics = (
        replace(peak, key=peak.key + "_changed"),
        replace(peak, minimum=peak.minimum + 1.0),
        replace(peak, maximum=peak.maximum + 1.0),
        replace(peak, label=peak.label + " changed"),
        replace(peak, unit="um"),
        replace(peak, interpretation=peak.interpretation + " changed"),
    )

    duplicate = replace(
        FDTD_COMPARISON_MANIFEST, metrics=(peak, peak)
    )
    assert manifest_for_asset(asset, duplicate) is None
    assert manifest_for_asset(
        asset,
        replace(
            FDTD_COMPARISON_MANIFEST,
            metrics=(correlation, peak),
        ),
    ) is None
    for changed in changed_metrics:
        candidate = replace(
            FDTD_COMPARISON_MANIFEST,
            metrics=(changed, correlation),
        )
        assert manifest_for_asset(asset, candidate) is None


def test_manifest_rejects_forged_asset_bytes_path_and_metadata():
    asset = resolve_fdtd_asset(Path.cwd())
    assert asset.available
    tampered = bytearray(asset.image_bytes)
    tampered[-16] ^= 0x01
    for forged in (
        replace(asset, relative_path="data/other.png"),
        replace(asset, image_bytes=bytes(tampered)),
        replace(asset, image_bytes=None),
        replace(asset, byte_count=asset.byte_count + 1),
        replace(asset, size=(1, 1)),
        replace(asset, mode="RGB"),
        replace(asset, image_format="JPEG"),
        replace(asset, sha256="0" * 64),
        replace(asset, detail="forged detail"),
        replace(asset, status="unavailable"),
        replace(asset, reason_code="forged"),
    ):
        assert isinstance(forged, FDTDAssetResult)
        assert manifest_for_asset(forged) is None


def test_manifest_requires_exact_frozen_top_level_fields():
    asset = resolve_fdtd_asset(Path.cwd())
    for candidate in (
        replace(FDTD_COMPARISON_MANIFEST, schema_version="changed"),
        replace(FDTD_COMPARISON_MANIFEST, revision="changed"),
        replace(FDTD_COMPARISON_MANIFEST, quantity="changed"),
        replace(FDTD_COMPARISON_MANIFEST, geometry="changed"),
        replace(FDTD_COMPARISON_MANIFEST, source="data/other.png"),
        replace(FDTD_COMPARISON_MANIFEST, asset_sha256="0" * 64),
        replace(FDTD_COMPARISON_MANIFEST, metrics=(FDTDMetric(
            key="other", label="other", minimum=0.0, maximum=1.0,
            unit="", interpretation="other",
        ),)),
    ):
        assert manifest_for_asset(asset, candidate) is None
