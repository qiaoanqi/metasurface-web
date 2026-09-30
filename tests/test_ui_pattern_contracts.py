import copy
import csv
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from ccm import CCM_COEFF_TABLE
from engine import MaterialLibrary
from ui_pattern_contracts import (
    PATTERN_ASSUMPTIONS,
    PATTERN_CONTRACT_VERSION,
    PATTERN_SESSION_KEY,
    PatternSnapshot,
    build_pattern_exports,
    build_pattern_payload,
    exact_ccm_registry,
    load_pattern_snapshot,
    make_pattern_contract,
    pattern_artifact_identity,
    pattern_context_fingerprint,
    pattern_dependency_hashes,
    store_pattern_snapshot,
    validate_pattern_exports,
    validate_pattern_snapshot_record,
)


UPLOAD_A = hashlib.sha256(b"pattern-a").hexdigest()
UPLOAD_B = hashlib.sha256(b"pattern-b").hexdigest()


def supported_contract(**changes):
    values = {
        "structure_type": "single",
        "structure_identity": "单柱",
        "material": "TiO2 (anatase)",
        "substrate": "SiO2 (fused silica)",
        "angle_deg": 0.0,
        "far_field_enabled": False,
    }
    values.update(changes)
    return make_pattern_contract(**values)


def arrays():
    original = np.array([
        [[0.0, 0.2, 0.4], [0.1, 0.3, 0.5], [0.2, 0.4, 0.6]],
        [[0.3, 0.5, 0.7], [0.4, 0.6, 0.8], [0.5, 0.7, 0.9]],
    ])
    mapped = np.clip(original + 0.02, 0.0, 1.0)
    params = np.empty((2, 3, 3), dtype=float)
    params[..., 0] = [[80, 100, 120], [140, 160, 180]]
    params[..., 1] = [[200, 220, 240], [260, 280, 300]]
    params[..., 2] = [[300, 320, 340], [360, 380, 400]]
    return original, mapped, params


def snapshot(contract=None, upload=UPLOAD_A, max_size=48):
    contract = contract or supported_contract()
    original, mapped, params = arrays()
    payload = build_pattern_payload(
        contract, upload, max_size, original, mapped, params)
    return PatternSnapshot.create(contract, upload, max_size, payload)


def contract_variant(contract=None, **changes):
    base = contract or supported_contract()
    dependencies_json = changes.get(
        "artifact_dependencies_json", base.artifact_dependencies_json)
    dependencies = json.loads(dependencies_json)
    registry_version = changes.get("registry_version", base.registry_version)
    model_identity = changes.get("model_identity", base.model_identity)
    model_version = changes.get("model_version", base.model_version)
    contract_version = changes.get("contract_version", base.contract_version)
    return replace(
        base,
        **changes,
        artifact_identity=pattern_artifact_identity(
            dependencies,
            registry_version=registry_version,
            model_identity=model_identity,
            model_version=model_version,
            contract_version=contract_version,
        ),
    )


def test_exact_registry_is_literal_ccm_table_and_only_one_key_is_ui_reachable():
    registry = exact_ccm_registry()
    assert registry == tuple(sorted(CCM_COEFF_TABLE))
    ui_materials = set(MaterialLibrary.pillar_materials())
    ui_substrates = set(MaterialLibrary.substrate_materials())
    reachable = {
        pair for pair in registry
        if pair[0] in ui_materials and pair[1] in ui_substrates
    }
    assert reachable == {("TiO2 (anatase)", "SiO2 (fused silica)")}


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"structure_type": "dual", "structure_identity": "双柱"}, "只支持独立单柱"),
        ({"structure_type": "fp", "structure_identity": "FP-DBR 腔"}, "只支持独立单柱"),
        ({"structure_type": "fp", "structure_identity": "FP-Ag 腔"}, "只支持独立单柱"),
        ({"far_field_enabled": True}, "固定 no-far-field"),
        ({"angle_deg": 5.0}, "只注册固定 0°"),
        ({"material": "a-Si (amorphous)"}, "未注册"),
        ({"substrate": "Si3N4 (nitride)"}, "未注册"),
    ],
)
def test_rejection_matrix_is_explicit_and_never_substitutes(changes, reason):
    contract = supported_contract(**changes)
    assert contract.available is False
    assert reason in contract.reason
    assert contract.assumptions == PATTERN_ASSUMPTIONS


def test_supported_contract_is_fixed_and_does_not_contain_preview_route_state():
    contract = supported_contract()
    assert contract.available is True
    assert contract.reason == ""
    assert contract.assumptions == {
        "structure_type": "single",
        "polarization": "TE (s-pol)",
        "angle_deg": 0.0,
        "far_field_enabled": False,
        "response": "scalar analytical",
        "preview_ml_inherited": False,
    }
    assert "ML" not in contract.model_identity


def test_context_fingerprint_covers_upload_size_contract_registry_model_and_material():
    contract = supported_contract()
    baseline = pattern_context_fingerprint(contract, UPLOAD_A, 48)
    variants = [
        pattern_context_fingerprint(contract, UPLOAD_B, 48),
        pattern_context_fingerprint(contract, UPLOAD_A, 32),
        pattern_context_fingerprint(
            contract_variant(
                contract, registry_version=contract.registry_version + "-new"),
            UPLOAD_A, 48),
        pattern_context_fingerprint(
            contract_variant(
                contract, model_version=contract.model_version + "-new"),
            UPLOAD_A, 48),
        pattern_context_fingerprint(
            contract_variant(
                contract, contract_version=PATTERN_CONTRACT_VERSION + "-new"),
            UPLOAD_A, 48),
    ]
    assert all(value != baseline for value in variants)
    assert supported_contract(material="a-Si (amorphous)").fingerprint != contract.fingerprint


def test_artifact_identity_covers_current_project_source_dependency_hashes():
    contract = supported_contract()
    expected = {
        filename: hashlib.sha256(Path(filename).read_bytes()).hexdigest()
        for filename in ("ccm.py", "color_utils.py", "engine.py")
    }
    assert pattern_dependency_hashes() == expected
    assert contract.artifact_dependencies == expected
    assert contract.artifact_identity == pattern_artifact_identity(
        expected,
        registry_version=contract.registry_version,
        model_identity=contract.model_identity,
        model_version=contract.model_version,
        contract_version=contract.contract_version,
    )


@pytest.mark.parametrize("filename", ["ccm.py", "color_utils.py", "engine.py"])
def test_each_dependency_hash_change_makes_old_snapshot_stale(filename):
    current = supported_contract()
    old_dependencies = dict(current.artifact_dependencies)
    old_dependencies[filename] = "0" * 64
    old_contract = contract_variant(
        current,
        artifact_dependencies_json=json.dumps(
            old_dependencies, sort_keys=True, separators=(",", ":")),
    )
    old_snapshot = snapshot(old_contract)
    loaded = load_pattern_snapshot(
        {PATTERN_SESSION_KEY: old_snapshot.to_record()},
        current, UPLOAD_A, 48,
    )
    assert loaded.state == "stale"


def test_non_square_snapshot_round_trip_and_fixed_session_key():
    current = supported_contract()
    value = snapshot(current)
    state = {}
    store_pattern_snapshot(state, value)
    loaded = load_pattern_snapshot(state, current, UPLOAD_A, 48)
    assert set(state) == {PATTERN_SESSION_KEY}
    assert loaded.state == "fresh"
    assert loaded.snapshot.payload["shape"] == [2, 3]
    assert loaded.snapshot.fingerprint == value.fingerprint


@pytest.mark.parametrize(
    ("contract", "upload", "max_size"),
    [
        (supported_contract(), UPLOAD_B, 48),
        (supported_contract(), UPLOAD_A, 32),
        (contract_variant(registry_version="registry-new"), UPLOAD_A, 48),
        (contract_variant(model_version="model-new"), UPLOAD_A, 48),
        (contract_variant(contract_version="contract-new"), UPLOAD_A, 48),
    ],
)
def test_context_changes_make_old_snapshot_stale(contract, upload, max_size):
    state = {PATTERN_SESSION_KEY: snapshot().to_record()}
    assert load_pattern_snapshot(state, contract, upload, max_size).state == "stale"


def test_exports_are_exactly_derived_from_the_snapshot_arrays():
    value = snapshot()
    exports = build_pattern_exports(value)
    payload = value.payload
    mapped = np.asarray(payload["mapped_rgb"])
    params = np.asarray(payload["parameters_nm"])

    png = np.asarray(Image.open(io.BytesIO(exports.mapped_png)).convert("RGB"))
    np.testing.assert_array_equal(png, np.rint(mapped * 255).astype(np.uint8))

    csv_text = exports.csv_bytes.decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    assert len(rows) == mapped.shape[0] * mapped.shape[1]
    for item in rows:
        row, col = int(item["row"]), int(item["col"])
        np.testing.assert_allclose(
            [float(item[name]) for name in ("R", "G", "B")], mapped[row, col])
        np.testing.assert_allclose(
            [float(item[name]) for name in ("D", "H", "P")], params[row, col])

    metadata = json.loads(exports.metadata_json)
    assert metadata["context_fingerprint"] == value.fingerprint
    assert metadata["array_sha256"] == payload["array_sha256"]
    assert metadata["payload_sha256"] == value.payload_sha256
    assert metadata["csv_row_count"] == len(rows)
    assert metadata["mapped_png_sha256"] == hashlib.sha256(exports.mapped_png).hexdigest()
    assert metadata["pixel_csv_sha256"] == hashlib.sha256(exports.csv_bytes).hexdigest()
    assert metadata["pixel_csv_byte_count"] == len(exports.csv_bytes)
    assert csv_text.splitlines()[0] == "row,col,R,G,B,D,H,P"


@pytest.mark.parametrize("tamper", ["csv_bytes", "csv_hash", "csv_byte_count"])
def test_csv_export_tamper_fails_closed(tamper):
    value = snapshot()
    exports = build_pattern_exports(value)
    if tamper == "csv_bytes":
        broken = replace(exports, csv_bytes=exports.csv_bytes + b"0")
    else:
        metadata = json.loads(exports.metadata_json)
        field = "pixel_csv_sha256" if tamper == "csv_hash" else "pixel_csv_byte_count"
        metadata[field] = "0" * 64 if tamper == "csv_hash" else 0
        broken = replace(
            exports,
            metadata_json=json.dumps(
                metadata, ensure_ascii=False, sort_keys=True, indent=2),
        )
    with pytest.raises(ValueError):
        validate_pattern_exports(value, broken)


@pytest.mark.parametrize("field", ["original_rgb", "mapped_rgb", "parameters_nm"])
def test_nan_inf_and_shape_fail_closed(field):
    contract = supported_contract()
    original, mapped, params = arrays()
    values = {"original_rgb": original, "mapped_rgb": mapped, "parameters_nm": params}
    broken = values[field].copy()
    broken[0, 0, 0] = np.nan if field != "parameters_nm" else np.inf
    values[field] = broken
    with pytest.raises(ValueError):
        build_pattern_payload(
            contract, UPLOAD_A, 48,
            values["original_rgb"], values["mapped_rgb"], values["parameters_nm"],
        )

    with pytest.raises(ValueError):
        build_pattern_payload(contract, UPLOAD_A, 48, original[:, :2], mapped, params)


def test_geometry_range_and_d_less_than_p_fail_closed():
    contract = supported_contract()
    original, mapped, params = arrays()
    params[0, 0] = [350.0, 300.0, 300.0]
    with pytest.raises(ValueError, match="geometry domain"):
        build_pattern_payload(contract, UPLOAD_A, 48, original, mapped, params)


@pytest.mark.parametrize(
    "tamper", ["array", "hash", "payload_hash", "context_hash", "dependency_hash"])
def test_snapshot_tamper_fails_closed(tamper):
    record = snapshot().to_record()
    if tamper == "array":
        record["payload"]["mapped_rgb"][0][0][0] += 0.1
    elif tamper == "hash":
        record["payload"]["array_sha256"]["mapped_rgb"] = "0" * 64
    elif tamper == "payload_hash":
        record["payload_sha256"] = "0" * 64
    elif tamper == "context_hash":
        record["context_fingerprint"] = "0" * 64
    else:
        record["context"]["contract"]["artifact_dependencies"]["color_utils.py"] = (
            "0" * 64)
    assert load_pattern_snapshot(
        {PATTERN_SESSION_KEY: record}, supported_contract(), UPLOAD_A, 48,
    ).state == "invalid"


def test_old_schema_is_invalid_and_old_registry_model_are_stale():
    old_schema = snapshot().to_record()
    old_schema["schema_version"] = 1
    assert load_pattern_snapshot(
        {PATTERN_SESSION_KEY: old_schema}, supported_contract(), UPLOAD_A, 48,
    ).state == "invalid"

    for field, value in (("registry_version", "old-registry"), ("model_version", "old-model")):
        old_contract = contract_variant(**{field: value})
        old_snapshot = snapshot(old_contract)
        assert load_pattern_snapshot(
            {PATTERN_SESSION_KEY: old_snapshot.to_record()},
            supported_contract(), UPLOAD_A, 48,
        ).state == "stale"


def test_record_validator_rejects_unexpected_context_fields():
    record = snapshot().to_record()
    record["context"]["legacy_dynamic_key"] = True
    record["context_fingerprint"] = hashlib.sha256(
        json.dumps(record["context"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with pytest.raises(ValueError):
        validate_pattern_snapshot_record(record)
