import copy
import json

import numpy as np
import pytest

from ui_analysis_snapshots import (
    ANALYSIS_SESSION_KEYS,
    AnalysisContext,
    AnalysisSnapshot,
    EngineStateMutationError,
    EngineStateRestoreError,
    analysis_engine_transaction,
    angle_payload_arrays,
    build_angle_payload,
    canonical_json,
    capture_engine_state,
    load_analysis_snapshot,
    require_engine_state_unchanged,
    source_artifact_identity,
    store_analysis_snapshot,
    validate_snapshot_record,
)


def context(kind="angle", **changes):
    values = dict(
        structure_type="single", geometry={"d": 180.0, "h": 300.0, "p": 400.0},
        material="TiO2 (anatase)", substrate="SiO2 (fused silica)",
        polarization="TE (s-pol)", angle_deg=0.0,
        route_id="lorentz_fano_fallback", model_version="torch_model-v1",
        artifact_version="sha256:fixture", registry_version="registry-v1",
        far_field_enabled=False, na=0.7, theta_obs_deg=30.0,
        sampling={"angles_deg": list(range(0, 85, 5)), "frozen_route": True},
    )
    values.update(changes)
    return AnalysisContext.create(kind, **values)


def payload(kind):
    if kind == "angle":
        angles = np.arange(0.0, 85.0, 5.0)
        rgb = np.full((17, 3), 0.25)
        return build_angle_payload(angles, rgb, np.ones(17, dtype=bool), ["route"] * 17)
    if kind == "sensitivity":
        side = {
            "status": "available", "requested_value": 175.0,
            "evaluated_value": 175.0, "actual_delta_nm": -5.0,
            "rgb": [0.2, 0.3, 0.4], "delta_e2000": 1.0, "reason": "",
        }
        return {
            "status": "available", "reason": "", "base_rgb": [0.2, 0.3, 0.4],
            "route_id": "route", "rows": [{
                "label": "D", "field": "d", "lower": side,
                "upper": {**side, "requested_value": 185.0, "evaluated_value": 185.0,
                          "actual_delta_nm": 5.0},
            }],
        }
    if kind == "mapping":
        return {
            "status": "available", "reason": "", "d_values": [80.0],
            "h_values": [200.0], "route_id": "route", "boundary": "fixture",
            "cells": [{"hi": 0, "di": 0, "status": "available",
                       "rgb": [0.2, 0.3, 0.4], "reason": ""}],
        }
    return {
        "status": "available", "reason": "", "delta_e2000": [1.0],
        "fano_rgb": [[0.2, 0.3, 0.4]], "generic_rgb": [[0.3, 0.4, 0.5]],
    }


def test_context_fingerprint_covers_required_fields_and_far_field_is_canonical(
        tmp_path, monkeypatch):
    base = context()
    assert base.na == 0.1 and base.theta_obs_deg == 0.0
    changes = [
        {"structure_type": "dual", "geometry": {"d1": 100, "h1": 200, "d2": 150, "h2": 250, "p": 400}},
        {"geometry": {"d": 181, "h": 300, "p": 400}},
        {"material": "a-Si (amorphous)"}, {"substrate": "Air"},
        {"polarization": "TM (p-pol)"}, {"angle_deg": 5.0},
        {"route_id": "fp_tmm"}, {"model_version": "v2"},
        {"artifact_version": "sha256:other"}, {"registry_version": "registry-v2"},
        {"far_field_enabled": True, "na": 0.5, "theta_obs_deg": 20.0},
        {"sampling": {"angles_deg": [0, 10], "frozen_route": True}},
    ]
    for changed in changes:
        assert context(**changed).fingerprint != base.fingerprint
    assert context(na=0.9, theta_obs_deg=70.0).fingerprint == base.fingerprint

    dependency = tmp_path / "route.py"
    dependency.write_bytes(b"route-A")
    identity_a = source_artifact_identity(tmp_path, ("route.py",))
    assert identity_a.available and identity_a.version.startswith("sha256:")
    assert source_artifact_identity(tmp_path, ("route.py",)) == identity_a
    dependency.write_bytes(b"route-B")
    identity_b = source_artifact_identity(tmp_path, ("route.py",))
    assert identity_b.available and identity_b.version != identity_a.version
    expected_mismatch = source_artifact_identity(
        tmp_path, ("route.py",),
        expected_sha256={"route.py": identity_a.dependencies[0].sha256},
    )
    assert not expected_mismatch.available
    assert expected_mismatch.dependencies[0].status == "hash_mismatch"

    state = {}
    old_context = context(artifact_version=identity_a.version)
    store_analysis_snapshot(state, AnalysisSnapshot.create(old_context, payload("angle")))
    assert load_analysis_snapshot(
        state, context(artifact_version=identity_b.version)).state == "stale"

    missing = source_artifact_identity(tmp_path, ("missing.py",))
    assert not missing.available and missing.version.startswith("unavailable:")
    assert source_artifact_identity(tmp_path, ("missing.py",)) == missing

    path_type = type(dependency)
    original_read_bytes = path_type.read_bytes

    def unreadable_read_bytes(path):
        if path == dependency:
            raise PermissionError("fixture")
        return original_read_bytes(path)

    monkeypatch.setattr(path_type, "read_bytes", unreadable_read_bytes)
    unreadable = source_artifact_identity(tmp_path, ("route.py",))
    assert not unreadable.available
    assert unreadable.dependencies[0].status == "unreadable"
    assert unreadable.version.startswith("unavailable:")

    monkeypatch.setattr(path_type, "read_bytes", original_read_bytes)
    original_resolve = path_type.resolve

    def looping_resolve(path, *args, **kwargs):
        if path == tmp_path:
            raise RuntimeError("fixture symlink loop")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(path_type, "resolve", looping_resolve)
    unresolvable = source_artifact_identity(tmp_path, ("route.py",))
    assert not unresolvable.available
    assert unresolvable.dependencies[0].status == "unresolvable"
    assert source_artifact_identity(tmp_path, ("route.py",)) == unresolvable

    monkeypatch.setattr(path_type, "resolve", original_resolve)
    monkeypatch.setattr(path_type, "read_bytes", lambda _path: "not-bytes")
    invalid_bytes = source_artifact_identity(tmp_path, ("route.py",))
    assert not invalid_bytes.available
    assert invalid_bytes.dependencies[0].status == "invalid_bytes"


@pytest.mark.parametrize("kind", list(ANALYSIS_SESSION_KEYS))
def test_store_load_reuse_stale_and_tamper_fail_closed(kind):
    current = context(kind, sampling={"kind": kind, "version": 1})
    snapshot = AnalysisSnapshot.create(current, payload(kind))
    state = {}
    store_analysis_snapshot(state, snapshot)
    assert set(state) == {ANALYSIS_SESSION_KEYS[kind]}
    loaded = load_analysis_snapshot(state, current)
    assert loaded.state == "fresh" and loaded.snapshot == snapshot

    stale = context(kind, geometry={"d": 181.0, "h": 300.0, "p": 400.0},
                    sampling={"kind": kind, "version": 1})
    assert load_analysis_snapshot(state, stale).state == "stale"

    record = copy.deepcopy(state[current.session_key])
    record["context"]["material"] = "tampered"
    assert load_analysis_snapshot({current.session_key: record}, current).state == "invalid"
    record = copy.deepcopy(state[current.session_key])
    record["payload_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="payload hash"):
        validate_snapshot_record(record)


def test_angle_unavailable_is_nan_raw_and_null_json_with_mask():
    angles = np.array([0.0, 5.0, 10.0])
    raw = np.array([[0.1, 0.2, 0.3], [np.nan, np.nan, np.nan], [0.4, 0.5, 0.6]])
    mask = np.array([True, False, True])
    built = build_angle_payload(angles, raw, mask, ["route"] * 3)
    assert built["rgb"][1] is None
    assert json.loads(canonical_json(built))["rgb"][1] is None
    _, restored, restored_mask, _ = angle_payload_arrays(built)
    assert np.isnan(restored[1]).all()
    assert restored_mask.tolist() == [True, False, True]


@pytest.mark.parametrize(
    "raw,mask",
    [
        (np.zeros((2, 3)), [True]),
        (np.array([[0.1, 0.2, 0.3], [0.5, 0.5, 0.5]]), [True, False]),
        (np.array([[np.nan, 0.2, 0.3]]), [True]),
    ],
)
def test_angle_shape_nan_and_gray_numeric_fail_closed(raw, mask):
    with pytest.raises(ValueError):
        build_angle_payload(np.arange(len(raw), dtype=float), raw, mask, ["route"] * len(raw))


def test_route_mismatch_can_be_recorded_as_all_unavailable_without_gray_numbers():
    raw = np.full((3, 3), np.nan)
    built = build_angle_payload([0, 5, 10], raw, [False] * 3, ["route_mismatch"] * 3)
    assert built["rgb"] == [None, None, None]
    assert built["available_mask"] == [False, False, False]


class Engine:
    def __init__(self):
        self._ui_session_contract_version = "fixture-v1"
        self._ui_session_owner = "owner-a"
        self._last_material = "TiO2"
        self._last_substrate = "SiO2"
        self._last_polarization = "TE"
        self._last_angle = 0.0
        self._enable_far_field = False
        self._na = 0.1
        self._theta_obs_deg = 0.0
        self._ui_library_identity = ("fixture",)
        self._cache = {("route",): (np.array([[0.1, 0.2, 0.3]]),)}
        self._coarse_grid_cache = {"coarse": np.array([1, 2], dtype=np.int16)}
        self.grid_params = np.array([[1.0, 2.0, 3.0]])
        self.grid_rgb = np.array([[0.1, 0.2, 0.3]])
        self.grid_lab = np.array([[1.0, 2.0, 3.0]])
        self.grid_xy = np.array([[0.2, 0.3]])


def test_engine_guard_detects_identity_or_grid_pollution():
    engine = Engine()
    before = capture_engine_state(engine)
    require_engine_state_unchanged(engine, before)
    engine.grid_rgb[0, 0] = 0.9
    with pytest.raises(RuntimeError, match="mutated"):
        require_engine_state_unchanged(engine, before)


def _mutate_engine(engine):
    engine._ui_session_owner = "polluted-owner"
    engine._ui_library_identity = ("polluted",)
    engine._last_material = "polluted-material"
    engine._last_substrate = "polluted-substrate"
    engine._last_polarization = "TM"
    engine._last_angle = 45.0
    engine._enable_far_field = True
    engine._na = 0.8
    engine._theta_obs_deg = 30.0
    engine._cache[("route",)][0][0, 0] = 0.9
    engine._coarse_grid_cache["coarse"][0] = 9
    engine.grid_params = np.ones((2, 3), dtype=np.float32)
    engine.grid_rgb = np.ones((2, 3), dtype=np.float32)
    engine.grid_lab = np.ones((2, 3), dtype=np.float32)
    engine.grid_xy = np.ones((2, 2), dtype=np.float32)


@pytest.mark.parametrize("raise_after_mutation", [False, True])
def test_engine_transaction_restores_mutate_then_return_or_raise(raise_after_mutation):
    engine = Engine()
    state = {"_ui_engine_instance_v1": engine}
    before = capture_engine_state(engine)
    original_arrays = {
        name: np.asarray(getattr(engine, name)).copy()
        for name in ("grid_params", "grid_rgb", "grid_lab", "grid_xy")
    }

    with pytest.raises(EngineStateMutationError, match="before-state restored"):
        with analysis_engine_transaction(engine, state):
            _mutate_engine(engine)
            state["_ui_engine_instance_v1"] = Engine()
            if raise_after_mutation:
                raise ValueError("fixture evaluator failure")

    assert state["_ui_engine_instance_v1"] is engine
    assert id(engine) == before.object_id
    require_engine_state_unchanged(engine, before)
    for name, expected in original_arrays.items():
        restored = np.asarray(getattr(engine, name))
        assert restored.shape == expected.shape
        assert restored.dtype == expected.dtype
        np.testing.assert_array_equal(restored, expected)
    assert engine._cache[("route",)][0][0, 0] == pytest.approx(0.1)
    np.testing.assert_array_equal(
        engine._coarse_grid_cache["coarse"], np.array([1, 2], dtype=np.int16))


def test_engine_transaction_normal_path_preserves_deep_snapshot():
    engine = Engine()
    state = {"_ui_engine_instance_v1": engine}
    before = capture_engine_state(engine)
    assert before.value("grid_rgb") is not engine.grid_rgb
    assert before.value("_cache") is not engine._cache

    with analysis_engine_transaction(engine, state):
        pass

    require_engine_state_unchanged(engine, before)
    assert state["_ui_engine_instance_v1"] is engine


def test_engine_transaction_restores_before_restore_verifier_raises(monkeypatch):
    import ui_analysis_snapshots as snapshots

    engine = Engine()
    state = {"_ui_engine_instance_v1": engine}
    before = capture_engine_state(engine)

    def verifier_failure(_engine, _before):
        raise RuntimeError("fixture verifier failure")

    monkeypatch.setattr(snapshots, "require_engine_state_unchanged", verifier_failure)
    with pytest.raises(EngineStateRestoreError, match="could not restore"):
        with snapshots.analysis_engine_transaction(engine, state):
            _mutate_engine(engine)
            state["_ui_engine_instance_v1"] = Engine()

    assert state["_ui_engine_instance_v1"] is engine
    assert id(engine) == before.object_id
    for name in ("grid_params", "grid_rgb", "grid_lab", "grid_xy"):
        expected = before.value(name)
        restored = np.asarray(getattr(engine, name))
        assert restored.shape == expected.shape
        assert restored.dtype == expected.dtype
        np.testing.assert_array_equal(restored, expected)


def test_two_session_snapshot_mappings_never_share_records():
    current = context()
    state_a, state_b = {}, {}
    store_analysis_snapshot(state_a, AnalysisSnapshot.create(current, payload("angle")))
    assert current.session_key not in state_b
    state_b[current.session_key] = copy.deepcopy(state_a[current.session_key])
    state_b[current.session_key]["payload"]["rgb"][0][0] = 0.9
    assert state_a[current.session_key]["payload"]["rgb"][0][0] == 0.25
