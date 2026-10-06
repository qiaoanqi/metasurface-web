import os
from contextlib import contextmanager
from functools import wraps
import hashlib

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _offline_small_app(monkeypatch):
    os.environ["HF_HUB_OFFLINE"] = "1"

    import engine
    import huggingface_hub
    import streamlit as st
    from ui_model_resources import _reset_resource_registry_for_tests
    from color_utils import rgb_to_lab, rgb_to_xy

    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("network disabled in UI tests")),
    )

    def small_library(_self):
        params = np.array([[180.0, 300.0, 400.0]])
        rgb = np.array([[0.2, 0.4, 0.6]])
        return params, rgb, rgb_to_lab(rgb), rgb_to_xy(rgb)

    monkeypatch.setattr(engine.MetaSurfaceColorEngine, "_build_library", small_library)

    download_calls = []
    delta_generator_type = type(st.sidebar)
    original_download_button = delta_generator_type.download_button
    original_top_level_download_button = st.download_button

    def record_download(args, kwargs):
        label = kwargs.get("label", args[0] if args else None)
        data = kwargs.get("data", args[1] if len(args) > 1 else None)
        file_name = kwargs.get("file_name", args[2] if len(args) > 2 else None)
        download_calls.append({"label": label, "file_name": file_name, "data": data})

    @wraps(original_download_button)
    def record_download_button(self, *args, **kwargs):
        record_download(args, kwargs)
        return original_download_button(self, *args, **kwargs)

    @wraps(original_top_level_download_button)
    def record_top_level_download_button(*args, **kwargs):
        record_download(args, kwargs)
        return original_top_level_download_button(*args, **kwargs)

    monkeypatch.setattr(delta_generator_type, "download_button", record_download_button)
    monkeypatch.setattr(st, "download_button", record_top_level_download_button)
    st.cache_data.clear()
    st.cache_resource.clear()
    _reset_resource_registry_for_tests()
    yield download_calls
    _reset_resource_registry_for_tests()
    st.cache_data.clear()
    st.cache_resource.clear()


def _run_app():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file("app.py").run(timeout=30)


def _markdown_text(at):
    return "\n".join(element.value for element in at.markdown)


def _all_text(at):
    return _markdown_text(at) + "\n" + "\n".join(item.value for item in at.caption)


def _unexpected_warnings(at):
    return [
        item.value for item in at.warning
        if "以下材料不能作为全波验证" not in item.value
        and not item.value.startswith("图案工具不可用：")
    ]


def _latest_download_names(download_calls):
    names = {}
    for call in download_calls:
        if call["label"] is not None:
            names[call["label"]] = call["file_name"]
    return names


def _latest_download_data(download_calls, label):
    return next(
        call["data"] for call in reversed(download_calls)
        if call["label"] == label
    )


def _control(at, collection, label):
    return next(item for item in getattr(at, collection) if item.label == label)


def _run_app_with_session_state(values):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py")
    for key, value in values.items():
        at.session_state[key] = value
    return at.run(timeout=30)


def _disable_rcwa_registry(monkeypatch):
    """Force the single-pillar ML route to the generic bound resource."""
    import ml_module

    for name in (
        "_RCWA_MODELS", "_RCWA_SUBSTRATE_MODELS", "_RCWA_WL_MODELS",
        "_RCWA_SESSIONS", "_RCWA_WL_SESSIONS",
    ):
        monkeypatch.setattr(ml_module, name, {})


def test_legacy_session_values_migrate_before_widgets_and_are_idempotent():
    from ui_session_migration import (
        SESSION_MIGRATION_KEY, SESSION_MIGRATION_SCHEMA, SESSION_MIGRATION_VERSION,
    )

    at = _run_app_with_session_state({
        "structure_type": "legacy-structure",
        "_pillar_material_pref": "legacy-material",
        "_substrate_pref": "legacy-substrate",
        "polarization": "legacy-polarization",
        "fp_mirror_type": "legacy-mirror",
        "far_field": "true",
        "a_val": 999,
        "theta_obs": float("nan"),
        "na_val": 9,
        "d_val": "legacy",
        "h_val": float("inf"),
        "p_val": float("-inf"),
        "d1_val": True,
        "h1_val": 999,
        "d2_val": False,
        "h2_val": -10,
        "fp_t_val": 999,
        "fp_target_wl": float("nan"),
    })
    assert len(at.exception) == 0
    expected = {
        "structure_type": "single",
        "_pillar_material_pref": "TiO2 (anatase)",
        "_substrate_pref": "SiO2 (fused silica)",
        "polarization": "TE (s-pol)",
        "fp_mirror_type": "介质 DBR (TiO2/SiO2)",
        "far_field": False,
        "a_val": 80.0,
        "theta_obs": 0.0,
        "na_val": 0.95,
        "d_val": 180.0,
        "h_val": 300.0,
        "p_val": 400.0,
        "d1_val": 120.0,
        "h1_val": 600.0,
        "d2_val": 200.0,
        "h2_val": 80.0,
        "fp_t_val": 600.0,
        "fp_target_wl": 450.0,
    }
    for key, value in expected.items():
        assert at.session_state[key] == value
    assert at.session_state[SESSION_MIGRATION_KEY] == {
        "schema": SESSION_MIGRATION_SCHEMA,
        "version": SESSION_MIGRATION_VERSION,
    }

    first = {key: at.session_state[key] for key in expected}
    at = at.run(timeout=30)
    assert len(at.exception) == 0
    assert {key: at.session_state[key] for key in expected} == first


def test_valid_legacy_session_values_are_preserved_exactly():
    at = _run_app_with_session_state({
        "structure_type": "dual",
        "_pillar_material_pref": "Si3N4 (nitride)",
        "_substrate_pref": "Al2O3 (sapphire)",
        "polarization": "TM (p-pol)",
        "far_field": True,
        "a_val": 15.0,
        "theta_obs": 20.0,
        "na_val": 0.5,
        "d1_val": 120.0,
        "h1_val": 250.0,
        "d2_val": 200.0,
        "h2_val": 350.0,
        "p_val": 450.0,
    })
    assert len(at.exception) == 0
    assert at.session_state["structure_type"] == "dual"
    assert at.session_state["_pillar_material_pref"] == "Si3N4 (nitride)"
    assert at.session_state["_substrate_pref"] == "Al2O3 (sapphire)"
    assert at.session_state["polarization"] == "TM (p-pol)"
    assert at.session_state["far_field"] is True
    assert at.session_state["a_val"] == 15.0
    assert at.session_state["theta_obs"] == 20.0
    assert at.session_state["na_val"] == 0.5
    assert at.session_state["p_val"] == 450.0


@pytest.mark.parametrize("raw", ["true", 1, float("nan")])
def test_invalid_legacy_ml_accel_is_false_and_stable(raw):
    from ui_session_migration import ML_ACCEL_PREFERENCE_INITIALIZED_KEY

    at = _run_app_with_session_state({"ml_accel": raw})
    assert len(at.exception) == 0
    assert at.session_state["ml_accel"] is False
    assert at.session_state["ml_accel_control"] is False
    assert at.session_state[ML_ACCEL_PREFERENCE_INITIALIZED_KEY] is True

    at = at.run(timeout=30)
    assert len(at.exception) == 0
    assert at.session_state["ml_accel"] is False
    assert at.session_state["ml_accel_control"] is False
    assert at.session_state[ML_ACCEL_PREFERENCE_INITIALIZED_KEY] is True


@pytest.mark.parametrize("value", [False, True])
def test_valid_legacy_ml_accel_bool_is_preserved(value):
    from ui_session_migration import ML_ACCEL_PREFERENCE_INITIALIZED_KEY

    at = _run_app_with_session_state({"ml_accel": value})
    assert len(at.exception) == 0
    assert at.session_state["ml_accel"] is value
    assert at.session_state["ml_accel_control"] is value
    assert at.session_state[ML_ACCEL_PREFERENCE_INITIALIZED_KEY] is True


def test_ml_checkbox_callback_and_forward_exports_follow_canonical_route(
        _offline_small_app):
    import csv
    import io
    import json
    from PIL import Image
    from color_utils import rgb_255
    from ui_session_migration import ML_ACCEL_PREFERENCE_INITIALIZED_KEY

    at = _run_app()
    at.session_state[ML_ACCEL_PREFERENCE_INITIALIZED_KEY] = False
    ml_checkbox = _control(at, "checkbox", "启用 ML 代理模型（快速候选预测）")
    ml_checkbox.set_value(False)
    at = at.run(timeout=30)
    assert len(at.exception) == 0
    assert at.session_state["ml_accel"] is False
    assert at.session_state["ml_accel_control"] is False
    assert at.session_state[ML_ACCEL_PREFERENCE_INITIALIZED_KEY] is True
    payload = json.loads(_latest_download_data(
        _offline_small_app, "下载当前结果 JSON"))
    csv_rows = list(csv.DictReader(io.StringIO(_latest_download_data(
        _offline_small_app, "下载光谱 CSV"))))
    assert payload["provenance"]["route_id"] == "lorentz_fano_fallback"
    assert json.loads(csv_rows[0]["provenance_json"]) == payload["provenance"]
    swatch = Image.open(io.BytesIO(_latest_download_data(
        _offline_small_app, "下载色板 PNG")))
    assert swatch.getpixel((0, 0)) == rgb_255(payload["rgb"])

    at = at.run(timeout=30)
    assert at.session_state["ml_accel"] is False
    assert at.session_state[ML_ACCEL_PREFERENCE_INITIALIZED_KEY] is True

    _control(
        at, "checkbox", "启用 ML 代理模型（快速候选预测）"
    ).set_value(True)
    at = at.run(timeout=30)
    assert len(at.exception) == 0
    assert at.session_state["ml_accel"] is True
    assert at.session_state["ml_accel_control"] is True
    payload = json.loads(_latest_download_data(
        _offline_small_app, "下载当前结果 JSON"))
    csv_rows = list(csv.DictReader(io.StringIO(_latest_download_data(
        _offline_small_app, "下载光谱 CSV"))))
    assert payload["provenance"]["route_id"] in {
        "rcwa_surrogate", "ml_surrogate",
    }
    assert payload["provenance"]["model_artifact_version"].startswith("sha256:")
    assert json.loads(csv_rows[0]["provenance_json"]) == payload["provenance"]
    swatch = Image.open(io.BytesIO(_latest_download_data(
        _offline_small_app, "下载色板 PNG")))
    assert swatch.getpixel((0, 0)) == rgb_255(payload["rgb"])


def test_ml_preview_and_sensitivity_snapshot_share_model_artifact(
        _offline_small_app):
    import json
    from ui_analysis_snapshots import ANALYSIS_SESSION_KEYS

    at = _run_app()
    preview = json.loads(_latest_download_data(
        _offline_small_app, "下载当前结果 JSON"))
    assert preview["provenance"]["route_id"] in {
        "ml_surrogate", "rcwa_surrogate"}
    next(
        button for button in at.button
        if button.label == "运行 / 加载当前灵敏度分析"
    ).click()
    at = at.run(timeout=30)
    snapshot = at.session_state[ANALYSIS_SESSION_KEYS["sensitivity"]]

    assert len(at.exception) == 0
    assert snapshot["payload"]["status"] == "available"
    assert snapshot["payload"]["analysis_artifact_version"] == (
        snapshot["context"]["artifact_version"])
    assert snapshot["payload"]["model_artifact_version"] == (
        preview["provenance"]["model_artifact_version"])


def test_angle_payload_uses_its_generic_context_not_rcwa_preview_identity(
        monkeypatch, _offline_small_app):
    from types import SimpleNamespace
    import app
    from ui_analysis_snapshots import AnalysisContext, AnalysisSnapshot, build_angle_payload
    from ui_model_resources import BoundModelResource

    rcwa_session = SimpleNamespace(_model_path="models/rcwa.onnx")
    generic_session = SimpleNamespace(_model_path="models/generic.onnx")
    rcwa_key = app._rcwa_resource_context(
        "TiO2 (anatase)", "SiO2 (fused silica)")[0]
    generic_key = app._primary_resource_context_key()
    monkeypatch.setattr(
        app, "_rcwa_runtime_binding",
        BoundModelResource.create(
            "rcwa", rcwa_key, "sha256:RCWA", ("models/rcwa.onnx",),
            ("models/rcwa.onnx",), (rcwa_session,)),
    )
    monkeypatch.setattr(
        app, "_primary_runtime_binding",
        BoundModelResource.create(
            "primary", generic_key, "sha256:GENERIC", ("models/generic.onnx",),
            ("models/generic.onnx",), (generic_session,)),
    )
    angle_context = AnalysisContext.create(
        "angle", structure_type="single",
        geometry={"D_nm": 180.0, "H_nm": 300.0, "P_nm": 400.0},
        material="TiO2 (anatase)", substrate="SiO2 (fused silica)",
        polarization="TE (s-pol)", angle_deg=0.0,
        route_id="ML surrogate (generic angle-conditioned)",
        model_version="generic", artifact_version="sha256:ANGLE",
        registry_version="fixture", far_field_enabled=False,
        na=0.1, theta_obs_deg=0.0,
        sampling={"angles_deg": list(range(0, 81, 5))},
    )
    payload = build_angle_payload(
        [0.0], [[0.2, 0.3, 0.4]], [True],
        ["ML surrogate (generic angle-conditioned)"],
    )
    payload.update({
        "analysis_artifact_version": angle_context.artifact_version,
        "model_artifact_version": app._analysis_model_artifact_version(angle_context),
    })
    snapshot = AnalysisSnapshot.create(angle_context, payload)
    assert snapshot.payload["analysis_artifact_version"] == angle_context.artifact_version
    assert snapshot.payload["model_artifact_version"] == "sha256:GENERIC"
    assert snapshot.payload["model_artifact_version"] != "sha256:RCWA"


def test_preview_session_drift_cannot_be_exported_as_generic_result(
        monkeypatch, _offline_small_app):
    import json
    import ml_module

    _disable_rcwa_registry(monkeypatch)
    original = ml_module.predict_generic_spectrum
    calls = {"generic": 0}

    def mutate_session(*args, **kwargs):
        calls["generic"] += 1
        spectrum = original(*args, **kwargs)
        ml_module._ORT_SESSION = object()
        return spectrum

    monkeypatch.setattr(ml_module, "predict_generic_spectrum", mutate_session)
    at = _run_app()
    preview = json.loads(_latest_download_data(
        _offline_small_app, "下载当前结果 JSON"))

    assert len(at.exception) == 0
    assert calls["generic"] == 1
    assert preview["provenance"]["route_id"] == "lorentz_fano_fallback"
    assert preview["provenance"]["model_artifact_version"] == "not_applicable"


def test_sensitivity_session_drift_stores_no_mutated_numeric_result(
        monkeypatch, _offline_small_app):
    import ml_module
    from ui_analysis_snapshots import ANALYSIS_SESSION_KEYS

    _disable_rcwa_registry(monkeypatch)
    original = ml_module.predict_generic_spectrum
    calls = {"perturbed": 0}

    def mutate_perturbation(d_nm, *args, **kwargs):
        spectrum = original(d_nm, *args, **kwargs)
        if float(d_nm) != 180.0:
            calls["perturbed"] += 1
            ml_module._ORT_SESSION = object()
            raise RuntimeError("fixture business error after session drift")
        return spectrum

    monkeypatch.setattr(ml_module, "predict_generic_spectrum", mutate_perturbation)
    at = _run_app()
    next(button for button in at.button
         if button.label == "运行 / 加载当前灵敏度分析").click()
    at = at.run(timeout=30)
    key = ANALYSIS_SESSION_KEYS["sensitivity"]

    assert len(at.exception) == 0
    assert calls["perturbed"] == 1
    assert key not in at.session_state.filtered_state
    assert any("灵敏度分析模型会话身份发生漂移" in item.value for item in at.error)


def test_mapping_session_drift_clears_snapshot_and_stores_no_cells(
        monkeypatch, _offline_small_app):
    import ml_module
    import ui_forward_routes as routes
    from ui_analysis_snapshots import ANALYSIS_SESSION_KEYS

    _disable_rcwa_registry(monkeypatch)
    monkeypatch.setitem(
        routes._MAPPING_DOMAIN_REGISTRY,
        ("TiO2 (anatase)", "SiO2 (fused silica)", "ml_surrogate"),
        (routes._ANALYTICAL_MAPPING_DOMAIN, "fixture generic mapping domain"),
    )
    original = ml_module.predict_generic_spectrum
    calls = {"perturbed": 0}

    def mutate_map(d_nm, *args, **kwargs):
        spectrum = original(d_nm, *args, **kwargs)
        if float(d_nm) != 180.0:
            calls["perturbed"] += 1
            ml_module._ORT_SESSION = object()
        return spectrum

    monkeypatch.setattr(ml_module, "predict_generic_spectrum", mutate_map)
    at = _run_app()
    next(button for button in at.button
         if button.label == "运行 / 加载当前 D-H 映射").click()
    at = at.run(timeout=30)
    key = ANALYSIS_SESSION_KEYS["mapping"]

    assert len(at.exception) == 0
    assert calls["perturbed"] >= 1
    assert key not in at.session_state.filtered_state
    assert any("D-H 映射模型会话身份发生漂移" in item.value for item in at.error)


def test_angle_session_drift_clears_snapshot_and_hides_export(
        monkeypatch, _offline_small_app):
    import ml_module
    from ui_analysis_snapshots import ANALYSIS_SESSION_KEYS

    original = ml_module.predict_generic_spectrum

    def mutate_angle(d_nm, h_nm, p_nm, angle_deg, *args, **kwargs):
        spectrum = original(d_nm, h_nm, p_nm, angle_deg, *args, **kwargs)
        if float(angle_deg) != 0.0:
            ml_module._ORT_SESSION = object()
        return spectrum

    monkeypatch.setattr(ml_module, "predict_generic_spectrum", mutate_angle)
    at = _run_app()
    next(button for button in at.button
         if button.label == "运行 / 加载当前入射角扫描").click()
    _offline_small_app.clear()
    at = at.run(timeout=30)
    key = ANALYSIS_SESSION_KEYS["angle"]

    assert len(at.exception) == 0
    assert key not in at.session_state.filtered_state
    assert any("角扫模型会话身份发生漂移" in item.value for item in at.error)
    assert not any(call["label"] == "下载角扫 JSON" for call in _offline_small_app)

def test_fresh_fp_then_single_initializes_ready_preference_once(
        monkeypatch, _offline_small_app):
    import json
    import ml_module
    from ui_session_migration import ML_ACCEL_PREFERENCE_INITIALIZED_KEY

    calls = {"primary": 0, "dual": 0, "rcwa": 0}
    real_init_ml = ml_module.init_ml

    def count(name):
        def inner(*_args, **_kwargs):
            calls[name] += 1
            return True
        return inner

    def init_primary():
        calls["primary"] += 1
        return real_init_ml()

    monkeypatch.setattr(ml_module, "init_ml", init_primary)
    monkeypatch.setattr(ml_module, "init_dual_ml", count("dual"))
    monkeypatch.setattr(ml_module, "init_rcwa_ml", count("rcwa"))
    at = _run_app_with_session_state({"structure_type": "fp"})

    assert len(at.exception) == 0
    assert calls == {"primary": 0, "dual": 0, "rcwa": 0}
    assert at.session_state["ml_accel"] is False
    assert at.session_state["ml_accel_control"] is False
    assert at.session_state[ML_ACCEL_PREFERENCE_INITIALIZED_KEY] is False
    assert "启用 ML 代理模型（快速候选预测）" not in [
        item.label for item in at.checkbox]
    payload = json.loads(_latest_download_data(
        _offline_small_app, "下载当前结果 JSON"))
    assert payload["provenance"]["route_id"] == "fp_tmm"

    at.radio[0].set_value("单柱")
    at = at.run(timeout=30)
    assert len(at.exception) == 0
    # The strict resource path no longer calls the legacy all-registry loader.
    # With no exact RCWA ONNX asset on disk, single correctly stays generic.
    assert calls == {"primary": 1, "dual": 0, "rcwa": 0}
    assert at.session_state["ml_accel"] is True
    assert at.session_state["ml_accel_control"] is True
    assert at.session_state[ML_ACCEL_PREFERENCE_INITIALIZED_KEY] is True

    at = at.run(timeout=30)
    assert calls == {"primary": 1, "dual": 0, "rcwa": 0}
    assert at.session_state["ml_accel"] is True
    assert at.session_state[ML_ACCEL_PREFERENCE_INITIALIZED_KEY] is True


def test_fresh_fp_keeps_grid_lazy_until_first_real_grid_consumer(
        monkeypatch, _offline_small_app):
    import engine as engine_module
    from ui_engine_session import (
        ENGINE_SESSION_KEY, LibraryIdentity, bind_engine_library,
        engine_library_matches,
    )

    calls = {"rebuild": 0}

    def rebuild(self, material, substrate, polarization, angle_deg):
        calls["rebuild"] += 1
        self._grid_library_initialized = True
        self._last_material = material
        self._last_substrate = substrate
        self._last_polarization = polarization
        self._last_angle = float(angle_deg)
        self.grid_params = np.array([[180.0, 300.0, 400.0]])
        self.grid_rgb = np.array([[0.2, 0.4, 0.6]])
        self.grid_lab = np.array([[42.0, 3.0, -7.0]])
        self.grid_xy = np.array([[0.25, 0.30]])

    monkeypatch.setattr(
        engine_module.MetaSurfaceColorEngine, "rebuild_library", rebuild)

    at = _run_app_with_session_state({"structure_type": "fp"})
    assert len(at.exception) == 0
    session_engine = at.session_state[ENGINE_SESSION_KEY]
    assert calls == {"rebuild": 0}
    assert session_engine.grid_params.shape == (0, 3)

    at.radio[0].set_value("单柱")
    at = at.run(timeout=30)
    assert len(at.exception) == 0
    assert at.session_state[ENGINE_SESSION_KEY] is session_engine
    assert calls == {"rebuild": 0}
    assert session_engine.grid_params.shape == (0, 3)

    target = LibraryIdentity(
        "TiO2 (anatase)", "SiO2 (fused silica)", "TE (s-pol)", 0.0)
    bind_engine_library(session_engine, target, at.session_state)
    assert calls == {"rebuild": 1}
    assert engine_library_matches(session_engine, target)

    bind_engine_library(session_engine, target, at.session_state)
    assert calls == {"rebuild": 1}


@pytest.mark.parametrize("legacy", ["invalid", False])
def test_legacy_fp_then_single_never_overwrites_false_preference(legacy):
    from ui_session_migration import ML_ACCEL_PREFERENCE_INITIALIZED_KEY

    at = _run_app_with_session_state({
        "structure_type": "fp",
        "ml_accel": legacy,
    })
    assert len(at.exception) == 0
    assert at.session_state["ml_accel"] is False
    assert at.session_state[ML_ACCEL_PREFERENCE_INITIALIZED_KEY] is True


@pytest.mark.parametrize(
    ("mirror", "expected_calls", "expected_geometry"),
    [
        (
            "介质 DBR (TiO2/SiO2)", {"dbr": 1, "ag": 0},
            {"T_nm", "center_wavelength_nm", "top_pairs", "bottom_pairs"},
        ),
        (
            "金属 Ag (减色)", {"dbr": 0, "ag": 1},
            {"T_nm", "top_ag_nm"},
        ),
    ],
)
def test_fp_forward_is_top_level_tmm_only_with_forbidden_routes_bombed(
        mirror, expected_calls, expected_geometry, monkeypatch, _offline_small_app):
    import csv
    import io
    import json

    import engine as engine_module
    import fp_cavity
    import ml_module
    import torch_model
    import ui_engine_session
    import ui_model_difference_contracts

    calls = {
        "dbr": 0, "ag": 0, "primary_init": 0, "dual_init": 0,
        "rcwa_init": 0, "generic_predict": 0, "dual_predict": 0,
        "rcwa_freeze": 0, "generic_freeze": 0, "physical": 0,
        "torch": 0, "far_field": 0, "grid_build": 0,
    }

    def bomb(name):
        def inner(*_args, **_kwargs):
            calls[name] += 1
            raise AssertionError(f"FP must not call {name}")
        return inner

    def dbr_stub(*_args, **_kwargs):
        calls["dbr"] += 1
        return np.linspace(380.0, 780.0, 81), np.full(81, 0.35)

    def ag_stub(*_args, **_kwargs):
        calls["ag"] += 1
        return np.linspace(380.0, 780.0, 81), np.full(81, 0.55)

    monkeypatch.setattr(ml_module, "init_ml", bomb("primary_init"))
    monkeypatch.setattr(ml_module, "init_dual_ml", bomb("dual_init"))
    monkeypatch.setattr(ml_module, "init_rcwa_ml", bomb("rcwa_init"))
    monkeypatch.setattr(
        ml_module, "predict_generic_spectrum", bomb("generic_predict"))
    monkeypatch.setattr(ml_module, "predict_dual_spectrum", bomb("dual_predict"))
    monkeypatch.setattr(
        ml_module, "freeze_rcwa_spectrum_predictor", bomb("rcwa_freeze"))
    monkeypatch.setattr(
        ui_model_difference_contracts, "freeze_generic_onnx_evaluator",
        bomb("generic_freeze"))
    monkeypatch.setattr(
        engine_module.MetaSurfaceColorEngine, "compute_spectrum", bomb("physical"))
    monkeypatch.setattr(
        engine_module.MetaSurfaceColorEngine, "_build_library", bomb("grid_build"))
    monkeypatch.setattr(
        engine_module.MetaSurfaceColorEngine, "_far_field_spectrum",
        bomb("far_field"))
    monkeypatch.setattr(
        engine_module.MetaSurfaceColorEngine, "_dual_far_field_spectrum",
        bomb("far_field"))
    monkeypatch.setattr(
        ui_engine_session, "configure_engine_far_field", bomb("far_field"))
    monkeypatch.setattr(
        torch_model, "batch_lorentzian_spectrum", bomb("torch"))
    monkeypatch.setattr(fp_cavity, "fp_dielectric_spectrum", dbr_stub)
    monkeypatch.setattr(fp_cavity, "fp_cavity_spectrum", ag_stub)

    at = _run_app_with_session_state({
        "structure_type": "fp",
        "fp_mirror_type": mirror,
        "far_field": True,
        "ml_accel": True,
    })
    assert len(at.exception) == 0
    assert {key: calls[key] for key in ("dbr", "ag")} == expected_calls
    assert all(calls[key] == 0 for key in calls if key not in {"dbr", "ag"})

    payload = json.loads(_latest_download_data(
        _offline_small_app, "下载当前结果 JSON"))
    provenance = payload["provenance"]
    assert provenance["route_id"] == "fp_tmm"
    assert provenance["structure_type"] == "fp"
    assert set(provenance["geometry"]) == expected_geometry
    assert not any(key.startswith(("D", "H", "P")) for key in provenance["geometry"])
    csv_rows = list(csv.DictReader(io.StringIO(_latest_download_data(
        _offline_small_app, "下载光谱 CSV"))))
    assert json.loads(csv_rows[0]["provenance_json"]) == provenance


def test_dual_route_loads_only_dual_model(monkeypatch, _offline_small_app):
    import json
    import os
    from pathlib import Path
    from types import SimpleNamespace
    import ml_module

    calls = {"primary": 0, "dual": 0, "rcwa": 0, "generic": 0, "dual_predict": 0}
    original_isfile = os.path.isfile
    original_stat = Path.stat
    original_read_bytes = Path.read_bytes
    project_root = Path(__file__).resolve().parents[1]
    dual_path = project_root / "models" / "dual_mlp_v3_multi.onnx"
    dual_bytes = b"dual-model-test-artifact"
    dual_session = SimpleNamespace(_model_path=str(dual_path))

    def counted(name):
        def inner(*_args, **_kwargs):
            calls[name] += 1
            return True
        return inner

    def isfile(path):
        if str(path).replace("\\", "/").endswith("models/dual_mlp_v3_multi.onnx"):
            return True
        return original_isfile(path)

    def stat(path, *args, **kwargs):
        if str(path).replace("\\", "/").endswith("models/dual_mlp_v3_multi.onnx"):
            return SimpleNamespace(st_size=len(dual_bytes))
        return original_stat(path, *args, **kwargs)

    def read_bytes(path):
        if str(path).replace("\\", "/").endswith("models/dual_mlp_v3_multi.onnx"):
            return dual_bytes
        return original_read_bytes(path)

    def init_dual():
        calls["dual"] += 1
        ml_module._DUAL_ORT_SESSION = dual_session
        ml_module._DUAL_ORT_AVAILABLE = True
        return True

    def dual_predict(*_args, **_kwargs):
        calls["dual_predict"] += 1
        return np.full(81, 0.4)

    monkeypatch.setattr(os.path, "isfile", isfile)
    monkeypatch.setattr(Path, "stat", stat)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(ml_module, "init_ml", counted("primary"))
    monkeypatch.setattr(ml_module, "init_dual_ml", init_dual)
    monkeypatch.setattr(ml_module, "init_rcwa_ml", counted("rcwa"))
    monkeypatch.setattr(
        ml_module, "predict_generic_spectrum", counted("generic"))
    monkeypatch.setattr(ml_module, "predict_dual_spectrum", dual_predict)

    at = _run_app_with_session_state({"structure_type": "dual"})
    assert len(at.exception) == 0
    assert calls == {
        "primary": 0, "dual": 1, "rcwa": 0,
        "generic": 0, "dual_predict": 1,
    }
    payload = json.loads(_latest_download_data(
        _offline_small_app, "下载当前结果 JSON"))
    assert payload["provenance"]["structure_type"] == "dual"
    assert payload["provenance"]["route_id"] == "ml_surrogate"


def test_single_fp_single_reuses_model_loaders_and_preference(
        monkeypatch, _offline_small_app):
    import ml_module
    from ui_session_migration import ML_ACCEL_PREFERENCE_INITIALIZED_KEY

    calls = {"primary": 0, "dual": 0, "rcwa": 0}
    real_init_ml = ml_module.init_ml

    def counted(name):
        def inner(*_args, **_kwargs):
            calls[name] += 1
            return True
        return inner

    def init_primary():
        calls["primary"] += 1
        return real_init_ml()

    monkeypatch.setattr(ml_module, "init_ml", init_primary)
    monkeypatch.setattr(ml_module, "init_dual_ml", counted("dual"))
    monkeypatch.setattr(ml_module, "init_rcwa_ml", counted("rcwa"))

    at = _run_app()
    assert calls == {"primary": 1, "dual": 0, "rcwa": 0}
    assert at.session_state["ml_accel"] is True
    assert at.session_state[ML_ACCEL_PREFERENCE_INITIALIZED_KEY] is True

    at.radio[0].set_value("FP 腔（Fabry-Pérot）")
    at = at.run(timeout=30)
    assert len(at.exception) == 0
    assert calls == {"primary": 1, "dual": 0, "rcwa": 0}
    assert at.session_state["ml_accel"] is True

    at.radio[0].set_value("单柱")
    at = at.run(timeout=30)
    assert len(at.exception) == 0
    assert calls == {"primary": 1, "dual": 0, "rcwa": 0}
    assert at.session_state["ml_accel"] is True
    assert at.session_state[ML_ACCEL_PREFERENCE_INITIALIZED_KEY] is True


def test_paired_controls_sync_in_one_rerun_and_forward_exports_use_canonical(
        _offline_small_app):
    import csv
    import io
    import json

    at = _run_app()
    _control(at, "slider", "直径 D (nm)").set_value(231.2)
    at = at.run(timeout=30)
    assert len(at.exception) == 0
    assert _control(at, "slider", "直径 D (nm)").value == 231.2
    assert _control(at, "number_input", "精确输入 D").value == 231.2
    assert at.session_state["d_val"] == 231.2
    payload = json.loads(_latest_download_data(
        _offline_small_app, "下载当前结果 JSON"))
    assert payload["provenance"]["geometry"]["D_nm"] == 231.2
    csv_rows = list(csv.DictReader(io.StringIO(_latest_download_data(
        _offline_small_app, "下载光谱 CSV"))))
    assert json.loads(csv_rows[0]["provenance_json"])["geometry"]["D_nm"] == 231.2

    _control(at, "number_input", "精确输入 D").set_value(246.7)
    at = at.run(timeout=30)
    assert _control(at, "slider", "直径 D (nm)").value == 246.7
    assert _control(at, "number_input", "精确输入 D").value == 246.7
    assert at.session_state["d_val"] == 246.7
    payload = json.loads(_latest_download_data(
        _offline_small_app, "下载当前结果 JSON"))
    assert payload["provenance"]["geometry"]["D_nm"] == 246.7

    _control(at, "number_input", "精确输入 角度").set_value(12.3)
    at = at.run(timeout=30)
    assert _control(at, "slider", "入射角 (°)").value == 12.3
    assert at.session_state["a_val"] == 12.3
    payload = json.loads(_latest_download_data(
        _offline_small_app, "下载当前结果 JSON"))
    assert payload["provenance"]["angle_deg"] == 12.3

    _control(at, "slider", "高度 H (nm)").set_value(315.0)
    _control(at, "number_input", "精确输入 P").set_value(430.0)
    at = at.run(timeout=30)
    assert at.session_state["h_val"] == 315.0
    assert at.session_state["single_h_input"] == 315.0
    assert at.session_state["p_val"] == 430.0
    assert at.session_state["single_p_slider"] == 430.0
    payload = json.loads(_latest_download_data(
        _offline_small_app, "下载当前结果 JSON"))
    assert payload["provenance"]["geometry"] == {
        "D_nm": 246.7, "H_nm": 315.0, "P_nm": 430.0,
    }


def test_far_field_theta_and_na_pairs_sync_in_one_rerun(_offline_small_app):
    import json

    at = _run_app()
    _control(
        at, "checkbox", "启用角谱远场传播 (Angular Spectrum)"
    ).set_value(True)
    at = at.run(timeout=30)
    _control(at, "slider", "观察角度 θ (°)").set_value(20.0)
    _control(at, "number_input", "精确 NA").set_value(0.5)
    at = at.run(timeout=30)

    assert len(at.exception) == 0
    assert at.session_state["far_field"] is True
    assert at.session_state["far_field_control"] is True
    assert at.session_state["theta_obs"] == 20.0
    assert _control(at, "number_input", "精确 θ").value == 20.0
    assert at.session_state["na_val"] == 0.5
    assert _control(at, "slider", "收集数值孔径 NA").value == 0.5
    payload = json.loads(_latest_download_data(
        _offline_small_app, "下载当前结果 JSON"))
    assert payload["provenance"]["theta_obs_deg"] == 20.0
    assert payload["provenance"]["na"] == 0.5


def test_single_dual_fp_paired_widget_keys_do_not_crosstalk():
    at = _run_app()
    _control(at, "number_input", "精确输入 P").set_value(425.0)
    at = at.run(timeout=30)
    assert at.session_state["p_val"] == 425.0
    assert at.session_state["single_p_slider"] == 425.0
    assert at.session_state["dual_p_slider"] == 425.0

    at.radio[0].set_value("双柱")
    at = at.run(timeout=30)
    assert _control(at, "slider", "周期 P (nm)").value == 425.0
    _control(at, "slider", "柱1直径 D1 (nm)").set_value(130.0)
    _control(at, "number_input", "精确输入 H1").set_value(260.0)
    _control(at, "slider", "柱2直径 D2 (nm)").set_value(210.0)
    _control(at, "number_input", "精确输入 H2").set_value(360.0)
    _control(at, "number_input", "精确输入 P").set_value(450.0)
    at = at.run(timeout=30)
    assert at.session_state["d1_val"] == 130.0
    assert at.session_state["dual_d1_input"] == 130.0
    assert at.session_state["h1_val"] == 260.0
    assert at.session_state["dual_h1_slider"] == 260.0
    assert at.session_state["d2_val"] == 210.0
    assert at.session_state["dual_d2_input"] == 210.0
    assert at.session_state["h2_val"] == 360.0
    assert at.session_state["dual_h2_slider"] == 360.0
    assert at.session_state["p_val"] == 450.0
    assert at.session_state["single_p_slider"] == 450.0

    at.radio[0].set_value("FP 腔（Fabry-Pérot）")
    at = at.run(timeout=30)
    _control(at, "slider", "腔长 T (nm)").set_value(275.0)
    _control(at, "number_input", "精确中心波长").set_value(525.0)
    at = at.run(timeout=30)
    assert at.session_state["fp_t_val"] == 275.0
    assert _control(at, "number_input", "精确输入 T").value == 275.0
    assert at.session_state["fp_target_wl"] == 525.0
    assert _control(at, "slider", "DBR 中心波长 (nm)").value == 525.0

    at.radio[0].set_value("单柱")
    at = at.run(timeout=30)
    assert at.session_state["d_val"] == 180.0
    assert _control(at, "slider", "周期 P (nm)").value == 450.0


def test_independent_apptest_sessions_have_private_engines_and_default_is_stable(
        _offline_small_app):
    import json
    from ui_engine_session import ENGINE_SESSION_KEY
    from streamlit.testing.v1 import AppTest

    app_a = AppTest.from_file("app.py").run(timeout=30)
    app_b = AppTest.from_file("app.py").run(timeout=30)
    engine_a = app_a.session_state[ENGINE_SESSION_KEY]
    engine_b = app_b.session_state[ENGINE_SESSION_KEY]
    assert engine_a is not engine_b

    baseline_b = json.loads(_latest_download_data(
        _offline_small_app, "下载当前结果 JSON"))
    far_field_a = next(
        item for item in app_a.checkbox
        if item.label == "启用角谱远场传播 (Angular Spectrum)")
    far_field_a.set_value(True)
    app_a = app_a.run(timeout=30)
    next(item for item in app_a.slider if item.label == "观察角度 θ (°)").set_value(20.0)
    next(item for item in app_a.slider if item.label == "收集数值孔径 NA").set_value(0.5)
    app_a = app_a.run(timeout=30)
    assert app_a.session_state[ENGINE_SESSION_KEY] is engine_a
    assert (engine_a._enable_far_field, engine_a._na, engine_a._theta_obs_deg) == (
        True, 0.5, 20.0)

    far_field_a = next(
        item for item in app_a.checkbox
        if item.label == "启用角谱远场传播 (Angular Spectrum)")
    far_field_a.set_value(False)
    app_a = app_a.run(timeout=30)
    assert (engine_a._enable_far_field, engine_a._na, engine_a._theta_obs_deg) == (
        False, 0.1, 0.0)

    app_b = app_b.run(timeout=30)
    after_b = json.loads(_latest_download_data(
        _offline_small_app, "下载当前结果 JSON"))
    assert app_b.session_state[ENGINE_SESSION_KEY] is engine_b
    assert (engine_b._enable_far_field, engine_b._na, engine_b._theta_obs_deg) == (
        False, 0.1, 0.0)
    assert after_b["reflectance"] == baseline_b["reflectance"]
    assert after_b["rgb"] == baseline_b["rgb"]


def test_pattern_generation_uses_local_fixed_identity_and_preserves_main_engine(
        monkeypatch, _offline_small_app, tmp_path):
    import io
    import json

    import engine as engine_module
    import streamlit as st
    from PIL import Image
    import ui_engine_session as engine_session
    from ui_analysis_snapshots import capture_engine_state, require_engine_state_unchanged
    from ui_engine_session import ENGINE_LIBRARY_KEY, ENGINE_SESSION_KEY
    from ui_pattern_contracts import PATTERN_SESSION_KEY

    payload = io.BytesIO()
    Image.new("RGB", (2, 2), (64, 128, 192)).save(payload, format="PNG")
    upload_bytes = payload.getvalue()
    monkeypatch.setattr(
        st, "file_uploader", lambda *_args, **_kwargs: io.BytesIO(upload_bytes))

    calls = {
        "bind": 0, "map": 0, "vectorized": 0,
        "disk_load": 0, "disk_save": 0,
    }
    identities = []
    local_engines = []
    original_bind = engine_session.bind_engine_library
    original_map = engine_module.MetaSurfaceColorEngine.image_to_metasurface_map
    original_vectorized = engine_module.MetaSurfaceColorEngine._build_library_vectorised

    def counted_bind(local_engine, identity, session_state=None):
        calls["bind"] += 1
        identities.append(identity)
        local_engines.append(local_engine)
        assert session_state is None
        return original_bind(local_engine, identity, session_state)

    def counted_map(local_engine, image, max_size=80):
        calls["map"] += 1
        return original_map(local_engine, image, max_size)

    monkeypatch.setattr(engine_session, "bind_engine_library", counted_bind)
    monkeypatch.setattr(
        engine_module.MetaSurfaceColorEngine, "image_to_metasurface_map", counted_map)

    at = _run_app()
    forward_before = json.loads(_latest_download_data(
        _offline_small_app, "下载当前结果 JSON"))
    main_engine = at.session_state[ENGINE_SESSION_KEY]
    main_before = capture_engine_state(main_engine)
    library_before = at.session_state.filtered_state.get(ENGINE_LIBRARY_KEY)
    structure_before = at.session_state["structure_type"]

    old_cache = tmp_path / "grid_old_bomb.pkl"
    old_cache.write_bytes(b"malicious-old-grid-pickle-must-not-be-read")
    old_cache_hash = hashlib.sha256(old_cache.read_bytes()).hexdigest()

    def bomb_load(*_args, **_kwargs):
        calls["disk_load"] += 1
        raise AssertionError("isolated pattern engine must not read disk pickle")

    def bomb_save(*_args, **_kwargs):
        calls["disk_save"] += 1
        raise AssertionError("isolated pattern engine must not write disk pickle")

    def counted_vectorized(local_engine, *args, **kwargs):
        calls["vectorized"] += 1
        return original_vectorized(local_engine, *args, **kwargs)

    def current_code_builder(local_engine):
        return local_engine._build_library_vectorised(
            np.array([180.0]), np.array([300.0]), np.array([400.0]), 2.3,
            local_engine._last_material, local_engine._last_substrate,
        )

    monkeypatch.setattr(
        engine_module.MetaSurfaceColorEngine, "_grid_cache_path",
        staticmethod(lambda _key: str(old_cache)),
    )
    monkeypatch.setattr(
        engine_module.MetaSurfaceColorEngine, "_load_grid_disk",
        staticmethod(bomb_load),
    )
    monkeypatch.setattr(
        engine_module.MetaSurfaceColorEngine, "_save_grid_disk",
        staticmethod(bomb_save),
    )
    monkeypatch.setattr(
        engine_module.MetaSurfaceColorEngine, "_build_library_vectorised",
        counted_vectorized,
    )
    monkeypatch.setattr(
        engine_module.MetaSurfaceColorEngine, "_build_library",
        current_code_builder,
    )
    next(button for button in at.button if button.label == "生成图案").click()
    at = at.run(timeout=30)

    assert len(at.exception) == 0
    assert calls == {
        "bind": 1, "map": 1, "vectorized": 1,
        "disk_load": 0, "disk_save": 0,
    }
    assert len(identities) == 1
    assert len(local_engines) == 1
    assert local_engines[0]._use_disk_grid_cache is False
    assert hashlib.sha256(old_cache.read_bytes()).hexdigest() == old_cache_hash
    identity = identities[0]
    assert (
        identity.material, identity.substrate, identity.polarization,
        identity.angle_deg, identity.far_field_enabled, identity.na,
        identity.theta_obs_deg,
    ) == (
        "TiO2 (anatase)", "SiO2 (fused silica)", "TE (s-pol)",
        0.0, False, 0.1, 0.0,
    )
    assert at.session_state[ENGINE_SESSION_KEY] is main_engine
    require_engine_state_unchanged(main_engine, main_before)
    assert at.session_state.filtered_state.get(ENGINE_LIBRARY_KEY) == library_before
    assert at.session_state["structure_type"] == structure_before == "single"
    assert PATTERN_SESSION_KEY in at.session_state.filtered_state
    forward_after = json.loads(_latest_download_data(
        _offline_small_app, "下载当前结果 JSON"))
    assert forward_after["provenance"] == forward_before["provenance"]
    assert forward_after["rgb"] == forward_before["rgb"]
    assert forward_after["reflectance"] == forward_before["reflectance"]

    downloads = _latest_download_names(_offline_small_app)
    assert {
        "下载 mapped PNG", "下载像素 CSV", "下载 metadata JSON",
    }.issubset(downloads)

    at = at.run(timeout=30)
    assert len(at.exception) == 0
    assert calls == {
        "bind": 1, "map": 1, "vectorized": 1,
        "disk_load": 0, "disk_save": 0,
    }
    assert hashlib.sha256(old_cache.read_bytes()).hexdigest() == old_cache_hash
    require_engine_state_unchanged(main_engine, main_before)


def test_alternating_sessions_keep_provenance_spectra_rgb_and_grids_private(
        monkeypatch, _offline_small_app):
    import json

    import engine as engine_module
    from color_utils import rgb_to_lab, rgb_to_xy
    from streamlit.testing.v1 import AppTest
    from ui_engine_session import (
        ENGINE_LIBRARY_KEY, ENGINE_SESSION_KEY, LibraryIdentity,
        bind_engine_library,
    )

    def grid_for(self):
        is_a = self._last_material == "Si3N4 (nitride)"
        rgb = np.array([[0.15, 0.35, 0.55] if is_a else [0.65, 0.45, 0.25]])
        params = np.array([[165.0, 315.0, 415.0] if is_a else [185.0, 295.0, 395.0]])
        return params, rgb, rgb_to_lab(rgb), rgb_to_xy(rgb)

    def rebuild(self, material, substrate, polarization, angle_deg):
        self._last_material = material
        self._last_substrate = substrate
        self._last_polarization = polarization
        self._last_angle = float(angle_deg)
        self.grid_params, self.grid_rgb, self.grid_lab, self.grid_xy = grid_for(self)

    def spectrum(self, param, wl_start=380.0, wl_end=780.0, n_pts=81):
        wavelengths = np.linspace(wl_start, wl_end, n_pts)
        base = 0.18 if param.material == "Si3N4 (nitride)" else 0.58
        reflectance = base + 0.04 * np.sin((wavelengths - 380.0) / 60.0)
        return wavelengths, reflectance

    monkeypatch.setattr(engine_module.MetaSurfaceColorEngine, "_build_library", grid_for)
    monkeypatch.setattr(engine_module.MetaSurfaceColorEngine, "rebuild_library", rebuild)
    monkeypatch.setattr(engine_module.MetaSurfaceColorEngine, "compute_spectrum", spectrum)

    app_a = AppTest.from_file("app.py").run(timeout=30)
    app_b = AppTest.from_file("app.py").run(timeout=30)
    for app in (app_a, app_b):
        next(
            item for item in app.checkbox
            if item.label == "启用 ML 代理模型（快速候选预测）"
        ).set_value(False)
    app_a = app_a.run(timeout=30)
    app_b = app_b.run(timeout=30)

    next(item for item in app_a.selectbox if item.label == "柱材料").set_value(
        "Si3N4 (nitride)")
    next(item for item in app_a.selectbox if item.label == "衬底材料").set_value(
        "Al2O3 (sapphire)")
    next(item for item in app_a.selectbox if item.label == "偏振").set_value(
        "TM (p-pol)")
    next(item for item in app_a.slider if item.label == "入射角 (°)").set_value(15.0)
    app_a = app_a.run(timeout=30)

    identity_a = LibraryIdentity(
        "Si3N4 (nitride)", "Al2O3 (sapphire)", "TM (p-pol)", 15.0)
    identity_b = LibraryIdentity(
        "TiO2 (anatase)", "SiO2 (fused silica)", "TE (s-pol)", 0.0)

    def capture(app, identity):
        engine = app.session_state[ENGINE_SESSION_KEY]
        bind_engine_library(engine, identity, app.session_state)
        payload = json.loads(_latest_download_data(
            _offline_small_app, "下载当前结果 JSON"))
        provenance = payload["provenance"]
        return {
            "provenance": {
                key: provenance[key]
                for key in ("material", "substrate", "polarization", "angle_deg")
            },
            "reflectance": tuple(payload["reflectance"]),
            "rgb": tuple(payload["rgb"]),
            "identity": app.session_state[ENGINE_LIBRARY_KEY],
            "grid_params": engine.grid_params.copy(),
            "grid_rgb": engine.grid_rgb.copy(),
            "grid_lab": engine.grid_lab.copy(),
            "grid_xy": engine.grid_xy.copy(),
        }

    a_first = capture(app_a, identity_a)
    app_b = app_b.run(timeout=30)
    b_first = capture(app_b, identity_b)
    app_a = app_a.run(timeout=30)
    a_second = capture(app_a, identity_a)
    app_b = app_b.run(timeout=30)
    b_second = capture(app_b, identity_b)

    assert a_first["identity"] == a_second["identity"] == identity_a
    assert b_first["identity"] == b_second["identity"] == identity_b
    assert a_first["provenance"] == a_second["provenance"] == {
        "material": "Si3N4 (nitride)",
        "substrate": "Al2O3 (sapphire)",
        "polarization": "TM (p-pol)",
        "angle_deg": 15.0,
    }
    assert b_first["provenance"] == b_second["provenance"] == {
        "material": "TiO2 (anatase)",
        "substrate": "SiO2 (fused silica)",
        "polarization": "TE (s-pol)",
        "angle_deg": 0.0,
    }
    assert a_first["reflectance"] == a_second["reflectance"]
    assert b_first["reflectance"] == b_second["reflectance"]
    assert a_first["reflectance"] != b_first["reflectance"]
    assert a_first["rgb"] == a_second["rgb"]
    assert b_first["rgb"] == b_second["rgb"]
    assert a_first["rgb"] != b_first["rgb"]
    for field in ("grid_params", "grid_rgb", "grid_lab", "grid_xy"):
        np.testing.assert_array_equal(a_first[field], a_second[field])
        np.testing.assert_array_equal(b_first[field], b_second[field])
        assert not np.shares_memory(a_first[field], b_first[field])
        assert not np.array_equal(a_first[field], b_first[field])


def test_default_available_result_card_matches_export_availability(
        _offline_small_app, monkeypatch):
    import csv
    import io
    import json
    from pathlib import Path
    from matplotlib.axes import Axes

    legend_labels = []
    original_plot = Axes.plot

    def plot_spy(self, *args, **kwargs):
        if kwargs.get("label"):
            legend_labels.append(str(kwargs["label"]))
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", plot_spy)
    at = _run_app()
    text = _all_text(at)
    labels = [element.label for element in at.get("download_button")]
    assert len(at.exception) == 0
    assert _unexpected_warnings(at) == []
    assert "可用；颜色由该光谱计算" in text
    assert "不可用；未用其它模型冒充" not in text
    assert "下载光谱 CSV" in labels
    assert "下载当前结果 JSON" in labels
    downloads = _latest_download_names(_offline_small_app)
    assert downloads["下载光谱 CSV"] == "spectrum_single_D-180_H-300_P-400.csv"
    assert downloads["下载当前结果 JSON"] == "forward_single_D-180_H-300_P-400.json"
    source = Path("app.py").read_text(encoding="utf-8")
    assert 'class="workflow-hint"' in source
    assert "使用顺序：</strong>输入目标颜色 → 调整结构 → 查看结果 → 导出记录。" in source
    assert any(
        "下一步：查看“光谱”页核对输出" in element.value
        for element in at.caption
    )
    assert any(expander.label == "查看来源、边界与证据详情" for expander in at.expander)
    assert any(button.label == "加载完整审计详情" for button in at.button)
    assert "loaded_and_called" not in text
    assert "sha256=" not in text
    assert "论文2池 manifest" not in text
    assert [tab.label for tab in at.tabs] == ["预览", "逆设计", "图案", "映射", "光谱"]
    assert at.radio[0].options == ["单柱", "双柱", "FP 腔（Fabry-Pérot）"]
    assert "结构：单柱；几何：D=180 nm, H=300 nm, P=400 nm" in text
    assert "D=180 nm, H=300 nm, P=400 nm" in legend_labels
    payload = json.loads(_latest_download_data(
        _offline_small_app, "下载当前结果 JSON"))
    provenance = payload["provenance"]
    assert provenance["structure_type"] == "single"
    assert provenance["structure_label"] == "单柱"
    assert provenance["geometry"] == {"D_nm": 180.0, "H_nm": 300.0, "P_nm": 400.0}
    csv_rows = list(csv.DictReader(io.StringIO(_latest_download_data(
        _offline_small_app, "下载光谱 CSV"))))
    assert json.loads(csv_rows[0]["provenance_json"]) == provenance


def test_tio2_gamut_notice_is_scoped_to_audited_single_pillar_range():
    at = _run_app()
    info_text = "\n".join(item.value for item in at.info)
    assert "既有审计覆盖的 TiO2 单柱范围" in info_text
    assert "当前参数范围" not in info_text

    at.radio[0].set_value("双柱")
    at = at.run(timeout=30)
    dual_info_text = "\n".join(item.value for item in at.info)
    assert "既有审计覆盖的 TiO2 单柱范围" not in dual_info_text


def test_proxy_mapping_hint_is_limited_to_registered_analytical_pair():
    from pathlib import Path
    at = _run_app()
    source = Path("app.py").read_text(encoding="utf-8")
    assert "当前预览使用 ML；D-H 映射使用已注册的 TiO₂/SiO₂ 解析路线" in source
    assert "不代表 ML/RCWA 代理输出" in source

    substrate = next(item for item in at.selectbox if item.label == "衬底材料")
    substrate.set_value("Si3N4 (nitride)")
    at = at.run(timeout=30)
    assert "如需查看有代码证据的解析 D-H 映射，可在侧栏关闭 ML" not in _all_text(at)


def test_far_field_controls_are_disclosed_only_when_enabled():
    at = _run_app()
    assert "远场未启用；启用后显示观察角度与 NA 控件。" in [item.value for item in at.caption]
    assert "观察角度 θ (°)" not in [item.label for item in at.slider]
    assert "收集数值孔径 NA" not in [item.label for item in at.slider]

    far_field = next(
        item for item in at.checkbox
        if item.label == "启用角谱远场传播 (Angular Spectrum)"
    )
    far_field.set_value(True)
    at = at.run(timeout=30)
    assert "观察角度 θ (°)" in [item.label for item in at.slider]
    assert "收集数值孔径 NA" in [item.label for item in at.slider]


def test_material_substrate_polarization_angle_and_dual_switch(
        _offline_small_app, monkeypatch):
    import csv
    import io
    import json
    from matplotlib.axes import Axes

    legend_labels = []
    original_plot = Axes.plot

    def plot_spy(self, *args, **kwargs):
        if kwargs.get("label"):
            legend_labels.append(str(kwargs["label"]))
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", plot_spy)
    at = _run_app()
    at.selectbox[0].set_value("Si3N4 (nitride)")
    at.selectbox[1].set_value("Al2O3 (sapphire)")
    at.selectbox[2].set_value("TM (p-pol)")
    at.slider[0].set_value(15.0)
    at.radio[0].set_value("双柱")
    at = at.run(timeout=30)
    text = _all_text(at)
    assert len(at.exception) == 0
    assert _unexpected_warnings(at) == []
    assert len(at.tabs) == 5
    assert at.radio[0].value == "双柱"
    for value in ("Si3N4 (nitride)", "Al2O3 (sapphire)", "TM (p-pol)", "15.0"):
        assert value in text
    dual_summary = "D1=120 nm, H1=250 nm, D2=200 nm, H2=350 nm, P=400 nm"
    assert f"结构：双柱；几何：{dual_summary}" in text
    assert dual_summary in legend_labels
    payload = json.loads(_latest_download_data(
        _offline_small_app, "下载当前结果 JSON"))
    provenance = payload["provenance"]
    assert provenance["structure_type"] == "dual"
    assert provenance["geometry"] == {
        "D1_nm": 120.0, "H1_nm": 250.0,
        "D2_nm": 200.0, "H2_nm": 350.0, "P_nm": 400.0,
    }
    csv_rows = list(csv.DictReader(io.StringIO(_latest_download_data(
        _offline_small_app, "下载光谱 CSV"))))
    assert json.loads(csv_rows[0]["provenance_json"]) == provenance


@pytest.mark.parametrize("mirror", ["介质 DBR (TiO2/SiO2)", "金属 Ag (减色)"])
def test_fp_mirror_routes_render_tmm_and_exports(
        mirror, _offline_small_app, monkeypatch):
    import csv
    import io
    import json
    from matplotlib.axes import Axes

    legend_labels = []
    original_plot = Axes.plot

    def plot_spy(self, *args, **kwargs):
        if kwargs.get("label"):
            legend_labels.append(str(kwargs["label"]))
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", plot_spy)
    at = _run_app()
    at.radio[0].set_value("FP 腔（Fabry-Pérot）")
    at = at.run(timeout=30)
    next(item for item in at.selectbox if item.label == "反射镜类型").set_value(mirror)
    at = at.run(timeout=30)
    text = _all_text(at)
    assert len(at.exception) == 0
    assert _unexpected_warnings(at) == []
    assert "FP cavity TMM" in text
    labels = [element.label for element in at.get("download_button")]
    assert "下载光谱 CSV" in labels
    assert "下载当前结果 JSON" in labels
    downloads = _latest_download_names(_offline_small_app)
    expected_base = "fp_dbr_T-200_C-450" if mirror.startswith("介质") else "fp_ag_T-200"
    assert downloads["下载光谱 CSV"] == f"spectrum_{expected_base}.csv"
    assert downloads["下载当前结果 JSON"] == f"forward_{expected_base}.json"
    assert "TiO2 (cavity layer)" in text
    assert ("SiO2 (DBR mirror stack)" in text
            if mirror.startswith("介质") else "Ag (metal mirrors)" in text)
    payload = json.loads(_latest_download_data(
        _offline_small_app, "下载当前结果 JSON"))
    provenance = payload["provenance"]
    assert provenance["structure_type"] == "fp"
    assert provenance["geometry"]["T_nm"] == 200.0
    assert "D_nm" not in provenance["geometry"]
    assert "H_nm" not in provenance["geometry"]
    assert "P_nm" not in provenance["geometry"]
    if mirror.startswith("介质"):
        expected_summary = "T=200 nm, center=450 nm, DBR pairs=3/5"
        assert provenance["structure_label"] == "FP-DBR 腔"
        assert provenance["mirror_type"] == mirror
        assert provenance["geometry"]["center_wavelength_nm"] == 450.0
        assert provenance["stack_identity"] == "(TiO2/SiO2)^3 / TiO2(T) / (SiO2/TiO2)^5"
    else:
        expected_summary = "T=200 nm, top Ag=30 nm, bottom Ag=bulk"
        assert provenance["structure_label"] == "FP-Ag 腔"
        assert provenance["mirror_type"] == mirror
        assert provenance["geometry"]["top_ag_nm"] == 30.0
        assert provenance["stack_identity"] == "Ag(30 nm) / TiO2(T) / Ag(bulk)"
    assert f"几何：{expected_summary}" in text
    assert expected_summary in legend_labels
    assert not any(label.startswith("D/H/P=") for label in legend_labels)
    csv_rows = list(csv.DictReader(io.StringIO(_latest_download_data(
        _offline_small_app, "下载光谱 CSV"))))
    assert json.loads(csv_rows[0]["provenance_json"]) == provenance


def test_single_diameter_greater_than_period_fails_closed():
    at = _run_app()
    next(item for item in at.slider if item.label == "直径 D (nm)").set_value(350.0)
    next(item for item in at.slider if item.label == "周期 P (nm)").set_value(200.0)
    at = at.run(timeout=30)
    text = _markdown_text(at)
    assert len(at.exception) == 0
    warnings = [element.value for element in at.warning]
    errors = [element.value for element in at.error]
    assert any("D > P" in value for value in warnings)
    assert any("当前 CIE 点不可用，未绘制占位点" in value for value in warnings)
    assert any("当前单柱参数无效" in value for value in errors)
    labels = [element.label for element in at.get("download_button")]
    assert "下载光谱 CSV" not in labels
    assert "下载当前结果 JSON" not in labels
    assert "下载色板 PNG" not in labels


def test_far_field_disables_ml_route_and_records_observation_context():
    at = _run_app()
    next(
        item for item in at.checkbox
        if item.label == "启用角谱远场传播 (Angular Spectrum)"
    ).set_value(True)
    at = at.run(timeout=30)
    next(item for item in at.slider if item.label == "观察角度 θ (°)").set_value(20.0)
    next(item for item in at.slider if item.label == "收集数值孔径 NA").set_value(0.5)
    at = at.run(timeout=30)
    text = _markdown_text(at)
    assert len(at.exception) == 0
    assert _unexpected_warnings(at) == []
    assert "Far-field post-processing" in text
    assert "0.5" in text
    assert "20.0" in text
    assert "Lorentz/Fano + CCM analytical response" in text


def test_far_field_hides_ml_control_and_restores_saved_preference():
    at = _run_app()
    ml_control = next(
        item for item in at.checkbox
        if item.label == "启用 ML 代理模型（快速候选预测）"
    )
    assert ml_control.value is True

    next(
        item for item in at.checkbox
        if item.label == "启用角谱远场传播 (Angular Spectrum)"
    ).set_value(True)
    at = at.run(timeout=30)
    text = _all_text(at)
    assert "启用 ML 代理模型（快速候选预测）" not in [item.label for item in at.checkbox]
    assert "本路线未使用 ML：远场路线使用解析局部响应 + 角谱/NA 后处理" in text
    assert "Far-field post-processing" in text

    next(
        item for item in at.checkbox
        if item.label == "启用角谱远场传播 (Angular Spectrum)"
    ).set_value(False)
    at = at.run(timeout=30)
    restored = next(
        item for item in at.checkbox
        if item.label == "启用 ML 代理模型（快速候选预测）"
    )
    assert restored.value is True
    assert "ML surrogate" in _markdown_text(at)


def test_dual_route_hides_single_ml_control_and_restores_it_on_return(_offline_small_app):
    at = _run_app()
    assert next(
        item for item in at.checkbox
        if item.label == "启用 ML 代理模型（快速候选预测）"
    ).value is True

    at.radio[0].set_value("双柱")
    at = at.run(timeout=30)
    text = _all_text(at)
    assert "启用 ML 代理模型（快速候选预测）" not in [item.label for item in at.checkbox]
    assert "本路线未使用 ML：双柱 ONNX 模型不可用" in text
    assert "Lorentz/Fano" in text
    assert "柱1直径 D1 (nm)" in [item.label for item in at.slider]
    downloads = _latest_download_names(_offline_small_app)
    dual_base = "dual_D1-120_H1-250_D2-200_H2-350_P-400"
    assert downloads["下载光谱 CSV"] == f"spectrum_{dual_base}.csv"
    assert downloads["下载当前结果 JSON"] == f"forward_{dual_base}.json"

    at.radio[0].set_value("单柱")
    at = at.run(timeout=30)
    restored = next(
        item for item in at.checkbox
        if item.label == "启用 ML 代理模型（快速候选预测）"
    )
    assert restored.value is True
    assert "直径 D (nm)" in [item.label for item in at.slider]


def test_fp_hides_irrelevant_global_controls_and_restores_preferences():
    at = _run_app()
    initial_material = at.selectbox[0].value
    initial_substrate = at.selectbox[1].value

    at.radio[0].set_value("FP 腔（Fabry-Pérot）")
    at = at.run(timeout=30)
    text = _all_text(at)
    labels = [item.label for item in at.selectbox]
    checkbox_labels = [item.label for item in at.checkbox]
    assert "柱材料" not in labels
    assert "衬底材料" not in labels
    assert "启用角谱远场传播 (Angular Spectrum)" not in checkbox_labels
    assert "启用 ML 代理模型（快速候选预测）" not in checkbox_labels
    assert "FP-TMM 使用腔体内固定材料" in text
    assert "本路线未使用 ML：FP-TMM 直接计算薄膜腔光谱" in text
    assert "FP cavity TMM" in text

    at.radio[0].set_value("单柱")
    at = at.run(timeout=30)
    restored = {item.label: item.value for item in at.selectbox}
    assert restored["柱材料"] == initial_material
    assert restored["衬底材料"] == initial_substrate
    assert next(
        item for item in at.checkbox
        if item.label == "启用 ML 代理模型（快速候选预测）"
    ).value is True


def test_inverse_workflow_disables_unavailable_methods_before_execution():
    at = _run_app()
    text = _all_text(at)
    primary = next(button for button in at.button if button.label.startswith("开始搜索 ·"))

    assert primary.disabled is False
    assert not any(button.label == "双柱梯度" for button in at.button)
    assert not any(button.label == "FP 腔搜索" for button in at.button)
    smart_secondary = [button for button in at.button if button.label == "智能网格"]
    if primary.label == "开始搜索 · 智能网格":
        assert smart_secondary == []
        assert "主路线：快速网格搜索 · 可用" in text
    else:
        assert primary.label == "开始搜索 · 单柱梯度"
        assert len(smart_secondary) == 1
        assert smart_secondary[0].disabled is True
        assert "缺少智能网格所需 .pt 权重" in text
    assert "跨结构比较（不作为推荐主方法）" in [item.label for item in at.expander]
    assert "#80C8FF" in text
    assert "RGB(128, 200, 255)" in text
    assert not any(button.label == "应用此候选" for button in at.button)


def test_dual_inverse_entry_never_falls_back_to_single_or_fp():
    from pathlib import Path
    at = _run_app()
    at.radio[0].set_value("双柱")
    at = at.run(timeout=30)
    primary = next(button for button in at.button if button.label.startswith("开始搜索 ·"))
    assert primary.label == "开始搜索 · 双柱解析"
    assert primary.disabled is False
    assert not any("单柱梯度" in button.label for button in at.button)
    assert not any("智能网格" in button.label for button in at.button)
    assert not any("FP 腔搜索" in button.label for button in at.button)
    assert "dual_physical" in Path("app.py").read_text(encoding="utf-8")
    assert "models/dual_mlp_v3_multi.onnx" in _all_text(at)
    assert "dual_mlp_v3_multi.pt" not in _all_text(at)


@pytest.mark.parametrize(
    ("mirror", "disabled"),
    [("介质 DBR (TiO2/SiO2)", False), ("金属 Ag (减色)", True)],
)
def test_fp_inverse_has_one_target_and_only_fp_primary_entry(mirror, disabled):
    at = _run_app()
    at.radio[0].set_value("FP 腔（Fabry-Pérot）")
    at = at.run(timeout=30)
    next(item for item in at.selectbox if item.label == "反射镜类型").set_value(mirror)
    at = at.run(timeout=30)
    primary = next(button for button in at.button if button.label.startswith("开始搜索 ·"))
    assert primary.label == "开始搜索 · FP 腔搜索"
    assert primary.disabled is disabled
    assert [item.label for item in at.get("color_picker")] == ["目标颜色"]
    assert not any("单柱梯度" in button.label for button in at.button)
    assert not any("双柱梯度" in button.label for button in at.button)
    assert not any("智能网格" in button.label for button in at.button)
    assert not any(button.label == "FP腔搜索" for button in at.button)
    if disabled:
        assert "Ag 反射镜尚无专属逆设计算法" in _all_text(at)


def test_primary_inverse_search_renders_available_candidate(monkeypatch):
    import ml_module

    monkeypatch.setattr(
        ml_module,
        "_inverse_design_ml_serial",
        lambda *args, **kwargs: (
            "Fano", 190.0, 310.0, 410.0,
            np.array([0.25, 0.5, 0.75]), 0.1,
        ),
    )
    at = _run_app()
    # Smart grid is the current primary route when its exact RCWA proxy is
    # loaded; the legacy gradient route remains visible as a secondary button.
    primary = next(
        button for button in at.button
        if button.label in {"开始搜索 · 单柱梯度", "单柱梯度"}
    )
    primary.click()
    at = at.run(timeout=30)
    text = _markdown_text(at)

    assert len(at.exception) == 0
    assert "单柱梯度候选搜索完成" in "\n".join(item.value for item in at.success)
    assert "Lorentz/Fano analytical fallback" in text
    apply_button = next(button for button in at.button if button.label == "应用此候选")
    apply_button.click()
    at = at.run(timeout=30)
    sliders = {item.label: item.value for item in at.slider}
    assert sliders["直径 D (nm)"] == 190.0
    assert sliders["高度 H (nm)"] == 310.0
    assert sliders["周期 P (nm)"] == 410.0
    assert "柱1直径 D1 (nm)" not in sliders


def _force_current_smart_registry(monkeypatch, *, exact_pair=True):
    import glob
    from pathlib import Path
    import ml_module
    from types import SimpleNamespace

    real_glob = glob.glob
    real_stat = Path.stat
    real_read_bytes = Path.read_bytes
    project_root = Path(__file__).resolve().parents[1]
    onnx_path = project_root / "models" / "fixture-smart.onnx"
    pt_path = project_root / "models" / "fixture-smart.pt"
    onnx_bytes = b"fixture-smart-onnx"

    class FakeInput:
        name = "input"

    class FakeSession:
        def __init__(self, model_path, providers=None, sess_options=None):
            del providers
            self._model_path = str(model_path)

        def get_inputs(self):
            return [FakeInput()]

        def run(self, *_args, **_kwargs):
            return [np.full((1, 81), 0.3, dtype=np.float32)]

    def fixture_glob(pattern):
        normalized = str(pattern).replace("\\", "/")
        if "forward_mlp_rcwa" in normalized and normalized.endswith(".onnx"):
            return [str(onnx_path)]
        if "forward_mlp_rcwa" in normalized and normalized.endswith(".pt"):
            return [str(pt_path)]
        return real_glob(pattern)

    def stat(path, *args, **kwargs):
        if str(path).replace("\\", "/").endswith("models/fixture-smart.onnx"):
            return SimpleNamespace(st_size=len(onnx_bytes))
        return real_stat(path, *args, **kwargs)

    def read_bytes(path):
        if str(path).replace("\\", "/").endswith("models/fixture-smart.onnx"):
            return onnx_bytes
        return real_read_bytes(path)

    monkeypatch.setattr(
        glob, "glob", fixture_glob,
    )
    monkeypatch.setattr(Path, "stat", stat)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    import onnxruntime as ort
    monkeypatch.setattr(ort, "InferenceSession", FakeSession)
    if exact_pair:
        monkeypatch.setattr(
            ml_module, "_RCWA_SUBSTRATE_MODELS",
            {("TiO2 (anatase)", "SiO2 (fused silica)"): [
                "forward_mlp_rcwa_TiO2_s?.onnx"]})


def test_smart_inverse_export_uses_all_canonical_candidates(
        monkeypatch, _offline_small_app):
    import json
    import ml_module
    import streamlit as st
    from color_utils import rgb_255, rgb_to_hex
    from engine import MetaSurfaceParam

    _force_current_smart_registry(monkeypatch)
    candidates = [
        (None, MetaSurfaceParam(180.0, 300.0, 400.0),
         np.array([0.2, 0.4, 0.6]), 3.0, 2.0),
        (None, MetaSurfaceParam(200.0, 320.0, 420.0),
         np.array([0.3, 0.5, 0.7]), 4.0, 2.8),
    ]
    monkeypatch.setattr(
        ml_module, "smart_grid_search", lambda *args, **kwargs: candidates)
    st.cache_resource.clear()

    at = _run_app()
    primary = next(
        button for button in at.button if button.label.startswith("开始搜索 ·"))
    assert primary.label == "开始搜索 · 智能网格"
    primary.click()
    at = at.run(timeout=30)

    labels = [element.label for element in at.get("download_button")]
    assert "💾 导出逆设计 CSV" in labels
    assert "💾 导出逆设计 JSON" in labels
    payload = json.loads(_latest_download_data(
        _offline_small_app, "💾 导出逆设计 JSON"))
    assert payload["method"] == {"id": "smart", "label": "智能网格"}
    assert len(payload["candidates"]) == len(candidates)
    assert all(item["method_id"] == "smart" for item in payload["candidates"])
    assert all(item["route_id"] == "rcwa_surrogate" for item in payload["candidates"])
    for record in payload["candidates"]:
        assert tuple(record["predicted_rgb255"]) == rgb_255(record["predicted_rgb"])
        assert record["predicted_hex"].lower() == rgb_to_hex(record["predicted_rgb"])

    # A download or appearance change reruns Streamlit. Cards and apply actions
    # must survive without rerunning the search, including the second candidate.
    monkeypatch.setattr(ml_module, "smart_grid_search", lambda *_args, **_kwargs: pytest.fail("unexpected repeated search"))
    at = at.run(timeout=30)
    assert "智能网格完成" in "\n".join(item.value for item in at.success)
    saved_payload = json.loads(_latest_download_data(
        _offline_small_app, "💾 导出逆设计 JSON"))
    assert saved_payload["candidates"] == payload["candidates"]
    apply_buttons = [button for button in at.button if button.label == "应用此候选"]
    assert len(apply_buttons) == 2
    apply_buttons[1].click()
    at = at.run(timeout=30)
    sliders = {item.label: item.value for item in at.slider}
    assert sliders["直径 D (nm)"] == 200.0
    assert sliders["高度 H (nm)"] == 320.0
    assert sliders["周期 P (nm)"] == 420.0


@pytest.mark.parametrize(
    ("control_label", "value"),
    [("偏振", "TM (p-pol)"), ("入射角 (°)", 0.1)],
)
def test_smart_inverse_is_disabled_outside_registered_te_normal_domain(
        monkeypatch, control_label, value):
    import streamlit as st

    _force_current_smart_registry(monkeypatch)
    st.cache_resource.clear()
    at = _run_app()
    primary = next(
        button for button in at.button if button.label.startswith("开始搜索 ·"))
    assert primary.label == "开始搜索 · 智能网格"
    assert primary.disabled is False

    if control_label == "偏振":
        next(item for item in at.selectbox if item.label == control_label).set_value(value)
    else:
        next(item for item in at.slider if item.label == control_label).set_value(value)
    at = at.run(timeout=30)
    primary = next(
        button for button in at.button if button.label.startswith("开始搜索 ·"))
    assert primary.label == "开始搜索 · 智能网格"
    assert primary.disabled is True
    assert "缺少 smart 代理对 TM 或非 0° 入射的版本化训练域证据" in _all_text(at)


def test_material_only_smart_session_cannot_cover_another_substrate(monkeypatch):
    import streamlit as st

    _force_current_smart_registry(monkeypatch, exact_pair=False)
    st.cache_resource.clear()
    at = _run_app()
    next(
        item for item in at.selectbox if item.label == "衬底材料"
    ).set_value("Al2O3 (sapphire)")
    at = at.run(timeout=30)

    assert not any(button.label == "智能网格" for button in at.button)
    assert "暂不可用" in _all_text(at)
    assert "未加载匹配的 RCWA 代理" in _all_text(at)


def test_successful_inverse_search_does_not_render_removed_fake_history(monkeypatch):
    import ml_module

    monkeypatch.setattr(
        ml_module, "_inverse_design_ml_serial",
        lambda *args, **kwargs: (
            "Fano", 190.0, 310.0, 410.0,
            np.array([0.25, 0.5, 0.75]), 0.1,
        ),
    )
    at = _run_app()
    next(button for button in at.button if "单柱梯度" in button.label).click()
    at = at.run(timeout=30)
    assert "搜索历史" not in _all_text(at)
    assert not any(button.label in {"查看", "关闭回看"} for button in at.button)


def test_compare_inverse_export_uses_cross_structure_run_candidates(
        monkeypatch, _offline_small_app):
    import engine
    import fp_cavity
    import json
    from engine import MetaSurfaceParam
    from ui_engine_session import (
        ENGINE_LIBRARY_KEY, ENGINE_SESSION_KEY, LibraryIdentity,
        bind_engine_library,
    )

    def rebuild_library_stub(self, material, substrate, polarization, angle_deg):
        self._last_material = material
        self._last_substrate = substrate
        self._last_polarization = polarization
        self._last_angle = float(angle_deg)

    monkeypatch.setattr(
        engine.MetaSurfaceColorEngine, "rebuild_library", rebuild_library_stub)
    monkeypatch.setattr(
        engine.MetaSurfaceColorEngine, "inverse_design",
        lambda *args, **kwargs: [
            (None, MetaSurfaceParam(180.0, 300.0, 400.0),
             np.array([0.2, 0.4, 0.6]), 3.0, 2.0),
        ],
    )

    def fp_stub(*args, **kwargs):
        wavelengths = np.linspace(380.0, 780.0, 81)
        return wavelengths, np.full_like(wavelengths, 0.35)

    monkeypatch.setattr(fp_cavity, "fp_dielectric_spectrum", fp_stub)
    at = _run_app()
    main_engine = at.session_state[ENGINE_SESSION_KEY]
    main_identity = LibraryIdentity(
        "TiO2 (anatase)", "SiO2 (fused silica)", "TE (s-pol)", 0.0)
    bind_engine_library(main_engine, main_identity, at.session_state)
    main_grids = tuple(
        getattr(main_engine, name).copy()
        for name in ("grid_params", "grid_rgb", "grid_lab", "grid_xy")
    )
    next(
        button for button in at.button
        if button.label == "跨结构方案对比"
    ).click()
    at = at.run(timeout=30)

    payload = json.loads(_latest_download_data(
        _offline_small_app, "💾 导出逆设计 JSON"))
    candidates = payload["candidates"]
    assert payload["method"] == {"id": "compare", "label": "跨结构方案对比"}
    assert len(candidates) == 3
    assert [item["structure_type"] for item in candidates].count("single") == 2
    assert [item["structure_type"] for item in candidates].count("fp") == 1
    assert {item["route_id"] for item in candidates} == {
        "lorentz_fano_fallback", "fp_tmm"}
    fp_candidate = next(
        item for item in candidates if item["structure_type"] == "fp")
    assert fp_candidate["candidate_context"] == {
        "structure_type": "fp",
        "material": "TiO2 (cavity layer)",
        "substrate": "SiO2 (DBR mirror stack)",
        "polarization": "TE (s-pol)",
        "angle_deg": 0.0,
        "mirror_type": "介质 DBR (TiO2/SiO2)",
        "n_pairs": 3,
        "algorithm_version": "fp-dbr-grid-v1",
    }
    assert at.session_state[ENGINE_SESSION_KEY] is main_engine
    assert at.session_state[ENGINE_LIBRARY_KEY] == main_identity
    assert main_engine._ui_library_identity == main_identity
    for name, before in zip(
            ("grid_params", "grid_rgb", "grid_lab", "grid_xy"), main_grids):
        np.testing.assert_array_equal(getattr(main_engine, name), before)


def test_fp_first_search_exports_and_second_identical_search_hits_cache(
        monkeypatch, _offline_small_app):
    import fp_cavity
    import json

    calls = []

    def fp_stub(t_nm, center_nm, *args, **kwargs):
        calls.append((float(t_nm), float(center_nm)))
        wavelengths = np.linspace(380.0, 780.0, 81)
        reflectance = np.full_like(
            wavelengths,
            0.2 + 0.2 * ((float(t_nm) + float(center_nm)) % 100.0) / 100.0,
        )
        return wavelengths, reflectance

    monkeypatch.setattr(fp_cavity, "fp_dielectric_spectrum", fp_stub)
    at = _run_app()
    at.radio[0].set_value("FP 腔（Fabry-Pérot）")
    at = at.run(timeout=30)
    before_first = len(calls)
    next(
        button for button in at.button
        if button.label == "开始搜索 · FP 腔搜索"
    ).click()
    at = at.run(timeout=30)
    first_search_calls = len(calls) - before_first

    labels = [element.label for element in at.get("download_button")]
    assert "💾 导出逆设计 CSV" in labels
    assert "💾 导出逆设计 JSON" in labels
    payload = json.loads(_latest_download_data(
        _offline_small_app, "💾 导出逆设计 JSON"))
    assert payload["method"] == {"id": "fp", "label": "FP 腔搜索"}
    assert len(payload["candidates"]) == 3
    assert first_search_calls > 100

    before_second = len(calls)
    next(
        button for button in at.button
        if button.label == "开始搜索 · FP 腔搜索"
    ).click()
    at = at.run(timeout=30)
    second_search_calls = len(calls) - before_second
    assert second_search_calls < first_search_calls / 10
    assert any("命中缓存，跳过重复计算" in item.value for item in at.success)

    next(
        item for item in at.get("color_picker")
        if item.label == "目标颜色"
    ).set_value("#FF0000")
    at = at.run(timeout=30)
    before_changed_target = len(calls)
    next(
        button for button in at.button
        if button.label == "开始搜索 · FP 腔搜索"
    ).click()
    at = at.run(timeout=30)
    assert len(calls) - before_changed_target > 100
    assert not any("命中缓存，跳过重复计算" in item.value for item in at.success)

    next(
        item for item in at.get("color_picker")
        if item.label == "目标颜色"
    ).set_value("#80C8FF")
    at = at.run(timeout=30)
    before_returned_target = len(calls)
    next(
        button for button in at.button
        if button.label == "开始搜索 · FP 腔搜索"
    ).click()
    at = at.run(timeout=30)
    assert len(calls) - before_returned_target < first_search_calls / 10
    assert any("命中缓存，跳过重复计算" in item.value for item in at.success)


def test_target_change_invalidates_old_inverse_run_and_hides_actions(monkeypatch):
    import ml_module

    monkeypatch.setattr(
        ml_module, "_inverse_design_ml_serial",
        lambda *args, **kwargs: (
            "Fano", 190.0, 310.0, 410.0,
            np.array([0.25, 0.5, 0.75]), 0.1,
        ),
    )
    at = _run_app()
    next(button for button in at.button if "单柱梯度" in button.label).click()
    at = at.run(timeout=30)
    assert any(button.label == "应用此候选" for button in at.button)

    next(item for item in at.get("color_picker") if item.label == "目标颜色").set_value("#ff0000")
    at = at.run(timeout=30)
    assert not any(button.label == "应用此候选" for button in at.button)
    assert not any("导出结果" in expander.label for expander in at.expander)
    assert any("旧候选、应用按钮与导出已失效" in item.value for item in at.info)


def test_pattern_page_default_state_explains_limits_route_and_outputs():
    at = _run_app()
    text = _markdown_text(at)
    uploader = next(item for item in at.get("file_uploader") if item.label == "选择图像")

    assert len(at.exception) == 0
    assert uploader is not None
    assert "独立单柱解析近似图案工具" in [item.value for item in at.get("subheader")]
    assert "single · TE · 0° · no-far-field · scalar analytical" in text
    assert "不继承预览 ML、偏振或角度" in text
    assert "禁止 fuzzy、材料单项、默认系数或衬底替代" in text
    assert "单文件不超过 8 MB" in text
    assert "输出最长边 20-64 像素" in text
    assert "单块理论临时工作区不超过 16 MiB" in text
    assert "Lab 三维平方欧氏距离" in text
    assert "与 ΔE76 的排序等价；这不是 ΔE00" in text
    assert "D/H/P 三张参数图" in "\n".join(item.value for item in at.caption)
    assert "约 700 MiB" not in text
    assert any("尚未选择图像" in item.value for item in at.info)
    assert not any(button.label == "生成图案" for button in at.button)


@pytest.mark.parametrize(
    "mode",
    ["dual", "fp_dbr", "fp_ag", "far_field", "nonzero_angle", "material", "substrate"],
)
def test_pattern_unavailable_matrix_hides_controls_results_and_never_binds_or_maps(
        mode, monkeypatch, _offline_small_app):
    import engine as engine_module
    import ui_engine_session as engine_session
    from ui_pattern_contracts import (
        PATTERN_SESSION_KEY, PatternSnapshot, build_pattern_payload,
        make_pattern_contract,
    )

    calls = {"bind": 0, "map": 0}

    def bomb_bind(*_args, **_kwargs):
        calls["bind"] += 1
        raise AssertionError("unavailable pattern must not bind")

    def bomb_map(*_args, **_kwargs):
        calls["map"] += 1
        raise AssertionError("unavailable pattern must not map")

    monkeypatch.setattr(engine_session, "bind_engine_library", bomb_bind)
    monkeypatch.setattr(
        engine_module.MetaSurfaceColorEngine, "image_to_metasurface_map", bomb_map)

    at = _run_app()
    contract = make_pattern_contract(
        structure_type="single", structure_identity="单柱",
        material="TiO2 (anatase)", substrate="SiO2 (fused silica)",
        angle_deg=0.0, far_field_enabled=False,
    )
    upload_hash = hashlib.sha256(b"fixture-old-pattern").hexdigest()
    original = np.array([[[0.1, 0.2, 0.3]]])
    mapped = np.array([[[0.2, 0.3, 0.4]]])
    params = np.array([[[180.0, 300.0, 400.0]]])
    old = PatternSnapshot.create(
        contract, upload_hash, 48,
        build_pattern_payload(contract, upload_hash, 48, original, mapped, params),
    )
    at.session_state[PATTERN_SESSION_KEY] = old.to_record()
    _offline_small_app.clear()

    if mode == "dual":
        at.radio[0].set_value("双柱")
        expected_structure = "dual"
    elif mode in {"fp_dbr", "fp_ag"}:
        at.radio[0].set_value("FP 腔（Fabry-Pérot）")
        expected_structure = "fp"
    elif mode == "far_field":
        next(
            item for item in at.checkbox
            if item.label == "启用角谱远场传播 (Angular Spectrum)"
        ).set_value(True)
        expected_structure = "single"
    elif mode == "nonzero_angle":
        next(item for item in at.slider if item.label == "入射角 (°)").set_value(5.0)
        expected_structure = "single"
    elif mode == "material":
        next(item for item in at.selectbox if item.label == "柱材料").set_value(
            "a-Si (amorphous)")
        expected_structure = "single"
    else:
        next(item for item in at.selectbox if item.label == "衬底材料").set_value(
            "Si3N4 (nitride)")
        expected_structure = "single"
    at = at.run(timeout=30)
    stale_was_reported = any(
        "视为 stale" in item.value for item in at.warning
    )
    if mode == "fp_ag":
        next(item for item in at.selectbox if item.label == "反射镜类型").set_value(
            "金属 Ag (减色)")
        at = at.run(timeout=30)

    assert len(at.exception) == 0
    assert calls == {"bind": 0, "map": 0}
    assert at.session_state["structure_type"] == expected_structure
    assert not any(item.label == "选择图像" for item in at.get("file_uploader"))
    assert not any(button.label == "生成图案" for button in at.button)
    warnings = "\n".join(item.value for item in at.warning)
    assert "图案工具不可用" in warnings
    assert stale_was_reported or "视为 stale" in warnings
    assert PATTERN_SESSION_KEY not in at.session_state.filtered_state
    assert not any(
        call["label"] in {"下载 mapped PNG", "下载像素 CSV", "下载 metadata JSON"}
        for call in _offline_small_app
    )


def test_pattern_upload_and_size_changes_stale_then_regenerate(monkeypatch, _offline_small_app):
    import io
    import engine as engine_module
    import streamlit as st
    from PIL import Image
    import ui_engine_session as engine_session

    def png_bytes(color):
        buffer = io.BytesIO()
        Image.new("RGB", (3, 2), color).save(buffer, format="PNG")
        return buffer.getvalue()

    current_upload = [png_bytes((40, 80, 120))]
    monkeypatch.setattr(
        st, "file_uploader",
        lambda *_args, **_kwargs: io.BytesIO(current_upload[0]),
    )
    calls = {"bind": 0, "map": 0}
    original_bind = engine_session.bind_engine_library
    original_map = engine_module.MetaSurfaceColorEngine.image_to_metasurface_map

    def counted_bind(*args, **kwargs):
        calls["bind"] += 1
        return original_bind(*args, **kwargs)

    def counted_map(*args, **kwargs):
        calls["map"] += 1
        return original_map(*args, **kwargs)

    monkeypatch.setattr(engine_session, "bind_engine_library", counted_bind)
    monkeypatch.setattr(
        engine_module.MetaSurfaceColorEngine, "image_to_metasurface_map", counted_map)

    at = _run_app()
    next(button for button in at.button if button.label == "生成图案").click()
    at = at.run(timeout=30)
    assert calls == {"bind": 1, "map": 1}

    _offline_small_app.clear()
    current_upload[0] = png_bytes((120, 80, 40))
    at = at.run(timeout=30)
    assert calls == {"bind": 1, "map": 1}
    assert any("图案快照已陈旧" in item.value for item in at.warning)
    assert not any(
        call["label"] in {"下载 mapped PNG", "下载像素 CSV", "下载 metadata JSON"}
        for call in _offline_small_app
    )

    next(button for button in at.button if button.label == "生成图案").click()
    at = at.run(timeout=30)
    assert calls == {"bind": 2, "map": 2}

    _offline_small_app.clear()
    next(item for item in at.slider if item.label == "输出最长边 (px)").set_value(32)
    at = at.run(timeout=30)
    assert calls == {"bind": 2, "map": 2}
    assert any("图案快照已陈旧" in item.value for item in at.warning)
    assert not any(
        call["label"] in {"下载 mapped PNG", "下载像素 CSV", "下载 metadata JSON"}
        for call in _offline_small_app
    )


def test_pattern_ui_exports_match_the_same_non_square_snapshot(
        monkeypatch, _offline_small_app):
    import csv
    import io
    import json
    import streamlit as st
    from PIL import Image
    from ui_pattern_contracts import (
        PATTERN_SESSION_KEY, build_pattern_exports, validate_pattern_snapshot_record,
    )

    upload = io.BytesIO()
    Image.new("RGB", (3, 2), (50, 100, 150)).save(upload, format="PNG")
    upload_bytes = upload.getvalue()
    monkeypatch.setattr(
        st, "file_uploader", lambda *_args, **_kwargs: io.BytesIO(upload_bytes))

    at = _run_app()
    next(button for button in at.button if button.label == "生成图案").click()
    at = at.run(timeout=30)
    stored = validate_pattern_snapshot_record(at.session_state[PATTERN_SESSION_KEY])
    expected = build_pattern_exports(stored)

    actual_png = _latest_download_data(_offline_small_app, "下载 mapped PNG")
    actual_csv = _latest_download_data(_offline_small_app, "下载像素 CSV")
    actual_json = _latest_download_data(_offline_small_app, "下载 metadata JSON")
    assert len(at.exception) == 0
    assert stored.payload["shape"] == [2, 3]
    assert actual_png == expected.mapped_png
    assert isinstance(actual_csv, bytes)
    assert actual_csv == expected.csv_bytes
    assert json.loads(actual_json) == json.loads(expected.metadata_json)
    decoded_csv = actual_csv.decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(decoded_csv)))
    metadata = json.loads(actual_json)
    assert len(rows) == 6
    assert decoded_csv.splitlines()[0] == "row,col,R,G,B,D,H,P"
    assert metadata["pixel_csv_sha256"] == hashlib.sha256(actual_csv).hexdigest()
    assert metadata["pixel_csv_byte_count"] == len(actual_csv)
    assert metadata["csv_row_count"] == len(rows)


@pytest.mark.parametrize(
    "identity_change", ["registry", "model", "contract", "dependency"])
def test_pattern_hot_identity_change_marks_snapshot_stale_without_recompute(
        identity_change, monkeypatch, _offline_small_app):
    import io
    import streamlit as st
    from PIL import Image
    import ui_pattern_contracts as pattern_contracts

    upload = io.BytesIO()
    Image.new("RGB", (2, 2), (70, 110, 150)).save(upload, format="PNG")
    upload_bytes = upload.getvalue()
    monkeypatch.setattr(
        st, "file_uploader", lambda *_args, **_kwargs: io.BytesIO(upload_bytes))

    at = _run_app()
    next(button for button in at.button if button.label == "生成图案").click()
    at = at.run(timeout=30)
    assert pattern_contracts.PATTERN_SESSION_KEY in at.session_state.filtered_state
    _offline_small_app.clear()

    if identity_change == "registry":
        old_registry = pattern_contracts.pattern_registry_version()
        monkeypatch.setattr(
            pattern_contracts, "pattern_registry_version",
            lambda: old_registry + "-hot-update",
        )
    elif identity_change == "model":
        monkeypatch.setattr(
            pattern_contracts, "PATTERN_MODEL_VERSION",
            pattern_contracts.PATTERN_MODEL_VERSION + "-hot-update",
        )
    elif identity_change == "contract":
        monkeypatch.setattr(
            pattern_contracts, "PATTERN_CONTRACT_VERSION",
            pattern_contracts.PATTERN_CONTRACT_VERSION + "-hot-update",
        )
    else:
        old_dependencies = pattern_contracts.pattern_dependency_hashes()
        changed_dependencies = dict(old_dependencies)
        changed_dependencies["color_utils.py"] = "0" * 64
        monkeypatch.setattr(
            pattern_contracts, "pattern_dependency_hashes",
            lambda: changed_dependencies,
        )
    at = at.run(timeout=30)

    assert len(at.exception) == 0
    assert any("图案快照已陈旧" in item.value for item in at.warning)
    assert not any(
        call["label"] in {"下载 mapped PNG", "下载像素 CSV", "下载 metadata JSON"}
        for call in _offline_small_app
    )


def test_spectrum_page_marks_verified_local_fdtd_as_limited_comparison(
        monkeypatch, _offline_small_app):
    import hashlib
    import streamlit as st
    from ui_fdtd_asset import FDTD_ASSET_SHA256

    image_calls = []
    original_image = st.image

    def record_image(*args, **kwargs):
        caption = kwargs.get("caption", "")
        if "历史透射振幅图" in str(caption):
            image_calls.append((args, kwargs))
        return original_image(*args, **kwargs)

    monkeypatch.setattr(st, "image", record_image)
    at = _run_app()
    at = at.run(timeout=30)
    text = _markdown_text(at)
    captions = "\n".join(item.value for item in at.caption)
    warnings = "\n".join(item.value for item in at.warning)
    subheaders = [item.value for item in at.get("subheader")]

    assert len(at.exception) == 0
    assert len(image_calls) == 2
    for args, kwargs in image_calls:
        assert hashlib.sha256(args[0]).hexdigest().upper() == FDTD_ASSET_SHA256
        assert "历史透射振幅图，非当前 Fano 反射全波验证" in kwargs["caption"]
        assert FDTD_ASSET_SHA256[:12] in kwargs["caption"]
    assert "Fano 近似与历史 FDTD 图：有限可比性对照" in subheaders
    assert "不能作为全波验证" in warnings
    assert "1-T ≠ R" in warnings
    assert "共振峰位偏差 16-24 nm" in text
    assert "光谱形状相关系数 0.48-0.51" in text
    assert "只表示有限形状一致性，不是定量精度或模型验证" in text
    assert "historical-fdtd-comparison-v1" in captions
    assert FDTD_ASSET_SHA256[:12] in captions
    assert not any(
        "FDTD" in str(call["label"]) or "历史" in str(call["label"])
        for call in _offline_small_app
    )


def test_fdtd_unavailable_hides_image_and_hash_bound_metrics(
        monkeypatch, _offline_small_app):
    import streamlit as st
    import ui_fdtd_asset as fdtd

    image_calls = []
    original_image = st.image

    def record_image(*args, **kwargs):
        if "历史透射振幅图" in str(kwargs.get("caption", "")):
            image_calls.append((args, kwargs))
        return original_image(*args, **kwargs)

    unavailable = fdtd.FDTDEvidence(
        asset=fdtd.FDTDAssetResult(
            status="unavailable",
            reason_code="missing",
            detail="项目内历史 FDTD 图不存在",
        ),
        manifest=None,
    )
    monkeypatch.setattr(fdtd, "resolve_fdtd_evidence", lambda: unavailable)
    monkeypatch.setattr(st, "image", record_image)

    at = _run_app()
    at = at.run(timeout=30)
    text = _markdown_text(at)
    infos = "\n".join(item.value for item in at.info)

    assert len(at.exception) == 0
    assert image_calls == []
    assert "页面未联网，当前正向结果不受影响" in infos
    assert "共振峰位偏差" not in text
    assert "光谱形状相关系数" not in text


def test_fdtd_duplicate_manifest_keeps_verified_image_but_hides_metrics(
        monkeypatch, _offline_small_app):
    from dataclasses import replace
    import streamlit as st
    import ui_fdtd_asset as fdtd

    image_calls = []
    original_image = st.image

    def record_image(*args, **kwargs):
        if "历史透射振幅图" in str(kwargs.get("caption", "")):
            image_calls.append((args, kwargs))
        return original_image(*args, **kwargs)

    verified_asset = fdtd.resolve_fdtd_asset()
    assert verified_asset.available
    duplicate_manifest = replace(
        fdtd.FDTD_COMPARISON_MANIFEST,
        metrics=(
            fdtd.FDTD_COMPARISON_MANIFEST.metrics[0],
            fdtd.FDTD_COMPARISON_MANIFEST.metrics[0],
        ),
    )
    assert fdtd.manifest_for_asset(
        verified_asset, duplicate_manifest
    ) is None
    monkeypatch.setattr(
        fdtd,
        "resolve_fdtd_evidence",
        lambda: fdtd.FDTDEvidence(
            asset=verified_asset,
            manifest=fdtd.manifest_for_asset(
                verified_asset, duplicate_manifest
            ),
        ),
    )
    monkeypatch.setattr(st, "image", record_image)

    at = _run_app()
    text = _markdown_text(at)
    infos = "\n".join(item.value for item in at.info)

    assert len(at.exception) == 0
    assert len(image_calls) == 1
    assert "指标合同不一致，已隐藏指标" in infos
    assert "指标合同与资产 hash 不一致" not in infos
    assert "页面未联网，当前正向结果不受影响" in infos
    assert "共振峰位偏差" not in text
    assert "光谱形状相关系数" not in text


def test_fdtd_default_and_rerun_never_download_create_dirs_or_write(
        monkeypatch, _offline_small_app):
    import os
    from pathlib import Path
    import huggingface_hub

    calls = {"hf": 0, "mkdir": 0, "write_bytes": 0, "write_text": 0}

    def bomb(name):
        def inner(*_args, **_kwargs):
            calls[name] += 1
            raise AssertionError(f"FDTD display must not call {name}")
        return inner

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", bomb("hf"))
    monkeypatch.setattr(os, "makedirs", bomb("mkdir"))
    monkeypatch.setattr(Path, "write_bytes", bomb("write_bytes"))
    monkeypatch.setattr(Path, "write_text", bomb("write_text"))

    at = _run_app()
    assert len(at.exception) == 0
    at = at.run(timeout=30)

    assert len(at.exception) == 0
    assert calls == {"hf": 0, "mkdir": 0, "write_bytes": 0, "write_text": 0}


def _model_difference_button(at):
    return next(
        button for button in at.button
        if button.label == "▶ 运行模型间差异分析"
    )


def _fake_model_difference_load(monkeypatch, output):
    import ui_model_difference_contracts as contracts

    class FakeEvaluator:
        contract = contracts.GENERIC_ONNX_ROUTE
        model_path = "C:/fixture/models/forward_mlp_v8_sub.onnx"
        source_pt_path = "C:/fixture/models/forward_mlp_v8_sub.pt"
        calls = 0

        def predict_spectra(self, geometries, **kwargs):
            self.calls += 1
            if isinstance(output, BaseException):
                raise output
            if callable(output):
                return output(geometries)
            return output

    evaluator = FakeEvaluator()
    evidence = contracts.ConversionEvidence(
        contracts.GENERIC_ONNX_ROUTE.conversion_protocol_relative_path,
        contracts.GENERIC_ONNX_ROUTE.conversion_result_relative_path,
        contracts.GENERIC_ONNX_ROUTE.conversion_protocol_sha256,
        contracts.GENERIC_ONNX_ROUTE.conversion_result_sha256,
        "ac1302131ea10919175b3131a981dd544a957f14e4da03890a9de1c6494b1cfd",
        2048,
        "ee11ce200dd8f81b18fa7897652e35646571be3c29eeee8e9db804fa5e45af58",
        3.129243850708008e-07,
        1e-6,
    )
    monkeypatch.setattr(
        contracts, "freeze_generic_onnx_evaluator",
        lambda *_args, **_kwargs: contracts.GenericEvaluatorLoad(
            "loaded", "", evaluator, evidence),
    )
    return evaluator


def test_model_difference_ml_disabled_does_not_load_generic(monkeypatch):
    import ui_model_difference_contracts as contracts
    from streamlit.testing.v1 import AppTest

    def unexpected_load(*args, **kwargs):
        raise AssertionError("generic evaluator must not load when ml_accel=False")

    monkeypatch.setattr(contracts, "freeze_generic_onnx_evaluator", unexpected_load)
    at = AppTest.from_file("app.py")
    at.session_state["ml_accel"] = False
    at = at.run(timeout=30)

    assert len(at.exception) == 0
    assert _model_difference_button(at).disabled
    assert "ML 加速已关闭" in "\n".join(item.value for item in at.warning)


def test_model_difference_missing_generic_model_disables_button(monkeypatch):
    import ui_model_difference_contracts as contracts

    calls = {"load": 0}

    def unavailable_load(*_args, **_kwargs):
        calls["load"] += 1
        return contracts.GenericEvaluatorLoad(
            "unavailable", "fixture generic model missing")

    monkeypatch.setattr(
        contracts, "freeze_generic_onnx_evaluator",
        unavailable_load,
    )
    at = _run_app()

    assert len(at.exception) == 0
    assert not _model_difference_button(at).disabled
    assert calls["load"] == 0
    _model_difference_button(at).click()
    at = at.run(timeout=30)
    assert calls["load"] == 1
    assert "fixture generic model missing" in "\n".join(
        item.value for item in at.warning)


def test_model_difference_unregistered_material_disables_button():
    at = _run_app()
    next(
        item for item in at.selectbox if item.label == "柱材料"
    ).set_value("GaN (wurtzite)")
    at = at.run(timeout=30)

    assert len(at.exception) == 0
    assert _model_difference_button(at).disabled
    assert "GaN (wurtzite) / SiO2 (fused silica) 未注册训练证据" in "\n".join(
        item.value for item in at.warning)


@pytest.mark.parametrize(
    ("substrate", "train_count", "validation_count"),
    [
        ("Si3N4 (nitride)", "2,600", "10,400"),
        ("Al2O3 (sapphire)", "0", "13,000"),
    ],
)
def test_model_difference_undertrained_al2o3_pairs_are_disabled(
    substrate, train_count, validation_count,
):
    at = _run_app()
    next(item for item in at.selectbox if item.label == "柱材料").set_value(
        "Al2O3 (sapphire)")
    next(item for item in at.selectbox if item.label == "衬底材料").set_value(
        substrate)
    at = at.run(timeout=30)
    warnings = "\n".join(item.value for item in at.warning)

    assert len(at.exception) == 0
    assert _model_difference_button(at).disabled
    assert f"train={train_count}" in warnings
    assert f"validation={validation_count}" in warnings
    assert "按保守规则禁用" in warnings


def test_model_difference_tm_is_disabled_as_polarization_insensitive_target():
    at = _run_app()
    next(item for item in at.selectbox if item.label == "偏振").set_value(
        "TM (p-pol)")
    at = at.run(timeout=30)

    assert len(at.exception) == 0
    assert _model_difference_button(at).disabled
    assert "训练 target single_spectrum 未使用偏振输入" in "\n".join(
        item.value for item in at.warning)
    assert "不能比较 TM 物理差异" in "\n".join(
        item.value for item in at.warning)


@pytest.mark.parametrize(
    "generic_output",
    [
        RuntimeError("fixture generic failure"),
        None,
        lambda geometries: np.full((len(geometries), 81), np.nan),
        lambda geometries: np.full((len(geometries), 80), 0.2),
    ],
    ids=["exception", "none", "nonfinite", "wrong-shape"],
)
def test_model_difference_generic_failure_is_unavailable(monkeypatch, generic_output):
    import torch
    import torch_model

    evaluator = _fake_model_difference_load(monkeypatch, generic_output)
    monkeypatch.setattr(
        torch_model, "batch_lorentzian_spectrum",
        lambda d, h, p, angle, pol, material, substrate: torch.full(
            (len(d), 81), 0.1),
    )
    at = _run_app()
    _model_difference_button(at).click()
    at = at.run(timeout=30)
    text = _markdown_text(at)

    assert len(at.exception) == 0
    assert evaluator.calls == 1
    assert any("模型间差异结果不可用" in item.value for item in at.warning)
    assert "**模型间差异统计**" not in text


def test_model_difference_success_uses_only_frozen_generic_route(monkeypatch):
    import ml_module
    import torch
    import torch_model
    import ui_model_difference_contracts as contracts

    evaluator = _fake_model_difference_load(
        monkeypatch,
        lambda geometries: np.full((len(geometries), 81), 0.3),
    )
    monkeypatch.setattr(
        torch_model, "batch_lorentzian_spectrum",
        lambda d, h, p, angle, pol, material, substrate: torch.full(
            (len(d), 81), 0.1),
    )
    monkeypatch.setattr(
        ml_module, "predict_rgb",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("predict_rgb auto-route must not be called")),
    )
    monkeypatch.setattr(ml_module, "init_rcwa_ml", lambda: True)
    monkeypatch.setattr(
        ml_module, "_RCWA_SESSIONS",
        {"TiO2 (anatase)": [object()]},
    )
    at = _run_app()
    _model_difference_button(at).click()
    at = at.run(timeout=30)
    text = _markdown_text(at)
    captions = "\n".join(item.value for item in at.caption)

    assert len(at.exception) == 0
    assert evaluator.calls == 1
    assert "**模型间差异统计**" in text
    assert "平均 CIEDE2000 差异" in text
    assert "不是 RCWA 真值误差" in text
    assert "不是 RCWA 精度、实验误差或人眼感知阈值" in captions
    assert "generic-fano-resmlp-v8-sub-onnx-only" in captions
    assert contracts.GENERIC_ONNX_ROUTE.model_sha256 in captions
    assert contracts.GENERIC_ONNX_ROUTE.source_pt_sha256 in captions
    assert "训练提交 2e8c170" in captions
    assert "转换提交 64b94de" in captions
    assert "PT-vs-ONNX max|Δ|=3.1292438507080078e-07" in captions
    assert "train=13,000，validation=0" in captions
    at = at.run(timeout=30)
    assert evaluator.calls == 1
    _model_difference_button(at).click()
    at = at.run(timeout=30)
    assert evaluator.calls == 1


def test_model_difference_bundle_tamper_stales_and_never_loads_evaluator(monkeypatch):
    import torch
    import torch_model
    import ui_analysis_snapshots as snapshots
    from ui_analysis_snapshots import SourceArtifactIdentity

    evaluator = _fake_model_difference_load(
        monkeypatch, lambda geometries: np.full((len(geometries), 81), 0.3))
    monkeypatch.setattr(
        torch_model, "batch_lorentzian_spectrum",
        lambda d, h, p, angle, pol, material, substrate: torch.full(
            (len(d), 81), 0.1),
    )
    real_identity = snapshots.source_artifact_identity
    state = {"tampered": False}

    def bundle_identity(*args, **kwargs):
        if kwargs.get("expected_sha256") and state["tampered"]:
            return SourceArtifactIdentity(
                False, "unavailable:tampered-bundle",
                "models/forward_mlp_v8_sub.onnx:hash_mismatch", (),
            )
        return real_identity(*args, **kwargs)

    monkeypatch.setattr(snapshots, "source_artifact_identity", bundle_identity)
    at = _run_app()
    _model_difference_button(at).click()
    at = at.run(timeout=30)
    assert len(at.exception) == 0
    assert evaluator.calls == 1
    assert "**模型间差异统计**" in _markdown_text(at)

    state["tampered"] = True
    at = at.run(timeout=30)
    assert len(at.exception) == 0
    assert evaluator.calls == 1
    assert any("模型差异快照已陈旧" in item.value for item in at.warning)
    assert "**模型间差异统计**" not in _markdown_text(at)
    assert _model_difference_button(at).disabled


def _configure_fast_benchmark(monkeypatch, single_result):
    import importlib.util
    import ml_module
    import os

    real_isfile = os.path.isfile
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        os.path, "isfile",
        lambda path: (
            True if str(path).replace("\\", "/").endswith(
                "models/forward_mlp_v8_sub.pt")
            else real_isfile(path)
        ),
    )
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name: object() if name == "torch" else real_find_spec(name),
    )
    monkeypatch.setattr(ml_module, "init_rcwa_ml", lambda: False)
    # The exact RCWA registry is now loaded independently by the app route;
    # make the missing-model fixture fail closed at that route boundary too.
    monkeypatch.setattr(ml_module, "_RCWA_MODELS", {})
    monkeypatch.setattr(ml_module, "_RCWA_SUBSTRATE_MODELS", {})
    monkeypatch.setattr(ml_module, "_RCWA_SESSIONS", {})
    monkeypatch.setattr(ml_module, "_RCWA_WL_SESSIONS", {})
    if isinstance(single_result, BaseException):
        def raise_single(*args, **kwargs):
            raise single_result
        monkeypatch.setattr(
            ml_module, "_inverse_design_ml_serial", raise_single)
    else:
        monkeypatch.setattr(
            ml_module, "_inverse_design_ml_serial",
            lambda *args, **kwargs: single_result,
        )


def _click_benchmark(at):
    next(
        button for button in at.button
        if button.label == "▶ 运行当前可用方法基准"
    ).click()
    return at.run(timeout=30)


def _run_app_with_benchmark_cache(value):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py")
    at.session_state["_bench_cache"] = value
    return at.run(timeout=30)


@pytest.mark.parametrize(
    "stale_cache",
    [
        {"single": {"status": "available"}},
        [],
        object(),
    ],
    ids=["legacy-dict", "mixed-list", "wrong-object"],
)
def test_benchmark_stale_cache_fails_closed_without_rendering(stale_cache):
    from ui_benchmark_contracts import BenchmarkRow

    if isinstance(stale_cache, list):
        stale_cache.extend([
            BenchmarkRow.unavailable(
                "rl", "RL Q-learning", route="实验性离散探索", detail="未运行。",
            ),
            {"status": "available"},
        ])
    at = _run_app_with_benchmark_cache(stale_cache)
    text = _markdown_text(at)

    assert len(at.exception) == 0
    assert any("旧基准缓存已失效，请重新运行" in item.value for item in at.warning)
    assert "| 方法 | 状态 | 耗时 | 候选 ΔE00 | 实际路线 |" not in text
    assert "_bench_cache" not in at.session_state


def test_benchmark_typed_cache_continues_to_render():
    from ui_benchmark_contracts import BenchmarkRow

    row = BenchmarkRow(
        method_id="fixture", label="合法缓存方法", status="available",
        elapsed_s=0.125, delta_e2000=1.5, route="测试代理路线",
        timing_source="测试计时器", metric_source="测试有限 sRGB",
        detail="合法 typed cache fixture。",
    )
    at = _run_app_with_benchmark_cache((row,))
    text = _markdown_text(at)

    assert len(at.exception) == 0
    assert not any("旧基准缓存已失效" in item.value for item in at.warning)
    assert "| 合法缓存方法 | 可用 | 0.125s | 1.500 | 测试代理路线 |" in text


def test_benchmark_empty_typed_cache_is_handled_without_rendering():
    at = _run_app_with_benchmark_cache(())
    text = _markdown_text(at)

    assert len(at.exception) == 0
    assert any("本次基准没有返回结果，请重新运行" in item.value for item in at.info)
    assert "| 方法 | 状态 | 耗时 | 候选 ΔE00 | 实际路线 |" not in text


def test_benchmark_click_missing_model_keeps_analytical_dual_baseline(monkeypatch):
    import ml_module
    import os

    real_isfile = os.path.isfile
    monkeypatch.setattr(
        os.path, "isfile",
        lambda path: (
            False if str(path).replace("\\", "/").endswith(
                "models/forward_mlp_v8_sub.pt")
            else real_isfile(path)
        ),
    )
    monkeypatch.setattr(ml_module, "init_rcwa_ml", lambda: False)
    monkeypatch.setattr(ml_module, "_RCWA_MODELS", {})
    monkeypatch.setattr(ml_module, "_RCWA_SUBSTRATE_MODELS", {})
    monkeypatch.setattr(ml_module, "_RCWA_SESSIONS", {})
    at = _click_benchmark(_run_app())
    text = _markdown_text(at)

    assert len(at.exception) == 0
    assert "| 单柱梯度 | 未运行/不可用 | — | — |" in text
    assert "| RL Q-learning | 未运行/不可用 | — | — |" in text
    assert "| 双柱梯度 | 未运行/不可用 | — | — |" in text
    assert "| 双柱解析基线 | 可用 |" in text
    assert "Lorentz/Fano analytical dual" in text
    assert "N/A" not in text
    assert "本次没有方法返回可验收的有限 sRGB" not in "\n".join(
        item.value for item in at.info)


def test_benchmark_click_uses_rgb_slot_four_from_single_six_tuple(monkeypatch):
    from color_utils import delta_e2000, rgb_to_lab

    predicted = np.array([0.25, 0.50, 0.75])
    _configure_fast_benchmark(
        monkeypatch,
        ("Fano", 190.0, 310.0, 410.0, predicted, 0.01),
    )
    at = _click_benchmark(_run_app())
    text = _markdown_text(at)
    target = np.array([122, 79, 34], dtype=float) / 255.0
    expected = delta_e2000(rgb_to_lab(target), rgb_to_lab(predicted))

    assert len(at.exception) == 0
    assert "| 单柱梯度 | 可用 |" in text
    assert f"| {expected:.3f} |" in text
    assert "返回标识 Fano" in text
    assert "color_utils.delta_e2000" in text
    assert "直接 RCWA 或全局最优" in text


def test_benchmark_click_method_exception_fails_closed(monkeypatch):
    _configure_fast_benchmark(monkeypatch, RuntimeError("fixture failure"))
    at = _click_benchmark(_run_app())
    text = _markdown_text(at)

    assert len(at.exception) == 0
    assert "| 单柱梯度 | 错误 | — | — |" in text
    assert "优化失败：RuntimeError" in text
    assert "未生成可验收的有限 sRGB" in text


def test_mapping_grid_uses_full_width_semantics_and_locates_current_geometry():
    at = _run_app()
    assert not any(
        'class="mapping-table-wrap"' in element.value for element in at.markdown)

    next(
        item for item in at.checkbox
        if item.label == "启用 ML 代理模型（快速候选预测）"
    ).set_value(False)
    next(item for item in at.slider if item.label == "直径 D (nm)").set_value(260.0)
    at = at.run(timeout=30)
    next(
        button for button in at.button
        if button.label == "运行 / 加载当前 D-H 映射"
    ).click()
    at = at.run(timeout=30)
    markup = next(
        element.value for element in at.markdown
        if 'class="mapping-table-wrap"' in element.value
    )

    assert len(at.exception) == 0
    assert markup.count('data-status="') == 48
    assert markup.count('<th scope="col">') == 9
    assert markup.count('<th scope="row">') == 6
    assert markup.count('role="img"') == 48
    assert markup.count('role="img" tabindex="0"') == 1
    assert markup.count('role="img" tabindex="-1"') == 47
    assert 'aria-current="true"' in markup
    assert "当前邻近可用格" in markup
    assert "当前参数" in markup
    assert "D=260.0 nm · H=300.0 nm · P=400.0 nm" in markup
    current_tag = next(
        tag.split(">", 1)[0] for tag in markup.split("<div ")
        if "mapping-cell--current" in tag
    )
    assert 'data-status="available"' in current_tag
    assert 'tabindex="0"' in current_tag
    assert 'aria-current="true"' in current_tag
    assert "D=269 nm" in current_tag
    assert not any(
        'data-status="unavailable"' in tag and 'aria-current="true"' in tag
        for tag in markup.split("<div ")
    )
    assert "D-H 颜色映射表，可横向滚动" in markup


def test_analysis_default_render_calls_no_evaluator_or_model_loader(monkeypatch):
    import ui_forward_routes as routes
    import ui_model_difference_contracts as difference

    calls = {"series": 0, "mapping": 0, "difference_load": 0}
    original_series = routes.evaluate_frozen_series
    original_mapping = routes.build_mapping_cells

    def counted_series(route, requests):
        calls["series"] += 1
        return original_series(route, requests)

    def counted_mapping(*args, **kwargs):
        calls["mapping"] += 1
        return original_mapping(*args, **kwargs)

    def unexpected_difference_load(*_args, **_kwargs):
        calls["difference_load"] += 1
        raise AssertionError("difference evaluator must stay lazy")

    monkeypatch.setattr(routes, "evaluate_frozen_series", counted_series)
    monkeypatch.setattr(routes, "build_mapping_cells", counted_mapping)
    monkeypatch.setattr(
        difference, "freeze_generic_onnx_evaluator", unexpected_difference_load)
    at = _run_app()
    state = at.session_state.filtered_state

    assert len(at.exception) == 0
    assert calls == {"series": 0, "mapping": 0, "difference_load": 0}
    assert not any(str(key).startswith("_ui_analysis_") for key in state)


@pytest.mark.parametrize(
    ("structure", "has_sensitivity"),
    [("单柱", True), ("双柱", True), ("FP 腔（Fabry-Pérot）", False)],
)
def test_analysis_controls_are_lazy_for_single_dual_and_fp(structure, has_sensitivity):
    at = _run_app()
    at.radio[0].set_value(structure)
    at = at.run(timeout=30)
    labels = {button.label for button in at.button}

    assert len(at.exception) == 0
    assert ("运行 / 加载当前灵敏度分析" in labels) is has_sensitivity
    assert "运行 / 加载当前 D-H 映射" in labels
    assert "运行 / 加载当前入射角扫描" in labels
    assert not any(
        str(key).startswith("_ui_analysis_")
        for key in at.session_state.filtered_state)


def test_angle_snapshot_null_mask_reuse_stale_engine_guard_and_session_isolation(
        monkeypatch, _offline_small_app):
    import json
    import streamlit as st
    import engine as engine_module
    import ui_forward_routes as routes
    from streamlit.testing.v1 import AppTest
    from ui_analysis_snapshots import ANALYSIS_SESSION_KEYS
    from ui_engine_session import ENGINE_SESSION_KEY

    st.cache_data.clear()
    calls = {"angle_series": 0}
    original_series = routes.evaluate_frozen_series

    def counted_series(route, requests):
        materialized = list(requests)
        if len(materialized) == 17:
            calls["angle_series"] += 1
        return original_series(route, materialized)

    def spectrum(self, param, wl_start=380.0, wl_end=780.0, n_pts=81):
        if abs(float(param.angle_deg) - 5.0) < 1e-9:
            raise RuntimeError("fixture unavailable angle")
        wavelengths = np.linspace(wl_start, wl_end, n_pts)
        return wavelengths, np.full(n_pts, 0.2 + float(param.angle_deg) / 1000.0)

    monkeypatch.setattr(routes, "evaluate_frozen_series", counted_series)
    monkeypatch.setattr(engine_module.MetaSurfaceColorEngine, "compute_spectrum", spectrum)

    app_a = AppTest.from_file("app.py")
    app_a.session_state["ml_accel"] = False
    app_a = app_a.run(timeout=30)
    app_b = AppTest.from_file("app.py")
    app_b.session_state["ml_accel"] = False
    app_b = app_b.run(timeout=30)
    engine_a = app_a.session_state[ENGINE_SESSION_KEY]
    engine_id = id(engine_a)
    grid_before = tuple(
        np.asarray(getattr(engine_a, name)).copy()
        for name in ("grid_params", "grid_rgb", "grid_lab", "grid_xy"))

    next(
        button for button in app_a.button
        if button.label == "运行 / 加载当前入射角扫描"
    ).click()
    app_a = app_a.run(timeout=30)
    key = ANALYSIS_SESSION_KEYS["angle"]
    record = app_a.session_state[key]
    payload = record["payload"]
    angle_json = json.loads(_latest_download_data(
        _offline_small_app, "下载角扫 JSON"))

    assert len(app_a.exception) == 0
    assert calls["angle_series"] == 1
    assert payload["available_mask"][1] is False
    assert payload["rgb"][1] is None
    assert angle_json["angle_scan"]["available_mask"][1] is False
    assert angle_json["angle_scan"]["rgb"][1] is None
    assert id(app_a.session_state[ENGINE_SESSION_KEY]) == engine_id
    for name, expected in zip(
        ("grid_params", "grid_rgb", "grid_lab", "grid_xy"), grid_before):
        np.testing.assert_array_equal(getattr(engine_a, name), expected)
    assert key not in app_b.session_state.filtered_state

    app_a = app_a.run(timeout=30)
    assert calls["angle_series"] == 1
    next(
        button for button in app_a.button
        if button.label == "运行 / 加载当前入射角扫描"
    ).click()
    app_a = app_a.run(timeout=30)
    assert calls["angle_series"] == 1

    next(item for item in app_a.slider if item.label == "直径 D (nm)").set_value(181.0)
    _offline_small_app.clear()
    app_a = app_a.run(timeout=30)
    assert calls["angle_series"] == 1
    assert any("角扫快照已陈旧" in item.value for item in app_a.warning)
    assert app_a.session_state[key]["context_fingerprint"] == record["context_fingerprint"]
    assert not any(
        call["label"] == "下载角扫 JSON" for call in _offline_small_app)


def test_invalid_geometry_disables_angle_and_hides_old_snapshot(
        monkeypatch, _offline_small_app):
    import ui_forward_routes as routes
    from streamlit.testing.v1 import AppTest

    calls = {"angle_series": 0}
    original = routes.evaluate_frozen_series

    def counted(route, requests):
        materialized = list(requests)
        if len(materialized) == 17:
            calls["angle_series"] += 1
        return original(route, materialized)

    monkeypatch.setattr(routes, "evaluate_frozen_series", counted)
    at = AppTest.from_file("app.py")
    at.session_state["ml_accel"] = False
    at = at.run(timeout=30)
    next(
        button for button in at.button
        if button.label == "运行 / 加载当前入射角扫描"
    ).click()
    at = at.run(timeout=30)
    assert calls["angle_series"] == 1

    next(item for item in at.slider if item.label == "直径 D (nm)").set_value(350.0)
    next(item for item in at.slider if item.label == "周期 P (nm)").set_value(200.0)
    _offline_small_app.clear()
    at = at.run(timeout=30)
    angle_button = next(
        button for button in at.button
        if button.label == "运行 / 加载当前入射角扫描")

    assert len(at.exception) == 0
    assert angle_button.disabled
    assert calls["angle_series"] == 1
    assert any("当前正向结果或几何不可用" in item.value for item in at.info)
    assert any("角扫快照已陈旧" in item.value for item in at.warning)
    assert not any(call["label"] == "下载角扫 JSON" for call in _offline_small_app)


def test_unavailable_forward_disables_angle_without_evaluator(
        monkeypatch, _offline_small_app):
    import engine as engine_module
    import ui_forward_routes as routes

    calls = {"series": 0}
    monkeypatch.setattr(
        routes, "evaluate_frozen_series",
        lambda *_args, **_kwargs: calls.__setitem__("series", calls["series"] + 1),
    )
    monkeypatch.setattr(
        engine_module.MetaSurfaceColorEngine, "compute_spectrum",
        lambda *_args, **_kwargs: (None, None),
    )
    at = _run_app_with_session_state({"ml_accel": False})
    angle_button = next(
        button for button in at.button
        if button.label == "运行 / 加载当前入射角扫描")

    assert len(at.exception) == 0
    assert angle_button.disabled
    assert calls["series"] == 0
    assert any("当前正向结果或几何不可用" in item.value for item in at.info)
    assert not any(call["label"] == "下载角扫 JSON" for call in _offline_small_app)


def test_sensitivity_snapshot_runs_once_reuses_and_hides_when_stale(monkeypatch):
    import streamlit as st
    import engine as engine_module
    import ui_forward_routes as routes
    from streamlit.testing.v1 import AppTest
    from ui_analysis_snapshots import ANALYSIS_SESSION_KEYS

    st.cache_data.clear()
    calls = {"series": 0}
    original_series = routes.evaluate_frozen_series

    def counted_series(route, requests):
        calls["series"] += 1
        return original_series(route, requests)

    def spectrum(self, param, wl_start=380.0, wl_end=780.0, n_pts=81):
        wavelengths = np.linspace(wl_start, wl_end, n_pts)
        return wavelengths, np.full(n_pts, 0.25)

    monkeypatch.setattr(routes, "evaluate_frozen_series", counted_series)
    monkeypatch.setattr(engine_module.MetaSurfaceColorEngine, "compute_spectrum", spectrum)
    at = AppTest.from_file("app.py")
    at.session_state["ml_accel"] = False
    at = at.run(timeout=30)
    next(
        button for button in at.button
        if button.label == "运行 / 加载当前灵敏度分析"
    ).click()
    at = at.run(timeout=30)

    assert len(at.exception) == 0
    assert calls["series"] == 3
    assert ANALYSIS_SESSION_KEYS["sensitivity"] in at.session_state.filtered_state
    at = at.run(timeout=30)
    assert calls["series"] == 3
    next(item for item in at.slider if item.label == "高度 H (nm)").set_value(301.0)
    at = at.run(timeout=30)
    assert calls["series"] == 3
    assert any("灵敏度快照已陈旧" in item.value for item in at.warning)


def test_mapping_snapshot_runs_once_reuses_and_hides_when_stale(monkeypatch):
    import streamlit as st
    import engine as engine_module
    import ui_analysis_snapshots as snapshots
    import ui_forward_routes as routes
    from streamlit.testing.v1 import AppTest
    from ui_analysis_snapshots import (
        ANALYSIS_SESSION_KEYS, SourceArtifactIdentity,
    )

    st.cache_data.clear()
    calls = {"mapping": 0, "mapping_points": 0}
    artifact = {"version": "A", "available": True}
    original_mapping = routes.build_mapping_cells

    def dynamic_artifact(*_args, **_kwargs):
        prefix = "sha256" if artifact["available"] else "unavailable"
        return SourceArtifactIdentity(
            artifact["available"], f"{prefix}:{artifact['version']}",
            "" if artifact["available"] else "fixture.py:unreadable", (),
        )

    def counted_mapping(*args, **kwargs):
        calls["mapping"] += 1
        return original_mapping(*args, **kwargs)

    def spectrum(self, param, wl_start=380.0, wl_end=780.0, n_pts=81):
        d_samples = np.linspace(80, 300, 8)
        h_samples = np.linspace(200, 600, 6)
        if (
            any(abs(float(param.diameter_nm) - value) < 1e-9 for value in d_samples)
            and any(abs(float(param.height_nm) - value) < 1e-9 for value in h_samples)
        ):
            calls["mapping_points"] += 1
        wavelengths = np.linspace(wl_start, wl_end, n_pts)
        return wavelengths, np.full(n_pts, 0.3)

    monkeypatch.setattr(snapshots, "source_artifact_identity", dynamic_artifact)
    monkeypatch.setattr(routes, "build_mapping_cells", counted_mapping)
    monkeypatch.setattr(engine_module.MetaSurfaceColorEngine, "compute_spectrum", spectrum)
    at = AppTest.from_file("app.py")
    at.session_state["ml_accel"] = False
    at = at.run(timeout=30)
    next(
        button for button in at.button
        if button.label == "运行 / 加载当前 D-H 映射"
    ).click()
    at = at.run(timeout=30)

    assert len(at.exception) == 0
    assert calls["mapping"] == 1
    assert calls["mapping_points"] == 48
    assert ANALYSIS_SESSION_KEYS["mapping"] in at.session_state.filtered_state
    assert any('class="mapping-table-wrap"' in item.value for item in at.markdown)
    at = at.run(timeout=30)
    assert calls["mapping"] == 1
    assert calls["mapping_points"] == 48

    artifact["version"] = "B"
    at = at.run(timeout=30)
    assert calls["mapping"] == 1
    assert calls["mapping_points"] == 48
    assert any("D-H 映射快照已陈旧" in item.value for item in at.warning)
    assert not any('class="mapping-table-wrap"' in item.value for item in at.markdown)

    next(
        button for button in at.button
        if button.label == "运行 / 加载当前 D-H 映射"
    ).click()
    at = at.run(timeout=30)
    assert len(at.exception) == 0
    assert calls["mapping"] == 2
    assert calls["mapping_points"] == 96
    assert any('class="mapping-table-wrap"' in item.value for item in at.markdown)

    artifact.update(version="missing", available=False)
    at = at.run(timeout=30)
    assert len(at.exception) == 0
    assert calls == {"mapping": 2, "mapping_points": 96}
    assert any("映射源码身份不可用" in item.value for item in at.info)
    assert any("D-H 映射快照已陈旧" in item.value for item in at.warning)
    assert not any('class="mapping-table-wrap"' in item.value for item in at.markdown)
    mapping_button = next(
        button for button in at.button
        if button.label == "运行 / 加载当前 D-H 映射"
    )
    assert mapping_button.disabled


def test_all_analysis_blocks_use_the_shared_engine_transaction():
    import ast
    from pathlib import Path

    tree = ast.parse(Path("app.py").read_text(encoding="utf-8"))
    transaction_withs = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.With)
        and any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Name)
            and item.context_expr.func.id == "analysis_engine_transaction"
            for item in node.items
        )
    ]
    # Four lazy analysis blocks plus the isolated pattern-generation operation.
    assert len(transaction_withs) == 5
    restore_handlers = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler)
        and isinstance(node.type, ast.Name)
        and node.type.id == "EngineStateRestoreError"
    ]
    mutation_handlers = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler)
        and isinstance(node.type, ast.Name)
        and node.type.id == "EngineStateMutationError"
    ]
    assert len(restore_handlers) == 5
    assert len(mutation_handlers) == 5
    assert "capture_engine_state(engine)" not in Path("app.py").read_text(encoding="utf-8")
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert "artifact_version" in {
        arg.arg for arg in functions["_cached_mapping_forward"].args.args
    }
    assert "artifact_version" in {
        arg.arg for arg in functions["_load_model_difference_evaluator_cached"].args.args
    }
    cached_hash = functions["_ui_code_artifact_version"]
    assert not any(
        isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr in {"cache_data", "cache_resource"}
        for decorator in cached_hash.decorator_list
    )


@pytest.mark.parametrize(
    ("button_label", "analysis_type"),
    [
        ("运行 / 加载当前灵敏度分析", "sensitivity"),
        ("运行 / 加载当前 D-H 映射", "mapping"),
        ("▶ 运行模型间差异分析", "difference"),
        ("运行 / 加载当前入射角扫描", "angle"),
    ],
)
def test_each_analysis_transaction_restores_mutation_and_writes_no_available_result(
        button_label, analysis_type, monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    import ui_analysis_snapshots as snapshots
    import ui_model_difference_contracts as difference
    from ui_engine_session import ENGINE_SESSION_KEY

    st.cache_data.clear()
    original_transaction = snapshots.analysis_engine_transaction
    entries = []

    @contextmanager
    def mutating_transaction(engine, session_state, *args, **kwargs):
        with original_transaction(engine, session_state, *args, **kwargs) as before:
            entries.append(id(engine))
            yield before
            engine._last_material = "fixture-pollution"
            engine._enable_far_field = not bool(engine._enable_far_field)
            engine._cache["fixture-pollution"] = np.array([9.0])
            engine.grid_rgb = np.ones((2, 3), dtype=np.float32)

    monkeypatch.setattr(
        snapshots, "analysis_engine_transaction", mutating_transaction)
    monkeypatch.setattr(
        difference, "freeze_generic_onnx_evaluator",
        lambda *_args, **_kwargs: difference.GenericEvaluatorLoad(
            "unavailable", "fixture evaluator disabled"),
    )

    at = AppTest.from_file("app.py").run(timeout=30)
    if analysis_type == "mapping":
        next(
            item for item in at.checkbox
            if item.label == "启用 ML 代理模型（快速候选预测）"
        ).set_value(False)
        at = at.run(timeout=30)
    engine = at.session_state[ENGINE_SESSION_KEY]
    before = snapshots.capture_engine_state(engine)
    next(button for button in at.button if button.label == button_label).click()
    at = at.run(timeout=60)

    record = at.session_state[snapshots.ANALYSIS_SESSION_KEYS[analysis_type]]
    assert len(at.exception) == 0
    assert entries == [id(engine)]
    assert at.session_state[ENGINE_SESSION_KEY] is engine
    assert id(engine) == before.object_id
    snapshots.require_engine_state_unchanged(engine, before)
    assert record["payload"]["status"] == "unavailable"
    assert "完整性失败" in record["payload"]["reason"]
    assert "fixture-pollution" not in engine._cache


def test_restore_failure_clears_stale_angle_snapshot_and_shows_explicit_error(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import ui_analysis_snapshots as snapshots

    at = AppTest.from_file("app.py").run(timeout=30)
    next(
        button for button in at.button
        if button.label == "运行 / 加载当前入射角扫描"
    ).click()
    at = at.run(timeout=30)
    key = snapshots.ANALYSIS_SESSION_KEYS["angle"]
    assert key in at.session_state.filtered_state

    next(item for item in at.slider if item.label == "直径 D (nm)").set_value(181.0)
    at = at.run(timeout=30)
    assert any("角扫快照已陈旧" in item.value for item in at.warning)

    class RestoreFailureContext:
        def __enter__(self):
            raise snapshots.EngineStateRestoreError("fixture restore failure")

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        snapshots, "analysis_engine_transaction",
        lambda *_args, **_kwargs: RestoreFailureContext(),
    )
    next(
        button for button in at.button
        if button.label == "运行 / 加载当前入射角扫描"
    ).click()
    at = at.run(timeout=30)

    assert len(at.exception) == 0
    assert key not in at.session_state.filtered_state
    assert any("会话引擎恢复失败" in item.value for item in at.error)


def test_unloaded_rcwa_candidate_contract_does_not_claim_model(monkeypatch):
    import app

    monkeypatch.setattr(app, "_rcwa_model_version", lambda *_args: "不可用")
    contract = app._inverse_candidate_contract(
        "smart_grid", "TiO2 (anatase)", "SiO2 (fused silica)", "TE (s-pol)", 0.0)
    assert "不可用" in contract["model"]
    assert "RCWA ResMLP .pt ensemble" not in contract["model"]
