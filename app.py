# ===================== Streamlit 版本：超表面结构色设计系统 =====================
from __future__ import annotations

import io, os, json, hashlib, glob, importlib.util
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import html
from dataclasses import replace
# NumPy 1.x/2.x compatibility
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz
from PIL import Image, ImageOps
import streamlit as st
import logging
import ml_module
import ui_model_difference_contracts as model_difference_contracts
from ui_fdtd_asset import resolve_fdtd_evidence
from ui_cie_contracts import (
    GamutSamples, build_cie_plot_data, evaluate_gamut, forward_provenance_caption,
)
from ui_engine_session import (
    ENGINE_LIBRARY_KEY, ENGINE_SESSION_KEY,
    LibraryIdentity, bind_engine_library, configure_engine_far_field,
    engine_library_matches, get_session_engine, make_local_bound_engine,
)
from ui_analysis_snapshots import (
    AnalysisContext, AnalysisSnapshot, angle_payload_arrays,
    build_angle_payload, canonical_sha256, analysis_engine_transaction,
    EngineStateMutationError, EngineStateRestoreError,
    SourceArtifactIdentity, load_analysis_snapshot, source_artifact_identity,
    store_analysis_snapshot,
)
from ui_model_resources import (
    BoundModelResource, ModelResourceDriftError, ModelResourceUnavailable,
    bound_model_context, exception_has_model_resource_drift,
    get_bound_resource, register_first_resource, resource_lock,
    validate_bound_resource,
)
from ui_pattern_contracts import (
    PATTERN_SESSION_KEY, PatternSnapshot, build_pattern_exports,
    build_pattern_payload, load_pattern_snapshot, make_pattern_contract,
    store_pattern_snapshot,
)
from ui_session_migration import (
    BoolControlSpec, EnumControlSpec, NumericControlSpec,
    ML_ACCEL_PREFERENCE_INITIALIZED_KEY, initialize_bool_preference_marker,
    migrate_session_state, set_bool_value, set_enum_value, set_numeric_value,
    sync_bool_from_widget, sync_enum_from_widget, sync_numeric_from_widget,
    sync_bool_preference_from_widget,
)
import rl_design  # RL agent for inverse design
from ui_forward_routes import (
    FrozenSpectrumRoute, evaluate_frozen_series, resolve_perturbation,
    route_results_consistent, make_result_provenance, build_forward_exports,
    ForwardResult, normalize_forward_result, sync_forward_status,
    inverse_candidates_available, mapping_domain_contract,
    build_mapping_cells, nearest_available_mapping_index, forward_export_basename,
    MappingCellResult,
)
from ui_inverse_contracts import (
    InverseContext, InverseMethodState, InverseRun, build_inverse_candidate,
    candidate_parameter_updates, fp_search_cache_key,
    invalidate_inverse_run, inverse_context_fingerprint,
    inverse_method_registry, inverse_run_matches, serialize_inverse_run,
)
from ui_benchmark_contracts import (
    BenchmarkRow,
    benchmark_row_from_rgb,
    validate_benchmark_cache,
)

# LLM 功能暂时隐藏；保留后端适配器，待供应商和审计边界确定后再启用。
ENABLE_LLM_FEATURES = False
# 三方案搜索本身是本地确定性计算，不依赖 LLM，继续保留。
ENABLE_MULTI_SCHEME_SEARCH = True
_FP_INVERSE_ALGORITHM_VERSION = "fp-dbr-grid-v1"
_DUAL_MODEL_RELATIVE_PATH = "models/dual_mlp_v3_multi.onnx"
_DUAL_MODEL_VERSION = "dual_mlp_v3_multi.onnx"
from color_utils import (
    CIE_WAVELENGTHS as _CIE_WAVELENGTHS, CIE_X as _CIE_X, CIE_Y as _CIE_Y, CIE_Z as _CIE_Z,
    WL as _WL, CIE_NORM as _CIE_NORM, D65, SRGB_M as _SRGB_M_NP,
    spectrum_to_xyz, xyz_to_srgb, spectrum_to_srgb, clamp01,
    srgb_to_linear, rgb_to_xyz, xyz_to_xy, rgb_to_xy,
    xyz_to_lab, rgb_to_lab, rgb_to_lab_scalar, rgb_to_hex, rgb_255,
    delta_e76, delta_e2000, delta_e2000_scalar,
)
from competition.reference_library import (
    ReferenceLibraryError,
    load_reference_library,
    reference_condition_summary,
)

# LLM module (DeepSeek API)
try:
    from llm.deepseek_client import analyze_color, suggest_params
    _LLM_AVAILABLE = True
except Exception as e:
    logging.warning(f"app fallback: {e}")
    _LLM_AVAILABLE = False
    def analyze_color(*a, **kw): return u'[LLM模块加载失败，请检查 llm/ 目录]'
    def suggest_params(*a, **kw): return u'[LLM模块加载失败]'


@st.cache_resource
def _get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    try:
        fm._load_fontmanager(try_read_cache=False)
    except Exception as e:
        logging.warning(f"font cache load: {e}")
    available = {f.name for f in fm.fontManager.ttflist}
    fonts = ['WenQuanYi Micro Hei', 'SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC', 'DejaVu Sans']
    chosen = 'DejaVu Sans'
    for fn in fonts:
        if fn in available:
            chosen = fn
            break
    plt.rcParams['font.sans-serif'] = [chosen, 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return plt
# matplotlib imported lazily to avoid cloud startup issues
from contextlib import nullcontext
from typing import Tuple, List

st.set_page_config(page_title="AI超表面结构色设计", layout="wide")


_PATTERN_UPLOAD_MAX_BYTES = 8 * 1024 * 1024
_PATTERN_SOURCE_MAX_PIXELS = 12_000_000
_PATTERN_SOURCE_MAX_SIDE = 5_000
_PATTERN_ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP"}


def _pattern_upload_metadata_error(file_size, image_format, width, height):
    """Validate pattern-upload metadata before decoding or rebuilding the library."""
    if int(file_size) > _PATTERN_UPLOAD_MAX_BYTES:
        return "文件超过 8 MB 上限，未读取图像内容。"
    fmt = str(image_format or "").upper()
    if fmt not in _PATTERN_ALLOWED_FORMATS:
        return "仅接受 PNG、JPEG 或 WebP 图像。"
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        return "图像尺寸无效。"
    if width > _PATTERN_SOURCE_MAX_SIDE or height > _PATTERN_SOURCE_MAX_SIDE:
        return "图像单边超过 5000 像素，未进入解码或映射。"
    if width * height > _PATTERN_SOURCE_MAX_PIXELS:
        return "图像超过 1200 万源像素，未进入解码或映射。"
    return ""


def _open_pattern_upload(uploaded_file):
    """Open a small validated local image and return (image, metadata, error)."""
    try:
        file_size = getattr(uploaded_file, "size", None)
        if file_size is None:
            file_size = uploaded_file.getbuffer().nbytes
        if int(file_size) > _PATTERN_UPLOAD_MAX_BYTES:
            return None, {}, _pattern_upload_metadata_error(file_size, "PNG", 1, 1)

        uploaded_file.seek(0)
        upload_bytes = uploaded_file.read()
        if len(upload_bytes) != int(file_size):
            return None, {}, "图像字节长度与上传元数据不一致。"
        upload_sha256 = hashlib.sha256(upload_bytes).hexdigest()
        probe = Image.open(io.BytesIO(upload_bytes))
        width, height = probe.size
        image_format = str(probe.format or "").upper()
        error = _pattern_upload_metadata_error(
            file_size, image_format, width, height)
        if error:
            return None, {
                "file_size": int(file_size), "format": image_format,
                "width": int(width), "height": int(height),
            }, error
        probe.verify()

        image = ImageOps.exif_transpose(Image.open(io.BytesIO(upload_bytes))).convert("RGB")
        return image, {
            "file_size": int(file_size), "format": image_format,
            "width": int(image.width), "height": int(image.height),
            "upload_sha256": upload_sha256,
        }, ""
    except Exception as exc:
        return None, {}, f"图像无法通过完整性校验：{type(exc).__name__}。"

# ===================== Constants & Helpers =====================
# D65 imported from color_utils

# CIE 1931 data imported from color_utils


# Engine module imported from engine.py
from engine import (
    MaterialLibrary, MetaSurfaceParam, DualPillarParam,
    _single_pillar_complex, MetaSurfaceColorEngine,
)

_PILLAR_MATERIAL_OPTIONS = tuple(MaterialLibrary.pillar_materials())
_SUBSTRATE_OPTIONS = tuple(MaterialLibrary.substrate_materials())
_POLARIZATION_OPTIONS = ('TE (s-pol)', 'TM (p-pol)')
_FP_MIRROR_OPTIONS = ('介质 DBR (TiO2/SiO2)', '金属 Ag (减色)')

_NUMERIC_CONTROLS = {
    "angle": NumericControlSpec(
        "a_val", ("angle_slider", "angle_input"), 0.0, 80.0, 0.0),
    "theta_obs": NumericControlSpec(
        "theta_obs", ("theta_obs_slider", "theta_obs_input"), 0.0, 80.0, 0.0),
    "na": NumericControlSpec(
        "na_val", ("na_slider", "na_input"), 0.05, 0.95, 0.1),
    "single_d": NumericControlSpec(
        "d_val", ("single_d_slider", "single_d_input"), 50.0, 350.0, 180.0),
    "single_h": NumericControlSpec(
        "h_val", ("single_h_slider", "single_h_input"), 80.0, 600.0, 300.0),
    "period": NumericControlSpec(
        "p_val", (
            "single_p_slider", "single_p_input",
            "dual_p_slider", "dual_p_input",
        ), 200.0, 600.0, 400.0),
    "dual_d1": NumericControlSpec(
        "d1_val", ("dual_d1_slider", "dual_d1_input"), 50.0, 350.0, 120.0),
    "dual_h1": NumericControlSpec(
        "h1_val", ("dual_h1_slider", "dual_h1_input"), 80.0, 600.0, 250.0),
    "dual_d2": NumericControlSpec(
        "d2_val", ("dual_d2_slider", "dual_d2_input"), 50.0, 350.0, 200.0),
    "dual_h2": NumericControlSpec(
        "h2_val", ("dual_h2_slider", "dual_h2_input"), 80.0, 600.0, 350.0),
    "fp_t": NumericControlSpec(
        "fp_t_val", ("fp_t_slider", "fp_t_input"), 50.0, 600.0, 200.0),
    "fp_center": NumericControlSpec(
        "fp_target_wl", ("fp_center_slider", "fp_center_input"),
        380.0, 780.0, 450.0),
}
_ENUM_CONTROLS = {
    "structure": EnumControlSpec(
        "structure_type", "structure_type_control", ('single', 'dual', 'fp'),
        'single', ('单柱', '双柱', 'FP 腔（Fabry-Pérot）')),
    "material": EnumControlSpec(
        "_pillar_material_pref", "pillar_material_control",
        _PILLAR_MATERIAL_OPTIONS, _PILLAR_MATERIAL_OPTIONS[1]),
    "substrate": EnumControlSpec(
        "_substrate_pref", "substrate_control",
        _SUBSTRATE_OPTIONS, _SUBSTRATE_OPTIONS[0]),
    "polarization": EnumControlSpec(
        "polarization", "polarization_control",
        _POLARIZATION_OPTIONS, _POLARIZATION_OPTIONS[0]),
    "fp_mirror": EnumControlSpec(
        "fp_mirror_type", "fp_mirror_type_control",
        _FP_MIRROR_OPTIONS, _FP_MIRROR_OPTIONS[0]),
}
_BOOL_CONTROLS = {
    "far_field": BoolControlSpec("far_field", "far_field_control", False),
    "ml_accel": BoolControlSpec("ml_accel", "ml_accel_control", False),
}

# This is a deterministic, exact-match reference example for the competition
# demo. It is intentionally a UI preset, not a recommendation or a training
# sample. The reference library remains the authority for the stored color.
_AUDITED_REFERENCE_SAMPLE = {
    "material": "TiO2 (anatase)",
    "substrate": "SiO2 (fused silica)",
    "polarization": "TM (p-pol)",
    "angle_deg": 0.0,
    "diameter_nm": 140.0,
    "height_nm": 281.0,
    "period_nm": 407.0,
    "target_hex": "#242d49",
}


def _load_audited_reference_sample() -> None:
    """Load the exact geometry used by the audited competition reference set."""
    set_enum_value(st.session_state, _ENUM_CONTROLS["structure"], "single")
    set_enum_value(st.session_state, _ENUM_CONTROLS["material"], _AUDITED_REFERENCE_SAMPLE["material"])
    set_enum_value(st.session_state, _ENUM_CONTROLS["substrate"], _AUDITED_REFERENCE_SAMPLE["substrate"])
    set_enum_value(st.session_state, _ENUM_CONTROLS["polarization"], _AUDITED_REFERENCE_SAMPLE["polarization"])
    set_numeric_value(st.session_state, _NUMERIC_CONTROLS["angle"], _AUDITED_REFERENCE_SAMPLE["angle_deg"])
    set_numeric_value(st.session_state, _NUMERIC_CONTROLS["single_d"], _AUDITED_REFERENCE_SAMPLE["diameter_nm"])
    set_numeric_value(st.session_state, _NUMERIC_CONTROLS["single_h"], _AUDITED_REFERENCE_SAMPLE["height_nm"])
    set_numeric_value(st.session_state, _NUMERIC_CONTROLS["period"], _AUDITED_REFERENCE_SAMPLE["period_nm"])
    set_bool_value(st.session_state, _BOOL_CONTROLS["far_field"], False)
    # Keep this demonstration on the transparent analytical route so the
    # comparison visibly separates the current route from the audited RCWA.
    set_bool_value(st.session_state, _BOOL_CONTROLS["ml_accel"], False)
    # The color picker is created later in the script.  It reads this existing
    # session value without also receiving a widget default, which avoids
    # Streamlit's "value set via Session State API" warning on callbacks.
    st.session_state["inverse_target_picker"] = _AUDITED_REFERENCE_SAMPLE["target_hex"]
    st.session_state["_audited_sample_notice"] = (
        "已加载可复核示例：TiO2/SiO2/air · TM · 0° · "
        "D/H/P=140/281/407 nm。展开“高保真参考对照”查看精确命中结果。"
    )
    st.session_state["_expand_reference_recheck_once"] = True
    _clear_inverse_results()

initialize_bool_preference_marker(
    st.session_state, _BOOL_CONTROLS["ml_accel"],
    ML_ACCEL_PREFERENCE_INITIALIZED_KEY,
)
migrate_session_state(
    st.session_state, tuple(_NUMERIC_CONTROLS.values()),
    tuple(_ENUM_CONTROLS.values()), tuple(_BOOL_CONTROLS.values()),
)

# ===================== UI appearance =====================
# Keep appearance preferences in this Streamlit session.  They are deliberately
# separate from the physical/model controls so changing a theme never changes
# a simulation input or invalidates a computed result.
_UI_THEME_OPTIONS = {
    "深色": "dark",
    "浅色": "light",
    "高对比": "contrast",
}
_UI_ACCENT_OPTIONS = {
    "琥珀橙": "amber",
    "靛蓝": "indigo",
    "青绿色": "teal",
    "紫罗兰": "violet",
}
_UI_THEME_BASE = {
    "dark": {
        "bg_deep": "#0C0812", "bg_surface": "#18121E", "bg_elevated": "#251F2B",
        "border_subtle": "#342D3C", "border_strong": "#51495C",
        "text_primary": "#F3F5F5", "text_secondary": "#AFB1B2", "text_muted": "#888B8C",
        "scrim": "rgba(12, 8, 18, .78)", "preview_start": "#1A1A2E", "preview_end": "#16213E",
        "substrate_start": "#3A3A5C", "substrate_end": "#252540", "highlight_surface": "#2A1C12",
        "button_bg": "#251F2B", "button_text": "#F3F5F5", "accent_ink": "#160D05", "code_bg": "#100C16",
        "tooltip_bg": "#251F2B", "tooltip_text": "#F3F5F5", "tooltip_border": "#51495C",
        "icon_color": "#AFB1B2", "status_good": "#5BD48A", "status_warn": "#F3C969", "status_bad": "#FF8B82",
    },
    "light": {
        "bg_deep": "#F4F7FB", "bg_surface": "#FFFFFF", "bg_elevated": "#EEF2F7",
        "border_subtle": "#D9E1EC", "border_strong": "#B7C4D5",
        "text_primary": "#182230", "text_secondary": "#475569", "text_muted": "#64748B",
        "scrim": "rgba(15, 23, 42, .72)", "preview_start": "#E8EEF8", "preview_end": "#DCE6F5",
        "substrate_start": "#A8B6CB", "substrate_end": "#7E8DA6", "highlight_surface": "#FFF4E8",
        "button_bg": "#FFFFFF", "button_text": "#182230", "accent_ink": "#FFFFFF", "code_bg": "#F8FAFC",
        "tooltip_bg": "#FFFFFF", "tooltip_text": "#182230", "tooltip_border": "#B7C4D5",
        "icon_color": "#475569", "status_good": "#15803D", "status_warn": "#B45309", "status_bad": "#B91C1C",
    },
    "contrast": {
        "bg_deep": "#000000", "bg_surface": "#0B0B0B", "bg_elevated": "#171717",
        "border_subtle": "#707070", "border_strong": "#D4D4D4",
        "text_primary": "#FFFFFF", "text_secondary": "#F5F5F5", "text_muted": "#D4D4D4",
        "scrim": "rgba(0, 0, 0, .84)", "preview_start": "#111111", "preview_end": "#252525",
        "substrate_start": "#666666", "substrate_end": "#333333", "highlight_surface": "#3A2B00",
        "button_bg": "#111111", "button_text": "#FFFFFF", "accent_ink": "#000000", "code_bg": "#050505",
        "tooltip_bg": "#0B0B0B", "tooltip_text": "#FFFFFF", "tooltip_border": "#D4D4D4",
        "icon_color": "#FFFFFF", "status_good": "#00FF85", "status_warn": "#FFD000", "status_bad": "#FF6B6B",
    },
}
_UI_ACCENT_PALETTE = {
    "amber": {
        "dark": {"accent": "#FDA765", "accent_strong": "#FC6B2E", "accent_soft": "#F3D259", "accent_cool": "#C191F5"},
        "light": {"accent": "#B45309", "accent_strong": "#C2410C", "accent_soft": "#9A3412", "accent_cool": "#6D28D9"},
        "contrast": {"accent": "#FFD000", "accent_strong": "#FF7A00", "accent_soft": "#FFFF00", "accent_cool": "#D8B4FE"},
    },
    "indigo": {
        "dark": {"accent": "#8AB4FF", "accent_strong": "#4F8CFF", "accent_soft": "#B9D3FF", "accent_cool": "#C4B5FD"},
        "light": {"accent": "#2563EB", "accent_strong": "#1D4ED8", "accent_soft": "#6D28D9", "accent_cool": "#0E7490"},
        "contrast": {"accent": "#8AB4FF", "accent_strong": "#4F8CFF", "accent_soft": "#D9E7FF", "accent_cool": "#C4B5FD"},
    },
    "teal": {
        "dark": {"accent": "#65D6C5", "accent_strong": "#14B8A6", "accent_soft": "#99F6E4", "accent_cool": "#7DD3FC"},
        "light": {"accent": "#0F766E", "accent_strong": "#0D9488", "accent_soft": "#115E59", "accent_cool": "#0369A1"},
        "contrast": {"accent": "#5EEAD4", "accent_strong": "#2DD4BF", "accent_soft": "#CCFBF1", "accent_cool": "#BAE6FD"},
    },
    "violet": {
        "dark": {"accent": "#C4B5FD", "accent_strong": "#8B5CF6", "accent_soft": "#DDD6FE", "accent_cool": "#93C5FD"},
        "light": {"accent": "#7C3AED", "accent_strong": "#6D28D9", "accent_soft": "#6B21A8", "accent_cool": "#2563EB"},
        "contrast": {"accent": "#C4B5FD", "accent_strong": "#A78BFA", "accent_soft": "#EDE9FE", "accent_cool": "#BFDBFE"},
    },
}

st.session_state.setdefault("ui_theme_control", "深色")
st.session_state.setdefault("ui_accent_control", "琥珀橙")

# Render these controls before the page CSS is emitted.  Streamlit reruns the
# script after a selection, so the current values are available to the CSS in
# the same render and persist for the rest of the session.
with st.sidebar:
    st.markdown("### 🎨 外观")
    st.segmented_control(
        "界面主题",
        options=list(_UI_THEME_OPTIONS),
        key="ui_theme_control",
        help="只改变界面颜色，不改变材料、结构参数或计算结果。",
    )
    st.segmented_control(
        "强调色",
        options=list(_UI_ACCENT_OPTIONS),
        key="ui_accent_control",
        help="选择按钮、边框和重点信息的强调色。",
    )
    st.caption("主题偏好仅保存在当前浏览器会话。")
    st.divider()

_UI_THEME_MODE = _UI_THEME_OPTIONS.get(
    st.session_state.get("ui_theme_control", "深色"), "dark")
_UI_ACCENT_MODE = _UI_ACCENT_OPTIONS.get(
    st.session_state.get("ui_accent_control", "琥珀橙"), "amber")
_UI_STYLE = dict(_UI_THEME_BASE[_UI_THEME_MODE])
_UI_STYLE.update(_UI_ACCENT_PALETTE[_UI_ACCENT_MODE][_UI_THEME_MODE])

# ===================== Streamlit UI =====================
def get_engine():
    """Return the mutable engine owned by this Streamlit session only."""
    # FP-TMM does not use the single-column grid.  Clear the mutable grid when
    # crossing a structure route boundary so a ~60k-row single-column grid
    # cannot survive into an FP session or be reused accidentally on return.
    # Keep the session engine object itself stable: analysis snapshots and
    # Streamlit tests rely on one session-owned engine identity.
    structure_mode = str(st.session_state.get("structure_type", "single"))
    previous_mode = st.session_state.get("_ui_engine_structure_mode")
    if previous_mode != structure_mode:
        existing_engine = st.session_state.get(ENGINE_SESSION_KEY)
        if existing_engine is not None:
            existing_engine.grid_params = np.zeros((0, 3))
            existing_engine.grid_rgb = np.zeros((0, 3))
            existing_engine.grid_lab = np.zeros((0, 3))
            existing_engine.grid_xy = np.zeros((0, 2))
            existing_engine._grid_library_initialized = False
            existing_engine._ui_library_identity = None
        st.session_state.pop(ENGINE_LIBRARY_KEY, None)
        st.session_state["_ui_engine_structure_mode"] = structure_mode
    initialize_grid = st.session_state.get("structure_type") != "fp"
    return get_session_engine(
        st.session_state,
        lambda: MetaSurfaceColorEngine(
            initialize_grid_library=initialize_grid),
    )

try:
    engine = get_engine()
except Exception as e:
    st.error(f"Engine init failed: {e}")
    import traceback; st.code(traceback.format_exc())
    st.stop()

# Lazy ML init - only load models when user enables ML acceleration
_ml_ready = False
_ml_is_v8 = False
_dual_ml_ready = False
_ml_tried = False
_dual_ml_tried = False
_ml_error = ""
_dual_ml_error = ""
_ml_state = {"state": "not_selected", "file_present": False, "loaded": False, "call_error": ""}
_dual_ml_state = {"state": "not_selected", "file_present": False, "loaded": False, "call_error": ""}
_rcwa_ml_ready = False
_rcwa_ml_tried = False
_rcwa_ml_error = ""
_primary_runtime_binding = None
_dual_runtime_binding = None
_rcwa_runtime_binding = None


_ANALYSIS_SOURCE_DEPENDENCIES = {
    "dual_ml": (
        "ui_forward_routes.py", "ml_module.py", "engine.py", "color_utils.py",
    ),
    "dual_physical": (
        "ui_forward_routes.py", "torch_model.py", "engine.py", "ccm.py",
        "color_utils.py",
    ),
    "fp_tmm": (
        "ui_forward_routes.py", "fp_cavity.py", "engine.py", "color_utils.py",
    ),
    "single_generic": (
        "ml_module.py", "ui_model_difference_contracts.py",
        "ui_forward_routes.py", "color_utils.py",
    ),
    "single_rcwa": (
        "ml_module.py", "ui_forward_routes.py", "color_utils.py",
    ),
    "single_physical": (
        "ui_forward_routes.py", "torch_model.py", "engine.py", "ccm.py",
        "color_utils.py",
    ),
}


def _local_model_exists(relative_path):
    return os.path.isfile(os.path.join(os.path.dirname(__file__), relative_path))


def _ui_code_artifact_identity(relative_paths):
    """Read current small route source bytes; never load a model/checkpoint."""
    dependencies = (
        "app.py", "ui_analysis_snapshots.py", "ui_model_resources.py",
        *tuple(relative_paths),
    )
    return source_artifact_identity(
        os.path.dirname(__file__), tuple(dict.fromkeys(dependencies)))


def _ui_code_artifact_version(relative_paths):
    return _ui_code_artifact_identity(relative_paths).version


def _combined_analysis_artifact(*identities, **fields):
    versions = [identity.version for identity in identities]
    available = bool(identities) and all(identity.available for identity in identities)
    prefix = "sha256:" if available else "unavailable:"
    return prefix + canonical_sha256({
        "artifact_identities": versions,
        **fields,
    })


def _artifact_identity_available(version):
    return not str(version).startswith("unavailable:")


def _project_relative_model_path(path):
    try:
        root = os.path.realpath(os.path.dirname(__file__))
        resolved = os.path.realpath(str(path))
        relative = os.path.relpath(resolved, root).replace("\\", "/")
    except (OSError, RuntimeError, ValueError):
        return ""
    if not resolved or relative == ".." or relative.startswith("../"):
        return ""
    return relative


def _session_model_paths(sessions):
    paths = []
    for session in tuple(sessions):
        relative = _project_relative_model_path(
            getattr(session, "_model_path", ""))
        if not relative:
            return ()
        paths.append(relative)
    return tuple(sorted(dict.fromkeys(paths)))


def _primary_model_disk_identity():
    contract = model_difference_contracts.GENERIC_ONNX_ROUTE
    paths = (contract.model_relative_path, contract.external_data_relative_path)
    return source_artifact_identity(
        os.path.dirname(__file__), paths, max_bytes=64 * 1024 * 1024,
        expected_sha256={
            contract.model_relative_path: contract.model_sha256,
            contract.external_data_relative_path: contract.external_data_sha256,
        },
    )


def _primary_resource_context_key():
    contract = model_difference_contracts.GENERIC_ONNX_ROUTE
    return "generic:" + canonical_sha256({
        "route": contract.route_id,
        "graph": contract.model_relative_path,
        "external_data": contract.external_data_relative_path,
    })


def _dual_resource_context_key():
    return "dual:" + canonical_sha256({
        "route": "dual_mlp", "model": _DUAL_MODEL_RELATIVE_PATH,
    })


def _dual_model_disk_identity():
    return source_artifact_identity(
        os.path.dirname(__file__), (_DUAL_MODEL_RELATIVE_PATH,),
        max_bytes=64 * 1024 * 1024,
    )


def _registered_rcwa_model_paths(material, substrate):
    """Resolve only the exact route family: wavelength model, else exact ensemble."""
    material = str(material)
    substrate = str(substrate)
    wavelength_name = getattr(ml_module, "_RCWA_WL_MODELS", {}).get(material)
    if wavelength_name:
        return (f"models/{str(wavelength_name).replace(chr(92), '/')}",)
    registry = getattr(ml_module, "_RCWA_SUBSTRATE_MODELS", {})
    patterns = registry.get((material, substrate))
    if patterns is None:
        patterns = getattr(ml_module, "_RCWA_MODELS", {}).get(material, ())
    root = os.path.dirname(__file__)
    paths = []
    for pattern in tuple(patterns or ()):
        pattern = str(pattern).replace("\\", "/")
        relative_pattern = f"models/{pattern}"
        if any(token in pattern for token in ("*", "?", "[")):
            matches = sorted(glob.glob(os.path.join(root, *relative_pattern.split("/"))))
            paths.extend(
                relative for match in matches
                if (relative := _project_relative_model_path(match))
            )
        elif os.path.isfile(os.path.join(root, *relative_pattern.split("/"))):
            paths.append(relative_pattern)
    return tuple(sorted(dict.fromkeys(paths)))


def _rcwa_resource_context(material, substrate):
    material = str(material)
    substrate = str(substrate)
    paths = _registered_rcwa_model_paths(material, substrate)
    if getattr(ml_module, "_RCWA_WL_MODELS", {}).get(material):
        selection_kind = "wavelength"
    elif (material, substrate) in getattr(
            ml_module, "_RCWA_SUBSTRATE_MODELS", {}):
        selection_kind = "exact_pair"
    else:
        selection_kind = "material_fallback"
    context_key = "rcwa:" + canonical_sha256({
        "material": material, "substrate": substrate,
        "selection_kind": selection_kind, "registered_paths": paths,
    })
    return context_key, selection_kind, paths


def _rcwa_model_disk_identity(material, substrate):
    paths = _registered_rcwa_model_paths(material, substrate)
    return source_artifact_identity(
        os.path.dirname(__file__), paths, max_bytes=64 * 1024 * 1024)


def _current_rcwa_sessions(material, substrate):
    wavelength = getattr(ml_module, "_RCWA_WL_SESSIONS", {}).get(str(material))
    if wavelength is not None:
        return (wavelength,)
    return tuple(getattr(ml_module, "_get_rcwa_sessions", lambda *_args: ())(
        str(material), str(substrate)))


def _capture_runtime_binding(
        family, context_key, requested_identity, registered_paths, sessions,
        current_identity):
    registered_paths = tuple(sorted(dict.fromkeys(registered_paths)))
    if not requested_identity.available:
        return None, requested_identity.reason or "requested model identity unavailable"
    if not current_identity.available:
        return None, current_identity.reason or "loaded model identity unavailable"
    session_paths = tuple(
        _project_relative_model_path(getattr(session, "_model_path", ""))
        for session in tuple(sessions))
    expected_session_paths = (
        (registered_paths[0],) if family == "primary" else registered_paths)
    if (
        not session_paths or any(not path for path in session_paths)
        or tuple(sorted(session_paths)) != tuple(sorted(expected_session_paths))
    ):
        return None, "loaded session path does not match the registered route"
    if current_identity.version != requested_identity.version:
        return None, "model bytes changed while the session was loading"
    try:
        candidate = BoundModelResource.create(
            str(family), str(context_key), current_identity.version, registered_paths,
            session_paths, tuple(sessions))
    except ValueError as exc:
        return None, str(exc)
    resource = register_first_resource(candidate)
    if (
        resource.loaded_identity != candidate.loaded_identity
        or resource.context_key != candidate.context_key
        or resource.session_ids != candidate.session_ids
        or resource.session_paths != candidate.session_paths
    ):
        return resource, "another runtime session is already bound for this route"
    return resource, ""


def _runtime_binding_issue(
        binding, current_identity, sessions, expected_session_paths,
        expected_context_key=None):
    issue = validate_bound_resource(
        binding, current_available=current_identity.available,
        current_identity=current_identity.version,
        current_reason=current_identity.reason,
        expected_context_key=expected_context_key)
    if issue:
        return issue
    expected_paths = tuple(sorted(dict.fromkeys(expected_session_paths)))
    if binding.registered_paths != expected_paths:
        return "bound resource paths differ from the registered model route"
    active = tuple(sessions)
    if (
        len(active) != len(binding.session_refs)
        or any(actual is not bound for actual, bound in zip(
            active, binding.session_refs))
    ):
        return "active runtime sessions differ from the bound resource"
    return ""


def _dual_model_artifact_identity():
    """Return current bytes only when they match the strong dual binding."""
    current = _dual_model_disk_identity()
    issue = validate_bound_resource(
        _dual_runtime_binding, current_available=current.available,
        current_identity=current.version, current_reason=current.reason,
        expected_context_key=_dual_resource_context_key())
    if issue:
        return SourceArtifactIdentity(
            False, "unavailable:" + canonical_sha256({
                "route": "dual", "reason": issue,
            }), issue, current.dependencies,
        )
    return SourceArtifactIdentity(
        True, _dual_runtime_binding.loaded_identity, "", current.dependencies)


def _rcwa_model_artifact_identity(material, substrate):
    """Return current exact-family bytes only when they match the strong binding."""
    current = _rcwa_model_disk_identity(material, substrate)
    issue = validate_bound_resource(
        _rcwa_runtime_binding, current_available=current.available,
        current_identity=current.version, current_reason=current.reason,
        expected_context_key=_rcwa_resource_context(material, substrate)[0])
    if issue:
        return SourceArtifactIdentity(
            False,
            "unavailable:" + canonical_sha256({
                "route": "rcwa_surrogate", "reason": issue,
            }),
            issue, current.dependencies,
        )
    return SourceArtifactIdentity(
        True, _rcwa_runtime_binding.loaded_identity, "", current.dependencies)


def _primary_model_artifact_identity():
    current = _primary_model_disk_identity()
    issue = validate_bound_resource(
        _primary_runtime_binding, current_available=current.available,
        current_identity=current.version, current_reason=current.reason,
        expected_context_key=_primary_resource_context_key())
    if issue:
        return SourceArtifactIdentity(
            False, "unavailable:" + canonical_sha256({
                "route": "generic", "reason": issue,
            }), issue, current.dependencies)
    return SourceArtifactIdentity(
        True, _primary_runtime_binding.loaded_identity, "", current.dependencies)


def _analysis_artifact_version(
        route_id, model_version, *, structure_type=None, material=None, substrate=None):
    """Return a route- and structure-specific immutable analysis identity."""
    route = str(route_id)
    structure = str(structure_type if structure_type is not None else globals().get("_structure_type", "single"))
    if structure == "dual" and route in {"ml_surrogate", "dual ML surrogate"}:
        dual_identity = _dual_model_artifact_identity()
        return _combined_analysis_artifact(
            _ui_code_artifact_identity(_ANALYSIS_SOURCE_DEPENDENCIES["dual_ml"]),
            dual_identity,
            structure_type="dual", route_id=route,
            model_version=str(model_version), dual_model=dual_identity.version,
        )
    if structure == "dual" and route in {
        "lorentz_fano_fallback", "dual physical fallback",
    }:
        return _ui_code_artifact_version(
            _ANALYSIS_SOURCE_DEPENDENCIES["dual_physical"])
    if structure == "dual" and route == "far_field_postprocessing":
        return _ui_code_artifact_version(
            _ANALYSIS_SOURCE_DEPENDENCIES["dual_physical"])
    if structure == "fp" and route in {"fp_tmm", "FP cavity TMM"}:
        return _ui_code_artifact_version(_ANALYSIS_SOURCE_DEPENDENCIES["fp_tmm"])
    if structure == "single" and route in {
        "ml_surrogate", "generic-fano-resmlp-v8-sub-onnx-only",
        "ML surrogate (generic angle-conditioned)",
    }:
        contract = model_difference_contracts.GENERIC_ONNX_ROUTE
        code_identity = _ui_code_artifact_identity(
            _ANALYSIS_SOURCE_DEPENDENCIES["single_generic"])
        bundle_identity = _generic_bundle_artifact_identity(contract)
        runtime_identity = _primary_model_artifact_identity()
        return _combined_analysis_artifact(
            code_identity, bundle_identity, runtime_identity,
            model=contract.model_sha256,
            external_data=contract.external_data_sha256,
            source_pt=contract.source_pt_sha256,
            conversion_result=contract.conversion_result_sha256,
            loaded_model_identity=runtime_identity.version,
        )
    if structure == "single" and route == "rcwa_surrogate":
        code_identity = _ui_code_artifact_identity(
            _ANALYSIS_SOURCE_DEPENDENCIES["single_rcwa"])
        model_identity = _rcwa_model_artifact_identity(material, substrate)
        return _combined_analysis_artifact(
            code_identity, model_identity,
            registry=getattr(ml_module, "_RCWA_MODELS", {}),
            substrate_registry=getattr(ml_module, "_RCWA_SUBSTRATE_MODELS", {}),
            wavelength_registry=getattr(ml_module, "_RCWA_WL_MODELS", {}),
            loaded_model_version=str(model_version),
            loaded_model_identity=model_identity.version,
        )
    if structure == "single" and route in {"physical/far-field fallback", "far_field_postprocessing"}:
        return _ui_code_artifact_version(
            _ANALYSIS_SOURCE_DEPENDENCIES["single_physical"])
    if structure == "single" and route == "lorentz_fano_fallback":
        return _ui_code_artifact_version(
            _ANALYSIS_SOURCE_DEPENDENCIES["single_physical"])
    return "unavailable:" + canonical_sha256({
        "structure_type": structure, "route_id": route,
        "model_version": str(model_version), "reason": "unregistered_analysis_route",
    })


def _generic_bundle_artifact_identity(contract):
    """Hash current generic bundle bytes and reject model files not matching evidence."""
    paths = (
        contract.model_relative_path,
        contract.external_data_relative_path,
        contract.source_pt_relative_path,
        contract.conversion_protocol_relative_path,
        contract.conversion_result_relative_path,
    )
    expected = {
        contract.model_relative_path: contract.model_sha256,
        contract.external_data_relative_path: contract.external_data_sha256,
        contract.source_pt_relative_path: contract.source_pt_sha256,
    }
    identity = source_artifact_identity(
        os.path.dirname(__file__), paths, max_bytes=64 * 1024 * 1024,
        expected_sha256=expected,
    )
    if not identity.available:
        return identity
    evidence, reason = model_difference_contracts.load_conversion_evidence(
        os.path.dirname(__file__), contract)
    if evidence is not None:
        return identity
    return SourceArtifactIdentity(
        False,
        "unavailable:" + canonical_sha256({
            "bundle_identity": identity.version, "evidence_error": str(reason),
        }),
        str(reason), identity.dependencies,
    )


def _difference_analysis_artifact_identity(contract):
    code_identity = _ui_code_artifact_identity((
        "ui_forward_routes.py", "ui_model_difference_contracts.py",
        "torch_model.py", "ccm.py", "color_utils.py",
    ))
    bundle_identity = _generic_bundle_artifact_identity(contract)
    version = _combined_analysis_artifact(
        code_identity, bundle_identity,
        graph=contract.model_sha256,
        external_data=contract.external_data_sha256,
        source_pt=contract.source_pt_sha256,
        conversion_result=contract.conversion_result_sha256,
    )
    return version, code_identity, bundle_identity


@st.cache_resource(show_spinner=False)
def _load_model_difference_evaluator_cached(artifact_version):
    """Freeze the exact local generic ONNX route used by model comparison."""
    del artifact_version  # Streamlit cache key binds the frozen evaluator to source identity.
    return model_difference_contracts.freeze_generic_onnx_evaluator(
        os.path.dirname(__file__))


def _model_state(relative_path, *, loaded=False, call_error="", selected=False):
    """Return explicit local-file/load/call state for provenance and UI."""
    present = _local_model_exists(relative_path) if relative_path else False
    if call_error:
        state = "call_failed"
    elif loaded:
        state = "loaded"
    elif present:
        state = "file_present_load_failed"
    elif selected:
        state = "file_missing"
    else:
        state = "not_selected"
    return {
        "state": state,
        "file_present": bool(present),
        "loaded": bool(loaded),
        "call_error": str(call_error or ""),
    }


def _model_provenance(load_state, *, called=False, call_error="", state_override=None):
    """Derive internally consistent provenance fields from load/call state."""
    load_state = dict(load_state or {})
    present = bool(load_state.get("file_present", False))
    loaded = bool(load_state.get("loaded", False))
    error = str(call_error or load_state.get("call_error", "") or "")
    if state_override:
        state = str(state_override)
    elif error:
        state = "call_failed"
    elif called and loaded:
        state = "loaded_and_called"
    else:
        state = str(load_state.get("state", "not_selected"))
    return {
        "model_state": state,
        "model_file_present": present,
        "model_loaded": loaded,
        "model_call_error": error,
    }


def _load_with_restored_globals(init_fn, attribute_names):
    """Capture one loader result while leaving process globals exactly as found."""
    with resource_lock():
        originals = {
            name: getattr(ml_module, name) for name in tuple(attribute_names)
        }
        try:
            ok = bool(init_fn())
            loaded = {name: getattr(ml_module, name) for name in originals}
            return ok, loaded
        finally:
            for name, value in originals.items():
                setattr(ml_module, name, value)


@st.cache_resource(show_spinner=False)
def _load_primary_ml_cached(artifact_version):
    """Load one primary session keyed by the verified graph+data identity."""
    requested = _primary_model_disk_identity()
    graph_path = model_difference_contracts.GENERIC_ONNX_ROUTE.model_relative_path
    context_key = _primary_resource_context_key()
    if not requested.available or requested.version != str(artifact_version):
        return False, False, "本地 ONNX 身份不可用或已变化", _model_state(
            graph_path, selected=True), None, None
    with resource_lock():
        existing = get_bound_resource("primary", context_key)
        if existing is not None:
            issue = validate_bound_resource(
                existing, current_available=requested.available,
                current_identity=requested.version, current_reason=requested.reason)
            ok = not issue
            return (
                ok, bool(getattr(ml_module, "_ORT_IS_V8", False)), issue,
                _model_state(graph_path, loaded=ok, call_error=issue),
                existing.session_refs[0] if existing.session_refs else None,
                existing,
            )
    try:
        ok, loaded = _load_with_restored_globals(
            ml_module.init_ml, ("_ORT_SESSION", "_ORT_AVAILABLE", "_ORT_IS_V8"))
        session = loaded["_ORT_SESSION"]
        current = _primary_model_disk_identity()
        binding, binding_issue = _capture_runtime_binding(
            "primary", context_key, requested, (graph_path,),
            (session,) if session is not None else (), current)
        ok = bool(ok and binding is not None and not binding_issue)
        error = "" if ok else (binding_issue or "ONNX Runtime 初始化失败")
        return ok, bool(loaded["_ORT_IS_V8"]), error, _model_state(
            graph_path, loaded=ok, call_error="" if ok else error), session, binding
    except Exception as exc:
        return False, False, f"ONNX Runtime 初始化失败: {type(exc).__name__}", _model_state(
            graph_path, call_error=type(exc).__name__), None, None


@st.cache_resource(show_spinner=False)
def _load_dual_ml_cached(artifact_version):
    """Load only the ONNX asset used by the actual dual-pillar runtime."""
    model_path = _DUAL_MODEL_RELATIVE_PATH
    context_key = _dual_resource_context_key()
    requested = _dual_model_disk_identity()
    if not requested.available or requested.version != str(artifact_version):
        return False, f"本地双柱 ONNX 身份不可用：{model_path}", _model_state(
            model_path, selected=True), None, None
    with resource_lock():
        existing = get_bound_resource("dual", context_key)
        if existing is not None:
            issue = validate_bound_resource(
                existing, current_available=requested.available,
                current_identity=requested.version, current_reason=requested.reason)
            ok = not issue
            return (
                ok, issue, _model_state(
                    model_path, loaded=ok, call_error=issue),
                existing.session_refs[0] if existing.session_refs else None,
                existing,
            )
    try:
        ok, loaded = _load_with_restored_globals(
            ml_module.init_dual_ml,
            ("_DUAL_ORT_SESSION", "_DUAL_ORT_AVAILABLE", "_DUAL_IS_V3"))
        session = loaded["_DUAL_ORT_SESSION"]
        current = _dual_model_disk_identity()
        binding, binding_issue = _capture_runtime_binding(
            "dual", context_key, requested, (model_path,),
            (session,) if session is not None else (), current)
        ok = bool(ok and binding is not None and not binding_issue)
        error = "" if ok else (binding_issue or "双柱 ONNX 初始化失败")
        return ok, error, _model_state(
            model_path, loaded=ok, call_error="" if ok else error), session, binding
    except Exception as exc:
        return False, f"双柱 ONNX 初始化失败: {type(exc).__name__}", _model_state(
            model_path, call_error=type(exc).__name__), None, None


@st.cache_resource(show_spinner=False)
def _load_rcwa_registry_cached(artifact_version, material, substrate):
    """Load only the exact RCWA family selected for this route identity."""
    requested = _rcwa_model_disk_identity(material, substrate)
    context_key, _selection_kind, paths = _rcwa_resource_context(material, substrate)
    family = "rcwa"
    if not requested.available or requested.version != str(artifact_version):
        return False, "RCWA exact model identity unavailable", (), False, None
    with resource_lock():
        existing = get_bound_resource(family, context_key)
        if existing is not None:
            issue = validate_bound_resource(
                existing, current_available=requested.available,
                current_identity=requested.version, current_reason=requested.reason)
            is_wavelength = bool(
                getattr(ml_module, "_RCWA_WL_MODELS", {}).get(str(material)))
            return not issue, issue, existing.session_refs, is_wavelength, existing
    try:
        import onnxruntime as ort
        sessions = tuple(
            ort.InferenceSession(
                os.path.join(os.path.dirname(__file__), *path.split("/")),
                providers=["CPUExecutionProvider"])
            for path in paths)
        current = _rcwa_model_disk_identity(material, substrate)
        is_wavelength = bool(
            getattr(ml_module, "_RCWA_WL_MODELS", {}).get(str(material)))
        binding, binding_issue = _capture_runtime_binding(
            family, context_key, requested, paths, sessions, current)
        ok = bool(sessions and binding is not None and not binding_issue)
        return ok, "" if ok else (binding_issue or "本地未发现可加载的 RCWA 代理模型"), sessions, is_wavelength, binding
    except Exception as exc:
        return False, f"RCWA exact registry 初始化失败: {type(exc).__name__}", (), False, None


def _ensure_ml():
    global _ml_ready, _ml_is_v8, _ml_tried, _ml_error, _ml_state, _primary_runtime_binding
    if _ml_tried: return _ml_ready
    _ml_tried = True
    identity = _primary_model_disk_identity()
    if not identity.available:
        _ml_ready, _ml_error = False, identity.reason
        _ml_state = _model_state(
            model_difference_contracts.GENERIC_ONNX_ROUTE.model_relative_path,
            selected=True, call_error=identity.reason)
        _primary_runtime_binding = None
        return False
    (
        _ml_ready, _ml_is_v8, _ml_error, _ml_state, session,
        _primary_runtime_binding,
    ) = _load_primary_ml_cached(identity.version)
    runtime_issue = _runtime_binding_issue(
        _primary_runtime_binding, identity,
        tuple(getattr(_primary_runtime_binding, "session_refs", ())),
        (model_difference_contracts.GENERIC_ONNX_ROUTE.model_relative_path,),
        _primary_resource_context_key())
    if runtime_issue:
        _ml_ready = False
        _ml_error = runtime_issue
        _ml_state = _model_state(
            model_difference_contracts.GENERIC_ONNX_ROUTE.model_relative_path,
            selected=True, call_error=runtime_issue)
    if _ml_error:
        logging.info("primary ML unavailable: %s", _ml_error)
    return _ml_ready

def _ensure_dual_ml():
    global _dual_ml_ready, _dual_ml_tried, _dual_ml_error, _dual_ml_state, _dual_runtime_binding
    if _dual_ml_tried: return _dual_ml_ready
    _dual_ml_tried = True
    identity = _dual_model_disk_identity()
    if not identity.available:
        _dual_ml_ready, _dual_ml_error = False, identity.reason
        _dual_ml_state = _model_state(
            _DUAL_MODEL_RELATIVE_PATH, selected=True, call_error=identity.reason)
        _dual_runtime_binding = None
        return False
    (
        _dual_ml_ready, _dual_ml_error, _dual_ml_state, session,
        _dual_runtime_binding,
    ) = _load_dual_ml_cached(identity.version)
    runtime_issue = _runtime_binding_issue(
        _dual_runtime_binding, identity,
        tuple(getattr(_dual_runtime_binding, "session_refs", ())),
        (_DUAL_MODEL_RELATIVE_PATH,), _dual_resource_context_key())
    if runtime_issue:
        _dual_ml_ready = False
        _dual_ml_error = runtime_issue
        _dual_ml_state = _model_state(
            _DUAL_MODEL_RELATIVE_PATH, selected=True, call_error=runtime_issue)
    if _dual_ml_error:
        logging.info("dual ML unavailable: %s", _dual_ml_error)
    return _dual_ml_ready

def _ensure_rcwa_ml(material=None, substrate=None):
    global _rcwa_ml_ready, _rcwa_ml_tried, _rcwa_ml_error, _rcwa_runtime_binding
    if _rcwa_ml_tried:
        return _rcwa_ml_ready
    _rcwa_ml_tried = True
    material = str(material if material is not None else globals().get("material", ""))
    substrate = str(substrate if substrate is not None else globals().get("substrate", ""))
    identity = _rcwa_model_disk_identity(material, substrate)
    if not identity.available:
        _rcwa_ml_ready, _rcwa_ml_error = False, identity.reason
        _rcwa_runtime_binding = None
        return False
    (
        _rcwa_ml_ready, _rcwa_ml_error, sessions, is_wavelength,
        _rcwa_runtime_binding,
    ) = _load_rcwa_registry_cached(identity.version, material, substrate)
    runtime_issue = _runtime_binding_issue(
        _rcwa_runtime_binding, identity,
        tuple(getattr(_rcwa_runtime_binding, "session_refs", ())),
        _registered_rcwa_model_paths(material, substrate),
        _rcwa_resource_context(material, substrate)[0])
    if runtime_issue:
        _rcwa_ml_ready = False
        _rcwa_ml_error = runtime_issue
    elif _rcwa_ml_ready and _rcwa_runtime_binding is not None:
        # Keep the exact loaded family visible to the ordinary predictor
        # registry as well as to the scoped binding context.  The inverse
        # route still calls through ``_bound_runtime_context``; this mirror
        # only gives availability checks a live registry to validate against.
        context_key, selection_kind, _ = _rcwa_resource_context(material, substrate)
        if getattr(ml_module, "_RCWA_WL_MODELS", {}).get(material):
            wavelength_sessions = dict(getattr(ml_module, "_RCWA_WL_SESSIONS", {}))
            wavelength_sessions[material] = _rcwa_runtime_binding.session_refs[0]
            ml_module._RCWA_WL_SESSIONS = wavelength_sessions
        else:
            route_key = (
                (material, substrate)
                if selection_kind == "exact_pair" else material
            )
            registry = dict(getattr(ml_module, "_RCWA_SESSIONS", {}))
            registry[route_key] = list(_rcwa_runtime_binding.session_refs)
            ml_module._RCWA_SESSIONS = registry
        ml_module._RCWA_AVAILABLE = True
    return _rcwa_ml_ready


def _analysis_runtime_identity_issue(context):
    """Verify context, current bytes, and the active runtime session as one identity."""
    current_artifact = _analysis_artifact_version(
        context.route_id, context.model_version,
        structure_type=context.structure_type, material=context.material,
        substrate=context.substrate)
    if current_artifact != context.artifact_version:
        return "analysis artifact changed after the snapshot context was created"
    route = str(context.route_id)
    if context.structure_type == "dual" and route in {"ml_surrogate", "dual ML surrogate"}:
        identity = _dual_model_disk_identity()
        return _runtime_binding_issue(
            _dual_runtime_binding, identity,
            tuple(getattr(_dual_runtime_binding, "session_refs", ())),
            (_DUAL_MODEL_RELATIVE_PATH,), _dual_resource_context_key())
    if context.structure_type == "single" and route in {
        "ml_surrogate", "generic-fano-resmlp-v8-sub-onnx-only",
        "ML surrogate (generic angle-conditioned)",
    }:
        contract = model_difference_contracts.GENERIC_ONNX_ROUTE
        identity = _primary_model_disk_identity()
        return _runtime_binding_issue(
            _primary_runtime_binding, identity,
            tuple(getattr(_primary_runtime_binding, "session_refs", ())),
            (contract.model_relative_path,), _primary_resource_context_key())
    if context.structure_type == "single" and route == "rcwa_surrogate":
        identity = _rcwa_model_disk_identity(context.material, context.substrate)
        return _runtime_binding_issue(
            _rcwa_runtime_binding, identity,
            tuple(getattr(_rcwa_runtime_binding, "session_refs", ())),
            _registered_rcwa_model_paths(context.material, context.substrate),
            _rcwa_resource_context(context.material, context.substrate)[0])
    return ""


def _analysis_model_artifact_version(context):
    """Return only the runtime model identity owned by this analysis route."""
    route = str(context.route_id)
    if context.structure_type == "dual" and route in {
            "ml_surrogate", "dual ML surrogate"}:
        return (
            _dual_runtime_binding.loaded_identity
            if _dual_runtime_binding is not None else "not_applicable")
    if context.structure_type == "single" and route in {
            "ml_surrogate", "generic-fano-resmlp-v8-sub-onnx-only",
            "ML surrogate (generic angle-conditioned)"}:
        return (
            _primary_runtime_binding.loaded_identity
            if _primary_runtime_binding is not None else "not_applicable")
    if context.structure_type == "single" and route == "rcwa_surrogate":
        return (
            _rcwa_runtime_binding.loaded_identity
            if _rcwa_runtime_binding is not None else "not_applicable")
    return "not_applicable"


def _bound_runtime_context(structure_type, route_id, material, substrate):
    """Install one strong resource only for the duration of a preview/analysis call."""
    structure = str(structure_type)
    route = str(route_id)
    if structure == "dual" and route in {"ml_surrogate", "dual ML surrogate"}:
        resource = _dual_runtime_binding
        if resource is None:
            raise ModelResourceUnavailable("dual runtime model is not bound")
        return bound_model_context(
            resource, ml_module,
            {
                "_DUAL_ORT_SESSION": resource.session_refs[0],
                "_DUAL_ORT_AVAILABLE": True,
                "_DUAL_IS_V3": True,
            },
            _dual_model_disk_identity,
            lambda: (getattr(ml_module, "_DUAL_ORT_SESSION", None),),
            _dual_resource_context_key(),
        )
    if structure == "single" and route in {
        "ml_surrogate", "generic-fano-resmlp-v8-sub-onnx-only",
        "ML surrogate (generic angle-conditioned)",
    }:
        resource = _primary_runtime_binding
        if resource is None:
            raise ModelResourceUnavailable("primary runtime model is not bound")
        return bound_model_context(
            resource, ml_module,
            {
                "_ORT_SESSION": resource.session_refs[0],
                "_ORT_AVAILABLE": True,
                "_ORT_IS_V8": True,
            },
            _primary_model_disk_identity,
            lambda: (getattr(ml_module, "_ORT_SESSION", None),),
            _primary_resource_context_key(),
        )
    if structure == "single" and route == "rcwa_surrogate":
        resource = _rcwa_runtime_binding
        if resource is None:
            raise ModelResourceUnavailable("RCWA runtime model is not bound")
        material = str(material)
        substrate = str(substrate)
        wavelength = bool(getattr(ml_module, "_RCWA_WL_MODELS", {}).get(material))
        if wavelength:
            installed_sessions = {}
            installed_wavelength = {material: resource.session_refs[0]}
        else:
            key = (material, substrate)
            route_key = key if key in getattr(
                ml_module, "_RCWA_SUBSTRATE_MODELS", {}) else material
            installed_sessions = {route_key: list(resource.session_refs)}
            installed_wavelength = {}
        return bound_model_context(
            resource, ml_module,
            {
                "_RCWA_SESSIONS": installed_sessions,
                "_RCWA_WL_SESSIONS": installed_wavelength,
                "_RCWA_AVAILABLE": True,
            },
            lambda: _rcwa_model_disk_identity(material, substrate),
            lambda: _current_rcwa_sessions(material, substrate),
            _rcwa_resource_context(material, substrate)[0],
        )
    return nullcontext(None)


def _rcwa_model_version(material, substrate):
    """Return loaded surrogate files without implying direct RCWA execution."""
    names = []
    sessions = getattr(ml_module, "_RCWA_SESSIONS", {})
    for key in ((material, substrate), material):
        for session in sessions.get(key, []):
            try:
                names.append(os.path.basename(session._model_path))
            except Exception:
                names.append("loaded RCWA ensemble member")
        if names:
            break
    wl_session = getattr(ml_module, "_RCWA_WL_SESSIONS", {}).get(material)
    if wl_session is not None:
        try:
            names.append(os.path.basename(wl_session._model_path))
        except Exception:
            names.append("loaded wavelength-conditioned RCWA surrogate")
    return ", ".join(dict.fromkeys(names)) if names else "不可用"


def _has_rcwa_ensemble(material, substrate):
    """Only label RGB as RCWA-trained when predict_rgb has an ensemble."""
    try:
        return bool(ml_module._get_rcwa_sessions(material, substrate))
    except Exception:
        return False


def _render_result_provenance(provenance):
    """Render a concise source summary with full audit evidence on demand."""
    def esc(value):
        return html.escape(str(value), quote=True)

    # 来源徽章色：色相保持原有语义编码（绿=代理/蓝=ML/紫=远场/橙=FP/红=不可用），
    # 只在 Oklch 里提亮到"深色文字可读"的明度，使小字对比度达 WCAG AA。
    # 原先徽章用白字压在中等明度底上，最低只有 3.19，不达标。
    route_colors = {
        # The main card is for people using the tool.  Keep the exact route and
        # model names in the audit expander below instead of leading with them.
        "rcwa_surrogate": ("#42B68C", "快速计算"),
        "ml_surrogate": ("#5E9FFB", "快速计算"),
        "far_field_postprocessing": ("#B97BFA", "远场修正"),
        "fp_tmm": ("#E5821F", "薄膜腔计算"),
        "lorentz_fano_fallback": ("#CA6728", "解析计算"),
        "invalid_geometry": ("#FF6559", "参数不可用"),
    }
    color, route_label = route_colors.get(
        provenance.get("route_id"), ("#8FA0B8", "当前路线")
    )
    technical_route_label = str(
        provenance.get("route_label") or provenance.get("route_id") or "未记录"
    )
    technical_chain = " → ".join(str(item) for item in provenance.get("chain", []))
    state_labels = {
        "loaded_and_called": "模型已调用，输出通过校验",
        "loaded": "模型已加载",
        "not_selected": "本路线不使用模型文件",
        "not_applicable": "不适用",
        "file_missing": "本地模型文件缺失",
        "file_present_load_failed": "模型文件存在，但未成功加载",
        "call_failed": "模型调用失败",
        "output_validation_failed": "模型输出未通过校验",
        "invalid_geometry": "几何参数越域，未执行",
    }
    model_state = str(provenance.get("model_state", "not_recorded"))
    model_state_label = state_labels.get(model_state, "状态已记录，见审计详情")
    spectrum_ok = bool(provenance.get("spectrum_available", False))
    result_status = "可用；颜色由该光谱计算" if spectrum_ok else "不可用；未跨模型补数"
    rows = [
        ("材料 / 衬底", f"{provenance['material']} / {provenance['substrate']}"),
        ("偏振 / 入射角", f"{provenance['polarization']} / {provenance['angle_deg']:.1f}°"),
        ("NA / 观察角", f"{provenance['na']} / {provenance['theta_obs_deg']:.1f}°"),
        ("模型版本", provenance["model_version"]),
        ("模型状态", model_state_label),
        ("光谱状态", result_status),
    ]
    details = "".join(
        f'<div><span style="color:var(--text-muted)">{esc(label)}</span><br>{esc(value)}</div>'
        for label, value in rows
    )
    reason = provenance.get("fallback_reason") or "无回退；当前路径按配置直接执行。"
    chain = " → ".join(provenance.get("chain", []))
    st.markdown(
        f"""
        <div class="source-summary" role="status" aria-label="结果来源摘要"
             title="技术路线：{esc(technical_route_label)}；实际链路：{esc(technical_chain or '未记录')}；观察角度：{esc(provenance.get('theta_obs_deg', 0.0))}°；NA：{esc(provenance.get('na', '未启用'))}"
             style="border-left-color:{color}">
           <div class="source-summary__head">
             <strong>当前结果</strong>
              <span class="route-badge" title="技术路线：{esc(technical_route_label)}"
                    style="border-radius:999px;padding:3px 9px;font-size:12px;font-weight:600">
                {esc(route_label)}</span>
             <span class="source-summary__status">{esc(result_status)}</span>
           </div>
           <div class="source-summary__context">
             {esc(provenance.get('structure_label', '当前结构'))} ·
             {esc(provenance['material'])} / {esc(provenance['substrate'])} ·
             {esc(provenance['polarization'])} · 入射 {float(provenance['angle_deg']):.1f}°
           </div>
         </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("查看来源、边界与证据详情", expanded=False):
        st.caption("需要复核时再展开；平时只看上面的结果和颜色。")
        if st.button("加载完整审计详情", key="load_audit_details"):
            st.session_state["show_audit_details"] = True
        if st.session_state.get("show_audit_details", False):
            # Read/hash small evidence only after explicit user request.
            asset_bits = []
            try:
                asset_bits.append(
                    "当前运行包不携带论文2池 manifest 或科研控制面；"
                    "当前结果来自上方标注的模型/解析路线"
                )
            except Exception as exc:
                asset_bits.append(f"证据状态不可用: {type(exc).__name__}")
            model_files = []
            for model_name in str(provenance.get("model_version", "")).split(","):
                model_name = model_name.strip()
                if not model_name or "/" in model_name or "\\" in model_name:
                    continue
                model_path = os.path.join(os.path.dirname(__file__), "models", model_name)
                if os.path.isfile(model_path):
                    try:
                        with open(model_path, "rb") as handle:
                            model_hash = hashlib.sha256(handle.read()).hexdigest()[:12]
                        model_files.append(f"{model_name} sha256={model_hash}…")
                    except OSError:
                        model_files.append(f"{model_name} (存在但无法读取哈希)")
                elif model_name.endswith((".onnx", ".pt")):
                    model_files.append(f"{model_name} (本地缺失，可能回退)")
            asset_text = "；".join(asset_bits + model_files)
            st.markdown(
                f"""
                <div class="audit-details" aria-label="完整来源与证据">
                  <div class="audit-details__grid">{details}</div>
                  <div class="audit-details__section">
                    <span>实际链路</span><br>{esc(chain or '未执行')}
                  </div>
                  <div class="audit-details__section">
                    <span>边界 / 回退</span><br>{esc(reason)}
                  </div>
                  <details class="audit-details__evidence">
                    <summary>技术证据（模型、池与色度学）</summary>
                    <div class="audit-details__evidence-content">{esc(asset_text)}</div>
                  </details>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_reference_recheck_details(
    *,
    structure_type,
    material,
    substrate,
    polarization,
    angle_deg,
    far_field_enabled,
    diameter_nm,
    height_nm,
    period_nm,
    forward,
):
    """Show an exact lookup against the audited competition RCWA reference set."""
    st.subheader("已审核 RCWA 参考复核")
    st.caption(
        "只读取已审核的 TiO2/SiO2/air 参考库；查询要求完整匹配几何和边界，"
        "不插值、不写入训练数据。"
    )
    try:
        reference_library = load_reference_library()
    except ReferenceLibraryError as exc:
        st.error(f"参考库绑定失败，已停止接入：{exc}")
        return
    except Exception as exc:
        st.error(f"参考库加载异常，已停止接入：{type(exc).__name__}")
        return

    supported, support_reason = reference_library.supports_display_conditions(
        material, substrate, polarization, angle_deg,
        structure_type=structure_type,
        far_field_enabled=far_field_enabled,
    )
    metadata = reference_library.metadata()
    if not supported:
        st.info(f"当前配置未进入参考查询：{support_reason}")
        st.caption(
            f"已绑定 {reference_library.unique_geometry_count:,} 个唯一单柱几何；"
            f"记录 SHA256={metadata['records_sha256']}"
        )
        return

    match = reference_library.lookup(diameter_nm, height_nm, period_nm)
    if match is None:
        st.info(
            f"当前几何 D={float(diameter_nm):g} / H={float(height_nm):g} / "
            f"P={float(period_nm):g} nm 未命中参考库。这里只接受 JSONL 中已存在的精确整数几何，"
            "不会用最近邻或模型输出补齐。"
        )
        st.caption(
            f"可查询条件：{reference_condition_summary()}；"
            f"已审核记录 {reference_library.record_count:,} 条，唯一几何 "
            f"{reference_library.unique_geometry_count:,} 个。"
        )
        return

    reference_rgb = np.asarray(match.srgb_display, dtype=float)
    reference_hex = rgb_to_hex(reference_rgb)
    reference_r255 = rgb_255(reference_rgb)
    record = match.record
    st.success(
        f"精确命中参考记录 #{record.get('index', '?')} · "
        f"batch {record.get('batch', '?')} · D/H/P={match.geometry[0]}/{match.geometry[1]}/{match.geometry[2]} nm"
    )
    swatch_left, swatch_right = st.columns([1, 3])
    with swatch_left:
        st.markdown(
            f"""<div style="height:86px;border-radius:10px;background:{reference_hex};
            border:1px solid rgba(255,255,255,.28);
            box-shadow:inset 0 1px 0 rgba(255,255,255,.25);"></div>""",
            unsafe_allow_html=True,
        )
    with swatch_right:
        st.markdown(
            f"""**参考颜色 {reference_hex}**

            RGB({reference_r255[0]}, {reference_r255[1]}, {reference_r255[2]})

            XYZ({', '.join(f'{value:.6f}' for value in match.xyz_d65)})

            Lab({', '.join(f'{value:.4f}' for value in match.lab_d65)})"""
        )

    st.caption(
        "参考条件：TiO2 / SiO2 / air · TM (p-pol) · 0° · nG=151 · Nxy=256 · "
        f"守恒最大误差 {match.max_abs_rt_error:.3e}；记录 SHA256={match.source_sha256}"
    )
    if not forward.spectrum_available:
        st.warning(f"当前路线没有可用光谱，暂不能做逐波长对照：{forward.error}")
        return

    try:
        current_wavelengths = np.asarray(forward.wavelengths_nm, dtype=float)
        current_reflectance = np.asarray(forward.reflectance, dtype=float)
        reference_wavelengths = match.wavelengths_nm
        if np.array_equal(current_wavelengths, reference_wavelengths):
            compared_reflectance = current_reflectance
        else:
            compared_reflectance = np.interp(
                reference_wavelengths, current_wavelengths, current_reflectance)
        difference = compared_reflectance - match.reflectance
        max_abs_difference = float(np.max(np.abs(difference)))
        rmse = float(np.sqrt(np.mean(np.square(difference))))
        current_rgb = np.asarray(forward.rgb, dtype=float)
        delta_e = float(delta_e2000(rgb_to_lab(current_rgb), match.lab_d65))
        metric_cols = st.columns(3)
        metric_cols[0].metric("逐波长最大 |ΔR|", f"{max_abs_difference:.4f}")
        metric_cols[1].metric("光谱 RMSE", f"{rmse:.4f}")
        metric_cols[2].metric("当前路线 vs 参考 ΔE2000", f"{delta_e:.2f}")

        plt = _get_plt()
        figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
        axes[0].plot(reference_wavelengths, match.reflectance, lw=2.0,
                     color="#F3D259", label="审计参考 RCWA")
        axes[0].plot(reference_wavelengths, compared_reflectance, lw=1.7,
                     color="#5E9FFB", label="当前路线")
        axes[0].set_title("反射光谱对照")
        axes[0].set_xlabel("波长 (nm)")
        axes[0].set_ylabel("反射率")
        axes[0].set_xlim(380, 780)
        axes[0].set_ylim(0, 1.08)
        axes[0].grid(True, alpha=0.22)
        axes[0].legend(fontsize=8, framealpha=0.85)
        axes[1].plot(reference_wavelengths, difference, lw=1.5, color="#FC6B2E")
        axes[1].axhline(0.0, color="#555", lw=0.8)
        axes[1].set_title("当前路线 - 参考")
        axes[1].set_xlabel("波长 (nm)")
        axes[1].set_ylabel("ΔR")
        axes[1].set_xlim(380, 780)
        axes[1].grid(True, alpha=0.22)
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)
        st.caption(
            "对照指标只描述当前显示路线与该条参考记录的差异；不等于代理模型的全域精度，"
            "也不代表 nG=151 已完成收敛证明。"
        )
    except Exception as exc:
        st.warning(f"参考光谱对照失败：{type(exc).__name__}")
        return

    st.download_button(
        "下载参考复核 JSON",
        json.dumps(
            {
                "reference": match.export_payload(),
                "current_route": forward.provenance,
                "comparison": {
                    "max_abs_reflectance_difference": max_abs_difference,
                    "reflectance_rmse": rmse,
                    "delta_e2000": delta_e,
                },
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        file_name=f"reference_recheck_D{match.geometry[0]}_H{match.geometry[1]}_P{match.geometry[2]}.json",
        mime="application/json",
        use_container_width=True,
        key="download_reference_recheck",
    )


def _render_reference_recheck(**kwargs):
    """Keep high-fidelity comparison available without taking over the preview."""
    expanded = bool(st.session_state.pop("_expand_reference_recheck_once", False))
    with st.expander("高保真参考对照（按需展开）", expanded=expanded):
        _render_reference_recheck_details(**kwargs)


def _inverse_candidate_contract(method, material, substrate, polarization, angle_deg):
    """Describe an inverse candidate's actual search route and its scientific boundary."""
    method_key = str(method).lower()
    if method_key in {
        "reference", "audited_reference", "reference_library",
    }:
        return {
            "method": "已审核 RCWA 参考库颜色近邻搜索",
            "model": "competition/tio2_air_reference_records_v1.jsonl",
            "route_id": "audited_rcwa_reference",
            "boundary": (
                "候选来自已审核 JSONL 中的完整 RCWA 记录；只按已存颜色排序，"
                "不插值、不训练、不把参考记录当作代理模型预测。"
            ),
        }
    if method_key in {"smart_grid", "rcwa", "rcwa surrogate", "rcwa-trained"}:
        model_name = _rcwa_model_version(material, substrate)
        if model_name == "不可用":
            model_name = "不可用（未加载 RCWA 代理）"
        return {
            "method": "RCWA-trained ML surrogate",
            "model": model_name,
            "route_id": "rcwa_surrogate",
            "boundary": "两阶段代理搜索；不是本次直接 RCWA，适用范围受训练数据约束。",
        }
    if method_key in {"rl", "rl q-learning", "q-learning"}:
        return {
            "method": "RL Q-learning",
            "model": "models/rl_qtable.npy",
            "route_id": "rl_qlearning",
            "boundary": (
                "本地 q-table 的离散探索；候选由当前 TiO2/SiO2 RCWA 代理复核，"
                "不宣称全局最优。"
            ),
        }
    if method_key in {"fano", "lorentz", "lorentz/fano", "analytical"}:
        return {
            "method": "Lorentz/Fano analytical fallback",
            "model": "engine.py inverse_design / torch_model.py",
            "route_id": "lorentz_fano_fallback",
            "boundary": "解析/半解析候选，不是直接 RCWA；需用已注册全波求解或实验独立复核。",
        }
    if method_key in {"dual_physical", "dual analytical", "dual baseline"}:
        return {
            "method": "双柱解析/半解析基线",
            "model": "torch_model.py::inverse_design_dual_v2",
            "route_id": "dual physical fallback",
            "boundary": "解析/半解析双柱候选，用于交互参考；不代表双柱 ONNX/RCWA 精度。",
        }
    if method_key in {"dual", "dual ml", "dual surrogate"}:
        return {
            "method": "Dual-pillar ML surrogate",
            "model": _DUAL_MODEL_RELATIVE_PATH,
            "route_id": "dual_ml_surrogate",
            "boundary": "代理候选，仅支持 SiO2 衬底；不是直接 RCWA。",
        }
    if method_key in {"fp", "fp tmm", "fp cavity"}:
        return {
            "method": "FP cavity TMM",
            "model": "fp_cavity.py",
            "route_id": "fp_tmm",
            "boundary": "薄膜腔传输矩阵候选，不是 RCWA 纳米柱求解。",
        }
    return {
        "method": "解析引擎候选",
        "model": "engine.py inverse_design",
        "route_id": "lorentz_fano_fallback",
        "boundary": "候选排序使用现有解析/半解析引擎，不宣称直接 RCWA。",
    }


def _smart_grid_has_local_weights(material, substrate):
    """Return whether smart-grid has a local RCWA model family to use.

    Smart-grid uses the same registered ONNX ensemble as preview. PyTorch
    weights are retained separately for the differentiable gradient route.
    """
    key = (material, substrate)
    patterns = getattr(ml_module, "_RCWA_SUBSTRATE_MODELS", {}).get(key)
    if patterns is None:
        patterns = getattr(ml_module, "_RCWA_MODELS", {}).get(material, [])
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    for pattern in patterns:
        if glob.glob(os.path.join(model_dir, pattern.replace(".onnx", ".pt"))):
            return True
        if glob.glob(os.path.join(model_dir, pattern)):
            return True
    return False


def _smart_grid_has_matching_rcwa_session(material, substrate):
    """Require the strong resource and the *currently installed* session pair.

    The binding captures the session objects at load time, but callers can
    clear or replace ``ml_module``'s registries during a rerun or test.  Use
    the live registry for the comparison so availability never outlives the
    predictor that ``predict_rgb`` would actually call.
    """
    context_key, selection_kind, paths = _rcwa_resource_context(
        material, substrate)
    if selection_kind != "exact_pair" or _rcwa_runtime_binding is None:
        return False
    current = _rcwa_model_disk_identity(material, substrate)
    active_sessions = _current_rcwa_sessions(material, substrate)
    return not _runtime_binding_issue(
        _rcwa_runtime_binding, current, active_sessions, paths, context_key)


def _rl_has_local_qtable():
    """Require the checked-in q-table and its trained-state sidecar."""
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    return all(
        os.path.isfile(os.path.join(model_dir, name))
        for name in ("rl_qtable.npy", "rl_qtable_meta.npy")
    )


def _rl_route_ready(material, substrate, polarization, angle_deg):
    """Return whether the local RL route has both its table and exact predictor."""
    return bool(
        _rl_has_local_qtable()
        and str(material) == "TiO2 (anatase)"
        and str(substrate) == "SiO2 (fused silica)"
        and str(polarization).startswith("TE")
        and abs(float(angle_deg)) < 1e-9
        and bool(_rcwa_ml_ready)
        and _smart_grid_has_matching_rcwa_session(material, substrate)
    )


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dual_domain_manifest_verified(context, *, manifest_path=None, model_path=None):
    """Bind dual-domain evidence to the loaded model, its hash, and this context."""
    if context.structure_type != "dual":
        return False
    base_dir = os.path.dirname(__file__)
    model_path = model_path or os.path.join(base_dir, _DUAL_MODEL_RELATIVE_PATH)
    manifest_path = manifest_path or os.path.join(
        base_dir, "models", "dual_mlp_v3_multi.manifest.json")
    try:
        loaded_session = getattr(ml_module, "_DUAL_ORT_SESSION", None)
        loaded_path = os.path.realpath(str(getattr(loaded_session, "_model_path", "")))
        resolved_model_path = os.path.realpath(model_path)
        if not loaded_path or loaded_path != resolved_model_path:
            return False
        if not os.path.isfile(resolved_model_path):
            return False
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        model = manifest["model"]
        domain = manifest["training_domain"]
        if int(manifest["schema_version"]) != 1:
            return False
        if manifest["model_version"] != _DUAL_MODEL_VERSION:
            return False
        if model["path"].replace("\\", "/") != _DUAL_MODEL_RELATIVE_PATH:
            return False
        expected_hash = str(model["sha256"]).strip().lower()
        if len(expected_hash) != 64 or expected_hash != _sha256_file(resolved_model_path):
            return False
        for field in ("materials", "substrates", "polarizations"):
            values = domain[field]
            if not isinstance(values, list) or not values:
                return False
            if any(not isinstance(value, str) or not value.strip() for value in values):
                return False
        if context.material not in domain["materials"]:
            return False
        if context.substrate not in domain["substrates"]:
            return False
        if context.polarization not in domain["polarizations"]:
            return False
        angle_domain = domain["angle_deg"]
        mode = angle_domain["mode"]
        if mode == "exact":
            angle_ok = float(context.angle_deg) in {
                float(value) for value in angle_domain["values"]}
        elif mode == "range":
            angle_value = float(context.angle_deg)
            minimum = float(angle_domain["min"])
            maximum = float(angle_domain["max"])
            angle_ok = minimum <= angle_value <= maximum and minimum <= maximum
        else:
            return False
        return bool(angle_ok)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _inverse_method_states(*, context, rcwa_ready, primary_torch_ready,
                           dual_ready, compare_enabled, fp_mirror_type="",
                           far_field_enabled=False, rl_ready=False):
    """Return the structure-specific registry for the current context."""
    states = inverse_method_registry(
        context,
        supported_material=context.material in ml_module.MATERIAL_CODES,
        supported_substrate=context.substrate in ml_module.SUBSTRATE_CODES,
        rcwa_ready=bool(
            rcwa_ready and _smart_grid_has_matching_rcwa_session(
                context.material, context.substrate)),
        smart_weights_ready=_smart_grid_has_local_weights(
            context.material, context.substrate),
        primary_torch_ready=primary_torch_ready,
        dual_ready=dual_ready,
        dual_domain_verified=_dual_domain_manifest_verified(context),
        compare_enabled=compare_enabled,
        rl_ready=bool(rl_ready),
        fp_mirror_type=fp_mirror_type,
    )
    # The audited reference library is a separate, data-backed route.  It is
    # intentionally offered for TM/normal-incidence single-pillar contexts
    # only; TE keeps the existing proxy registry unchanged.
    if context.structure_type == "single" and not str(context.polarization).startswith("TE"):
        try:
            reference_library = load_reference_library()
            supported, support_reason = reference_library.supports_display_conditions(
                context.material, context.substrate, context.polarization,
                context.angle_deg, structure_type="single",
                far_field_enabled=bool(far_field_enabled),
            )
            reference_available = bool(context.geometry_valid and supported)
            if not context.geometry_valid:
                reference_reason = "不可用：当前几何无效。"
            elif not supported:
                reference_reason = f"不可用：{support_reason}"
            else:
                reference_reason = (
                    f"可用：已绑定 {reference_library.unique_geometry_count:,} 个唯一几何；"
                    "结果来自已审核 RCWA 记录。"
                )
            states["reference"] = InverseMethodState(
                "reference", "已审核参考库",
                "按 ΔE2000 从已审核 RCWA 记录中排序候选",
                reference_available, reference_reason, scope="reference_library",
            )
        except ReferenceLibraryError as exc:
            states["reference"] = InverseMethodState(
                "reference", "已审核参考库",
                "按 ΔE2000 从已审核 RCWA 记录中排序候选",
                False, f"不可用：参考库审核绑定失败（{exc}）。",
                scope="reference_library",
            )
        except Exception as exc:
            states["reference"] = InverseMethodState(
                "reference", "已审核参考库",
                "按 ΔE2000 从已审核 RCWA 记录中排序候选",
                False, f"不可用：参考库加载异常（{type(exc).__name__}）。",
                scope="reference_library",
            )
    return states


_INVERSE_RESULT_KEYS = (
    "_sg_candidates", "_sg_d", "_sg_h", "_sg_p", "_sg_hex", "_sg_de",
    "_rl_d", "_rl_h", "_rl_p", "_rl_hex", "_rl_de", "_rl_rgb",
    "_gd_d", "_gd_h", "_gd_p", "_gd_hex", "_gd_de", "_gd_rgb",
    "_dual_gd_d1", "_dual_gd_h1", "_dual_gd_d2", "_dual_gd_h2",
    "_dual_gd_p", "_dual_gd_hex", "_dual_gd_de", "_dual_gd_rgb",
    "_ai_candidates", "_reference_matches",
)


def _clear_inverse_results():
    for key in _INVERSE_RESULT_KEYS:
        st.session_state.pop(key, None)
    st.session_state.pop("_inverse_run", None)


def _store_inverse_run(context, method_id, method_label, candidates):
    run = InverseRun.create(context, method_id, method_label, candidates)
    st.session_state._inverse_run = run
    return run


def _apply_inverse_candidate(context, candidate_structure, parameters):
    run = st.session_state.get("_inverse_run")
    if not inverse_run_matches(run, context):
        return
    updates = candidate_parameter_updates(
        context, candidate_structure, parameters)
    for key, value in updates.items():
        st.session_state[key] = value


def _candidate_context(structure_type, material, substrate, polarization, angle_deg,
                       *, mirror_type="", n_pairs=None, algorithm_version=""):
    context = {
        "structure_type": str(structure_type),
        "material": str(material),
        "substrate": str(substrate),
        "polarization": str(polarization),
        "angle_deg": float(angle_deg),
    }
    if mirror_type:
        context["mirror_type"] = str(mirror_type)
    if n_pairs is not None:
        context["n_pairs"] = int(n_pairs)
    if algorithm_version:
        context["algorithm_version"] = str(algorithm_version)
    return context


def _normalized_smart_candidates(context, candidates):
    contract = _inverse_candidate_contract(
        "smart_grid", context.material, context.substrate,
        context.polarization, context.angle_deg)
    records = []
    for rank, candidate in enumerate(candidates, start=1):
        _, params, rgb, de76, de2000 = candidate
        records.append(build_inverse_candidate(
            context,
            method_id="smart", method_label="智能网格", rank=rank,
            structure_type="single",
            candidate_context=_candidate_context(
                "single", context.material, context.substrate,
                context.polarization, context.angle_deg),
            route_id=contract["route_id"], route_label=contract["method"],
            model_version=contract["model"], boundary=contract["boundary"],
            parameters={
                "d": params.diameter_nm, "h": params.height_nm,
                "p": params.period_nm,
            },
            predicted_rgb=rgb, delta_e76=de76, delta_e2000=de2000,
        ))
    return tuple(records)


def _normalized_rl_candidates(context, candidate, predicted_rgb=None):
    """Build one strict inverse-run record from the local RL result."""
    d_nm, h_nm, p_nm, _hex_value, de2000 = candidate
    contract = _inverse_candidate_contract(
        "rl", context.material, context.substrate,
        context.polarization, context.angle_deg)
    if predicted_rgb is None:
        predicted_rgb = ml_module.predict_rgb(
            float(d_nm), float(h_nm), float(p_nm),
            float(context.angle_deg), context.polarization,
            context.material, context.substrate,
        )
    predicted_rgb = np.asarray(predicted_rgb, dtype=float)
    return (
        build_inverse_candidate(
            context,
            method_id="rl", method_label="RL Q-learning", rank=1,
            structure_type="single",
            candidate_context=_candidate_context(
                "single", context.material, context.substrate,
                context.polarization, context.angle_deg),
            route_id=contract["route_id"], route_label=contract["method"],
            model_version=contract["model"], boundary=contract["boundary"],
            parameters={"d": d_nm, "h": h_nm, "p": p_nm},
            predicted_rgb=predicted_rgb, delta_e2000=float(de2000),
        ),
    )


def _reference_library_candidates(context, *, limit=5):
    """Return exact audited reference matches ranked by CIEDE2000.

    The library stores the audited Lab/sRGB values beside every full spectrum.
    This route only ranks those existing records; it never interpolates a
    missing geometry or calls a surrogate model.
    """
    if context.structure_type != "single":
        raise ValueError("已审核参考库只支持单柱结构")
    reference_library = load_reference_library()
    supported, support_reason = reference_library.supports_display_conditions(
        context.material, context.substrate, context.polarization,
        context.angle_deg, structure_type=context.structure_type,
        far_field_enabled=False,
    )
    if not supported:
        raise ValueError(support_reason)
    target_lab = rgb_to_lab(np.asarray(context.target_rgb, dtype=float) / 255.0)
    scored = [
        (
            float(delta_e2000_scalar(target_lab, match.lab_d65)),
            match,
        )
        for match in reference_library.by_geometry.values()
    ]
    scored.sort(key=lambda item: (item[0], item[1].geometry))
    return tuple(scored[:max(1, int(limit))])


def _normalized_reference_candidates(context, scored_candidates):
    """Build strict inverse-run records for audited reference candidates."""
    contract = _inverse_candidate_contract(
        "audited_reference", context.material, context.substrate,
        context.polarization, context.angle_deg,
    )
    target_lab = rgb_to_lab(np.asarray(context.target_rgb, dtype=float) / 255.0)
    records = []
    for rank, (de2000, match) in enumerate(scored_candidates, start=1):
        rgb = np.asarray(match.srgb_display, dtype=float)
        records.append(build_inverse_candidate(
            context,
            method_id="reference", method_label="已审核参考库", rank=rank,
            structure_type="single",
            candidate_context=_candidate_context(
                "single", context.material, context.substrate,
                context.polarization, context.angle_deg,
            ),
            route_id=contract["route_id"], route_label=contract["method"],
            model_version=contract["model"], boundary=contract["boundary"],
            parameters={
                "d": match.geometry[0], "h": match.geometry[1],
                "p": match.geometry[2],
            },
            predicted_rgb=rgb,
            delta_e76=delta_e76(target_lab, match.lab_d65),
            delta_e2000=de2000,
        ))
    return tuple(records)


def _normalized_compare_candidates(context, candidates):
    records = []
    for rank, (de2000, name, data, candidate_type) in enumerate(candidates, start=1):
        if candidate_type == "meta":
            _, params, rgb, de76, candidate_de2000 = data[0]
            candidate_material = (
                "TiO2 (anatase)" if "TiO2" in name else
                "a-Si (amorphous)" if "a-Si" in name else context.material)
            contract = _inverse_candidate_contract(
                "analytical", candidate_material, context.substrate,
                context.polarization, context.angle_deg)
            records.append(build_inverse_candidate(
                context,
                method_id="compare", method_label="跨结构方案对比", rank=rank,
                structure_type="single",
                candidate_context=_candidate_context(
                    "single", candidate_material, context.substrate,
                    context.polarization, context.angle_deg),
                route_id=contract["route_id"], route_label=contract["method"],
                model_version=contract["model"], boundary=contract["boundary"],
                parameters={
                    "d": params.diameter_nm, "h": params.height_nm,
                    "p": params.period_nm,
                },
                predicted_rgb=rgb, delta_e76=de76,
                delta_e2000=candidate_de2000, scheme=str(name),
            ))
        elif candidate_type == "fp":
            candidate_de2000, t_nm, center_nm, rgb = data
            contract = _inverse_candidate_contract(
                "fp", "TiO2 (cavity layer)", "SiO2 (DBR mirror stack)",
                context.polarization, context.angle_deg)
            records.append(build_inverse_candidate(
                context,
                method_id="compare", method_label="跨结构方案对比", rank=rank,
                structure_type="fp",
                candidate_context=_candidate_context(
                    "fp", "TiO2 (cavity layer)", "SiO2 (DBR mirror stack)",
                    context.polarization, context.angle_deg,
                    mirror_type="介质 DBR (TiO2/SiO2)", n_pairs=3,
                    algorithm_version=_FP_INVERSE_ALGORITHM_VERSION),
                route_id=contract["route_id"], route_label=contract["method"],
                model_version=(
                    f"{contract['model']} | {_FP_INVERSE_ALGORITHM_VERSION}"),
                boundary=contract["boundary"],
                parameters={"t": t_nm, "center_wavelength": center_nm},
                predicted_rgb=rgb, delta_e2000=candidate_de2000,
                scheme=str(name),
            ))
        else:
            raise ValueError(
                f"unsupported compare candidate type: {candidate_type}")
    return tuple(records)


def _normalized_fp_candidates(context, candidates):
    contract = _inverse_candidate_contract(
        "fp", context.material, context.substrate,
        context.polarization, context.angle_deg)
    return tuple(build_inverse_candidate(
        context,
        method_id="fp", method_label="FP 腔搜索", rank=rank,
        structure_type="fp",
        candidate_context=_candidate_context(
            "fp", context.material, context.substrate,
            context.polarization, context.angle_deg,
            mirror_type=context.fp_mirror_type, n_pairs=3,
            algorithm_version=_FP_INVERSE_ALGORITHM_VERSION),
        route_id=contract["route_id"], route_label=contract["method"],
        model_version=f"{contract['model']} | {_FP_INVERSE_ALGORITHM_VERSION}",
        boundary=contract["boundary"],
        parameters={"t": t_nm, "center_wavelength": center_nm},
        predicted_rgb=rgb, delta_e2000=de2000,
    ) for rank, (de2000, center_nm, t_nm, rgb) in enumerate(candidates, start=1))


def _render_inverse_exports(context):
    run = st.session_state.get("_inverse_run")
    if not inverse_run_matches(run, context):
        return False
    try:
        exports = serialize_inverse_run(run, context)
    except (TypeError, ValueError) as exc:
        logging.warning("inverse export contract rejected current run: %s", exc)
        st.warning("当前逆设计结果缺少完整追溯字段，已禁用导出；未从旧状态补造候选。")
        return False
    basename = f"inverse_{run.method_id}_{inverse_context_fingerprint(context)[:12]}"
    with st.expander("📥 导出结果 (CSV/JSON)", expanded=False):
        left, right = st.columns(2)
        with left:
            st.download_button(
                "💾 导出逆设计 CSV", exports.csv_text,
                file_name=f"{basename}.csv", mime="text/csv",
                use_container_width=True,
            )
        with right:
            st.download_button(
                "💾 导出逆设计 JSON", exports.json_text,
                file_name=f"{basename}.json", mime="application/json",
                use_container_width=True,
            )
    return True


def _inverse_display_route_label(route):
    """Return a short route label for the main interaction surface."""
    route_id = str((route or {}).get("route_id", ""))
    return {
        "rcwa_surrogate": "快速计算",
        "ml_surrogate": "快速计算",
        "rl_qlearning": "离散探索",
        "far_field_postprocessing": "远场修正",
        "fp_tmm": "薄膜腔计算",
        "lorentz_fano_fallback": "解析计算",
        "invalid_geometry": "参数不可用",
    }.get(route_id, "当前路线")


def _inverse_display_structure_label(structure_type):
    return {
        "single": "单柱",
        "dual": "双柱",
        "fp": "FP 腔",
    }.get(str(structure_type), str(structure_type))


def _render_inverse_context(context, route):
    """Show the fixed search context once so candidate cards do not repeat ambiguous state."""
    st.markdown(
        f"""
        <div class="inverse-context" aria-label="当前搜索配置">
          <strong>当前搜索配置</strong>
          <div class="inverse-context__grid">
            <span><b>结构</b>{html.escape(_inverse_display_structure_label(context.structure_type))}</span>
            <span><b>材料 / 衬底</b>{html.escape(context.material)} / {html.escape(context.substrate)}</span>
            <span><b>偏振 / 入射角</b>{html.escape(context.polarization)} / {float(context.angle_deg):.1f}°</span>
            <span><b>当前路线</b>{html.escape(_inverse_display_route_label(route))}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("先看颜色和色差，满意后应用候选。")
    # Keep the boundary visible but short: TM/0° has a usable audited-library
    # route, while the learned proxy methods remain registered for TE/0°.
    if context.structure_type == "single":
        is_te = str(context.polarization).startswith("TE")
        is_normal = abs(float(context.angle_deg)) < 1e-9
        if not is_te and is_normal:
            st.caption(
                "TM / 0°：已审核参考库可用；代理模型仍限 TE / 0°。"
            )
        elif not is_te or not is_normal:
            st.info(
                f"当前条件为 {html.escape(str(context.polarization))} / "
                f"{float(context.angle_deg):.1f}°；可用搜索路线受注册条件限制。"
            )


def _render_inverse_candidate_card(rank, hex_value, rgb_value, de2000, params_text,
                                   contract, material, substrate, polarization, angle_deg,
                                   apply_key=None, apply_callback=None):
    """Render a scan-friendly candidate card with technical details on demand."""
    rgb_text = ", ".join(str(int(v)) for v in rgb_value)
    card_color = "var(--highlight-surface)" if rank == 1 else "var(--bg-surface)"
    st.markdown(
        f"""
        <div style="width:100%;max-width:100%;min-width:0;overflow-wrap:anywhere;word-break:break-word;
             border:1px solid var(--border-subtle);border-left:4px solid {'var(--accent)' if rank == 1 else 'var(--border-strong)'};
             border-radius:10px;padding:12px 14px;margin:8px 0;background:{card_color};">
          <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
            <span style="font-weight:700">#{int(rank)}</span>
            <span style="width:28px;height:28px;border-radius:6px;background:{html.escape(hex_value)};
                         border:1px solid rgba(243,245,245,.42);display:inline-block"></span>
            <strong>{html.escape(hex_value)}</strong>
            <span style="color:var(--text-secondary)">RGB({html.escape(rgb_text)})</span>
            <strong style="margin-left:auto;color:var(--accent-soft)">ΔE2000 {float(de2000):.2f}</strong>
          </div>
          <div style="min-width:0;margin-top:8px;font-size:12px;line-height:1.5;color:var(--text-secondary);
                      overflow-wrap:anywhere;word-break:break-word">
            <b style="color:var(--text-primary)">参数：</b>{html.escape(params_text)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander(f"查看候选来源与适用边界 · #{int(rank)}", expanded=False):
        st.markdown(
            f"**方法**：{contract['method']}  \n"
            f"**模型**：{contract['model']}  \n"
            f"**上下文**：{material} / {substrate} · {polarization} · θ={float(angle_deg):.1f}°  \n"
            f"**边界**：{contract['boundary']}"
        )
    if apply_key and apply_callback:
        st.button("应用此候选", key=apply_key, on_click=apply_callback, use_container_width=True)


def _render_saved_inverse_candidates(context):
    """Keep usable candidate cards across downloads and ordinary reruns."""
    run = st.session_state.get("_inverse_run")
    if not inverse_run_matches(run, context) or run.method_id not in {
            "smart", "single", "rl", "dual", "dual_physical", "fp"}:
        return
    # Render the same validated records used by JSON/CSV exports. Never infer
    # candidates from leftover widget values or invoke a search during rerender.
    candidates = serialize_inverse_run(run, context).payload['candidates']
    if not candidates:
        return
    best = candidates[0]
    if run.method_id == 'smart':
        st.success(f"智能网格完成 · 本次搜索排名第一 {best['predicted_hex']} · ΔE2000={best['delta_e2000']:.1f}")
    else:
        st.success(f"{run.method_label} · 已保存 {len(candidates)} 个候选")
    st.caption(f"目标色：{context.target_hex} · 先比较颜色，再应用候选。")
    first_keys = {
        'smart': 'apply_sg_result', 'single': 'apply_gd_result',
        'rl': 'apply_rl_result', 'dual': 'apply_dual_gd_result',
        'dual_physical': 'apply_dual_physical_result', 'fp': 'fp_apply_0',
    }
    labels = {'d': 'D', 'h': 'H', 'p': 'P', 'd1': 'D1', 'h1': 'H1',
              'd2': 'D2', 'h2': 'H2', 't': 'T', 'center_wavelength': 'λ₀'}
    for index, candidate in enumerate(candidates):
        parameters = candidate['parameters']
        params_text = ' · '.join(
            f"{labels.get(key, key)}={value:.1f}nm" for key, value in parameters.items())
        contract = {'method': candidate['route_label'], 'model': candidate['model_version'],
                    'boundary': candidate['boundary']}

        def apply_saved(_parameters=dict(parameters), _structure=candidate['structure_type']):
            _apply_inverse_candidate(context, _structure, _parameters)

        _render_inverse_candidate_card(
            candidate['rank'], candidate['predicted_hex'], candidate['predicted_rgb255'],
            candidate['delta_e2000'], params_text, contract,
            context.material, context.substrate, context.polarization, context.angle_deg,
            apply_key=first_keys[run.method_id] if index == 0 else f"apply_saved_{run.method_id}_{index}",
            apply_callback=apply_saved,
        )
    if best['delta_e2000'] > 20:
        st.warning("目标色与候选仍有较大差异，可换目标色、材料或结构继续搜索。")


st.title("超表面结构色设计")
st.caption("输入目标颜色，搜索结构，查看光谱并导出结果。")
st.markdown(
    """
    <style>
      /* ===== 设计令牌:与 competition 展示页同源 =====
         强调色全部取自本项目正向求解器实际可生成的结构色
         (TiO2(anatase)/SiO2 单柱,engine.compute_spectrum 只读采样),
         括号内为该颜色对应的纳米柱几何。 */
      :root {
        --bg-deep: #0C0812;
        --bg-surface: #18121E;
        --bg-elevated: #251F2B;
        --border-subtle: #342D3C;
        --border-strong: #51495C;
        --text-primary: #F3F5F5;
        --text-secondary: #AFB1B2;
        --text-muted: #888B8C;
        --accent:        #FDA765;  /* 琥珀 D=250 H=140 P=300nm */
        --accent-strong: #FC6B2E;  /* 焰橙 D=300 H=100 P=420nm */
        --accent-soft:   #F3D259;  /* 金   D=220 H=100 P=300nm */
        --accent-cool:   #C191F5;  /* 紫   D=80  H=540 P=300nm */
        --focus-ring:    #FDA765;
        --focus-halo:    rgba(253, 167, 101, .26);
        --scrim:         rgba(12, 8, 18, .78);
      }
      .block-container { padding-top: clamp(1rem, 3vw, 2rem) !important; padding-left: clamp(.75rem, 3vw, 3rem) !important; padding-right: clamp(.75rem, 3vw, 3rem) !important; }
      [data-testid="stAppViewContainer"] { overflow-x: hidden; }
      [data-testid="stSidebar"] * { overflow-wrap: anywhere; word-break: break-word; }
      h1 { font-size: clamp(2rem, 3.1vw, 2.75rem) !important; line-height: 1.12 !important; overflow-wrap: anywhere; word-break: break-word; margin-bottom: .25rem !important; letter-spacing: 0; }
      .competition-banner { display:flex; align-items:center; gap:10px; flex-wrap:wrap; min-width:0; margin:10px 0 14px; padding:10px 12px; border:1px solid var(--border-strong); border-left:4px solid var(--accent); border-radius:10px; background:linear-gradient(90deg, #1A1024, var(--bg-surface)); color:var(--text-primary); font-size:13px; line-height:1.5; }
      .competition-banner strong { color:var(--accent-soft); }
      [data-baseweb="tab-list"] { gap: 2px; overflow-x: auto; flex-wrap: nowrap; scrollbar-width: none; -ms-overflow-style:none; }
      [data-baseweb="tab-list"]::-webkit-scrollbar { display:none; }
      [data-baseweb="tab"] { flex:0 0 auto; min-width:max-content; padding:7px 12px; white-space:nowrap; }
      [data-baseweb="tab"], [data-baseweb="tab"] p, [role="tab"] { color: var(--text-primary) !important; }
      [data-baseweb="tab"][aria-selected="true"],
      [data-baseweb="tab"][aria-selected="true"] p,
      [role="tab"][aria-selected="true"] { color: var(--accent) !important; }
      [data-baseweb="tab-highlight"] { background: var(--accent) !important; }
      .workflow-hint { min-width:0; margin:8px 0 14px; color:var(--text-muted); font-size:12px; line-height:1.5; overflow-wrap:anywhere; }
      .source-summary { width:100%; max-width:100%; min-width:0; box-sizing:border-box; overflow-wrap:anywhere; word-break:break-word; border:1px solid var(--border-subtle); border-left:4px solid var(--border-strong); border-radius:10px; padding:12px 14px; margin:0 0 10px; background:var(--bg-surface); color:var(--text-primary); }
      .source-summary__head { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
      .source-summary__status { color:var(--text-secondary); font-size:12px; margin-left:auto; }
      .source-summary__context { color:var(--text-muted); font-size:12px; line-height:1.5; margin-top:7px; }
      .inverse-context { min-width:0; border:1px solid var(--border-subtle); border-radius:8px; padding:10px 12px; margin:8px 0; background:var(--bg-elevated); color:var(--text-primary); }
      .inverse-context__grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:7px 14px; margin-top:8px; font-size:12px; }
      .inverse-context__grid span { min-width:0; overflow-wrap:anywhere; word-break:break-word; }
      .inverse-context__grid b { display:block; color:var(--text-muted); font-weight:600; margin-bottom:1px; }
      .inverse-target-card { display:flex; align-items:center; gap:16px; width:100%; min-width:0; box-sizing:border-box; border:1px solid var(--border-subtle); border-radius:8px; padding:12px; background:var(--bg-surface); color:var(--text-primary); }
      .inverse-target-card__swatch { width:88px; height:88px; flex:0 0 88px; border-radius:8px; border:2px solid rgba(255,255,255,.35); box-shadow:0 4px 14px rgba(0,0,0,.2); }
      .inverse-target-card__text { min-width:0; overflow-wrap:anywhere; word-break:break-word; }
      .inverse-target-card__hex { font-size:20px; font-weight:700; color:var(--text-primary); }
      .inverse-target-card__rgb { margin-top:4px; color:var(--text-secondary); font-size:13px; }
      .inverse-method-row { min-width:0; padding:5px 0 2px; }
      .inverse-method-row strong { color:var(--text-primary); }
      .inverse-method-row__status { display:inline-block; margin-left:6px; padding:1px 6px; border:1px solid var(--border-strong); border-radius:999px; color:var(--text-secondary); font-size:11px; font-style:normal; }
      .inverse-method-row span { display:block; color:var(--text-muted); font-size:12px; line-height:1.45; overflow-wrap:anywhere; word-break:break-word; }
      .pattern-boundary { min-width:0; border-left:4px solid var(--accent-strong); padding:9px 12px; margin:8px 0 14px; background:var(--bg-elevated); color:var(--text-primary); line-height:1.5; overflow-wrap:anywhere; word-break:break-word; }
      .pattern-boundary strong { color:var(--text-primary); }
      .pattern-boundary span { color:var(--text-muted); font-size:12px; }
      .pattern-empty { min-width:0; padding:10px 0 4px; color:var(--text-secondary); line-height:1.55; }
      .pattern-empty strong { color:var(--text-primary); }
      .pattern-empty ol { margin:7px 0 0; padding-left:1.35rem; }
      .pattern-empty li { margin:4px 0; }
      .mapping-summary { display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:8px; margin:8px 0 10px; }
      .mapping-summary__item { min-width:0; border:1px solid var(--border-subtle); border-radius:8px; padding:9px 11px; background:var(--bg-elevated); color:var(--text-secondary); font-size:12px; line-height:1.5; overflow-wrap:anywhere; }
      .mapping-summary__item strong { display:block; color:var(--text-primary); margin-bottom:2px; }
      .mapping-legend { display:flex; align-items:center; gap:8px 16px; flex-wrap:wrap; margin:7px 0 10px; color:var(--text-secondary); font-size:12px; }
      .mapping-legend span { display:inline-flex; align-items:center; gap:6px; }
      /* 图例"可用"色块用真实色域切片,而不是一个本系统生成不出来的蓝 */
      .mapping-legend i { width:18px; height:14px; border-radius:3px; display:inline-block; border:1px solid var(--border-strong); background:linear-gradient(135deg,#E05100,#BB9B00,#758E47,#463757); }
      .mapping-legend .mapping-legend__unavailable { background:var(--bg-elevated); border-style:dashed; }
      .mapping-legend .mapping-legend__current { background:var(--bg-elevated); border:2px solid var(--accent-soft); }
      .mapping-table-wrap { width:100%; max-width:100%; overflow-x:auto; padding:0 0 4px; scrollbar-color:var(--border-strong) var(--bg-surface); }
      .mapping-grid { width:100%; min-width:720px; border-collapse:separate; border-spacing:4px; table-layout:fixed; }
      .mapping-grid caption { position:absolute; width:1px; height:1px; padding:0; margin:-1px; overflow:hidden; clip:rect(0,0,0,0); white-space:nowrap; border:0; }
      .mapping-grid th { color:var(--text-secondary); font-size:11px; font-weight:650; text-align:center; padding:3px 2px; }
      .mapping-grid th[scope="row"] { width:64px; }
      .mapping-cell { position:relative; box-sizing:border-box; width:100%; min-height:64px; border:1px solid rgba(255,255,255,.16); border-radius:6px; display:flex; align-items:flex-end; justify-content:center; padding:5px; overflow:hidden; outline:none; }
      .mapping-cell:focus-visible { outline:3px solid var(--focus-ring); outline-offset:2px; }
      /* Mapping labels sit on a deliberately dark translucent scrim.  They
         therefore need an overlay-specific foreground instead of the page
         text token; the latter is dark in light mode and becomes unreadable
         on the saturated mapping colors. */
      .mapping-cell__value { width:100%; border-radius:4px; padding:3px 4px; background:var(--scrim); color:#fff !important; font-size:10px; line-height:1.25; text-align:center; text-shadow:0 1px 2px rgba(0,0,0,.48); }
      .mapping-cell--unavailable { align-items:center; border:1px dashed var(--border-strong); background:var(--bg-elevated); }
      .mapping-cell--unavailable .mapping-cell__value { background:transparent; color:var(--text-secondary); text-shadow:none; }
      .mapping-cell--current { border:3px solid var(--accent-soft); box-shadow:0 0 0 1px var(--scrim); }
      .mapping-cell__current-label { position:absolute; top:4px; right:4px; border-radius:3px; padding:2px 4px; background:var(--accent-soft); color:var(--bg-deep); font-size:9px; font-weight:800; line-height:1.1; }
      .audit-details { min-width:0; overflow-wrap:anywhere; word-break:break-word; color:var(--text-primary); font-size:12px; line-height:1.5; background:var(--bg-elevated); border:1px solid var(--border-subtle); border-radius:10px; padding:12px 14px; }
      .audit-details__grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(145px,1fr)); gap:8px 14px; }
      .audit-details__section { border-top:1px solid var(--border-subtle); margin-top:10px; padding:10px 0 0; color:var(--text-primary); }
      .audit-details__section span { color:var(--text-muted); font-weight:600; }
      .audit-details__evidence { margin-top:12px; border:1px solid var(--border-subtle); border-radius:8px; background:var(--bg-deep); overflow:hidden; }
      .audit-details__evidence > summary { cursor:pointer; color:var(--text-primary); font-weight:600; padding:9px 10px; list-style-position:inside; }
      .audit-details__evidence > summary:hover { background:var(--bg-elevated); }
      .audit-details__evidence-content { border-top:1px solid var(--border-subtle); padding:10px; color:var(--text-primary); font-family:ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace; white-space:pre-wrap; overflow-wrap:anywhere; word-break:break-word; overflow-x:auto; }
      [data-testid="stExpander"] details > summary { border-color:var(--border-subtle) !important; box-shadow:none !important; }
      [data-testid="stExpander"] details > summary:focus,
      [data-testid="stExpander"] details > summary:focus-visible,
      .audit-details__evidence > summary:focus,
      .audit-details__evidence > summary:focus-visible { outline:2px solid var(--focus-ring) !important; outline-offset:2px; border-color:var(--focus-ring) !important; box-shadow:0 0 0 2px var(--focus-halo) !important; }
      [data-testid="stButton"] button:focus,
      [data-testid="stButton"] button:focus-visible { outline:2px solid var(--focus-ring) !important; outline-offset:2px; border-color:var(--focus-ring) !important; box-shadow:0 0 0 2px var(--focus-halo) !important; }
      @media (max-width: 560px) {
        .block-container { padding-top:4.25rem !important; }
        h1 { font-size:2rem !important; line-height:1.16 !important; }
        h3 { font-size:1.35rem !important; line-height:1.25 !important; }
        .source-summary__status { margin-left:0; width:100%; }
        .inverse-target-card { align-items:flex-start; gap:12px; }
        .inverse-target-card__swatch { width:72px; height:72px; flex-basis:72px; }
        .mapping-summary { grid-template-columns:1fr; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <style id="ui-theme-vars">
      /* The selected palette is emitted after the shared stylesheet so it
         overrides the default tokens without changing the existing layout. */
      :root {{
        --bg-deep: {_UI_STYLE['bg_deep']};
        --bg-surface: {_UI_STYLE['bg_surface']};
        --bg-elevated: {_UI_STYLE['bg_elevated']};
        --border-subtle: {_UI_STYLE['border_subtle']};
        --border-strong: {_UI_STYLE['border_strong']};
        --text-primary: {_UI_STYLE['text_primary']};
        --text-secondary: {_UI_STYLE['text_secondary']};
        --text-muted: {_UI_STYLE['text_muted']};
        --accent: {_UI_STYLE['accent']};
        --accent-strong: {_UI_STYLE['accent_strong']};
        --accent-soft: {_UI_STYLE['accent_soft']};
        --accent-cool: {_UI_STYLE['accent_cool']};
        --focus-ring: {_UI_STYLE['accent']};
        --focus-halo: color-mix(in srgb, {_UI_STYLE['accent']} 28%, transparent);
        --scrim: {_UI_STYLE['scrim']};
        --preview-start: {_UI_STYLE['preview_start']};
        --preview-end: {_UI_STYLE['preview_end']};
        --substrate-start: {_UI_STYLE['substrate_start']};
        --substrate-end: {_UI_STYLE['substrate_end']};
        --highlight-surface: {_UI_STYLE['highlight_surface']};
        --button-bg: {_UI_STYLE['button_bg']};
        --button-text: {_UI_STYLE['button_text']};
        --accent-ink: {_UI_STYLE['accent_ink']};
        --code-bg: {_UI_STYLE['code_bg']};
        --tooltip-bg: {_UI_STYLE['tooltip_bg']};
        --tooltip-text: {_UI_STYLE['tooltip_text']};
        --tooltip-border: {_UI_STYLE['tooltip_border']};
        --icon-color: {_UI_STYLE['icon_color']};
        --status-good: {_UI_STYLE['status_good']};
        --status-warn: {_UI_STYLE['status_warn']};
        --status-bad: {_UI_STYLE['status_bad']};
        /* Kept as compatibility tokens for the audited light-mode palette
           (#e5e7eb, #4f9f9a); components still consume theme-aware variables. */
      }}
      html, body, .stApp {{ background: var(--bg-deep) !important; color: var(--text-primary); }}
      [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {{ background: var(--bg-deep) !important; color: var(--text-primary); }}
      [data-testid="stHeader"] {{ background: var(--bg-deep) !important; }}
      [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"],
      [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
      label, legend, .stRadio p, .stSelectbox p {{ color: var(--text-primary); }}
      [data-testid="stCaptionContainer"] {{ color: var(--text-muted) !important; }}
      [data-testid="stButton"] button, [data-testid="stDownloadButton"] button {{
        background: var(--button-bg) !important;
        color: var(--button-text) !important;
        border-color: var(--border-strong) !important;
      }}
       [data-testid="stButton"] button:hover, [data-testid="stDownloadButton"] button:hover {{
         border-color: var(--accent) !important;
         color: var(--accent) !important;
       }}
       [data-baseweb="button-group"] {{
         background: var(--bg-surface) !important;
         border-radius: 8px;
         display: flex !important;
         flex-wrap: nowrap !important;
         width: 100% !important;
         overflow: visible !important;
       }}
       [data-baseweb="button-group"] > button {{
         flex: 1 1 0 !important;
         min-width: 0 !important;
         width: auto !important;
         padding-left: 5px !important;
         padding-right: 5px !important;
         white-space: nowrap !important;
         font-size: 13px !important;
       }}
       [data-baseweb="button-group"] > button p {{
         margin: 0 !important;
         white-space: nowrap !important;
         font-size: inherit !important;
       }}
       /* The sidebar has four compact theme/accent choices.  Keep the labels
          readable at narrow widths instead of letting BaseWeb clip them. */
       [data-testid="stSidebar"] [data-baseweb="button-group"] > button {{
         padding-left: 3px !important;
         padding-right: 3px !important;
         font-size: 12px !important;
       }}
       [data-testid="stSidebar"] [data-baseweb="button-group"] > button,
       [data-testid="stSidebar"] [data-baseweb="button-group"] > button > div,
       [data-testid="stSidebar"] [data-baseweb="button-group"] > button span,
       [data-testid="stSidebar"] [data-baseweb="button-group"] > button p {{
         overflow: visible !important;
         text-overflow: clip !important;
       }}
       /* Streamlit 1.4x exposes segmented controls as button-group; keep a
          semantic fallback for versions that omit data-baseweb on the group. */
       [data-testid="stSidebar"] [role="radiogroup"]:has(> button) {{
         display: flex !important;
         flex-wrap: nowrap !important;
         width: 100% !important;
         overflow: visible !important;
       }}
       [data-testid="stSidebar"] [role="radiogroup"]:has(> button) > button {{
         flex: 1 1 0 !important;
         min-width: 0 !important;
         width: auto !important;
         padding-left: 5px !important;
         padding-right: 5px !important;
         white-space: nowrap !important;
         font-size: 13px !important;
       }}
       [data-testid="stSidebar"] [role="radiogroup"]:has(> button) > button,
       [data-testid="stSidebar"] [role="radiogroup"]:has(> button) > button > div,
       [data-testid="stSidebar"] [role="radiogroup"]:has(> button) > button span,
       [data-testid="stSidebar"] [role="radiogroup"]:has(> button) > button p {{
         overflow: visible !important;
         text-overflow: clip !important;
       }}
       [data-baseweb="button-group"] button[data-testid^="stBaseButton-segmented_control"] {{
         background: var(--button-bg) !important;
         color: var(--button-text) !important;
         border: 1px solid var(--border-strong) !important;
       }}
       [data-baseweb="button-group"] button[data-testid="stBaseButton-segmented_controlActive"] {{
         background: var(--accent) !important;
         color: var(--accent-ink) !important;
         border-color: var(--accent) !important;
       }}
       [data-baseweb="button-group"] button[data-testid^="stBaseButton-segmented_control"]:hover {{
         border-color: var(--accent) !important;
         color: var(--accent-ink) !important;
         background: var(--accent) !important;
       }}
       /* Streamlit's native toggles use BaseWeb-generated class names. Keep
          the semantic data attributes as the stable theme hook. */
       [data-baseweb="checkbox"] > span,
       [data-baseweb="radio"] > div > div:first-child {{
         box-sizing: border-box !important;
         background: var(--button-bg) !important;
         border: 1px solid var(--border-strong) !important;
         color: var(--text-secondary) !important;
         width: 16px !important;
         height: 16px !important;
         border-radius: 50% !important;
       }}
       [data-baseweb="checkbox"]:has(input:checked) > span,
       [data-baseweb="radio"]:has(input:checked) > div > div:first-child {{
         background: var(--accent) !important;
         border-color: var(--accent) !important;
         color: var(--accent-ink) !important;
       }}
       [data-baseweb="checkbox"]:has(input:checked) > span::after {{
         content: "" !important;
         display: block !important;
         width: 7px !important;
         height: 11px !important;
         margin: 1px 0 0 4px !important;
         border: solid var(--accent-ink) !important;
         border-width: 0 2px 2px 0 !important;
         transform: rotate(45deg) !important;
       }}
       [data-baseweb="radio"]:has(input:checked) > div > div:first-child::after {{
         content: "" !important;
         display: block !important;
         width: 6px !important;
         height: 6px !important;
         margin: 4px !important;
         border-radius: 50% !important;
         background: var(--accent-ink) !important;
       }}
       [data-baseweb="checkbox"] input:disabled ~ div,
       [data-baseweb="radio"] input:disabled ~ div {{
         color: var(--text-primary) !important;
       }}
       [data-testid="stFileUploaderDropzone"] {{
         background: var(--bg-elevated) !important;
         border: 1px dashed var(--border-strong) !important;
         color: var(--text-secondary) !important;
       }}
       [data-testid="stFileUploaderDropzone"] button {{
         background: var(--button-bg) !important;
         color: var(--button-text) !important;
         border-color: var(--border-strong) !important;
       }}
       [data-testid="stFileUploaderDropzone"] button:hover {{
         border-color: var(--accent) !important;
         color: var(--accent) !important;
       }}
       [data-testid="stFileUploaderDropzoneInstructions"] span {{
         color: var(--text-muted) !important;
       }}
       [data-baseweb="select"] > div, [data-baseweb="input"] > div,
      [data-baseweb="textarea"] > div {{
        background: var(--bg-surface) !important;
        border-color: var(--border-subtle) !important;
        color: var(--text-primary) !important;
      }}
       [data-baseweb="select"] input, [data-baseweb="input"] input,
       [data-baseweb="textarea"] textarea {{ color: var(--text-primary) !important; }}
       [data-baseweb="select"] svg, [data-baseweb="select"] [role="img"] {{
         color: var(--icon-color) !important; stroke: currentColor !important; fill: currentColor !important;
       }}
       [data-baseweb="tooltip"], [data-baseweb="tooltip"] > div,
       [data-testid="stTooltipContent"], [data-testid="stTooltipContent"] * {{
         background: var(--tooltip-bg) !important; color: var(--tooltip-text) !important;
         border-color: var(--tooltip-border) !important;
       }}
       [data-baseweb="tooltip"] {{ box-shadow: 0 8px 24px rgba(0,0,0,.24) !important; z-index: 1000 !important; }}
       [role="radiogroup"] label {{ color: var(--text-primary) !important; }}
       /* Keep the structure selector readable in the narrow sidebar.  The
          default BaseWeb rule lets labels shrink below their text width,
          which turns the long FP label into a vertical strip. */
       /* The structure selector is a narrow-sidebar control.  Keep it as a
          clean vertical list so the long FP label never wraps onto a second
          row or inherits BaseWeb's selected text chip. */
        [data-testid="stSidebar"] .stRadio [role="radiogroup"] {{
          display: grid !important;
          grid-template-columns: minmax(0, 1fr) !important;
          gap: 6px !important;
          align-items: stretch !important;
          width: 100% !important;
          max-width: 100% !important;
          overflow: visible !important;
        }}
        [data-testid="stSidebar"] .stRadio [role="radiogroup"] > label {{
          display: flex !important;
          flex: 0 0 auto !important;
          align-items: center !important;
          gap: 8px !important;
          box-sizing: border-box !important;
          width: 100% !important;
          min-width: 0 !important;
          max-width: 100% !important;
          min-height: 30px !important;
          margin: 0 !important;
          padding: 6px 8px !important;
          border: 1px solid transparent !important;
          border-radius: 8px !important;
          background: transparent !important;
          white-space: nowrap !important;
          overflow: visible !important;
        }}
        [data-testid="stSidebar"] .stRadio [role="radiogroup"] > label:has(input:checked) {{
          background: color-mix(in srgb, var(--accent) 12%, var(--bg-surface)) !important;
          border-color: color-mix(in srgb, var(--accent) 62%, var(--border-subtle)) !important;
        }}
        [data-testid="stSidebar"] .stRadio [role="radiogroup"] > label > div:first-child {{
          flex: 0 0 16px !important;
          width: 16px !important;
          height: 16px !important;
          margin: 0 !important;
        }}
        [data-testid="stSidebar"] .stRadio [role="radiogroup"] > label > div:last-child {{
          flex: 1 1 auto !important;
          width: auto !important;
          min-width: 0 !important;
          max-width: none !important;
          padding: 0 !important;
          background: transparent !important;
          border: 0 !important;
          color: var(--text-primary) !important;
          white-space: nowrap !important;
          overflow: visible !important;
        }}
        [data-testid="stSidebar"] .stRadio [role="radiogroup"] > label > div:last-child > [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] .stRadio [role="radiogroup"] > label > div:last-child p {{
          width: auto !important;
          min-width: 0 !important;
          max-width: none !important;
          margin: 0 !important;
          padding: 0 !important;
          background: transparent !important;
          border: 0 !important;
          color: var(--text-primary) !important;
          white-space: nowrap !important;
          overflow: visible !important;
        }}
       /* Override BaseWeb's fixed translucent swatches on both layers so
          light/dark/high-contrast modes use the same theme tokens. */
       [data-baseweb="checkbox"] > span,
       [data-baseweb="radio"] > div > div:first-child {{
         background: var(--button-bg) !important;
         border-color: var(--border-strong) !important;
       }}
       /* BaseWeb radios have a second outer shell around the visible dot.
          Theme both layers; otherwise the shell keeps Streamlit's default
          amber/dark color when the palette changes. */
       [data-baseweb="radio"] > div:first-child {{
         box-sizing: border-box !important;
         background: var(--button-bg) !important;
         border-color: var(--border-strong) !important;
       }}
       [data-baseweb="checkbox"]:has(input:checked) > span,
       [data-baseweb="radio"]:has(input:checked) > div > div:first-child {{
         background: var(--accent) !important;
         border-color: var(--accent) !important;
       }}
       [data-baseweb="radio"]:has(input:checked) > div:first-child {{
         background: var(--accent) !important;
         border-color: var(--accent) !important;
       }}
        [data-testid="stExpander"] details {{
          background: var(--bg-surface) !important;
          color: var(--text-primary) !important;
          border: 1px solid var(--border-subtle) !important;
          box-shadow: none !important;
        }}
        [data-testid="stExpander"] details > summary {{
          background: var(--bg-surface) !important;
          color: var(--text-primary) !important;
          border: 0 !important;
          box-shadow: none !important;
        }}
        [data-testid="stExpander"] details:has(> summary:focus),
        [data-testid="stExpander"] details:has(> summary:focus-visible) {{
          border-color: var(--focus-ring) !important;
          box-shadow: 0 0 0 2px var(--focus-halo) !important;
        }}
        [data-testid="stExpander"] details > summary:focus,
        [data-testid="stExpander"] details > summary:focus-visible {{
          outline: 2px solid var(--focus-ring) !important;
          outline-offset: -2px !important;
          border-color: transparent !important;
          box-shadow: none !important;
        }}
       /* The benchmark table is intentionally wider than a narrow half-column;
          keep numeric columns on one line and let the route column wrap at
          phrase boundaries instead of splitting every character. */
       [data-testid="stMarkdownContainer"]:has(> table:not(.mapping-grid)) {{
         max-width: 100% !important;
         overflow-x: auto !important;
       }}
       [data-testid="stMarkdownContainer"] > table:not(.mapping-grid) {{
         width: 100% !important;
         min-width: 620px !important;
         table-layout: auto !important;
         border-collapse: collapse !important;
       }}
       [data-testid="stMarkdownContainer"] > table:not(.mapping-grid) th,
       [data-testid="stMarkdownContainer"] > table:not(.mapping-grid) td {{
         padding: 7px 10px !important;
         vertical-align: top !important;
         word-break: normal !important;
         overflow-wrap: anywhere !important;
       }}
       [data-testid="stMarkdownContainer"] > table:not(.mapping-grid) th:not(:last-child),
       [data-testid="stMarkdownContainer"] > table:not(.mapping-grid) td:not(:last-child) {{
         white-space: nowrap !important;
       }}
       [data-testid="stMarkdownContainer"] > table:not(.mapping-grid) th:last-child,
       [data-testid="stMarkdownContainer"] > table:not(.mapping-grid) td:last-child {{
         min-width: 230px !important;
       }}
       .competition-banner {{ background: linear-gradient(90deg, var(--bg-elevated), var(--bg-surface)); }}
       .theme-swatch {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px; background:var(--accent); }}
       /* Streamlit alerts keep a generated blue/yellow surface unless both
          the container and its semantic kind are themed explicitly. */
       [data-testid="stAlertContainer"] {{
         background:var(--bg-elevated) !important;
         color:var(--text-primary) !important;
         border:1px solid var(--border-subtle) !important;
         border-left:3px solid var(--accent-cool) !important;
         border-radius:8px !important;
       }}
       [data-testid="stAlertContainer"] [data-testid="stMarkdownContainer"],
       [data-testid="stAlertContainer"] [data-testid="stMarkdownContainer"] p {{
         color:var(--text-primary) !important;
       }}
       [data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {{
         background:color-mix(in srgb, var(--status-warn) 12%, var(--bg-surface)) !important;
         border-left-color:var(--status-warn) !important;
       }}
       [data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) {{
         background:color-mix(in srgb, var(--status-bad) 12%, var(--bg-surface)) !important;
         border-left-color:var(--status-bad) !important;
       }}
       [data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {{
         background:color-mix(in srgb, var(--status-good) 12%, var(--bg-surface)) !important;
         border-left-color:var(--status-good) !important;
       }}
       [data-baseweb="select"] svg, [data-baseweb="select"] [role="img"] {{ color:var(--icon-color) !important; stroke:currentColor !important; fill:currentColor !important; }}
       [data-baseweb="tooltip"], [data-baseweb="tooltip"] > div,
       [data-testid="stTooltipContent"], [data-testid="stTooltipContent"] * {{
         background:var(--tooltip-bg) !important; color:var(--tooltip-text) !important;
         border-color:var(--tooltip-border) !important;
       }}
       [data-baseweb="tooltip"] {{ box-shadow:0 8px 24px rgba(0,0,0,.24) !important; z-index:1000 !important; }}
       .status-good {{ color:var(--status-good) !important; }}
       .status-warn {{ color:var(--status-warn) !important; }}
       .status-bad {{ color:var(--status-bad) !important; }}
       .route-badge {{ background:var(--accent) !important; color:var(--accent-ink) !important; }}
     </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="workflow-hint"
         aria-label="工作流：侧栏设置参数 → 预览颜色与来源 → 查看结果 → 导出记录。">
      <strong>使用顺序：</strong>输入目标颜色 → 调整结构 → 查看结果 → 导出记录。
    </div>
    """,
    unsafe_allow_html=True,
)
_audited_sample_notice = st.session_state.pop("_audited_sample_notice", "")
if _audited_sample_notice:
    st.success(_audited_sample_notice)

# Sidebar controls
with st.sidebar:
    st.header('⚙️ 参数控制')
    st.button(
        "加载可复核示例",
        key="load_audited_reference_sample",
        use_container_width=True,
        on_click=_load_audited_reference_sample,
        help=(
            "点击后自动填入一组已存在于审核 RCWA 参考库中的精确几何，"
            "并立即可在预览、光谱和参考复核中查看；不会插值、训练或修改数据。"
        ),
    )
    _structure_options = ['单柱', '双柱', 'FP 腔（Fabry-Pérot）']
    st.radio(
        '📏 结构类型',
        _structure_options,
        key="structure_type_control",
        on_change=sync_enum_from_widget,
        args=(st.session_state, _ENUM_CONTROLS["structure"]),
        horizontal=False,
        help='单柱/双柱纳米柱或法布里-珀罗腔'
    )
    is_fp = st.session_state.structure_type == 'fp'
    is_dual = st.session_state.structure_type == 'dual'
    st.session_state.dual_pillar = is_dual

    _pillar_materials = list(_PILLAR_MATERIAL_OPTIONS)
    _substrate_materials = list(_SUBSTRATE_OPTIONS)

    st.divider()
    if is_fp:
        material = st.session_state._pillar_material_pref
        substrate = st.session_state._substrate_pref
        st.caption(
            "FP-TMM 使用腔体内固定材料；柱材料、衬底与 Δn 控件本路线未使用。"
            "返回单柱或双柱后恢复先前选择。"
        )
    else:
        st.selectbox(
            '柱材料', _pillar_materials, key="pillar_material_control",
            on_change=sync_enum_from_widget,
            args=(st.session_state, _ENUM_CONTROLS["material"]),
        )
        st.selectbox(
            '衬底材料', _substrate_materials, key="substrate_control",
            on_change=sync_enum_from_widget,
            args=(st.session_state, _ENUM_CONTROLS["substrate"]),
        )
        material = st.session_state._pillar_material_pref
        substrate = st.session_state._substrate_pref

        # --- Delta-n indicator (paper Fig.2 criterion) ---
        n_pillar = MaterialLibrary.n_at_wavelength(material, 550)
        n_sub = MaterialLibrary.n_at_wavelength(substrate, 550)
        delta_n = n_pillar - n_sub
        if delta_n > 0.6:
            dn_class, dn_label = "status-good", f"折射率差 Δn = {delta_n:.2f}｜预计色域较宽"
        elif delta_n > 0.4:
            dn_class, dn_label = "status-warn", f"折射率差 Δn = {delta_n:.2f}｜接近截止，色域受限"
        else:
            dn_class, dn_label = "status-bad", f"折射率差 Δn = {delta_n:.2f}｜低于截止，难以形成结构色"
        st.markdown(
            f"""<div class="{dn_class}" style="background:color-mix(in srgb, currentColor 10%, var(--bg-surface)); border-left:3px solid currentColor;
            padding:6px 10px; border-radius:4px; margin:4px 0; font-size:0.82rem;">
            {dn_label}</div>""",
            unsafe_allow_html=True
        )

    st.selectbox(
        '偏振', list(_POLARIZATION_OPTIONS), key="polarization_control",
        on_change=sync_enum_from_widget,
        args=(st.session_state, _ENUM_CONTROLS["polarization"]),
    )
    polarization = st.session_state.polarization
    col_a1, col_a2 = st.columns([3, 1])
    with col_a1:
        st.slider(
            '入射角 (°)', 0.0, 80.0, step=0.1, key="angle_slider",
            on_change=sync_numeric_from_widget,
            args=(st.session_state, _NUMERIC_CONTROLS["angle"], "angle_slider"),
        )
    with col_a2:
        st.number_input(
            '精确输入 角度', 0.0, 80.0, step=0.1, key="angle_input",
            on_change=sync_numeric_from_widget,
            args=(st.session_state, _NUMERIC_CONTROLS["angle"], "angle_input"),
        )
    angle = float(st.session_state.a_val)

    st.divider()
    st.header('🔭 远场传播 (角谱理论)')
    if is_fp:
        st.caption(
            "FP-TMM 是平面薄膜腔路线，本路线未使用角谱远场。"
            f"先前远场偏好已保留为{'开启' if st.session_state.far_field else '关闭'}。"
        )
    else:
        st.checkbox(
            '启用角谱远场传播 (Angular Spectrum)', key="far_field_control",
            on_change=sync_bool_from_widget,
            args=(st.session_state, _BOOL_CONTROLS["far_field"]),
            help='N×N超表面阵列FFT角谱 + NA锥积分，计算探测器实际接收光谱',
        )
    if not is_fp and st.session_state.far_field:
        col_t1, col_t2 = st.columns([3, 1])
        with col_t1:
            st.slider(
                '观察角度 θ (°)', 0.0, 80.0, step=1.0, key="theta_obs_slider",
                on_change=sync_numeric_from_widget,
                args=(st.session_state, _NUMERIC_CONTROLS["theta_obs"], "theta_obs_slider"),
                help='观察方向偏离法线的角度, 影响角谱中心位置'
            )
        with col_t2:
            st.number_input(
                '精确 θ', 0.0, 80.0, step=1.0, key="theta_obs_input",
                on_change=sync_numeric_from_widget,
                args=(st.session_state, _NUMERIC_CONTROLS["theta_obs"], "theta_obs_input"),
            )
        col_n1, col_n2 = st.columns([3, 1])
        with col_n1:
            st.slider(
                '收集数值孔径 NA', 0.05, 0.95, step=0.01, key="na_slider",
                on_change=sync_numeric_from_widget,
                args=(st.session_state, _NUMERIC_CONTROLS["na"], "na_slider"),
                help='NA=0.1人眼瞳孔, NA=0.5显微镜20×, NA=0.95油镜100×'
            )
        with col_n2:
            st.number_input(
                '精确 NA', 0.05, 0.95, step=0.01, key="na_input",
                on_change=sync_numeric_from_widget,
                args=(st.session_state, _NUMERIC_CONTROLS["na"], "na_input"),
            )
        st.caption('👁 NA=0.1人眼 | 🔬 NA=0.5显微镜 | 🔍 NA=0.95油镜')
    elif not is_fp:
        st.caption('远场未启用；启用后显示观察角度与 NA 控件。')

    st.divider()
    st.header("ML 加速")
    _ml_loader_completed = False
    _current_ml_ready = False
    if (
        not is_fp
        and not st.session_state.far_field
        and material in ml_module.MATERIAL_CODES
    ):
        if is_dual:
            _dual_ready = bool(_ensure_dual_ml())
            _current_ml_ready = bool(
                _dual_ready and substrate == "SiO2 (fused silica)"
            )
        else:
            _current_ml_ready = bool(_ensure_ml())
        _ml_loader_completed = True
    if (
        not st.session_state[ML_ACCEL_PREFERENCE_INITIALIZED_KEY]
        and _ml_loader_completed
    ):
        set_bool_value(
            st.session_state, _BOOL_CONTROLS["ml_accel"], _current_ml_ready)
        st.session_state[ML_ACCEL_PREFERENCE_INITIALIZED_KEY] = True
    _ml_preference_label = '开启' if st.session_state.ml_accel else '关闭'
    _ml_route_applicable = bool(
        not is_fp
        and not st.session_state.far_field
        and material in ml_module.MATERIAL_CODES
        and ((_dual_ml_ready and substrate == "SiO2 (fused silica)") if is_dual else _ml_ready)
    )
    if _ml_route_applicable:
        _ml_control_label = (
            '启用双柱 ML 代理模型' if is_dual
            else '启用 ML 代理模型（快速候选预测）'
        )
        st.checkbox(
            _ml_control_label, key="ml_accel_control",
            on_change=sync_bool_preference_from_widget,
            args=(
                st.session_state, _BOOL_CONTROLS["ml_accel"],
                ML_ACCEL_PREFERENCE_INITIALIZED_KEY,
            ),
            help='使用当前结构路线注册的神经网络代理模型',
        )
        if is_dual:
            st.caption("双柱快速预测已就绪；仅支持已注册的 SiO₂ 衬底。")
        elif _ml_is_v8:
            st.caption("快速预测已就绪。")
        else:
            st.caption("快速预测已就绪。")
    else:
        if is_fp:
            _ml_inactive_reason = "FP-TMM 直接计算薄膜腔光谱"
        elif st.session_state.far_field:
            _ml_inactive_reason = "远场路线使用解析局部响应 + 角谱/NA 后处理"
        elif is_dual and not _dual_ml_ready:
            _ml_inactive_reason = f"双柱 ONNX 模型不可用（{_dual_ml_error or '未加载'}）"
        elif is_dual and substrate != "SiO2 (fused silica)":
            _ml_inactive_reason = "双柱 ML 仅注册 SiO2 衬底"
        elif material not in ml_module.MATERIAL_CODES:
            _ml_inactive_reason = f"{material} 不在 ML 注册材料中"
        else:
            _ml_inactive_reason = _ml_error or "本地 ML 模型不可用"
        st.caption(
            f"本路线未使用 ML：{_ml_inactive_reason}。"
            f"用户 ML 偏好已保留为{_ml_preference_label}，返回适用路线后恢复。"
        )
    st.divider()
    st.header('📏 纳米柱尺寸')
    _single_geometry_invalid = False

    if st.session_state.dual_pillar:
        # 预验证: 在渲染滑块前先修正参数, 确保滑块显示修正后的值
        try:
            pre = DualPillarParam(
                st.session_state.get('d1_val', 120.0),
                st.session_state.get('h1_val', 250.0),
                st.session_state.get('d2_val', 200.0),
                st.session_state.get('h2_val', 350.0),
                st.session_state.p_val,
                material, substrate, polarization, angle
            )
            if pre._corrected:
                set_numeric_value(
                    st.session_state, _NUMERIC_CONTROLS["period"], pre.period_nm)
                set_numeric_value(
                    st.session_state, _NUMERIC_CONTROLS["dual_d1"], pre.d1_nm)
                set_numeric_value(
                    st.session_state, _NUMERIC_CONTROLS["dual_d2"], pre.d2_nm)
                st.session_state._dual_correction = pre._correction_msg
                st.session_state._prev_dual_params = (pre.d1_nm, pre.d2_nm, pre.period_nm)
        except Exception as e:
            logging.warning(f"dual param preview: {e}")

        # --- Dual-Pillar Controls ---
        col_d1, col_d2 = st.columns([3, 1])
        with col_d1:
            st.slider(
                '柱1直径 D1 (nm)', 50.0, 350.0, step=0.1,
                key="dual_d1_slider", on_change=sync_numeric_from_widget,
                args=(st.session_state, _NUMERIC_CONTROLS["dual_d1"], "dual_d1_slider"),
            )
        with col_d2:
            st.number_input(
                '精确输入 D1', 50.0, 350.0, step=0.1,
                key="dual_d1_input", on_change=sync_numeric_from_widget,
                args=(st.session_state, _NUMERIC_CONTROLS["dual_d1"], "dual_d1_input"),
            )

        col_h1, col_h2 = st.columns([3, 1])
        with col_h1:
            st.slider(
                '柱1高度 H1 (nm)', 80.0, 600.0, step=0.1,
                key="dual_h1_slider", on_change=sync_numeric_from_widget,
                args=(st.session_state, _NUMERIC_CONTROLS["dual_h1"], "dual_h1_slider"),
            )
        with col_h2:
            st.number_input(
                '精确输入 H1', 80.0, 600.0, step=0.1,
                key="dual_h1_input", on_change=sync_numeric_from_widget,
                args=(st.session_state, _NUMERIC_CONTROLS["dual_h1"], "dual_h1_input"),
            )

        col_d3, col_d4 = st.columns([3, 1])
        with col_d3:
            st.slider(
                '柱2直径 D2 (nm)', 50.0, 350.0, step=0.1,
                key="dual_d2_slider", on_change=sync_numeric_from_widget,
                args=(st.session_state, _NUMERIC_CONTROLS["dual_d2"], "dual_d2_slider"),
            )
        with col_d4:
            st.number_input(
                '精确输入 D2', 50.0, 350.0, step=0.1,
                key="dual_d2_input", on_change=sync_numeric_from_widget,
                args=(st.session_state, _NUMERIC_CONTROLS["dual_d2"], "dual_d2_input"),
            )

        col_h3, col_h4 = st.columns([3, 1])
        with col_h3:
            st.slider(
                '柱2高度 H2 (nm)', 80.0, 600.0, step=0.1,
                key="dual_h2_slider", on_change=sync_numeric_from_widget,
                args=(st.session_state, _NUMERIC_CONTROLS["dual_h2"], "dual_h2_slider"),
            )
        with col_h4:
            st.number_input(
                '精确输入 H2', 80.0, 600.0, step=0.1,
                key="dual_h2_input", on_change=sync_numeric_from_widget,
                args=(st.session_state, _NUMERIC_CONTROLS["dual_h2"], "dual_h2_input"),
            )

        col_p1, col_p2 = st.columns([3, 1])
        with col_p1:
            st.slider(
                '周期 P (nm)', 200.0, 600.0, step=0.1,
                key="dual_p_slider", on_change=sync_numeric_from_widget,
                args=(st.session_state, _NUMERIC_CONTROLS["period"], "dual_p_slider"),
            )
        with col_p2:
            st.number_input(
                '精确输入 P', 200.0, 600.0, step=0.1,
                key="dual_p_input", on_change=sync_numeric_from_widget,
                args=(st.session_state, _NUMERIC_CONTROLS["period"], "dual_p_input"),
            )

        diameter = st.session_state.d1_val  # for backward compat
        height = st.session_state.h1_val
        period = st.session_state.p_val

        # Validation
        d1, d2, pv = st.session_state.d1_val, st.session_state.d2_val, st.session_state.p_val
        fill1 = np.pi*(d1/2)**2/(pv**2)
        fill2 = np.pi*(d2/2)**2/(pv**2)
        if d1 >= pv or d2 >= pv:
            st.warning('⚠️ D1={:.0f} D2={:.0f} >= P={:.0f}: 超出单元边界'.format(d1, d2, pv))
        if fill1 + fill2 > 0.85:
            st.warning('⚠️ 占空比总和 {:.2f} > 0.85: 纳米柱可能重叠'.format(fill1+fill2))
    elif is_fp:
        # --- FP Cavity Controls ---
        st.selectbox(
            '反射镜类型', list(_FP_MIRROR_OPTIONS),
            key="fp_mirror_type_control", on_change=sync_enum_from_widget,
            args=(st.session_state, _ENUM_CONTROLS["fp_mirror"]),
        )

        if st.session_state.fp_mirror_type.startswith('介质'):
            col_c1, col_c2 = st.columns([3, 1])
            with col_c1:
                st.slider(
                    'DBR 中心波长 (nm)', 380.0, 780.0, step=5.0,
                    key="fp_center_slider", on_change=sync_numeric_from_widget,
                    args=(st.session_state, _NUMERIC_CONTROLS["fp_center"], "fp_center_slider"),
                )
            with col_c2:
                st.number_input(
                    '精确中心波长', 380.0, 780.0, step=5.0,
                    key="fp_center_input", on_change=sync_numeric_from_widget,
                    args=(st.session_state, _NUMERIC_CONTROLS["fp_center"], "fp_center_input"),
                )

        col_t1, col_t2 = st.columns([3, 1])
        with col_t1:
            st.slider(
                '腔长 T (nm)', 50.0, 600.0, step=1.0,
                key="fp_t_slider", on_change=sync_numeric_from_widget,
                args=(st.session_state, _NUMERIC_CONTROLS["fp_t"], "fp_t_slider"),
            )
        with col_t2:
            st.number_input(
                '精确输入 T', 50.0, 600.0, step=1.0,
                key="fp_t_input", on_change=sync_numeric_from_widget,
                args=(st.session_state, _NUMERIC_CONTROLS["fp_t"], "fp_t_input"),
            )
        diameter = 0; height = st.session_state.fp_t_val; period = 0
        if st.session_state.fp_mirror_type.startswith('介质'):
            st.caption('FP腔 (DBR): (TiO2/SiO2)3 / TiO2(T) / (SiO2/TiO2)5 | 高饱和度')
        else:
            st.caption('FP腔 (Ag): Ag(30nm) / TiO2(T) / Ag(bulk) | 减色型 | 颜色偏淡')
    else:
        # --- Single-Pillar Controls (original) ---
        col_d1, col_d2 = st.columns([3, 1])
        with col_d1:
            st.slider(
                '直径 D (nm)', 50.0, 350.0, step=0.1,
                key="single_d_slider", on_change=sync_numeric_from_widget,
                args=(st.session_state, _NUMERIC_CONTROLS["single_d"], "single_d_slider"),
            )
        with col_d2:
            st.number_input(
                '精确输入 D', 50.0, 350.0, step=0.1,
                key="single_d_input", on_change=sync_numeric_from_widget,
                args=(st.session_state, _NUMERIC_CONTROLS["single_d"], "single_d_input"),
            )
        diameter = st.session_state.d_val

        col_h1, col_h2 = st.columns([3, 1])
        with col_h1:
            st.slider(
                '高度 H (nm)', 80.0, 600.0, step=0.1,
                key="single_h_slider", on_change=sync_numeric_from_widget,
                args=(st.session_state, _NUMERIC_CONTROLS["single_h"], "single_h_slider"),
            )
        with col_h2:
            st.number_input(
                '精确输入 H', 80.0, 600.0, step=0.1,
                key="single_h_input", on_change=sync_numeric_from_widget,
                args=(st.session_state, _NUMERIC_CONTROLS["single_h"], "single_h_input"),
            )
        height = st.session_state.h_val

        col_p1, col_p2 = st.columns([3, 1])
        with col_p1:
            st.slider(
                '周期 P (nm)', 200.0, 600.0, step=0.1,
                key="single_p_slider", on_change=sync_numeric_from_widget,
                args=(st.session_state, _NUMERIC_CONTROLS["period"], "single_p_slider"),
            )
        with col_p2:
            st.number_input(
                '精确输入 P', 200.0, 600.0, step=0.1,
                key="single_p_input", on_change=sync_numeric_from_widget,
                args=(st.session_state, _NUMERIC_CONTROLS["period"], "single_p_input"),
            )
        period = st.session_state.p_val

        if diameter > period:
            st.warning('⚠️ D > P：纳米柱会重叠，请调整')

    _single_geometry_invalid = (
        not is_fp and not is_dual and float(diameter) > float(period)
    )

    st.divider()
    presets = {
        '紫罗兰': (150, 100), '蓝色': (80, 250), '青色': (140, 200),
        '翠绿': (180, 250), '黄色': (250, 220), '橙色': (290, 200), '红色': (310, 160),
    }
    # Quick presets removed: dual-Lorentzian model limits prevent accurate preset colors.
    # Use the inverse design tab for precise color matching.
    st.caption('颜色匹配请进入「逆设计」。')
    # cols = st.columns(4)
    # for i, (name, (d_val, h_val)) in enumerate(presets.items()):
    #     with cols[i % 4]:
    #         if st.button(name, key=f'preset_{name}', use_container_width=True,
    #                      help=f'D={d_val}nm H={h_val}nm'):
    #             st.session_state.d_val = float(d_val)
    #             st.session_state.h_val = float(h_val)
    #             st.rerun()


# Build param
if st.session_state.get('dual_pillar', False):
    param = DualPillarParam(
        d1_nm=st.session_state.d1_val, h1_nm=st.session_state.h1_val,
        d2_nm=st.session_state.d2_val, h2_nm=st.session_state.h2_val,
        period_nm=st.session_state.p_val,
        material=material, substrate=substrate,
        polarization=polarization, angle_deg=angle
    )
    # 检测用户手动拖滑块 → 清除旧修正提示
    if st.session_state.get('_dual_correction'):
        prev = st.session_state.get('_prev_dual_params', ())
        curr = (st.session_state.d1_val, st.session_state.d2_val, st.session_state.p_val)
        if curr != prev:
            st.session_state._dual_correction = ''
    st.session_state._prev_dual_params = (
        st.session_state.d1_val, st.session_state.d2_val, st.session_state.p_val
    )
elif not is_fp:
    param = MetaSurfaceParam(diameter, height, period, material, substrate, polarization, angle)
def _cached_physical_forward(d_nm, h_nm, p_nm, mat, sub, pol, ang, d2_nm, h2_nm,
                             dual, far_field, na, theta_obs):
    """Return the physical route's actual spectrum; color is derived from it."""
    configure_engine_far_field(engine, far_field, na, theta_obs)
    if dual:
        p = DualPillarParam(d1_nm=d_nm, h1_nm=h_nm, d2_nm=d2_nm, h2_nm=h2_nm,
                            period_nm=p_nm, material=mat, substrate=sub,
                            polarization=pol, angle_deg=ang)
    else:
        p = MetaSurfaceParam(d_nm, h_nm, p_nm, mat, sub, pol, ang)
    return engine.compute_spectrum(p, 380, 780, 81)


def _forward_result(wavelengths, reflectance, provenance, *, rgb=None, error=""):
    """Normalize one route output for every UI surface and export."""
    return normalize_forward_result(
        wavelengths, reflectance, provenance, rgb=rgb, error=error)


@st.cache_data(show_spinner=False)
def _cached_mapping_forward(
    route_id, model_version, artifact_version, d_nm, h_nm, p_nm, material, substrate,
    polarization, angle_deg, far_field, na, theta_obs,
):
    """Evaluate one map point through the selected forward family only."""
    del model_version, artifact_version  # Both remain part of the Streamlit cache key.
    try:
        if route_id == "rcwa_surrogate":
            with _bound_runtime_context(
                    "single", "rcwa_surrogate", material, substrate):
                spectrum = _predict_rcwa_spectrum_strict(
                    d_nm, h_nm, p_nm, angle_deg, polarization, material, substrate)
            if spectrum is None:
                return ForwardResult(None, None, None, {}, False, "RCWA 代理未返回光谱")
            return _forward_result(ml_module.WL, spectrum, {})
        if route_id == "ml_surrogate":
            with _bound_runtime_context(
                    "single", "ml_surrogate", material, substrate):
                spectrum = ml_module.predict_generic_spectrum(
                    d_nm, h_nm, p_nm, angle_deg, polarization, material, substrate)
            if spectrum is None:
                return ForwardResult(None, None, None, {}, False, "通用 ML 代理未返回光谱")
            return _forward_result(ml_module.WL, spectrum, {})
        if route_id in {"lorentz_fano_fallback", "far_field_postprocessing"}:
            wavelengths, reflectance = _cached_physical_forward(
                d_nm, h_nm, p_nm, material, substrate, polarization, angle_deg,
                0.0, 0.0, False, far_field, na, theta_obs)
            return _forward_result(wavelengths, reflectance, {})
    except (ModelResourceDriftError, ModelResourceUnavailable):
        raise
    except Exception as exc:
        return ForwardResult(
            None, None, None, {}, False,
            f"映射路线求值失败: {type(exc).__name__}")
    return ForwardResult(None, None, None, {}, False, f"映射不支持路线 {route_id}")


def _predict_rcwa_spectrum_strict(d_nm, h_nm, p_nm, angle_deg, polarization, material, substrate,
                                  predictor=None):
    """Use only the registered RCWA surrogate; never fall through to generic ML."""
    try:
        if predictor is not None:
            return predictor(d_nm, h_nm, p_nm, angle_deg=angle_deg, polarization=polarization)
        frozen = getattr(ml_module, "freeze_rcwa_spectrum_predictor", lambda *a, **k: None)(
            material, substrate, angle_deg)
        if frozen is not None:
            return frozen(d_nm, h_nm, p_nm, angle_deg=angle_deg, polarization=polarization)
        if material in getattr(ml_module, "_RCWA_WL_SESSIONS", {}) and abs(float(angle_deg)) < 5:
            spec = ml_module._predict_rcwa_wavelength(material, substrate, d_nm, h_nm, p_nm)
            return None if spec is None else np.clip(spec, 0, None)
        if ml_module._should_use_rcwa(material, substrate, angle_deg):
            x = ml_module._build_rcwa_input(d_nm, h_nm, p_nm, angle_deg, polarization, material, substrate)
            spec = ml_module._ensemble_predict(material, substrate, x)
            return None if spec is None else np.clip(spec, 0, None)
    except Exception as exc:
        logging.debug("strict RCWA surrogate failed: %s", exc)
    return None


# FP cavity module imported from fp_cavity.py
from fp_cavity import (
    _AG_NK_TABLE, _ag_nk, _ag_nk_vec, _n_sio2_sellmeier,
    fp_cavity_spectrum, fp_dielectric_spectrum,
)

is_dbr_fp = st.session_state.get('fp_mirror_type', '').startswith('介质')
_single_ml_route_applicable = bool(
    not is_fp
    and not is_dual
    and not _single_geometry_invalid
    and st.session_state.get("ml_accel", False)
    and _ml_ready
    and not st.session_state.get('far_field', False)
    and material in ml_module.MATERIAL_CODES
)
# The RCWA registry belongs only to an applicable single-pillar ML route.
if _single_ml_route_applicable:
    _ensure_rcwa_ml()

_far_field_enabled = False
if not is_fp:
    _far_field_enabled = bool(st.session_state.get('far_field', False))
    configure_engine_far_field(
        engine, _far_field_enabled,
        float(st.session_state.get("na_val", 0.1)),
        float(st.session_state.get("theta_obs", 0.0)),
    )
_route_id = "lorentz_fano_fallback"
_route_label = "Lorentz/Fano fallback"
_route_chain = ["Lorentz/Fano + CCM analytical response"]
_route_reason = "当前配置未启用 ML 加速。"
_route_model_version = "torch_model.py batch_lorentzian_spectrum"
_route_boundary = "解析/半解析近似，不是直接 RCWA"
_route_model_state = "not_selected"
_route_call_error = ""
_forward = None

use_dual_ml = bool(
    is_dual
    and st.session_state.get('ml_accel', False)
    and _dual_ml_ready
    and not st.session_state.get('far_field', False)
    and material in ml_module.MATERIAL_CODES
    and substrate == "SiO2 (fused silica)"
)
use_single_ml = _single_ml_route_applicable
use_ml = use_dual_ml if is_dual else use_single_ml

if is_fp:
    if is_dbr_fp:
        _twl = st.session_state.get('fp_target_wl', 450.0)
        _wls, _refl = fp_dielectric_spectrum(
            st.session_state.fp_t_val, _twl, 3, 5, angle,
            polarization.startswith('TE'))
    else:
        _wls, _refl = fp_cavity_spectrum(
            st.session_state.fp_t_val, angle, polarization.startswith('TE'))
    _forward = _forward_result(_wls, _refl, {})
    _route_id = "fp_tmm"
    _route_label = "FP cavity TMM"
    _route_chain = ["FP cavity transfer-matrix spectrum", "CIE D65 colorimetry"]
    _route_reason = (
        "FP 腔使用顶层互斥 TMM 路线；未进入纳米柱代理、解析或远场路径。")
    _route_model_version = "fp_cavity.py"
    _route_boundary = "薄膜腔 TMM，不是 RCWA 纳米柱求解"
    _route_model_state = "not_applicable"
elif _single_geometry_invalid:
    _forward = ForwardResult(None, None, None, {}, False, "单柱几何越域：D > P")
    _route_id = "invalid_geometry"
    _route_label = "Invalid geometry"
    _route_chain = []
    _route_reason = f"几何不可行：D={float(diameter):g} 大于 P={float(period):g}；已停止正向计算。"
    _route_model_version = "未执行"
    _route_boundary = "单柱要求 D ≤ P；当前配置不在模型/物理域内"
    _route_model_state = "invalid_geometry"
    _route_call_error = _route_reason
    _forward = ForwardResult(None, None, None, {}, False, _route_reason)
elif is_dual:
    if use_dual_ml:
        d1v = st.session_state.get('d1_val', diameter)
        h1v = st.session_state.get('h1_val', height)
        d2v = st.session_state.get('d2_val', diameter)
        h2v = st.session_state.get('h2_val', height)
        try:
            with _bound_runtime_context(
                    "dual", "ml_surrogate", material, substrate):
                ml_spec = ml_module.predict_dual_spectrum(
                    d1v, h1v, d2v, h2v, period, angle, polarization,
                    material, substrate)
        except Exception as exc:
            logging.warning("dual ML spectrum failed: %s", exc)
            _route_call_error = type(exc).__name__
            ml_spec = None
        if ml_spec is not None:
            _route_id = "ml_surrogate"
            _route_label = "ML surrogate"
            _route_chain = [_DUAL_MODEL_RELATIVE_PATH, "CIE D65 colorimetry"]
            _route_reason = "无回退；双柱 ML 模型已成功加载并返回预测。"
            _route_model_state = "loaded_and_called"
            _route_model_version = _DUAL_MODEL_RELATIVE_PATH
            _route_boundary = "代理预测，不是直接 RCWA；仅支持 SiO2 衬底"
            _forward = _forward_result(ml_module.WL, ml_spec, {})
        else:
            _route_reason = "双柱 ML 模型调用失败或未返回光谱，已回退到解析/半解析引擎。"
            _route_model_state = "call_failed"
    else:
        if st.session_state.get('far_field', False):
            _route_id = "far_field_postprocessing"
            _route_label = "Far-field post-processing"
            _route_chain = [
                "Lorentz/Fano + CCM analytical response",
                "angular-spectrum NA integration",
            ]
            _route_reason = "启用远场后，双柱 ML 直通禁用，使用解析局部响应与远场后处理。"
            _route_model_version = "engine.py _dual_far_field_spectrum"
            _route_boundary = "远场后处理链，不是直接 RCWA"
        elif not st.session_state.get('ml_accel', False):
            _route_reason = "用户未启用双柱 ML 加速，使用解析/半解析引擎。"
            _route_model_state = "not_selected"
        elif not _dual_ml_ready:
            _route_reason = f"双柱 ML 未使用：{_dual_ml_error or '双柱 ONNX 模型不可用'}；当前使用解析/半解析引擎。"
            _route_model_state = _dual_ml_state.get("state", "file_present_load_failed")
        elif substrate != "SiO2 (fused silica)":
            _route_reason = "双柱 ML 仅支持 SiO2 衬底，当前组合使用解析/半解析引擎。"
    if _forward is None:
        _wls, _refl = _cached_physical_forward(
            round(st.session_state.d1_val, 1), round(st.session_state.h1_val, 1), round(st.session_state.p_val, 1),
            material, substrate, polarization, round(angle, 1),
            round(st.session_state.d2_val, 1), round(st.session_state.h2_val, 1), True,
            st.session_state.get('far_field', False),
            round(st.session_state.get('na_val', 0.1), 2),
            round(st.session_state.get('theta_obs', 0.0), 1))
        _forward = _forward_result(_wls, _refl, {})
else:
    if use_single_ml:
        _rcwa_route_eligible = bool(
            _rcwa_ml_ready and _rcwa_runtime_binding is not None
            and abs(float(angle)) < 5
        )
        _rcwa_spectrum_predictor = None
        try:
            if _rcwa_route_eligible:
                with _bound_runtime_context(
                        "single", "rcwa_surrogate", material, substrate):
                    _rcwa_spectrum_predictor = ml_module.freeze_rcwa_spectrum_predictor(
                        material, substrate, angle)
                    ml_spec = _predict_rcwa_spectrum_strict(
                        diameter, height, period, angle, polarization, material,
                        substrate, predictor=_rcwa_spectrum_predictor)
            else:
                with _bound_runtime_context(
                        "single", "ml_surrogate", material, substrate):
                    ml_spec = ml_module.predict_generic_spectrum(
                        diameter, height, period, angle, polarization, material,
                        substrate)
        except Exception as exc:
            logging.warning("ML spectrum failed: %s", exc)
            _route_call_error = type(exc).__name__
            ml_spec = None
        _single_rcwa_route = bool(
            _rcwa_route_eligible and _rcwa_spectrum_predictor is not None)
        if ml_spec is not None:
            if _single_rcwa_route:
                _route_id = "rcwa_surrogate"
                _route_label = "RCWA-trained ML surrogate"
                _route_chain = ["RCWA-trained ensemble", "CIE D65 colorimetry"]
                _route_model_version = ", ".join(
                    os.path.basename(path)
                    for path in _rcwa_runtime_binding.session_paths)
                _route_boundary = "RCWA 训练代理，不是本次直接 RCWA 求解"
                _route_reason = "无回退；RCWA 代理已成功加载并返回预测。"
                _route_model_state = "loaded_and_called"
            else:
                _route_id = "ml_surrogate"
                _route_label = "ML surrogate"
                _route_chain = ["forward_mlp_v8_sub", "CIE D65 colorimetry"]
                _route_model_version = "forward_mlp_v8_sub.onnx"
                _route_boundary = "代理预测，不是直接 RCWA"
                _route_reason = "无回退；通用 ML 已成功加载并返回预测。"
                _route_model_state = "loaded_and_called"
            _forward = _forward_result(ml_module.WL, ml_spec, {})
        else:
            _route_reason = "ML 代理调用失败或未返回光谱，已回退到解析/半解析引擎。"
            _route_model_state = "call_failed"
    else:
        if st.session_state.get('far_field', False):
            _route_id = "far_field_postprocessing"
            _route_label = "Far-field post-processing"
            _route_chain = ["Lorentz/Fano + CCM analytical response", "angular-spectrum NA integration"]
            _route_reason = "启用远场后，当前实现禁用 ML 直通，先计算局部响应再做角谱/NA 后处理。"
            _route_model_version = "engine.py _far_field_spectrum"
            _route_boundary = "远场后处理链，不是直接 RCWA"
        elif not st.session_state.get('ml_accel', False):
            _route_reason = "用户未启用 ML 加速，使用解析/半解析引擎。"
            _route_model_state = "not_selected"
        elif not _ml_ready:
            _route_reason = f"ML 模型不可用或未加载，使用解析/半解析引擎。原因：{_ml_error or '未知'}"
            _route_model_state = _ml_state.get("state", "load_failed")
        elif material not in ml_module.MATERIAL_CODES:
            _route_reason = f"材料 {material} 不在 ML 训练集合内，使用解析/半解析引擎。"
    if _forward is None:
        _wls, _refl = _cached_physical_forward(
            round(diameter, 1), round(height, 1), round(period, 1),
            material, substrate, polarization, round(angle, 1),
            0.0, 0.0, False,
            st.session_state.get('far_field', False),
            round(st.session_state.get('na_val', 0.1), 2),
            round(st.session_state.get('theta_obs', 0.0), 1))
        _forward = _forward_result(_wls, _refl, {})

# A non-None model return still must pass spectrum and color validation.
if (_forward is not None and _route_model_state == "loaded_and_called"
        and not _forward.spectrum_available):
    _route_model_state = "output_validation_failed"
    _route_call_error = _forward.error

_provenance_material = "TiO2 (cavity layer)" if is_fp else material
_provenance_substrate = (
    "SiO2 (DBR mirror stack)" if is_fp and is_dbr_fp
    else "Ag (metal mirrors)" if is_fp
    else substrate
)
if is_fp:
    _structure_type = "fp"
    _mirror_type = str(st.session_state.fp_mirror_type)
    _t_nm = float(st.session_state.fp_t_val)
    if is_dbr_fp:
        _structure_label = "FP-DBR 腔"
        _center_nm = float(st.session_state.get("fp_target_wl", 450.0))
        _geometry = {
            "T_nm": _t_nm,
            "center_wavelength_nm": _center_nm,
            "top_pairs": 3.0,
            "bottom_pairs": 5.0,
        }
        _geometry_summary = (
            f"T={_t_nm:g} nm, center={_center_nm:g} nm, DBR pairs=3/5")
        _stack_identity = (
            "(TiO2/SiO2)^3 / TiO2(T) / (SiO2/TiO2)^5")
    else:
        _structure_label = "FP-Ag 腔"
        _geometry = {"T_nm": _t_nm, "top_ag_nm": 30.0}
        _geometry_summary = f"T={_t_nm:g} nm, top Ag=30 nm, bottom Ag=bulk"
        _stack_identity = "Ag(30 nm) / TiO2(T) / Ag(bulk)"
elif is_dual:
    _structure_type = "dual"
    _structure_label = "双柱"
    _mirror_type = ""
    _stack_identity = ""
    _geometry = {
        "D1_nm": float(st.session_state.d1_val),
        "H1_nm": float(st.session_state.h1_val),
        "D2_nm": float(st.session_state.d2_val),
        "H2_nm": float(st.session_state.h2_val),
        "P_nm": float(period),
    }
    _geometry_summary = (
        f"D1={_geometry['D1_nm']:g} nm, H1={_geometry['H1_nm']:g} nm, "
        f"D2={_geometry['D2_nm']:g} nm, H2={_geometry['H2_nm']:g} nm, "
        f"P={_geometry['P_nm']:g} nm")
else:
    _structure_type = "single"
    _structure_label = "单柱"
    _mirror_type = ""
    _stack_identity = ""
    _geometry = {
        "D_nm": float(diameter),
        "H_nm": float(height),
        "P_nm": float(period),
    }
    _geometry_summary = (
        f"D={_geometry['D_nm']:g} nm, H={_geometry['H_nm']:g} nm, "
        f"P={_geometry['P_nm']:g} nm")
_result_provenance = make_result_provenance(
    _route_id, _route_label, _route_chain, _route_boundary, _route_reason,
    _route_model_version, _provenance_material, _provenance_substrate, polarization, angle,
    round(float(st.session_state.get('na_val', 0.1)), 2) if _far_field_enabled else "未启用",
    float(st.session_state.get('theta_obs', 0.0)) if _far_field_enabled else 0.0,
    structure_type=_structure_type,
    structure_label=_structure_label,
    geometry=_geometry,
    geometry_summary=_geometry_summary,
    mirror_type=_mirror_type,
    stack_identity=_stack_identity,
)
_active_model_resource = (
    _dual_runtime_binding if _structure_type == "dual" and _route_id == "ml_surrogate"
    else _primary_runtime_binding if _structure_type == "single" and _route_id == "ml_surrogate"
    else _rcwa_runtime_binding if _structure_type == "single" and _route_id == "rcwa_surrogate"
    else None
)
_result_provenance["model_artifact_version"] = (
    _active_model_resource.loaded_identity
    if _active_model_resource is not None else "not_applicable")
if _route_id in {"ml_surrogate", "rcwa_surrogate"}:
    _model_fields = _model_provenance(
        _dual_ml_state if is_dual else _ml_state,
        called=_route_model_state == "loaded_and_called",
        call_error=_route_call_error, state_override=_route_model_state)
else:
    _model_fields = {
        "model_state": _route_model_state,
        "model_file_present": "not_applicable",
        "model_loaded": "not_applicable",
        "model_call_error": _route_call_error,
    }
_result_provenance.update(_model_fields)
if _forward is None:
    _forward = ForwardResult(None, None, None, _result_provenance, False, "当前路由未生成结果")
else:
    _forward.provenance.update(_result_provenance)
_result_provenance = sync_forward_status(_forward).provenance


def _make_analysis_context(
    analysis_type, sampling, *, route_id=None, model_version=None,
    artifact_version=None, registry_version="ui-analysis-registry-v1",
    angle_deg=None,
):
    """Freeze every user-visible input before an expensive analysis runs."""
    context_route = str(route_id if route_id is not None else _route_id)
    context_model = str(
        model_version if model_version is not None else _route_model_version)
    return AnalysisContext.create(
        analysis_type,
        structure_type=_structure_type,
        geometry=dict(_geometry),
        material=str(_provenance_material),
        substrate=str(_provenance_substrate),
        polarization=str(polarization),
        angle_deg=float(angle if angle_deg is None else angle_deg),
        route_id=context_route,
        model_version=context_model,
        artifact_version=str(
            artifact_version
            if artifact_version is not None
            else _analysis_artifact_version(
                context_route, context_model, structure_type=_structure_type,
                material=_provenance_material, substrate=_provenance_substrate)),
        registry_version=str(registry_version),
        far_field_enabled=bool(_far_field_enabled),
        na=float(st.session_state.get("na_val", 0.1)),
        theta_obs_deg=float(st.session_state.get("theta_obs", 0.0)),
        sampling=dict(sampling),
    )


rgb = _forward.rgb
if rgb is None:
    rgb = np.array([0.0, 0.0, 0.0], dtype=float)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "预览", "逆设计", "图案", "映射", "光谱"
])

# Tab 1: 实时预览
with tab1:
    _color_available = bool(_forward.spectrum_available and _forward.rgb is not None)
    hex_color = rgb_to_hex(rgb) if _color_available else "#251F2B"
    r255, g255, b255 = rgb_255(rgb)

    _render_result_provenance(_forward.provenance)

    # --- Correction hint for dual-pillar ---
    if st.session_state.get('dual_pillar', False):
        if st.session_state.get('_dual_success_msg'):
            st.success(st.session_state._dual_success_msg)
            st.session_state._dual_success_msg = ''
        if st.session_state.get('_dual_correction'):
            st.caption(f"参数已修正: {st.session_state._dual_correction}")

    # --- Color swatch card ---
    if is_fp:
        mirror_label = st.session_state.get('fp_mirror_type', '介质 DBR (TiO2/SiO2)')
        param_info = f"腔长 T={st.session_state.fp_t_val:.0f}nm | FP腔: {mirror_label}"
    elif st.session_state.get('dual_pillar', False):
        param_info = f"D1={st.session_state.d1_val:.0f}nm H1={st.session_state.h1_val:.0f}nm | D2={st.session_state.d2_val:.0f}nm H2={st.session_state.h2_val:.0f}nm | P={period:.0f}nm"
    else:
        param_info = f"D={diameter:.0f}nm  H={height:.0f}nm  P={period:.0f}nm"
    st.markdown(f"""
    <div style="display:flex;align-items:center;flex-wrap:wrap;gap:14px;padding:16px;
                background:linear-gradient(135deg, var(--preview-start) 0%, var(--preview-end) 100%);
                border-radius:16px;margin-bottom:20px;">
      <div style="width:clamp(88px,30vw,130px);height:clamp(88px,30vw,130px);background:{hex_color};
                  border-radius:16px;box-shadow:0 8px 32px {hex_color}66,
                  inset 0 1px 0 rgba(255,255,255,0.3);flex-shrink:0;"></div>
      <div style="color:var(--text-primary);min-width:0;overflow-wrap:anywhere;word-break:break-word;">
        <div style="font-size:12px;opacity:0.72;margin-bottom:4px;">当前路线颜色 · 由当前路线光谱计算</div>
        <div style="font-size:24px;font-weight:700;margin-bottom:6px;">{"不可用" if not _color_available else hex_color}</div>
        <div style="font-size:14px;opacity:0.85;">{"当前路由未生成可用颜色" if not _color_available else f"RGB({r255}, {g255}, {b255})"}</div>
        <div style="margin-top:10px;font-size:13px;opacity:0.6;line-height:1.6;">
          {(_forward.provenance.get('material', material) if is_fp else material)} on {(_forward.provenance.get('substrate', substrate) if is_fp else substrate)}<br>
          {param_info}<br>
          {polarization} &nbsp; &theta;={angle:.0f}&deg;
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    _render_reference_recheck(
        structure_type=_structure_type,
        material=material,
        substrate=substrate,
        polarization=polarization,
        angle_deg=angle,
        far_field_enabled=_far_field_enabled,
        diameter_nm=diameter,
        height_nm=height,
        period_nm=period,
        forward=_forward,
    )
    st.caption("下一步：查看“光谱”页核对输出，或进入“逆设计”页匹配目标色；导出入口在侧栏。")

    # --- AI Analysis (temporarily hidden) ---
    if ENABLE_LLM_FEATURES:
        col_ai1, col_ai2 = st.columns([3, 1])
        with col_ai2:
            ai_clicked = st.button(u"🤖 AI 分析", key="ai_analyze_color", use_container_width=True,
                         help=u"使用配置的大模型分析当前颜色结果")
        if ai_clicked:
            with st.spinner(u"AI 分析中..."):
                params = {}
                if is_fp:
                    params = {u"腔长 T": f"{st.session_state.fp_t_val:.0f}nm",
                              u"反射镜": st.session_state.get('fp_mirror_type', 'DBR')}
                else:
                    params = {u"D": f"{diameter:.0f}nm", u"H": f"{height:.0f}nm", u"P": f"{period:.0f}nm"}
                params[u"材料"] = material
                params[u"衬底"] = substrate
                params[u"偏振"] = polarization
                params[u"角度"] = f"{angle:.0f}°"
                result = analyze_color(hex_color, params)
                st.session_state._ai_result = result
            st.rerun()
        if st.session_state.get('_ai_result'):
            st.info(st.session_state._ai_result)
            if st.button('✕ 清除', key='clear_ai_result'):
                st.session_state.pop('_ai_result', None)
                st.rerun()

    # --- Color gamut notice (audited single-pillar scope only) ---
    if not is_fp and not is_dual and material == "TiO2 (anatase)":
        st.info(
            "既有审计覆盖的 TiO2 单柱范围（D 60-267 nm，H 80-600 nm）内，"
            "未覆盖高饱和青蓝色或纯红色。该提示不代表当前全部控制范围，"
            "也不代表本次运行执行了直接 RCWA 验证。"
            "提示：1) 切换到 a-Si/Si3N4 材料；2) 在侧栏顶部将结构类型切换为 FP 腔。"
        )

    # --- Pillar visualization with pure CSS (non-FP only) ---
    if not is_fp:
        scale = 160.0 / max(height, 100)
        pw = max(diameter * scale * 0.45, 20)
        ph = height * scale * 0.45
        sh = 45
        period_w = period * scale * 0.45

        st.markdown(f"""
        <div style="background:var(--bg-surface);border:1px solid var(--border-subtle);border-radius:16px;padding:24px 24px 16px 24px;">
          <div style="text-align:center;color:var(--text-muted);font-size:12px;margin-bottom:16px;
                      letter-spacing:0.5px;">
            CROSS-SECTION &nbsp;&middot;&nbsp; {param_info}
          </div>
          <div style="display:flex;justify-content:center;align-items:flex-end;
                      height:230px;position:relative;">
            <div style="position:absolute;bottom:0;left:50%;transform:translateX(-50%);
                        width:{period_w*2.2:.0f}px;height:{sh}px;
                        background:linear-gradient(180deg, var(--substrate-start), var(--substrate-end));
                        border-radius:4px 4px 0 0;"></div>
            <div style="position:absolute;bottom:{sh}px;left:50%;transform:translateX(-50%);
                        width:{period_w*2.2:.0f}px;height:2px;
                        background:rgba(255,255,255,0.06);"></div>
            <div style="width:{pw:.0f}px;height:{ph:.0f}px;
                        background:linear-gradient(180deg, {hex_color}ee, {hex_color}77, {hex_color}cc);
                        border-radius:6px 6px 3px 3px;
                        box-shadow:0 6px 24px {hex_color}33, inset 0 1px 0 rgba(255,255,255,0.12);
                        position:relative;z-index:2;margin-bottom:{sh}px;
                        transition:all 0.3s ease;"></div>
            <!-- Height dimension line -->
            <div style="position:absolute;bottom:{sh}px;left:calc(50% + {pw/2+16:.0f}px);
                        width:1px;height:{ph:.0f}px;background:rgba(255,255,255,0.15);"></div>
            <div style="position:absolute;bottom:{sh+ph/2:.0f}px;left:calc(50% + {pw/2+22:.0f}px);
                        color:var(--text-muted);font-size:10px;">{height:.0f}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Parameter sensitivity: explicit, session-local snapshot (non-FP only).
    if not is_fp:
        st.divider()
        st.subheader("参数灵敏度 (工艺容差 +/-5 nm)")
        tol = 5.0
        if is_dual:
            base_geometry = {
                "d1": float(st.session_state.get("d1_val", diameter)),
                "h1": float(st.session_state.get("h1_val", height)),
                "d2": float(st.session_state.get("d2_val", diameter)),
                "h2": float(st.session_state.get("h2_val", height)),
                "p": float(period),
            }
            params = [("D1", "d1"), ("H1", "h1"), ("D2", "d2"), ("H2", "h2"), ("P", "p")]
        else:
            base_geometry = {"d": float(diameter), "h": float(height), "p": float(period)}
            params = [("D", "d"), ("H", "h"), ("P", "p")]
        _sensitivity_context = _make_analysis_context(
            "sensitivity",
            {"tolerance_nm": tol, "fields": [key for _, key in params], "sides": [-1, 1]},
            registry_version="sensitivity-domain-v1",
        )
        _sensitivity_artifact_available = _artifact_identity_available(
            _sensitivity_context.artifact_version)
        _sensitivity_runtime_issue = _analysis_runtime_identity_issue(
            _sensitivity_context)
        _sensitivity_execution_available = bool(
            _forward.spectrum_available and _sensitivity_artifact_available
            and not _sensitivity_runtime_issue)
        _sensitivity_load = load_analysis_snapshot(st.session_state, _sensitivity_context)
        run_sensitivity = st.button(
            "运行 / 加载当前灵敏度分析", key="run_sensitivity_analysis",
            use_container_width=True,
            disabled=not _sensitivity_execution_available,
        )
        if not _sensitivity_artifact_available:
            st.info("灵敏度分析源码身份不可用；旧结果已隐藏，当前不会运行或跨路线补数。")
        elif _sensitivity_runtime_issue:
            st.info(
                f"灵敏度分析模型会话身份不可用：{_sensitivity_runtime_issue}；"
                "旧结果已隐藏，当前不会运行 evaluator。")
        if _sensitivity_load.state == "stale":
            st.warning("灵敏度快照已陈旧；当前参数或路线已变化，旧结果已隐藏。")
        elif _sensitivity_load.state == "invalid":
            st.warning("灵敏度快照未通过完整性校验，旧结果已隐藏。")
        elif _sensitivity_load.state == "missing":
            st.info("点击按钮后才计算当前路线的工艺扰动；普通页面刷新不会运行分析。")

        if (
            run_sensitivity and _sensitivity_load.state != "fresh"
            and _sensitivity_execution_available
            and not _analysis_runtime_identity_issue(_sensitivity_context)
        ):
            _analysis_material = str(material)
            _analysis_substrate = str(substrate)
            _analysis_polarization = str(polarization)
            _analysis_angle = float(angle)
            _analysis_far_field = bool(_sensitivity_context.far_field_enabled)
            _analysis_na = float(_sensitivity_context.na)
            _analysis_theta = float(_sensitivity_context.theta_obs_deg)
            _analysis_route_id = str(_route_id)
            _analysis_route_label = str(_route_label)
            _analysis_is_dual = bool(is_dual)
            _analysis_use_dual_ml = bool(use_dual_ml)
            _analysis_rcwa_predictor = globals().get("_rcwa_spectrum_predictor")

            def _sensitivity_forward(geometry):
                if not geometry:
                    return None
                if _analysis_is_dual:
                    if _analysis_route_id == "ml_surrogate" and _analysis_use_dual_ml:
                        with _bound_runtime_context(
                                "dual", "ml_surrogate", _analysis_material,
                                _analysis_substrate):
                            spec = ml_module.predict_dual_spectrum(
                                geometry["d1"], geometry["h1"], geometry["d2"], geometry["h2"],
                                geometry["p"], _analysis_angle, _analysis_polarization,
                                _analysis_material, _analysis_substrate)
                        return _forward_result(ml_module.WL, spec, {}) if spec is not None else None
                    wls, refl = _cached_physical_forward(
                        geometry["d1"], geometry["h1"], geometry["p"],
                        _analysis_material, _analysis_substrate, _analysis_polarization,
                        _analysis_angle, geometry["d2"], geometry["h2"], True,
                        _analysis_far_field, _analysis_na, _analysis_theta)
                    return _forward_result(wls, refl, {})
                if _analysis_route_id == "rcwa_surrogate":
                    with _bound_runtime_context(
                            "single", "rcwa_surrogate", _analysis_material,
                            _analysis_substrate):
                        spec = _predict_rcwa_spectrum_strict(
                            geometry["d"], geometry["h"], geometry["p"],
                            _analysis_angle, _analysis_polarization, _analysis_material,
                            _analysis_substrate, predictor=_analysis_rcwa_predictor)
                    return _forward_result(ml_module.WL, spec, {}) if spec is not None else None
                if _analysis_route_id == "ml_surrogate":
                    with _bound_runtime_context(
                            "single", "ml_surrogate", _analysis_material,
                            _analysis_substrate):
                        spec = ml_module.predict_generic_spectrum(
                            geometry["d"], geometry["h"], geometry["p"],
                            _analysis_angle, _analysis_polarization,
                            _analysis_material, _analysis_substrate)
                    return _forward_result(ml_module.WL, spec, {}) if spec is not None else None
                wls, refl = _cached_physical_forward(
                    geometry["d"], geometry["h"], geometry["p"],
                    _analysis_material, _analysis_substrate, _analysis_polarization,
                    _analysis_angle, 0.0, 0.0, False, _analysis_far_field,
                    _analysis_na, _analysis_theta)
                return _forward_result(wls, refl, {})

            _store_sensitivity_snapshot = True
            try:
                with (
                    analysis_engine_transaction(engine, st.session_state),
                    _bound_runtime_context(
                        _sensitivity_context.structure_type,
                        _sensitivity_context.route_id,
                        _sensitivity_context.material,
                        _sensitivity_context.substrate),
                ):
                    _sensitivity_route = FrozenSpectrumRoute(
                        _analysis_route_id, _analysis_route_label,
                        lambda geometry: _sensitivity_forward(geometry))
                    base_lab = rgb_to_lab(np.asarray(rgb, dtype=float))
                    snapshot_rows = []
                    for label, key in params:
                        resolved = [
                            resolve_perturbation(base_geometry, key, -tol),
                            resolve_perturbation(base_geometry, key, tol),
                        ]
                        routed = evaluate_frozen_series(
                            _sensitivity_route,
                            [{"geometry": resolved[0][0]}, {"geometry": resolved[1][0]}],
                        )
                        route_ok = route_results_consistent(_analysis_route_id, routed)
                        if not route_ok:
                            routed = [(_analysis_route_id, None), (_analysis_route_id, None)]
                        sides = {}
                        for side_name, (_, result), (_, resolution) in zip(
                            ("lower", "upper"), routed, resolved,
                        ):
                            if (
                                route_ok and resolution.status == "available"
                                and result is not None and result.spectrum_available
                                and result.rgb is not None
                            ):
                                varied_rgb = np.asarray(result.rgb, dtype=float)
                                sides[side_name] = {
                                    "status": "available",
                                    "requested_value": resolution.requested_value,
                                    "evaluated_value": resolution.evaluated_value,
                                    "actual_delta_nm": resolution.actual_delta_nm,
                                    "rgb": varied_rgb.tolist(),
                                    "delta_e2000": float(delta_e2000(base_lab, rgb_to_lab(varied_rgb))),
                                    "reason": "",
                                }
                            else:
                                sides[side_name] = {
                                    "status": "unavailable",
                                    "requested_value": resolution.requested_value,
                                    "evaluated_value": None, "actual_delta_nm": None,
                                    "rgb": None, "delta_e2000": None,
                                    "reason": (
                                        resolution.reason if resolution.status != "available"
                                        else "冻结路线未返回可用结果" if route_ok
                                        else "分析路线身份不一致"),
                                }
                        snapshot_rows.append({"label": label, "field": key, **sides})
                    _sensitivity_payload = {
                        "status": "available", "reason": "",
                        "base_rgb": np.asarray(rgb, dtype=float).tolist(),
                        "route_id": _analysis_route_id,
                        "analysis_artifact_version": (
                            _sensitivity_context.artifact_version),
                        "model_artifact_version": (
                            _analysis_model_artifact_version(_sensitivity_context)),
                        "rows": snapshot_rows,
                    }
            except (ModelResourceDriftError, ModelResourceUnavailable) as exc:
                logging.error("sensitivity model resource drift: %s", exc)
                st.session_state.pop(_sensitivity_context.session_key, None)
                st.error("灵敏度分析模型会话身份发生漂移；结果与导出已清除，本次未保存。")
                _store_sensitivity_snapshot = False
            except EngineStateRestoreError as exc:
                logging.error("sensitivity engine restore failed: %s", exc)
                st.session_state.pop(_sensitivity_context.session_key, None)
                st.error("灵敏度分析会话引擎恢复失败；旧结果已清除，本次结果未保存。")
                _store_sensitivity_snapshot = False
            except EngineStateMutationError as exc:
                logging.error("sensitivity engine mutation restored: %s", exc)
                _sensitivity_payload = {
                    "status": "unavailable",
                    "reason": "会话引擎状态被分析修改；已恢复原状态，本次结果因完整性失败而作废",
                }
            except Exception as exc:
                if exception_has_model_resource_drift(exc):
                    logging.error("sensitivity business error with model drift: %s", exc)
                    st.session_state.pop(_sensitivity_context.session_key, None)
                    st.error("灵敏度分析模型会话身份发生漂移；结果与导出已清除，本次未保存。")
                    _store_sensitivity_snapshot = False
                else:
                    logging.warning("sensitivity analysis failed: %s", exc)
                    _sensitivity_payload = {
                        "status": "unavailable",
                        "reason": f"当前路线分析失败：{type(exc).__name__}",
                    }
            _post_sensitivity_issue = _analysis_runtime_identity_issue(
                _sensitivity_context)
            if _post_sensitivity_issue:
                st.session_state.pop(_sensitivity_context.session_key, None)
                st.error(
                    "灵敏度分析期间模型/源码身份发生变化；本次结果未保存。")
                _store_sensitivity_snapshot = False
            if _store_sensitivity_snapshot:
                store_analysis_snapshot(
                    st.session_state,
                    AnalysisSnapshot.create(_sensitivity_context, _sensitivity_payload),
                )
            _sensitivity_load = load_analysis_snapshot(st.session_state, _sensitivity_context)

        if _sensitivity_load.state == "fresh" and _sensitivity_execution_available:
            _sensitivity_payload = _sensitivity_load.snapshot.payload
            if _sensitivity_payload["status"] == "unavailable":
                st.warning(f"灵敏度分析不可用：{_sensitivity_payload['reason']}。未跨路线补数。")
            else:
                cols = st.columns(4)
                cols[0].markdown("**参数**")
                cols[1].markdown(f"**-{tol:.0f}nm**")
                cols[2].markdown("**当前**")
                cols[3].markdown(f"**+{tol:.0f}nm**")
                base_hex = rgb_to_hex(_sensitivity_payload["base_rgb"])
                for row in _sensitivity_payload["rows"]:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.markdown(f"**{row['label']}**")
                    c3.markdown(
                        f'<div style="width:40px;height:24px;background:{base_hex};border-radius:4px;border:2px solid white;"></div>',
                        unsafe_allow_html=True)
                    for column, entry in ((c2, row["lower"]), (c4, row["upper"])):
                        if entry["status"] != "available":
                            column.markdown(
                                f"<small>不可用<br>请求 {entry['requested_value']:.1f}nm</small>",
                                unsafe_allow_html=True)
                        else:
                            varied_hex = rgb_to_hex(entry["rgb"])
                            column.markdown(
                                f'<div style="width:40px;height:24px;background:{varied_hex};border-radius:4px;"></div>'
                                f'<small>ΔE00={entry["delta_e2000"]:.2f}<br>实际 {entry["evaluated_value"]:.1f}nm '
                                f'(Δ={entry["actual_delta_nm"]:+.1f}nm)</small>',
                                unsafe_allow_html=True)
                st.caption(
                    f"固定模型路线：{_route_label}；快照 fingerprint={_sensitivity_context.fingerprint[:12]}…；"
                    "色差为 CIEDE2000（ΔE00），未把单一数值当作通用感知阈值。"
                )

# Tab 2: Inverse Design
with tab2:
    st.subheader("逆设计")
    st.caption("选目标色，运行推荐搜索；结果可应用后再复核。")

    col_pick, col_target = st.columns([1, 2])
    with col_pick:
        st.session_state.setdefault("inverse_target_picker", "#80c8ff")
        picker_hex = st.color_picker("目标颜色", key="inverse_target_picker")
    target_r = int(picker_hex[1:3], 16)
    target_g = int(picker_hex[3:5], 16)
    target_b = int(picker_hex[5:7], 16)
    with col_target:
        st.markdown(
            f"""
            <div class="inverse-target-card" aria-label="目标颜色 {html.escape(picker_hex)} RGB {target_r} {target_g} {target_b}">
              <span class="inverse-target-card__swatch" style="background:{html.escape(picker_hex)}"></span>
              <div class="inverse-target-card__text">
                <div style="color:var(--text-muted);font-size:12px;font-weight:600">目标颜色</div>
                <div class="inverse-target-card__hex">{html.escape(picker_hex.upper())}</div>
                <div class="inverse-target-card__rgb">RGB({target_r}, {target_g}, {target_b})</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    _inverse_structure = "fp" if is_fp else "dual" if is_dual else "single"
    if _inverse_structure == "single":
        _inverse_geometry_valid = not _single_geometry_invalid
    elif _inverse_structure == "dual":
        _inverse_geometry_valid = bool(
            float(st.session_state.d1_val) < float(period)
            and float(st.session_state.d2_val) < float(period)
        )
    else:
        _inverse_geometry_valid = bool(50.0 <= float(st.session_state.fp_t_val) <= 600.0)
    _inverse_material = str(
        _forward.provenance.get("material", material) if is_fp else material)
    _inverse_substrate = str(
        _forward.provenance.get("substrate", substrate) if is_fp else substrate)
    _inverse_context = InverseContext(
        structure_type=_inverse_structure,
        material=_inverse_material,
        substrate=_inverse_substrate,
        polarization=str(polarization),
        angle_deg=float(angle),
        target_rgb=(target_r, target_g, target_b),
        target_hex=picker_hex,
        preview_route_id=str(_forward.provenance.get("route_id", "unknown")),
        preview_model_version=str(
            _forward.provenance.get("model_version", "unknown")),
        geometry_valid=_inverse_geometry_valid,
        fp_mirror_type=(
            str(st.session_state.get("fp_mirror_type", "")) if is_fp else ""),
    )
    _previous_inverse_run = st.session_state.get("_inverse_run")
    _context_was_initialized = bool(
        st.session_state.get("_inverse_context_initialized", False))
    if _previous_inverse_run is not None and invalidate_inverse_run(
            _previous_inverse_run, _inverse_context) is None:
        _clear_inverse_results()
        st.session_state.pop("_inverse_run", None)
        st.info("搜索上下文已变化；旧候选、应用按钮与导出已失效并隐藏。")
    elif not _context_was_initialized:
        _clear_inverse_results()
        st.session_state.pop("_inverse_run", None)
    st.session_state._inverse_context_initialized = True
    _render_inverse_context(_inverse_context, _forward.provenance)

    # Inverse search owns its local RCWA proxy independently of the preview
    # acceleration toggle.  Load the exact TiO2/SiO2 family here so turning
    # preview ML off does not make the real search controls appear dead.
    if (
        _inverse_structure == "single"
        and _inverse_geometry_valid
        and str(polarization).startswith("TE")
        and abs(float(angle)) < 1e-9
        and not _far_field_enabled
        and material in ml_module.MATERIAL_CODES
        and substrate in ml_module.SUBSTRATE_CODES
    ):
        _ensure_rcwa_ml(material, substrate)
    _rl_ready = _rl_route_ready(
        _inverse_material, _inverse_substrate, polarization, angle)

    _single_gradient_ready, _single_gradient_reason = (
        ml_module.single_inverse_model_status(_inverse_material, _inverse_substrate)
        if _inverse_structure == 'single' else (False, '当前不是单柱结构。'))
    _method_states = _inverse_method_states(
        context=_inverse_context,
        rcwa_ready=_rcwa_ml_ready,
        primary_torch_ready=_single_gradient_ready,
        dual_ready=_dual_ml_ready,
        compare_enabled=ENABLE_MULTI_SCHEME_SEARCH,
        fp_mirror_type=st.session_state.get("fp_mirror_type", ""),
        far_field_enabled=bool(_far_field_enabled),
        rl_ready=_rl_ready,
    )
    if 'single' in _method_states and not _single_gradient_ready:
        _method_states['single'] = replace(
            _method_states['single'], reason=_single_gradient_reason)
    # The dual ONNX route remains fail-closed when its manifest is absent, but
    # the project already ships a runnable analytical dual baseline.  Expose
    # that baseline as a real action so the structure selector never leads to
    # a page whose only visible control is a dead disabled button.
    if _inverse_structure == "dual" and "dual_physical" not in _method_states:
        _dual_physical_ready = bool(
            _inverse_context.geometry_valid
            and importlib.util.find_spec("torch") is not None
        )
        _method_states["dual_physical"] = InverseMethodState(
            "dual_physical", "双柱解析", "双柱五参数解析/半解析基线搜索",
            _dual_physical_ready,
            "可用：使用本地解析/半解析双柱基线；结果用于交互参考，不代表双柱 ONNX/RCWA 精度。"
            if _dual_physical_ready else
            "不可用：当前 Python 环境未安装 PyTorch。",
            scope="analytical_baseline",
        )
    # Far-field and geometry validity are intentionally outside
    # InverseContext's export identity.  Drop a reference-library run when
    # its current display conditions no longer support that exact route.
    _existing_inverse_run = st.session_state.get("_inverse_run")
    _reference_state = _method_states.get("reference")
    if (
        isinstance(_existing_inverse_run, InverseRun)
        and _existing_inverse_run.method_id == "reference"
        and (_reference_state is None or not _reference_state.available)
    ):
        _clear_inverse_results()
    # If a live, exact-pair RCWA proxy session is already bound, preserve its
    # existing primary entry and keep the audited-library route in the
    # secondary list.  On the normal local demo path no such session exists,
    # so TM/0° promotes the fully usable audited route to the main button.
    _reference_primary = (
        "reference" in _method_states and not bool(_rcwa_ml_ready)
    )
    if _inverse_structure == "single":
        _single_order = ["reference", "smart"] if _reference_primary else ["smart"]
        if _rl_ready:
            _single_order.append("rl")
        _single_order.append("single")
        _preferred_order = _single_order
    else:
        _preferred_order = {
            "dual": ["dual", "dual_physical"], "fp": ["fp"]
        }[_inverse_structure]
    _primary_method = next(
        (key for key in _preferred_order if _method_states[key].available),
        _preferred_order[0],
    )
    _primary_state = _method_states[_primary_method]
    _method_plain_summary = {
        "smart": "快速网格搜索",
        "rl": "离散探索（Q-learning）",
        "single": "单柱参数搜索",
        "dual": "双柱联合搜索",
        "dual_physical": "双柱解析搜索",
        "fp": "腔长与中心波长搜索",
        "compare": "比较不同结构路线",
    }
    primary_btn = st.button(
        f"开始搜索 · {_primary_state.label}",
        type="primary",
        use_container_width=True,
        disabled=not _primary_state.available,
        help=_primary_state.summary,
    )
    _primary_summary = _method_plain_summary.get(_primary_method, _primary_state.label)
    if _primary_state.available:
        st.caption(f"主路线：{_primary_summary} · 可用")
    else:
        st.caption(f"主路线：{_primary_summary} · 暂不可用。{_primary_state.reason}")

    reference_btn = primary_btn if _primary_method == "reference" else False
    smart_btn = primary_btn if _primary_method == "smart" else False
    rl_btn = primary_btn if _primary_method == "rl" else False
    gd_btn = primary_btn if _primary_method == "single" else False
    dual_gd_btn = primary_btn if _primary_method == "dual" else False
    dual_physical_btn = primary_btn if _primary_method == "dual_physical" else False
    fp_search_btn = primary_btn if _primary_method == "fp" else False
    ai_btn = primary_btn if _primary_method == "compare" else False
    _other_method_keys = [
        key for key in _preferred_order if key != _primary_method
    ]
    if _other_method_keys:
        st.caption("其他路线")
        with st.expander("其他搜索方法与可用性", expanded=False):
            for _method_key in _other_method_keys:
                _method = _method_states[_method_key]
                _method_status = "可用" if _method.available else "暂不可用"
                if _method.available:
                    _method_text, _method_action = st.columns([3, 1])
                else:
                    _method_text = st.container()
                    _method_action = None
                with _method_text:
                    st.markdown(
                        f"""
                        <div class="inverse-method-row">
                          <strong>{html.escape(_method.label)}</strong>
                          <em class="inverse-method-row__status">{_method_status}</em>
                          <span>{html.escape(_method_plain_summary.get(_method_key, _method.summary))}</span>
                          <span>{html.escape(_method.reason)}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                if _method.available:
                    with _method_action:
                        _clicked = st.button(
                            _method.label,
                            key=f"inverse_secondary_{_method_key}",
                            use_container_width=True,
                            help=_method.reason,
                        )
                else:
                    _clicked = False
                if _method_key == "smart":
                    smart_btn = _clicked
                elif _method_key == "rl":
                    rl_btn = _clicked
                elif _method_key == "reference":
                    reference_btn = _clicked
                elif _method_key == "single":
                    gd_btn = _clicked
                elif _method_key == "dual":
                    dual_gd_btn = _clicked
                elif _method_key == "dual_physical":
                    dual_physical_btn = _clicked
                elif _method_key == "fp":
                    fp_search_btn = _clicked

    if "compare" in _method_states:
        _compare_method = _method_states["compare"]
        with st.expander("跨结构方案对比", expanded=False):
            st.caption(f"{_compare_method.summary}。用于查看不同结构路线的差异。")
            if _compare_method.available:
                ai_btn = st.button(
                    _compare_method.label, key="inverse_cross_structure_compare",
                    use_container_width=True, help=_compare_method.reason,
                )
            else:
                ai_btn = False
                st.caption("当前条件下不可运行，已隐藏操作按钮。")

    if _inverse_structure == "single" and "TiO2" in material and target_b > 150 and target_b > target_r + 20 and target_b > target_g + 20:
        st.caption("TiO₂ 难以覆盖高饱和蓝/青色；可切换到 **a-Si**，或在侧栏顶部选择 **FP 腔**。")
    elif _inverse_structure == "single" and "TiO2" in material and target_r > 180 and target_r > target_g + 30 and target_r > target_b + 30:
        st.caption("💡 TiO₂ 做不出纯红色，建议切换到 **a-Si** + Si₃N₄ 衬底")

    target_rgb_norm = np.array([target_r, target_g, target_b]) / 255.0

    if not _inverse_context.geometry_valid:
        if _inverse_structure == "single":
            st.error("当前单柱参数无效：D > P 会导致纳米柱越过单元边界。已禁用逆设计候选和结果导出，请先调小 D 或增大 P。")
        else:
            st.error("当前结构参数无效；已禁用逆设计候选和结果导出，请先修正侧栏几何参数。")

    if reference_btn and _inverse_context.geometry_valid:
        _clear_inverse_results()
        with st.spinner("🎯 从已审核 RCWA 参考库中按颜色排序..."):
            try:
                _reference_matches = _reference_library_candidates(
                    _inverse_context, limit=5)
                if not _reference_matches:
                    st.warning("参考库中没有可用记录。")
                else:
                    st.session_state._reference_matches = _reference_matches
                    _reference_records = _normalized_reference_candidates(
                        _inverse_context, _reference_matches)
                    _store_inverse_run(
                        _inverse_context, "reference", "已审核参考库",
                        _reference_records,
                    )
                    st.success(
                        f"已审核参考库搜索完成 · 返回 {len(_reference_matches)} 个精确 RCWA 候选"
                        f" · 最佳 ΔE2000={_reference_matches[0][0]:.2f}"
                    )
            except ReferenceLibraryError as exc:
                st.error(f"参考库审核绑定失败，未生成候选：{exc}")
            except Exception as exc:
                st.warning(f"已审核参考库搜索失败：{type(exc).__name__}：{exc}")

    _reference_matches = st.session_state.get("_reference_matches")
    if _reference_matches and inverse_run_matches(
            st.session_state.get("_inverse_run"), _inverse_context):
        _reference_contract = _inverse_candidate_contract(
            "audited_reference", material, substrate, polarization, angle)
        st.caption(
            "以下候选均为参考库中已有的完整 RCWA 记录；只做颜色排序，不是插值或代理模型预测。"
        )
        for _rank, (_de, _match) in enumerate(_reference_matches, start=1):
            _reference_rgb = np.asarray(_match.srgb_display, dtype=float)
            _reference_hex = rgb_to_hex(_reference_rgb)
            _reference_rgb255 = rgb_255(_reference_rgb)
            _reference_params_text = (
                f"D={_match.geometry[0]}nm · H={_match.geometry[1]}nm · "
                f"P={_match.geometry[2]}nm · 记录 #{_match.record.get('index', '?')}"
            )

            def _apply_reference_cb(
                _d=_match.geometry[0], _h=_match.geometry[1],
                _p=_match.geometry[2],
            ):
                _apply_inverse_candidate(
                    _inverse_context, "single", {"d": _d, "h": _h, "p": _p})

            _render_inverse_candidate_card(
                _rank, _reference_hex, _reference_rgb255, _de,
                _reference_params_text, _reference_contract,
                material, substrate, polarization, angle,
                apply_key=f"apply_reference_result_{_rank}",
                apply_callback=_apply_reference_cb,
            )

    if smart_btn and _inverse_context.geometry_valid:
        _clear_inverse_results()
        with st.spinner("🎯 智能网格搜索中 (两阶段: 粗→精)..."):
            try:
                with _bound_runtime_context(
                        "single", "rcwa_surrogate", material, substrate):
                    result = ml_module.smart_grid_search(
                        target_rgb_norm, material=material, substrate=substrate,
                        angle_deg=angle, polarization=polarization,
                        # Keep the click path responsive on the shipped CPU runtime.
                        # The registered models use a fixed batch of one, so a larger
                        # grid multiplies Python-to-ONNX calls instead of vectorizing.
                        coarse_n=8, top_k=3, fine_steps=3, fine_range=8.0
                    )
                if not inverse_candidates_available(result):
                    st.warning("智能网格搜索不可用: 需要 RCWA/ML 模型")
                else:
                    # result is list of (None, MetaSurfaceParam, [r,g,b], de76, de2000)
                    best = result[0]
                    bp = best[1]
                    pred_rgb = best[2]
                    de_sg = best[4]
                    d_sg = bp.diameter_nm
                    h_sg = bp.height_nm
                    p_sg = bp.period_nm

                    rc = list(rgb_255(pred_rgb))
                    hex_sg = f"#{rc[0]:02x}{rc[1]:02x}{rc[2]:02x}"
                    st.session_state._sg_d = float(d_sg)
                    st.session_state._sg_h = float(h_sg)
                    st.session_state._sg_p = float(p_sg)
                    st.session_state._sg_hex = hex_sg
                    st.session_state._sg_de = float(de_sg)
                    st.session_state._sg_rgb = tuple(rc)
                    st.session_state._sg_candidates = result
                    _store_inverse_run(
                        _inverse_context, "smart", "智能网格",
                        _normalized_smart_candidates(_inverse_context, result))

            except Exception as e:
                st.warning(f"智能网格搜索失败: {e}")

    if rl_btn and _inverse_context.geometry_valid:
        _clear_inverse_results()
        with st.spinner("🎯 Q-learning 离散探索中（本地 q-table）..."):
            try:
                if not _rl_route_ready(material, substrate, polarization, angle):
                    raise RuntimeError("本地 q-table 或 TiO2/SiO2 RCWA 代理未就绪")
                with _bound_runtime_context(
                        "single", "rcwa_surrogate", material, substrate):
                    rl_agent = rl_design.get_trained_rl()
                    d_rl, h_rl, p_rl, _rl_hex, _ = rl_agent.search(
                        picker_hex, steps=30, restarts=5)
                    pred_rl = ml_module.predict_rgb(
                        float(d_rl), float(h_rl), float(p_rl),
                        float(angle), polarization, material, substrate)
                    if pred_rl is None:
                        raise RuntimeError("RCWA 代理未返回有限颜色")
                    pred_rl = np.asarray(pred_rl, dtype=float)
                if pred_rl.shape != (3,) or not np.all(np.isfinite(pred_rl)):
                    raise ValueError("RL 候选颜色不是有限 sRGB")
                pred_rl = np.clip(pred_rl, 0.0, 1.0)
                de_rl = float(delta_e2000(
                    rgb_to_lab(target_rgb_norm), rgb_to_lab(pred_rl)))
                hex_rl = rgb_to_hex(pred_rl)
                st.session_state._rl_d = float(d_rl)
                st.session_state._rl_h = float(h_rl)
                st.session_state._rl_p = float(p_rl)
                st.session_state._rl_hex = hex_rl
                st.session_state._rl_de = de_rl
                st.session_state._rl_rgb = tuple(rgb_255(pred_rl))
                _rl_candidate = (d_rl, h_rl, p_rl, hex_rl, de_rl)
                _store_inverse_run(
                    _inverse_context, "rl", "RL Q-learning",
                    _normalized_rl_candidates(
                        _inverse_context, _rl_candidate, pred_rl))
                _rl_contract = _inverse_candidate_contract(
                    "rl", material, substrate, polarization, angle)

                def _apply_rl_cb():
                    _apply_inverse_candidate(
                        _inverse_context, "single",
                        {"d": d_rl, "h": h_rl, "p": p_rl})

                st.success(
                    f"🎯 Q-learning 完成 · 返回 {hex_rl} · ΔE2000={de_rl:.2f}")
                _render_inverse_candidate_card(
                    1, hex_rl, rgb_255(pred_rl), de_rl,
                    f"D={d_rl:.1f}nm · H={h_rl:.1f}nm · P={p_rl:.1f}nm",
                    _rl_contract, material, substrate, polarization, angle,
                    apply_key="apply_rl_result", apply_callback=_apply_rl_cb,
                )
                st.caption(
                    "候选来自本地 q-table 的离散步进，并由当前 TiO₂/SiO₂ 代理重新计算颜色；"
                    "应用后建议回到预览页复核光谱。"
                )
            except Exception as exc:
                st.warning(f"Q-learning 搜索失败：{type(exc).__name__}：{exc}")

    if gd_btn and _inverse_context.geometry_valid:
        _clear_inverse_results()
        with st.spinner("单柱梯度搜索中..."):
            try:
                # torch autograd (fast with optimized torch on server)
                result = ml_module._inverse_design_ml_serial(
                    target_rgb_norm, n_steps=150, n_restarts=8,
                    material=material, substrate=substrate
                )
                if result is None:
                    st.warning("单柱梯度不可用: 需要ONNX模型")
                else:
                    if len(result) == 6 and isinstance(result[0], str):
                        _gd_method, d_gd, h_gd, p_gd, pred_rgb, loss = result
                    else:
                        _gd_method = "fano"
                        d_gd, h_gd, p_gd, pred_rgb, loss = result
                    rc = list(rgb_255(pred_rgb))
                    hex_gd = f"#{rc[0]:02x}{rc[1]:02x}{rc[2]:02x}"
                    from color_utils import rgb_to_lab_scalar, delta_e2000_scalar
                    de_gd = delta_e2000_scalar(rgb_to_lab_scalar(pred_rgb), rgb_to_lab_scalar(target_rgb_norm))
                    st.session_state._gd_d = float(d_gd)
                    st.session_state._gd_h = float(h_gd)
                    st.session_state._gd_p = float(p_gd)
                    st.session_state._gd_hex = hex_gd
                    st.session_state._gd_de = float(de_gd)
                    st.session_state._gd_rgb = tuple(rc)
                    _gd_contract = _inverse_candidate_contract(
                        _gd_method,
                        material, substrate, polarization, angle
                    )
                    _store_inverse_run(
                        _inverse_context, "single", "单柱梯度",
                        (build_inverse_candidate(
                            _inverse_context,
                            method_id="single", method_label="单柱梯度", rank=1,
                            structure_type="single",
                            candidate_context=_candidate_context(
                                "single", material, substrate, polarization, angle),
                            route_id=_gd_contract["route_id"],
                            route_label=_gd_contract["method"],
                            model_version=_gd_contract["model"],
                            boundary=_gd_contract["boundary"],
                            parameters={"d": d_gd, "h": h_gd, "p": p_gd},
                            predicted_rgb=pred_rgb, delta_e2000=de_gd,
                        ),))
                    st.success(f"🎉 单柱梯度候选搜索完成 · 本次返回 {hex_gd} · ΔE2000={de_gd:.1f}")
                    if de_gd > 20:
                        st.warning(
                            f"当前目标色与本次返回候选仍有较大色差（ΔE2000={de_gd:.1f}）。"
                            "这表示目标可能超出当前材料/模型色域；可尝试切换材料、衬底或 FP 腔，"
                            "并在应用前进行高保真复核。"
                        )
                    _render_inverse_candidate_card(
                        1, hex_gd, rc, de_gd,
                        f"D={d_gd:.1f}nm · H={h_gd:.1f}nm · P={p_gd:.1f}nm",
                        _gd_contract, material, substrate, polarization, angle,
                    )
                    def _apply_gd_cb():
                        _apply_inverse_candidate(
                            _inverse_context, "single",
                            {"d": d_gd, "h": h_gd, "p": p_gd})
                    st.button("应用此候选", on_click=_apply_gd_cb, key="apply_gd_result", use_container_width=True)
                    st.caption("✳️ 这是当前搜索返回的候选；如需继续探索，可用 RL 搜索或手动微调")
            except Exception as e:
                logging.warning(f"app fallback: {e}")
                st.warning(f"单柱梯度优化失败: {e}")

    if _inverse_structure == "dual" and dual_gd_btn and _inverse_context.geometry_valid:
        _clear_inverse_results()
        with st.spinner("📊 双柱梯度候选搜索中（numpy Adam）..."):
            try:
                # numpy finite-difference (no torch needed)
                result = ml_module._inverse_design_dual_numpy(
                    target_rgb_norm, n_steps=150, n_restarts=8,
                    material=material, substrate=substrate, theta=angle
                )
                if result is None:
                    st.warning("双柱梯度不可用: 需要ONNX双柱模型")
                else:
                    d1_gd, h1_gd, d2_gd, h2_gd, p_gd, pred_rgb, loss = result
                    rc = list(rgb_255(pred_rgb))
                    hex_gd = f"#{rc[0]:02x}{rc[1]:02x}{rc[2]:02x}"
                    from color_utils import rgb_to_lab_scalar, delta_e2000_scalar
                    de_gd = delta_e2000_scalar(rgb_to_lab_scalar(pred_rgb), rgb_to_lab_scalar(target_rgb_norm))
                    st.session_state._dual_gd_d1 = float(d1_gd)
                    st.session_state._dual_gd_h1 = float(h1_gd)
                    st.session_state._dual_gd_d2 = float(d2_gd)
                    st.session_state._dual_gd_h2 = float(h2_gd)
                    st.session_state._dual_gd_p = float(p_gd)
                    st.session_state._dual_gd_hex = hex_gd
                    st.session_state._dual_gd_de = float(de_gd)
                    st.session_state._dual_gd_rgb = tuple(rc)
                    _dual_contract = _inverse_candidate_contract(
                        "dual", material, substrate, polarization, angle
                    )
                    _store_inverse_run(
                        _inverse_context, "dual", "双柱梯度",
                        (build_inverse_candidate(
                            _inverse_context,
                            method_id="dual", method_label="双柱梯度", rank=1,
                            structure_type="dual",
                            candidate_context=_candidate_context(
                                "dual", material, substrate, polarization, angle),
                            route_id=_dual_contract["route_id"],
                            route_label=_dual_contract["method"],
                            model_version=_dual_contract["model"],
                            boundary=_dual_contract["boundary"],
                            parameters={
                                "d1": d1_gd, "h1": h1_gd, "d2": d2_gd,
                                "h2": h2_gd, "p": p_gd,
                            },
                            predicted_rgb=pred_rgb, delta_e2000=de_gd,
                        ),))
                    st.success(f"🎉 双柱梯度候选搜索完成 · 本次返回 {hex_gd} · ΔE2000={de_gd:.1f}")
                    if de_gd > 20:
                        st.warning(
                            f"当前目标色与本次返回候选仍有较大色差（ΔE2000={de_gd:.1f}）。"
                            "这表示目标可能超出当前材料/模型色域；可尝试切换材料、衬底或 FP 腔，"
                            "并在应用前进行高保真复核。"
                        )
                    _render_inverse_candidate_card(
                        1, hex_gd, rc, de_gd,
                        f"D1={d1_gd:.1f}nm · H1={h1_gd:.1f}nm · D2={d2_gd:.1f}nm · H2={h2_gd:.1f}nm · P={p_gd:.1f}nm",
                        _dual_contract, material, substrate, polarization, angle,
                    )
                    def _apply_dual_gd_cb():
                        _apply_inverse_candidate(
                            _inverse_context, "dual",
                            {"d1": d1_gd, "h1": h1_gd, "d2": d2_gd,
                             "h2": h2_gd, "p": p_gd})
                    st.button("应用此候选", on_click=_apply_dual_gd_cb, key="apply_dual_gd_result", use_container_width=True)
                    st.caption("✳️ 这是当前搜索返回的五参数候选；如需继续探索，可手动微调")
            except Exception as e:
                logging.warning(f"app fallback: {e}")
                st.warning(f"双柱梯度优化失败: {e}")

    if _inverse_structure == "dual" and dual_physical_btn and _inverse_context.geometry_valid:
        _clear_inverse_results()
        with st.spinner("📊 双柱解析候选搜索中..."):
            try:
                from torch_model import inverse_design_dual_v2, batch_dual_pillar_rgb
                result = inverse_design_dual_v2(
                    target_rgb_norm, n_restarts=8, steps=80, lr=0.05,
                    material=material, substrate=substrate, theta=float(angle),
                    pol_TE=polarization.startswith("TE"), p_fixed=None,
                    loss_type="de2000",
                )
                d1_gd, h1_gd, d2_gd, h2_gd, p_gd, pred_rgb, _loss = result
                # The analytical optimizer is intentionally permissive about P;
                # enforce the UI's non-overlap contract and recompute the shown
                # color whenever the repair changes the period.
                p_valid = max(float(p_gd), float(d1_gd), float(d2_gd))
                if p_valid != float(p_gd):
                    p_gd = p_valid
                    import torch
                    with torch.no_grad():
                        pred_rgb = batch_dual_pillar_rgb(
                            torch.tensor([d1_gd]), torch.tensor([h1_gd]),
                            torch.tensor([d2_gd]), torch.tensor([h2_gd]),
                            torch.tensor([p_gd]), float(angle),
                            polarization.startswith("TE"), material, substrate,
                        )[0].cpu().numpy()
                pred_rgb = np.asarray(pred_rgb, dtype=float)
                rc = list(rgb_255(pred_rgb))
                hex_gd = f"#{rc[0]:02x}{rc[1]:02x}{rc[2]:02x}"
                de_gd = delta_e2000_scalar(
                    rgb_to_lab_scalar(pred_rgb), rgb_to_lab_scalar(target_rgb_norm))
                st.session_state._dual_gd_d1 = float(d1_gd)
                st.session_state._dual_gd_h1 = float(h1_gd)
                st.session_state._dual_gd_d2 = float(d2_gd)
                st.session_state._dual_gd_h2 = float(h2_gd)
                st.session_state._dual_gd_p = float(p_gd)
                st.session_state._dual_gd_hex = hex_gd
                st.session_state._dual_gd_de = float(de_gd)
                st.session_state._dual_gd_rgb = tuple(rc)
                _dual_contract = _inverse_candidate_contract(
                    "dual_physical", material, substrate, polarization, angle)
                _store_inverse_run(
                    _inverse_context, "dual_physical", "双柱解析",
                    (build_inverse_candidate(
                        _inverse_context,
                        method_id="dual_physical", method_label="双柱解析", rank=1,
                        structure_type="dual",
                        candidate_context=_candidate_context(
                            "dual", material, substrate, polarization, angle),
                        route_id=_dual_contract["route_id"],
                        route_label=_dual_contract["method"],
                        model_version=_dual_contract["model"],
                        boundary=_dual_contract["boundary"],
                        parameters={
                            "d1": d1_gd, "h1": h1_gd, "d2": d2_gd,
                            "h2": h2_gd, "p": p_gd,
                        },
                        predicted_rgb=pred_rgb, delta_e2000=de_gd,
                    ),))
                st.success(f"🎉 双柱解析候选搜索完成 · {hex_gd} · ΔE2000={de_gd:.1f}")
                _render_inverse_candidate_card(
                    1, hex_gd, rc, de_gd,
                    f"D1={d1_gd:.1f}nm · H1={h1_gd:.1f}nm · D2={d2_gd:.1f}nm · H2={h2_gd:.1f}nm · P={p_gd:.1f}nm",
                    _dual_contract, material, substrate, polarization, angle,
                )
                def _apply_dual_physical_cb():
                    _apply_inverse_candidate(
                        _inverse_context, "dual", {
                            "d1": d1_gd, "h1": h1_gd, "d2": d2_gd,
                            "h2": h2_gd, "p": p_gd})
                st.button(
                    "应用此候选", on_click=_apply_dual_physical_cb,
                    key="apply_dual_physical_result", use_container_width=True)
                st.caption("这是解析/半解析基线候选；应用后请回到预览页复核当前正向结果。")
            except Exception as e:
                logging.warning(f"dual analytical inverse failed: {e}")
                st.warning(f"双柱解析搜索失败：{type(e).__name__}：{e}")
    if ai_btn and _inverse_structure == "single" and _inverse_context.geometry_valid:
        _clear_inverse_results()
        with st.spinner("🤖 三方案并行搜索中 (TiO2 / a-Si / FP腔)..."):
            candidates = []
            _compare_far_field = bool(_far_field_enabled)
            _compare_na = float(st.session_state.get("na_val", 0.1))
            _compare_theta = float(st.session_state.get("theta_obs", 0.0))
            _compare_engine = None
            # TiO2
            try:
                _compare_engine = make_local_bound_engine(
                    MetaSurfaceColorEngine,
                    LibraryIdentity(
                        "TiO2 (anatase)", substrate, polarization, float(angle),
                        _compare_far_field, _compare_na, _compare_theta,
                    ),
                )
                r = _compare_engine.inverse_design(target_rgb_norm)
                if r:
                    candidates.append((r[0][4], "TiO2 纳米柱", r, "meta"))
            except Exception as e: logging.warning(f"AI TiO2: {e}")
            # a-Si
            try:
                if _compare_engine is None:
                    _compare_engine = MetaSurfaceColorEngine()
                bind_engine_library(
                    _compare_engine,
                    LibraryIdentity(
                        "a-Si (amorphous)", substrate, polarization, float(angle),
                        _compare_far_field, _compare_na, _compare_theta,
                    ),
                )
                r = _compare_engine.inverse_design(target_rgb_norm)
                if r:
                    candidates.append((r[0][4], "a-Si 纳米柱", r, "meta"))
            except Exception as e: logging.warning(f"AI a-Si: {e}")
            # FP DBR (same pattern as FP cavity tab search)
            try:
                from color_utils import spectrum_to_srgb, rgb_to_lab, delta_e2000
                tl = rgb_to_lab(target_rgb_norm)
                pol_te = polarization.startswith("TE")
                best = None; best_de = 999
                # Coarse grid matching existing FP search pattern
                for wl in range(380, 785, 20):
                    for t in range(50, 605, 20):
                        wls, refl = fp_dielectric_spectrum(t, float(wl), 3, 5, angle, pol_te)
                        rgb_c = spectrum_to_srgb(wls, refl)
                        de = delta_e2000(tl, rgb_to_lab(rgb_c))
                        if de < best_de: best_de = de; best = (de, t, wl, rgb_c)
                if best:
                    candidates.append((best_de, "FP DBR 腔", best, "fp"))
            except Exception as e:
                logging.warning(f"AI FP: {e}")
                import traceback; logging.warning(traceback.format_exc())
            # Sort and store
            candidates.sort(key=lambda x: x[0])
            st.session_state._ai_candidates = candidates
            if candidates:
                _store_inverse_run(
                    _inverse_context, "compare", "跨结构方案对比",
                    _normalized_compare_candidates(_inverse_context, candidates))
            # AI commentary
            if ENABLE_LLM_FEATURES and _LLM_AVAILABLE and candidates:
                try:
                    best_name = candidates[0][1]
                    best_de = candidates[0][0]
                    adv = analyze_color(picker_hex, {"方案": best_name, "ΔE2000": f"{best_de:.1f}"})
                    st.info(f"🤖 AI: {adv}")
                except Exception as e: logging.warning(f"ai: {e}")
            st.rerun()

    # --- Export results: valid InverseRun is the only source of truth. ---
    _valid_inverse_run = inverse_run_matches(
        st.session_state.get("_inverse_run"), _inverse_context)

    # --- AI smart search results ---
    if _valid_inverse_run and '_ai_candidates' in st.session_state:
        st.markdown("---")
        st.markdown("**📊 三方案对比结果**")
        st.caption("候选按 ΔE2000 排序；每张卡保留独立方法来源和适用边界。")
        for rank, (de, name, data, typ) in enumerate(st.session_state._ai_candidates):
            if typ == "meta":
                _, bp, brgb, _, _ = data[0]
                hx = rgb_to_hex(brgb)
                r,g,b = rgb_255(brgb)
                _candidate_material = (
                    "TiO2 (anatase)" if "TiO2" in name else
                    "a-Si (amorphous)" if "a-Si" in name else material
                )
                _contract = _inverse_candidate_contract(
                    "analytical", _candidate_material, substrate, polarization, angle
                )
                _render_inverse_candidate_card(
                    rank + 1, hx, [r, g, b], de,
                    f"D={bp.diameter_nm:.1f}nm · H={bp.height_nm:.1f}nm · P={bp.period_nm:.1f}nm",
                    _contract, _candidate_material, substrate, polarization, angle,
                )
            else:
                de_val, ft, fw, frgb = data
                hx = rgb_to_hex(frgb)
                r,g,b = rgb_255(frgb)
                _contract = _inverse_candidate_contract(
                    "fp", name, substrate, polarization, angle
                )
                _render_inverse_candidate_card(
                    rank + 1, hx, [r, g, b], de,
                    f"T={ft:.1f}nm · λc={fw:.1f}nm",
                    _contract, "TiO2/SiO2 DBR", substrate, polarization, angle,
                )
        if st.button('✕ 清除结果', key='clear_ai_results'):
            st.session_state.pop('_ai_candidates', None)
            st.rerun()

    # --- FP Cavity Inverse Search: uses the shared target and primary action. ---
    if is_fp and fp_search_btn:
        _clear_inverse_results()
        st.caption("FP 搜索使用页面顶部同一目标色；扫描 DBR 中心波长与腔长 T。")
        fp_target_hex = picker_hex
        if fp_search_btn:
            fp_tr = int(fp_target_hex[1:3], 16)
            fp_tg = int(fp_target_hex[3:5], 16)
            fp_tb = int(fp_target_hex[5:7], 16)

            # --- Cache check ---
            cache_key = fp_search_cache_key(
                _inverse_context,
                mirror_type=st.session_state.get("fp_mirror_type", ""),
                algorithm_version=_FP_INVERSE_ALGORITHM_VERSION,
            )
            if "fp_search_cache" not in st.session_state:
                st.session_state.fp_search_cache = {}
            if cache_key in st.session_state.fp_search_cache:
                top3 = st.session_state.fp_search_cache[cache_key]
                st.success(f"命中缓存，跳过重复计算 · 共 {len(st.session_state.fp_search_cache)} 组缓存")
            else:
                target_rgb = np.array([fp_tr, fp_tg, fp_tb]) / 255.0
                target_lab = rgb_to_lab(target_rgb)

                # Coarse grid: step 20nm
                wl_coarse = np.arange(380, 785, 20)
                t_coarse = np.arange(50, 605, 20)
                total = len(wl_coarse) * len(t_coarse)
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                count = 0
                pol_te = polarization.startswith("TE")

                # Pre-compile the spectrum function to avoid closure overhead
                _fp_spec = fp_dielectric_spectrum

                for wl in wl_coarse:
                    for t in t_coarse:
                        wls, refl = _fp_spec(t, float(wl), 3, 5, angle, pol_te)
                        rgb_c = spectrum_to_srgb(wls, refl)
                        de = delta_e2000(target_lab, rgb_to_lab(rgb_c))
                        results.append((de, float(wl), float(t), rgb_c))
                        count += 1
                    progress_bar.progress(count / total)
                    status_text.caption(f"粗搜索 {count}/{total} (步长 20nm)")

                results.sort(key=lambda x: x[0])
                top3_coarse = results[:3]

                # Fine refinement around top 3: step 4nm, +/-18nm
                fine_results = list(top3_coarse)
                for _, wl_c, t_c, _ in top3_coarse:
                    for dw in range(-18, 19, 4):
                        for dt in range(-18, 19, 4):
                            wl_f = max(380, min(780, wl_c + dw))
                            t_f = max(50, min(600, t_c + dt))
                            wls, refl = _fp_spec(t_f, wl_f, 3, 5, angle, pol_te)
                            rgb_f = spectrum_to_srgb(wls, refl)
                            de = delta_e2000(target_lab, rgb_to_lab(rgb_f))
                            fine_results.append((de, wl_f, t_f, rgb_f))

                fine_results.sort(key=lambda x: x[0])
                # Deduplicate nearby results
                top3 = []
                for de, wl, t, rgb in fine_results:
                    dup = False
                    for _, ew, et, _ in top3:
                        if abs(wl - ew) < 5 and abs(t - et) < 5:
                            dup = True; break
                    if not dup:
                        top3.append((de, wl, t, rgb))
                    if len(top3) >= 3:
                        break

                st.session_state.fp_search_cache[cache_key] = top3
                status_text.caption(f"搜索完成! 粗扫 {total} + 精细 {len(fine_results)-3} 组")

            if top3:
                _store_inverse_run(
                    _inverse_context, "fp", "FP 腔搜索",
                    _normalized_fp_candidates(_inverse_context, top3))

            for rank, (de, wl, t, rgb) in enumerate(top3):
                hex_c = rgb_to_hex(rgb)
                r255, g255, b255 = rgb_255(rgb)
                with st.container():
                    c1, c2 = st.columns([1, 4])
                    with c1:
                        st.markdown(f'<div style="width:50px;height:50px;background:{hex_c};border-radius:8px;"></div>', unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"**#{rank+1} {hex_c}** | ΔE2000={de:.1f} | λ₀={wl:.0f}nm T={t:.0f}nm | RGB({r255},{g255},{b255})")
                        st.button(
                            "应用此候选", key=f"fp_apply_{rank}",
                            on_click=_apply_inverse_candidate,
                            args=(
                                _inverse_context, "fp",
                                {"t": t, "center_wavelength": wl},
                            ),
                        )

    if smart_btn or not any((reference_btn, rl_btn, gd_btn, dual_gd_btn,
                             dual_physical_btn, ai_btn, fp_search_btn)):
        _render_saved_inverse_candidates(_inverse_context)

    # The sole export renderer runs after every search branch, including FP.
    _render_inverse_exports(_inverse_context)


# Tab 3: Pattern Generation
with tab3:
    st.subheader("独立单柱解析近似图案工具")
    st.caption("上传图片，生成结构色映射与 D/H/P 参数图。")
    _pattern_contract = make_pattern_contract(
        structure_type=_structure_type,
        structure_identity=_structure_label,
        material=material,
        substrate=substrate,
        angle_deg=float(angle),
        far_field_enabled=bool(_far_field_enabled),
    )
    with st.expander("图案设置说明", expanded=False):
        st.markdown(
            f"""
            <div class="pattern-boundary" role="status" aria-label="图案映射来源边界">
              <strong>固定合同：single · TE · 0° · no-far-field · scalar analytical</strong><br>
              <span>精确材料键：{html.escape(material)} / {html.escape(substrate)}。
              本工具不继承预览 ML、偏振或角度，不支持双柱或 FP，也不是逐像素直接 RCWA。</span><br>
              <span>模型：{html.escape(_pattern_contract.model_version)} ·
              registry：{html.escape(_pattern_contract.registry_version[:24])}…</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "图像只在当前本机 Python 进程中处理；不会调用 DeepSeek 或上传到外部服务。"
            "可用结果包含缩放原图、映射图和 D/H/P 三张参数图。"
        )
        st.markdown(
            """
            <div class="pattern-empty">
              <strong>输入与计算边界</strong>
              <ol>
                <li>上传 PNG / JPEG / WebP：单文件不超过 8 MB，源图不超过 1200 万像素且单边不超过 5000 px。</li>
                <li>选择输出最长边 20-64 像素；建议 32-48。映射按目标像素 64 个、颜色库 4096 条分块穷举，单块理论临时工作区不超过 16 MiB，不构造“全部像素 × 全颜色库”矩阵。</li>
                <li>最近邻排序使用 Lab 三维平方欧氏距离，与 ΔE76 的排序等价；这不是 ΔE00。重复或等距颜色保持颜色库中最早索引优先。</li>
                <li>点击“生成图案”后才构建 operation-local 颜色库并逐像素匹配；不会修改主会话 engine。</li>
                <li>只有 CCM 精确注册键可用；禁止 fuzzy、材料单项、默认系数或衬底替代。</li>
                <li>输出是解析近似候选，仍需另行全波或实验复核。</li>
              </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )
    if not _pattern_contract.available:
        st.warning(f"图案工具不可用：{_pattern_contract.reason}")
        if PATTERN_SESSION_KEY in st.session_state:
            st.warning("旧图案快照与当前结构或固定合同不一致，已视为 stale；结果和导出均已隐藏。")
            st.session_state.pop(PATTERN_SESSION_KEY, None)
        st.caption("上传、生成、旧结果和导出均已隐藏；不会自动切换全局结构或计算路线。")
    else:
        uploaded = st.file_uploader(
            "选择图像",
            type=["png", "jpg", "jpeg", "webp"],
            max_upload_size=8,
            key="pattern_upload_v1",
            help="PNG、JPEG 或 WebP；<=8 MB，<=1200 万像素，单边<=5000 px。",
        )

        if uploaded is None:
            st.info("尚未选择图像。上传并通过校验后，分辨率和生成操作才会出现。")
        else:
            image, upload_meta, upload_error = _open_pattern_upload(uploaded)
            if upload_error:
                st.error(f"图像未通过校验：{upload_error}")
            else:
                st.success(
                    f"图像已通过校验：{upload_meta['format']} · "
                    f"{upload_meta['width']}×{upload_meta['height']} px · "
                    f"{upload_meta['file_size'] / 1024:.1f} KiB · "
                    f"SHA-256 {upload_meta['upload_sha256'][:12]}…"
                )
                preview_col, settings_col = st.columns([1, 2])
                with preview_col:
                    st.image(image, caption="本地输入预览", width=220)
                with settings_col:
                    max_s = st.slider(
                        "输出最长边 (px)", 20, 64, 48,
                        key="pattern_max_size_v1",
                        help="保持原始宽高比缩放；较高分辨率增加逐像素匹配时间。",
                    )
                    st.caption(
                        f"最长边将缩放到不超过 {max_s}px；每个输出像素映射为一组 D/H/P 候选。"
                    )
                    generate_pattern = st.button(
                        "生成图案", type="primary", use_container_width=True,
                        key="generate_pattern_v1")

                _pattern_load = load_pattern_snapshot(
                    st.session_state, _pattern_contract,
                    upload_meta["upload_sha256"], max_s,
                )
                if _pattern_load.state == "stale":
                    st.warning("图案快照已陈旧；上传、尺寸或精确合同已变化，旧结果和导出已隐藏。")
                    st.session_state.pop(PATTERN_SESSION_KEY, None)
                elif _pattern_load.state == "invalid":
                    st.warning("图案快照未通过 schema、同源或哈希校验，旧结果和导出已隐藏。")
                    st.session_state.pop(PATTERN_SESSION_KEY, None)

                if generate_pattern and _pattern_load.state == "fresh":
                    st.success("已复用当前 fingerprint 的 session-local 图案快照；未重新 bind 或 map。")
                elif generate_pattern:
                    _store_pattern_result = True
                    try:
                        with analysis_engine_transaction(engine, st.session_state):
                            with st.spinner("构建隔离的单柱解析近似颜色库并逐像素匹配..."):
                                _pattern_identity = LibraryIdentity(
                                    material, substrate, "TE (s-pol)", 0.0,
                                    False, 0.1, 0.0,
                                )
                                _pattern_engine = MetaSurfaceColorEngine(
                                    use_disk_grid_cache=False)
                                bind_engine_library(_pattern_engine, _pattern_identity)
                                if not engine_library_matches(_pattern_engine, _pattern_identity):
                                    raise RuntimeError("隔离图案颜色库身份校验失败")
                                orig, mapped, params_arr = (
                                    _pattern_engine.image_to_metasurface_map(image, max_s))
                        _pattern_payload = build_pattern_payload(
                            _pattern_contract, upload_meta["upload_sha256"], max_s,
                            orig, mapped, params_arr,
                        )
                        _pattern_snapshot = PatternSnapshot.create(
                            _pattern_contract, upload_meta["upload_sha256"], max_s,
                            _pattern_payload,
                        )
                    except EngineStateRestoreError as exc:
                        logging.error("pattern main-engine restore failed: %s", exc)
                        st.session_state.pop(PATTERN_SESSION_KEY, None)
                        st.error("图案生成后主会话 engine 恢复失败；旧结果已清除，本次结果未保存。")
                        _store_pattern_result = False
                    except EngineStateMutationError as exc:
                        logging.error("pattern polluted main engine but was restored: %s", exc)
                        st.session_state.pop(PATTERN_SESSION_KEY, None)
                        st.error("图案生成触及了主会话 engine；状态已恢复，本次结果因完整性失败而作废。")
                        _store_pattern_result = False
                    except Exception as exc:
                        logging.warning("pattern mapping failed: %s", exc)
                        st.session_state.pop(PATTERN_SESSION_KEY, None)
                        st.error(f"图案生成失败：{type(exc).__name__}。未保存部分结果。")
                        _store_pattern_result = False
                    if _store_pattern_result:
                        store_pattern_snapshot(st.session_state, _pattern_snapshot)
                    _pattern_load = load_pattern_snapshot(
                        st.session_state, _pattern_contract,
                        upload_meta["upload_sha256"], max_s,
                    )

                if _pattern_load.state == "fresh":
                    _pattern_snapshot = _pattern_load.snapshot
                    _pattern_payload = _pattern_snapshot.payload
                    orig = np.asarray(_pattern_payload["original_rgb"], dtype=float)
                    mapped = np.asarray(_pattern_payload["mapped_rgb"], dtype=float)
                    params_arr = np.asarray(_pattern_payload["parameters_nm"], dtype=float)
                    result_left, result_right = st.columns(2)
                    with result_left:
                        st.image(orig, caption="缩放原图", width="stretch")
                    with result_right:
                        st.image(mapped, caption="解析近似映射图", width="stretch")

                    fig_params, axes_params = _get_plt().subplots(1, 3, figsize=(12, 3.4))
                    for axis, channel, title in zip(
                        axes_params, range(3), ("D (nm)", "H (nm)", "P (nm)")):
                        image_map = axis.imshow(params_arr[:, :, channel], cmap="viridis")
                        axis.set_title(title)
                        axis.axis("off")
                        _get_plt().colorbar(image_map, ax=axis, fraction=0.046)
                    fig_params.tight_layout()
                    st.pyplot(fig_params)
                    _get_plt().close(fig_params)

                    st.info(
                        f"输出 {params_arr.shape[1]}×{params_arr.shape[0]} 像素 · "
                        f"平均归一化 sRGB 欧氏误差 {_pattern_payload['mean_srgb_error']:.4f} · "
                        f"fingerprint {_pattern_snapshot.fingerprint[:12]}…"
                    )
                    try:
                        _pattern_exports = build_pattern_exports(_pattern_snapshot)
                    except (TypeError, ValueError) as exc:
                        logging.warning("pattern export contract rejected snapshot: %s", exc)
                        st.error("图案快照导出校验失败；所有导出已禁用。")
                    else:
                        export_png, export_csv, export_json = st.columns(3)
                        basename = f"pattern_{_pattern_snapshot.fingerprint[:12]}"
                        with export_png:
                            st.download_button(
                                "下载 mapped PNG", _pattern_exports.mapped_png,
                                file_name=f"{basename}_mapped.png", mime="image/png",
                                use_container_width=True,
                            )
                        with export_csv:
                            st.download_button(
                                "下载像素 CSV", _pattern_exports.csv_bytes,
                                file_name=f"{basename}_pixels.csv", mime="text/csv",
                                use_container_width=True,
                            )
                        with export_json:
                            st.download_button(
                                "下载 metadata JSON", _pattern_exports.metadata_json,
                                file_name=f"{basename}_metadata.json", mime="application/json",
                                use_container_width=True,
                            )

# Tab 4: Color Palette
with tab4:
    st.subheader("D-H 颜色映射")

    # Preserve the registered numerical grid; only its presentation is
    # transposed so diameter runs across the wider desktop axis.
    d_sample = np.linspace(80, 300, 8)
    h_sample = np.linspace(200, 600, 6)

    _mapping_route_id = str(_forward.provenance.get("route_id", ""))
    _mapping_analytical_fallback = False
    if is_dual:
        _mapping_route_id = "dual_pillar_mapping_not_registered"
    elif is_fp:
        _mapping_route_id = "fp_cavity_mapping_not_registered"
    elif (
        material == "TiO2 (anatase)"
        and substrate == "SiO2 (fused silica)"
        and _mapping_route_id in {"rcwa_surrogate", "ml_surrogate"}
        and not mapping_domain_contract(material, substrate, _mapping_route_id).available
    ):
        # The ML/RCWA registries do not publish a D-H training-domain manifest.
        # Use the exact registered analytical pair instead of leaving the map
        # dead; the label below makes the route change explicit to the user.
        _mapping_route_id = "lorentz_fano_fallback"
        _mapping_analytical_fallback = True
    _mapping_contract = mapping_domain_contract(
        material, substrate, _mapping_route_id)
    _mapping_route_label = (
        "解析映射（当前预览为 ML）" if _mapping_analytical_fallback
        else str(_forward.provenance.get("route_label", _mapping_route_id))
    )
    _mapping_model_version = (
        "engine.py / CCM analytical response" if _mapping_analytical_fallback
        else str(_forward.provenance.get("model_version", "不可用"))
    )
    _mapping_current = {
        "d": float(diameter), "h": float(height), "p": float(period),
    }
    _mapping_registry_version = canonical_sha256({
        "available": _mapping_contract.available,
        "domains": _mapping_contract.domains,
        "boundary": _mapping_contract.boundary,
        "reason": _mapping_contract.reason,
        "version": "mapping-domain-registry-v1",
    })
    _mapping_artifact_version = _analysis_artifact_version(
        _mapping_route_id, _mapping_model_version, structure_type=_structure_type,
        material=_provenance_material, substrate=_provenance_substrate)
    _mapping_artifact_available = _artifact_identity_available(_mapping_artifact_version)
    _mapping_context = _make_analysis_context(
        "mapping",
        {
            "domain_registry_version": "mapping-domain-registry-v1",
            "d_values_nm": d_sample.tolist(), "h_values_nm": h_sample.tolist(),
            "period_nm": float(period), "sample_count": int(d_sample.size * h_sample.size),
        },
        route_id=_mapping_route_id, model_version=_mapping_model_version,
        artifact_version=_mapping_artifact_version,
        registry_version=_mapping_registry_version,
    )
    _mapping_runtime_issue = _analysis_runtime_identity_issue(_mapping_context)
    _mapping_execution_available = bool(
        _mapping_contract.available and _mapping_artifact_available
        and not _mapping_runtime_issue)
    _mapping_load = load_analysis_snapshot(st.session_state, _mapping_context)
    run_mapping = st.button(
        "运行 / 加载当前 D-H 映射", key="run_mapping_analysis",
        use_container_width=True,
        disabled=not _mapping_execution_available,
    )
    if _mapping_analytical_fallback:
        st.info("当前预览使用 ML；D-H 映射使用已注册的 TiO₂/SiO₂ 解析路线。该映射用于结构趋势查看，不代表 ML/RCWA 代理输出。")
    if not _mapping_contract.available:
        st.info(f"当前映射不可用：{_mapping_contract.reason}")
        if (
            not is_dual and not is_fp
            and _mapping_route_id in {"rcwa_surrogate", "ml_surrogate"}
            and material == "TiO2 (anatase)"
            and substrate == "SiO2 (fused silica)"
        ):
            st.caption(
                "如需查看有代码证据的解析 D-H 映射，可在侧栏关闭 ML。"
                "这会切换计算路线，不会把代理模型的适用域借给解析路线。")
    if not _mapping_artifact_available:
        st.info("D-H 映射源码身份不可用；旧结果已隐藏，当前不会运行 evaluator。")
    elif _mapping_runtime_issue:
        st.info(
            f"D-H 映射模型会话身份不可用：{_mapping_runtime_issue}；"
            "旧结果已隐藏，当前不会运行 evaluator。")
    if _mapping_load.state == "stale":
        st.warning("D-H 映射快照已陈旧；当前参数、路线或采样域已变化，旧结果已隐藏。")
    elif _mapping_load.state == "invalid":
        st.warning("D-H 映射快照未通过完整性校验，旧结果已隐藏。")
    elif (
        _mapping_load.state == "missing" and _mapping_contract.available
        and _mapping_artifact_available
    ):
        st.info("点击按钮后才计算 48 个映射格；普通页面刷新不会运行 mapping evaluator。")

    if (
        run_mapping and _mapping_load.state != "fresh"
        and _mapping_execution_available
        and not _analysis_runtime_identity_issue(_mapping_context)
    ):
        _map_route = str(_mapping_route_id)
        _map_model = str(_mapping_model_version)
        _map_material = str(material)
        _map_substrate = str(substrate)
        _map_polarization = str(polarization)
        _map_angle = float(angle)
        _map_far_field = bool(_mapping_context.far_field_enabled)
        _map_na = float(_mapping_context.na)
        _map_theta = float(_mapping_context.theta_obs_deg)

        def _evaluate_map_geometry(geometry):
            return _cached_mapping_forward(
                _map_route, _map_model, _mapping_context.artifact_version,
                geometry["d"], geometry["h"], geometry["p"],
                _map_material, _map_substrate, _map_polarization, _map_angle,
                _map_far_field, _map_na, _map_theta,
            )

        _store_mapping_snapshot = True
        try:
            with (
                analysis_engine_transaction(engine, st.session_state),
                _bound_runtime_context(
                    _mapping_context.structure_type, _mapping_context.route_id,
                    _mapping_context.material, _mapping_context.substrate),
            ):
                _computed_cells = build_mapping_cells(
                    d_sample, h_sample, float(period),
                    _mapping_contract, _evaluate_map_geometry)
                _cell_payload = []
                for hi in range(len(h_sample)):
                    for di in range(len(d_sample)):
                        cell = _computed_cells[(hi, di)]
                        if cell is not None and cell.status == "available":
                            _cell_payload.append({
                                "hi": hi, "di": di, "status": "available",
                                "rgb": np.asarray(cell.rgb, dtype=float).tolist(), "reason": "",
                            })
                        else:
                            _cell_payload.append({
                                "hi": hi, "di": di, "status": "unavailable", "rgb": None,
                                "reason": cell.reason if cell is not None else _mapping_contract.reason,
                            })
                _mapping_payload = {
                    "status": "available", "reason": "",
                    "d_values": d_sample.tolist(), "h_values": h_sample.tolist(),
                    "route_id": _map_route, "boundary": _mapping_contract.boundary,
                    "analysis_artifact_version": _mapping_context.artifact_version,
                    "model_artifact_version": (
                        _analysis_model_artifact_version(_mapping_context)),
                    "cells": _cell_payload,
                }
        except (ModelResourceDriftError, ModelResourceUnavailable) as exc:
            logging.error("mapping model resource drift: %s", exc)
            st.session_state.pop(_mapping_context.session_key, None)
            st.error("D-H 映射模型会话身份发生漂移；结果与导出已清除，本次未保存。")
            _store_mapping_snapshot = False
        except EngineStateRestoreError as exc:
            logging.error("mapping engine restore failed: %s", exc)
            st.session_state.pop(_mapping_context.session_key, None)
            st.error("D-H 映射会话引擎恢复失败；旧结果已清除，本次结果未保存。")
            _store_mapping_snapshot = False
        except EngineStateMutationError as exc:
            logging.error("mapping engine mutation restored: %s", exc)
            _mapping_payload = {
                "status": "unavailable",
                "reason": "会话引擎状态被分析修改；已恢复原状态，本次结果因完整性失败而作废",
            }
        except Exception as exc:
            if exception_has_model_resource_drift(exc):
                logging.error("mapping business error with model drift: %s", exc)
                st.session_state.pop(_mapping_context.session_key, None)
                st.error("D-H 映射模型会话身份发生漂移；结果与导出已清除，本次未保存。")
                _store_mapping_snapshot = False
            else:
                logging.warning("mapping analysis failed: %s", exc)
                _mapping_payload = {
                    "status": "unavailable", "reason": f"映射计算失败：{type(exc).__name__}",
                }
        _post_mapping_issue = _analysis_runtime_identity_issue(_mapping_context)
        if _post_mapping_issue:
            st.session_state.pop(_mapping_context.session_key, None)
            st.error("D-H 映射期间模型/源码身份发生变化；本次结果未保存。")
            _store_mapping_snapshot = False
        if _store_mapping_snapshot:
            store_analysis_snapshot(
                st.session_state, AnalysisSnapshot.create(_mapping_context, _mapping_payload))
        _mapping_load = load_analysis_snapshot(st.session_state, _mapping_context)

    if _mapping_load.state == "fresh" and _mapping_execution_available:
        _mapping_payload = _mapping_load.snapshot.payload
        if _mapping_payload["status"] == "unavailable":
            st.warning(f"D-H 映射不可用：{_mapping_payload['reason']}。未跨路线补数。")
        else:
            _mapping_cells = {
                (cell["hi"], cell["di"]): MappingCellResult(
                    "available", np.asarray(cell["rgb"], dtype=float), "", None)
                if cell["status"] == "available"
                else MappingCellResult("unavailable_execution", None, cell["reason"], None)
                for cell in _mapping_payload["cells"]
            }
            _mapping_current_domain_ok = bool(
                _mapping_contract.available and not is_dual and not is_fp
                and all(
                    float(_mapping_contract.domains[key][0]) <= _mapping_current[key]
                    <= float(_mapping_contract.domains[key][1])
                    for key in ("d", "h", "p")
                )
            )
            _mapping_current_displayed = bool(
                _mapping_current_domain_ok
                and float(d_sample[0]) <= _mapping_current["d"] <= float(d_sample[-1])
                and float(h_sample[0]) <= _mapping_current["h"] <= float(h_sample[-1])
            )
            _mapping_available_indices = sorted(
                index for index, cell in _mapping_cells.items()
                if cell.status == "available")
            _mapping_current_index = (
                nearest_available_mapping_index(
                    _mapping_current["d"], _mapping_current["h"],
                    d_sample, h_sample, _mapping_cells)
                if _mapping_current_displayed else None)
            _mapping_focus_index = (
                _mapping_current_index if _mapping_current_index is not None
                else (_mapping_available_indices[0] if _mapping_available_indices else None))
            if _mapping_current_index is not None:
                _nearest_h = float(h_sample[_mapping_current_index[0]])
                _nearest_d = float(d_sample[_mapping_current_index[1]])
                _mapping_position_text = (
                    f"当前 D/H 位于注册域内；黄色边框标出最近可用采样格 D={_nearest_d:.0f} nm、"
                    f"H={_nearest_h:.0f} nm，不代表参数已吸附到该格。")
            elif not _mapping_current_domain_ok:
                _mapping_position_text = "当前 D/H/P 至少一项位于注册域外。"
            elif not _mapping_available_indices:
                _mapping_position_text = "本快照 48 个采样格均不可用。"
            else:
                _mapping_position_text = "当前参数未对应到本页离散显示范围。"

            _route_text = html.escape(f"{_mapping_route_label}（{_mapping_model_version}）")
            _boundary_text = html.escape(_mapping_contract.boundary)
            rows_html = (
                '<div class="mapping-summary" role="status" aria-live="polite">'
                '<div class="mapping-summary__item"><strong>当前参数</strong>'
                f'D={_mapping_current["d"]:.1f} nm · H={_mapping_current["h"]:.1f} nm · '
                f'P={_mapping_current["p"]:.1f} nm<br>{html.escape(_mapping_position_text)}</div>'
                '<div class="mapping-summary__item"><strong>映射来源与边界</strong>'
                f'{_route_text}<br>{_boundary_text}</div></div>'
                '<div class="mapping-legend" role="list" aria-label="映射图例">'
                '<span role="listitem"><i></i>可用：显示 RGB 与来源</span>'
                '<span role="listitem"><i class="mapping-legend__unavailable"></i>不可用：未生成伪颜色</span>'
                '<span role="listitem"><i class="mapping-legend__current"></i>当前参数最近可用采样格</span>'
                '</div><div class="mapping-table-wrap" role="region" aria-label="D-H 颜色映射表，可横向滚动" tabindex="0">'
                '<table class="mapping-grid"><caption>D-H 颜色映射；列为直径 D，行为高度 H</caption>'
                '<thead><tr><th scope="col">H（行）<br>D（列）</th>')
            for d in d_sample:
                rows_html += f'<th scope="col">D {d:.0f}<br>nm</th>'
            rows_html += '</tr></thead><tbody>'
            for hi, h in enumerate(h_sample):
                rows_html += f'<tr><th scope="row">H {h:.0f}<br>nm</th>'
                for di, d in enumerate(d_sample):
                    cell = _mapping_cells[(hi, di)]
                    is_current = bool(_mapping_current_index == (hi, di) and cell.status == "available")
                    _cell_tabindex = 0 if _mapping_focus_index == (hi, di) else -1
                    current_class = " mapping-cell--current" if is_current else ""
                    current_attribute = ' aria-current="true"' if is_current else ""
                    current_badge = '<span class="mapping-cell__current-label">当前邻近可用格</span>' if is_current else ""
                    if cell.status == "available":
                        hex_t = rgb_to_hex(cell.rgb)
                        tr, tg, tb = rgb_255(cell.rgb)
                        label = html.escape(
                            f"可用；D={d:.0f} nm，H={h:.0f} nm，RGB({tr}, {tg}, {tb})；来源={_mapping_route_label}",
                            quote=True)
                        rows_html += (
                            f'<td><div class="mapping-cell{current_class}" data-status="available" '
                            f'role="img" tabindex="{_cell_tabindex}" aria-label="{label}" title="{label}"'
                            f'{current_attribute} style="background:{hex_t};">{current_badge}'
                            f'<span class="mapping-cell__value">D {d:.0f} · H {h:.0f}<br>{hex_t}</span></div></td>')
                    else:
                        label = html.escape(
                            f"不可用；D={d:.0f} nm，H={h:.0f} nm；原因：{cell.reason}", quote=True)
                        rows_html += (
                            f'<td><div class="mapping-cell mapping-cell--unavailable{current_class}" '
                            f'data-status="unavailable" role="img" tabindex="{_cell_tabindex}" aria-label="{label}" '
                            f'title="{label}"{current_attribute}>{current_badge}'
                            f'<span class="mapping-cell__value">不可用<br>D {d:.0f} · H {h:.0f}</span></div></td>')
                rows_html += '</tr>'
            rows_html += '</tbody></table></div>'
            st.markdown(rows_html, unsafe_allow_html=True)
            st.caption(
                f"快照 fingerprint={_mapping_context.fingerprint[:12]}…；"
                f"可用格来源：{_mapping_route_label}（{_mapping_model_version}）；"
                "标为“不可用”的格子未生成伪颜色。")
    st.caption(f"材料：{material} | 衬底：{substrate} | 周期 P={period:.0f}nm")
    st.caption("横轴：直径 D (nm)｜纵轴：高度 H (nm)")
    # CIE 1931 chromaticity diagram (v5.1 - color-filled + wavelength labels)
    st.divider()
    st.subheader("CIE 1931 色度图")
    try:
        plt = _get_plt()
        fig, ax = plt.subplots(figsize=(6, 5.5))

        # CIE 1931 spectrum locus (standard data, 81 pts 380-780nm step 5nm)
        cie_xy = [
            (0.1741,0.0050),(0.1740,0.0050),(0.1738,0.0049),(0.1736,0.0049),(0.1733,0.0048),
            (0.1730,0.0048),(0.1726,0.0048),(0.1721,0.0048),(0.1714,0.0051),(0.1703,0.0058),
            (0.1689,0.0069),(0.1669,0.0086),(0.1644,0.0109),(0.1611,0.0138),(0.1566,0.0177),
            (0.1510,0.0227),(0.1440,0.0297),(0.1355,0.0399),(0.1241,0.0578),(0.1096,0.0868),
            (0.0913,0.1327),(0.0687,0.2007),(0.0454,0.2950),(0.0235,0.4127),(0.0082,0.5384),
            (0.0039,0.6548),(0.0139,0.7502),(0.0389,0.8120),(0.0743,0.8338),(0.1142,0.8262),
            (0.1547,0.8059),(0.1929,0.7816),(0.2296,0.7543),(0.2658,0.7243),(0.3016,0.6923),
            (0.3373,0.6589),(0.3731,0.6245),(0.4087,0.5896),(0.4441,0.5547),(0.4788,0.5202),
            (0.5125,0.4866),(0.5448,0.4544),(0.5752,0.4242),(0.6029,0.3965),(0.6270,0.3725),
            (0.6482,0.3514),(0.6658,0.3340),(0.6801,0.3197),(0.6915,0.3083),(0.7006,0.2993),
            (0.7079,0.2920),(0.7140,0.2859),(0.7190,0.2809),(0.7230,0.2770),(0.7260,0.2740),
            (0.7283,0.2717),(0.7300,0.2700),(0.7311,0.2689),(0.7320,0.2680),(0.7327,0.2673),
            (0.7334,0.2666),(0.7340,0.2660),(0.7344,0.2656),(0.7346,0.2654),(0.7347,0.2653),
            (0.7347,0.2653),(0.7347,0.2653),(0.7346,0.2654),(0.7344,0.2656),(0.7340,0.2660),
            (0.7334,0.2666),(0.7327,0.2673),(0.7320,0.2680),(0.7311,0.2689),(0.7300,0.2700),
            (0.7283,0.2717),(0.7260,0.2740),(0.7230,0.2770),(0.7190,0.2809),(0.7140,0.2859),
            (0.7079,0.2920),(0.7006,0.2993),(0.6915,0.3083),(0.6801,0.3197),(0.6658,0.3340),
            (0.6482,0.3514),(0.6270,0.3725),(0.6029,0.3965),(0.5752,0.4242),(0.5448,0.4544),
            (0.5125,0.4866),(0.4788,0.5202),(0.4441,0.5547),(0.4087,0.5896),(0.3731,0.6245),
            (0.3373,0.6589),(0.3016,0.6923),(0.2658,0.7243),(0.2296,0.7543),(0.1929,0.7816),
            (0.1547,0.8059),(0.1142,0.8262),(0.0743,0.8338),(0.0389,0.8120),(0.0139,0.7502),
            (0.0039,0.6548),(0.0082,0.5384),(0.0235,0.4127),(0.0454,0.2950),(0.0687,0.2007),
            (0.0913,0.1327),(0.1096,0.0868),(0.1241,0.0578),(0.1355,0.0399),(0.1440,0.0297),
            (0.1510,0.0227),(0.1566,0.0177),(0.1611,0.0138),(0.1644,0.0109),(0.1669,0.0086),
            (0.1689,0.0069),(0.1703,0.0058),(0.1714,0.0051),(0.1721,0.0048),(0.1726,0.0048),
            (0.1730,0.0048),(0.1733,0.0048),(0.1736,0.0049),(0.1738,0.0049),(0.1740,0.0050),
        ]
        cx = [p[0] for p in cie_xy]; cy = [p[1] for p in cie_xy]
        # === CIE 1931 colour fill via ray-intersection from D65 white point ===
        import matplotlib.path as mpath_mod

        @st.cache_data(show_spinner=False)
        def _cie_colour_image():
            n_half = len(cx) // 2
            cx_h = np.array(cx[:n_half], dtype=np.float64)
            cy_h = np.array(cy[:n_half], dtype=np.float64)
            wl_h = np.linspace(380, 780, n_half)
            cx_cl = np.append(cx_h, cx_h[0])
            cy_cl = np.append(cy_h, cy_h[0])
            lp = mpath_mod.Path(np.column_stack([cx_cl, cy_cl]))
            wx, wy = 0.3333, 0.3333
            res = 750
            gxv = np.linspace(0, 0.80, res)
            gyv = np.linspace(0, 0.90, res)
            gx_m, gy_m = np.meshgrid(gxv, gyv)
            pts = np.column_stack([gx_m.ravel(), gy_m.ravel()])
            inside = lp.contains_points(pts).reshape(res, res)

            def _wl2rgb(w):
                w = max(380, min(780, w))
                if w < 440:
                    t = (w-380)/60; return 0.5*(1-t)+0.3*t**0.8, 0.0, 1-0.5*(1-t)
                if w < 490:
                    t = (w-440)/50; return 0.0, t*0.9, 1.0
                if w < 510:
                    t = (w-490)/20; return 0.0, 0.9+0.1*t, 1-t
                if w < 580:
                    t = (w-510)/70; return t**0.7, 1.0, 0.0
                if w < 645:
                    t = (w-580)/65; return 1.0, 1-t**0.6, 0.0
                t = min((w-645)/135, 1.0); return 1-0.4*t, 0.0, 0.0

            r = np.ones((res, res)); g = np.ones((res, res)); b = np.ones((res, res))
            iy, ix = np.where(inside)
            ni = len(iy)
            if ni == 0:
                return np.stack([r, g, b], axis=-1), inside
            px = gxv[ix]; py = gyv[iy]
            dx = px - wx; dy = py - wy
            dp = np.sqrt(dx*dx+dy*dy); dp[dp<1e-12] = 1e-12
            dx /= dp; dy /= dp
            best_t = np.full(ni, np.inf)
            best_w = np.full(ni, 550.0)
            for k in range(n_half - 1):
                x1, y1 = cx_h[k], cy_h[k]
                x2, y2 = cx_h[k+1], cy_h[k+1]
                det = dx*(y1-y2) - dy*(x1-x2)
                ok = np.abs(det) > 1e-12
                if not ok.any():
                    continue
                t = np.full(ni, np.inf)
                s = np.full(ni, -1.0)
                det_o = det[ok]
                t[ok] = ((x1-wx)*(y1-y2)-(y1-wy)*(x1-x2)) / det_o
                s[ok] = (dx[ok]*(y1-wy)-dy[ok]*(x1-wx)) / det_o
                better = (t>0) & (s>=0) & (s<=1) & (t<best_t)
                if better.any():
                    best_t[better] = t[better]
                    best_w[better] = wl_h[k] + s[better]*(wl_h[k+1]-wl_h[k])
            nohit = np.isinf(best_t)
            if nohit.any():
                x1, y1 = cx_h[0], cy_h[0]
                x2, y2 = cx_h[-1], cy_h[-1]
                det_p = dx[nohit]*(y1-y2) - dy[nohit]*(x1-x2)
                ok_p = np.abs(det_p) > 1e-12
                if ok_p.any():
                    no_idx = np.where(nohit)[0][ok_p]
                    t_p = ((x1-wx)*(y1-y2)-(y1-wy)*(x1-x2)) / det_p
                    s_p = (dx[no_idx]*(y1-wy)-dy[no_idx]*(x1-wx)) / det_p
                    good_p = (t_p>0) & (s_p>=0) & (s_p<=1)
                    if good_p.any():
                        gi = np.where(nohit)[0][good_p]
                        t_g = t_p[good_p]; s_g = s_p[good_p]
                        r1, g1, b1 = _wl2rgb(380)
                        r2, g2, b2 = _wl2rgb(700)
                        for m in range(len(gi)):
                            rr = (1-s_g[m])*r1 + s_g[m]*r2
                            gg = (1-s_g[m])*g1 + s_g[m]*g2
                            bb = (1-s_g[m])*b1 + s_g[m]*b2
                            pur = dp[gi[m]]/t_g[m]
                            r[iy[gi[m]],ix[gi[m]]] = pur*rr+(1-pur)*0.98
                            g[iy[gi[m]],ix[gi[m]]] = pur*gg+(1-pur)*0.98
                            b[iy[gi[m]],ix[gi[m]]] = pur*bb+(1-pur)*0.98
            hit = ~nohit
            if hit.any():
                hi = np.where(hit)[0]
                pur = dp[hi]/best_t[hi]
                for m in range(len(hi)):
                    rr, gg, bb = _wl2rgb(best_w[hi[m]])
                    r[iy[hi[m]],ix[hi[m]]] = pur[m]*rr+(1-pur[m])*0.98
                    g[iy[hi[m]],ix[hi[m]]] = pur[m]*gg+(1-pur[m])*0.98
                    b[iy[hi[m]],ix[hi[m]]] = pur[m]*bb+(1-pur[m])*0.98
            return np.stack([r, g, b], axis=-1), inside

        rgb_img, ccmask = _cie_colour_image()
        ax.imshow(rgb_img, extent=[0, 0.80, 0, 0.90], origin="lower", aspect="auto", zorder=0)

        # Locus outline
        ax.fill(cx, cy, color="#f0f0f0", alpha=0.15, zorder=1)
        ax.plot(cx, cy, "k-", linewidth=1.0, zorder=2)

        # Purple line: straight line connecting 380nm and 700nm endpoints
        ax.plot([cx[0], cx[-1]], [cy[0], cy[-1]], "k-", linewidth=0.8, alpha=0.4, zorder=2)

        # sRGB gamut triangle (standard coordinates)
        sR, sG, sB = (0.640, 0.330), (0.300, 0.600), (0.150, 0.060)
        ax.plot([sR[0], sG[0]], [sR[1], sG[1]], "k--", linewidth=0.5, zorder=3)
        ax.plot([sG[0], sB[0]], [sG[1], sB[1]], "k--", linewidth=0.5, zorder=3)
        ax.plot([sB[0], sR[0]], [sB[1], sR[1]], "k--", linewidth=0.5, zorder=3)
        ax.text(sR[0], sR[1] + 0.02, "R", fontsize=8, ha="center", zorder=4)
        ax.text(sG[0] - 0.02, sG[1] + 0.02, "G", fontsize=8, ha="center", zorder=4)
        ax.text(sB[0], sB[1] - 0.03, "B", fontsize=8, ha="center", zorder=4)

        # Wavelength labels on spectral locus (standard positions)
        # Index mapping: 81 pts, 380-780nm step 5nm
        wl_labels = {
            380: ("380 nm", 18, -2),
            460: ("460 nm", -30, 28),
            520: ("520 nm", 0, 15),
            580: ("580 nm", 12, 5),
            620: ("620 nm", 18, 14),
            700: ("700 nm", 18, -22),
        }
        for wl_nm, (label, dx, dy) in wl_labels.items():
            idx = int((wl_nm - 380) / 5)
            if 0 <= idx < len(cie_xy):
                px, py = cie_xy[idx]
                ax.annotate(label, (px, py), textcoords="offset points",
                           xytext=(dx, dy), fontsize=7, color="#444",
                           ha="center", va="center", zorder=5,
                           bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))

        # Derive the current point from the same validated ForwardResult used by
        # the preview, spectrum plot, and CSV export.
        try:
            if _forward.spectrum_available:
                _xyz = np.asarray(spectrum_to_xyz(_forward.wavelengths_nm, _forward.reflectance), dtype=float)
                _xy = xyz_to_xy(_xyz)
                px, py = float(_xy[0]), float(_xy[1])
                if np.isfinite(px) and np.isfinite(py):
                    ax.plot(px, py, "o", color="#FF1744", markersize=9, markeredgecolor="white", markeredgewidth=1.2, zorder=6)
                    ax.annotate(f"({px:.3f}, {py:.3f})", (px, py),
                               textcoords="offset points", xytext=(10, 10),
                               fontsize=8, color="#333", zorder=6)
            else:
                st.caption(f"当前路由没有可用光谱，无法标注色度点：{_forward.error}")
        except Exception as e:
            st.caption(f"色度点计算失败：{e}")

        ax.set_xlim(0, 0.80)
        ax.set_ylim(0, 0.90)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(material.replace("TiO2", "TiO$_2$") + f"  |  D={diameter:.0f}nm H={height:.0f}nm P={period:.0f}nm", fontsize=9)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15, lw=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("灰色区域: CIE 1931 色度图马蹄轨迹 | 虚线三角: sRGB 色域 | 红点: 当前预测色坐标 | 波长标注: 单色光位置")
    except Exception as e:
        st.caption(f"色度图渲染失败: {e}")

# Tab 5: Spectrum & CIE Chromaticity
with tab5:
    st.subheader("光谱与颜色位置")
    st.caption("上方看反射强弱，下方看颜色在标准色度图中的位置。")
    with st.expander("查看详细计算条件", expanded=False):
        st.caption(forward_provenance_caption(_forward.provenance))

    # Keep the two plots in one vertical reading flow.  The old 3:2 columns
    # compressed both figures on the normal 1114px browser viewport and made
    # their labels compete with each other.
    st.markdown("#### 反射光谱 · 380–780 nm")
    wls, refl = _forward.wavelengths_nm, _forward.reflectance
    if not _forward.spectrum_available:
        st.warning(f"当前结果没有可用光谱，无法绘制：{_forward.error}")
        wls, refl = np.array([]), np.array([])

    fig5, ax5 = _get_plt().subplots(figsize=(8, 5))
    # Color the spectrum curve with the actual computed color
    hex_c = rgb_to_hex(_forward.rgb) if _forward.spectrum_available else "#777777"
    if len(wls):
        ax5.plot(wls, refl, color="#333", lw=2.5,
                 label=str(_forward.provenance.get("geometry_summary") or "unknown"))
        ax5.fill_between(wls, 0, refl, alpha=0.12, color=hex_c)
    ax5.set_xlabel("波长 (nm)")
    ax5.set_ylabel("反射率")
    ax5.set_title("反射光谱")
    ax5.set_xlim(380, 780)
    ax5.set_ylim(0, 1.08)
    ax5.set_xticks(np.arange(380, 781, 80))
    ax5.tick_params(axis="both", labelsize=8)
    ax5.grid(True, alpha=0.25)
    if len(wls):
        ax5.legend(loc="upper right", fontsize=8, framealpha=0.85)
    fig5.tight_layout(pad=1.1)
    st.pyplot(fig5); _get_plt().close(fig5)

    st.divider()
    st.markdown("#### CIE 1931 色度位置")
    st.caption("红点是当前颜色；马蹄形轨迹和三角形用于定位与比较。")
    try:
        _cie_plot = build_cie_plot_data(_forward, _CIE_X, _CIE_Y, _CIE_Z)
    except Exception as exc:
        st.error(f"CIE 标准轨迹不可用：{type(exc).__name__}: {exc}")
        _cie_plot = None
    if _cie_plot is not None:
        fig_cie, ax_cie = _get_plt().subplots(figsize=(8, 6))
        locus = _cie_plot.spectral_locus_xy
        ax_cie.plot(
            locus[:, 0], locus[:, 1], "k-", lw=1.2, alpha=0.8,
            label="CIE 1931 光谱轨迹",
        )
        ax_cie.fill(locus[:, 0], locus[:, 1], alpha=0.05, color="gray")
        srgb_primaries_xy = _cie_plot.srgb_triangle_xy
        ax_cie.plot(
            srgb_primaries_xy[:, 0], srgb_primaries_xy[:, 1],
            "k--", lw=0.8, alpha=0.5, label="sRGB 色域",
        )
        ax_cie.plot(0.3127, 0.3290, "k+", ms=8, alpha=0.5, label="D65")
        if _cie_plot.current.available:
            xy = _cie_plot.current.xy
            ax_cie.plot(
                xy[0], xy[1], "o", color=hex_c, ms=10,
                markeredgecolor="white", markeredgewidth=1.5,
                label="当前颜色",
            )
            ax_cie.plot(xy[0], xy[1], "o", color=hex_c, ms=14, alpha=0.3)
            ax_cie.set_title(f"当前颜色位置  ({xy[0]:.4f}, {xy[1]:.4f})")
        else:
            ax_cie.set_title("当前颜色位置不可用")
            st.warning(f"当前 CIE 点不可用，未绘制占位点：{_cie_plot.current.reason}")
        ax_cie.set_xlabel("x")
        ax_cie.set_ylabel("y")
        ax_cie.set_xlim(0, 0.75)
        ax_cie.set_ylim(0, 0.85)
        ax_cie.set_aspect("equal")
        ax_cie.grid(True, alpha=0.2)
        ax_cie.legend(fontsize=8, loc="lower left")
        fig_cie.tight_layout()
        st.pyplot(fig_cie)
        _get_plt().close(fig_cie)


    # === Declared-route gamut comparison ===
    st.divider()
    st.subheader("模型路线色域对比")

    @st.cache_data
    def _lorentz_fano_gamut_points(material, substrate):
        """Sample the declared Lorentz/Fano analytical approximation."""
        try:
            import torch_model as _tm
        except Exception as exc:
            raise RuntimeError("torch_model unavailable") from exc
        D = np.arange(60, 340, 20, dtype=np.float32)
        H = np.arange(100, 580, 40, dtype=np.float32)
        P = np.arange(220, 580, 40, dtype=np.float32)
        pol_TE = True
        xy_list = []
        # Process in batches to avoid OOM
        for d in D:
            d_batch = []; h_batch = []; p_batch = []
            for h in H:
                for p in P:
                    if d >= p:
                        continue
                    fr = np.pi*(d/2)**2/(p**2)
                    if fr < 0.03 or fr > 0.70:
                        continue
                    d_batch.append(d); h_batch.append(h); p_batch.append(p)
            if not d_batch:
                continue
            d_t = _tm.torch.tensor(d_batch, dtype=_tm.torch.float32)
            h_t = _tm.torch.tensor(h_batch, dtype=_tm.torch.float32)
            p_t = _tm.torch.tensor(p_batch, dtype=_tm.torch.float32)
            spec = _tm.batch_lorentzian_spectrum(d_t, h_t, p_t,
                _tm.torch.zeros(len(d_batch)), pol_TE,
                material=material, substrate=substrate)
            rgb = _tm.batch_spectrum_to_rgb(spec).numpy()
            for r in rgb:
                r = np.clip(r, 0, 1)
                xy_list.append(rgb_to_xy(r))
        return np.array(xy_list) if xy_list else np.zeros((0, 2))

    @st.cache_data
    def _fp_dbr_gamut_points():
        """Sample the dielectric-mirror FP-TMM route."""
        from fp_cavity import fp_dielectric_spectrum
        from color_utils import spectrum_to_srgb as _g2s, rgb_to_xy as _g2xy
        pts = []
        failed_samples = 0
        for t in range(50, 601, 10):
            for wl_c in range(380, 781, 10):
                try:
                    wls, refl = fp_dielectric_spectrum(t, float(wl_c), 3, 5, 0.0, True)
                    rgb = _g2s(wls, np.clip(refl, 0, None))
                    pts.append(_g2xy(rgb))
                except Exception:
                    failed_samples += 1
        return GamutSamples(np.asarray(pts, dtype=float), failed_samples)

    @st.cache_data
    def _fp_ag_gamut_points():
        """Sample the Ag-mirror FP-TMM route."""
        from fp_cavity import fp_cavity_spectrum
        from color_utils import spectrum_to_srgb as _g2s, rgb_to_xy as _g2xy
        pts = []
        failed_samples = 0
        for t in range(50, 601, 10):
            for angle in range(0, 81, 10):
                try:
                    wls, refl = fp_cavity_spectrum(t, float(angle), True)
                    rgb = _g2s(wls, np.clip(refl, 0, None))
                    pts.append(_g2xy(rgb))
                except Exception:
                    failed_samples += 1
        return GamutSamples(np.asarray(pts, dtype=float), failed_samples)

    show_gamut = st.checkbox(
        "显示模型路线色域对比图", value=False,
        help=("仅比较 Lorentz/Fano analytical approximation 与 FP-TMM 的采样结果；"
              "不是直接 RCWA、代理模型精度或实验色域。"),
    )
    if show_gamut:
        with st.spinner("按声明路线计算色域采样（Lorentz/Fano approximation、FP-TMM）..."):
            gamut_results = [
                evaluate_gamut(
                    "Lorentz/Fano analytical approximation", "TiO2 / SiO2",
                    lambda: _lorentz_fano_gamut_points(
                        "TiO2 (anatase)", "SiO2 (fused silica)")),
                evaluate_gamut(
                    "Lorentz/Fano analytical approximation", "a-Si / SiO2",
                    lambda: _lorentz_fano_gamut_points(
                        "a-Si (amorphous)", "SiO2 (fused silica)")),
                evaluate_gamut("FP-TMM", "TiO2 cavity / SiO2 DBR", _fp_dbr_gamut_points),
                evaluate_gamut("FP-TMM", "TiO2 cavity / Ag mirrors", _fp_ag_gamut_points),
            ]

        for gamut_result in gamut_results:
            if gamut_result.error:
                st.warning(
                    f"{gamut_result.route_label} / {gamut_result.system_label} 不可用："
                    f"{gamut_result.error}")
            if gamut_result.warning:
                st.warning(
                    f"{gamut_result.route_label} / {gamut_result.system_label}："
                    f"{gamut_result.warning}")

        try:
            from scipy.spatial import ConvexHull
        except Exception as exc:
            ConvexHull = None
            st.warning(f"色域凸包不可用：{type(exc).__name__}: {exc}")

        fig_gamut, ax_g = _get_plt().subplots(figsize=(6.5, 6.5))
        plot_base = build_cie_plot_data(_forward, _CIE_X, _CIE_Y, _CIE_Z)
        x_xy = plot_base.spectral_locus_xy[:, 0]
        y_xy = plot_base.spectral_locus_xy[:, 1]
        srgb_primaries_xy = plot_base.srgb_triangle_xy
        ax_g.plot(x_xy, y_xy, "k-", lw=1.0, alpha=0.7)
        ax_g.fill(x_xy, y_xy, alpha=0.04, color="gray")
        ax_g.plot(0.3127, 0.3290, "k+", ms=8, label="D65")
        ax_g.plot(srgb_primaries_xy[:, 0], srgb_primaries_xy[:, 1],
                  "k--", lw=1.0, alpha=0.7, label="sRGB 色域")

        gamut_colors = ["#ff6b35", "#00aaff", "#44aa44", "#a84aa8"]
        for gamut_result, color in zip(gamut_results, gamut_colors):
            if not gamut_result.available:
                continue
            pts = gamut_result.points_xy
            label = f"{gamut_result.system_label} · {gamut_result.route_label}"
            step = max(1, len(pts) // 1200)
            ax_g.scatter(pts[::step, 0], pts[::step, 1], c=color, s=2, alpha=0.20, label=label)
            if ConvexHull is not None:
                try:
                    hull = ConvexHull(pts)
                    hull_pts = np.append(hull.vertices, hull.vertices[0])
                    ax_g.plot(pts[hull_pts, 0], pts[hull_pts, 1], "-", color=color, lw=2.0, alpha=0.90)
                except Exception as exc:
                    st.warning(
                        f"{gamut_result.system_label} 凸包不可用："
                        f"{type(exc).__name__}: {exc}")

        ax_g.legend(fontsize=7, loc="lower left", framealpha=0.85, ncol=1)
        ax_g.set_xlabel("x"); ax_g.set_ylabel("y")
        ax_g.set_title("模型路线色域对比（TE，0 deg）")
        ax_g.set_xlim(0, 0.75); ax_g.set_ylim(0, 0.85)
        ax_g.set_aspect("equal")
        ax_g.grid(True, alpha=0.15)
        fig_gamut.tight_layout()
        st.pyplot(fig_gamut)
        _get_plt().close(fig_gamut)

    def _benchmark_methods():
        """Benchmark only locally eligible methods against one fixed target."""
        import time
        from fp_cavity import fp_cavity_spectrum
        from color_utils import spectrum_to_srgb, delta_e2000, rgb_to_lab
        target_rgb = np.array([122, 79, 34], dtype=float) / 255.0  # #7A4F22
        target_lab = rgb_to_lab(target_rgb)
        mat = "TiO2 (anatase)"
        sub = "SiO2 (fused silica)"
        rows = []

        # 1. Smart grid: local RCWA-trained surrogate, never direct RCWA.
        if not _smart_grid_has_matching_rcwa_session(mat, sub):
            rows.append(BenchmarkRow.unavailable(
                "smart", "智能网格", route="RCWA-trained ML surrogate",
                detail="未加载当前 TiO2/SiO2 精确配对的 RCWA 训练代理 session。",
            ))
        elif not _smart_grid_has_local_weights(mat, sub):
            rows.append(BenchmarkRow.unavailable(
                "smart", "智能网格", route="RCWA-trained ML surrogate",
                detail="缺少当前 TiO2/SiO2 配对的本地模型文件。",
            ))
        else:
            t0 = time.perf_counter()
            try:
                with _bound_runtime_context("single", "rcwa_surrogate", mat, sub):
                    result = ml_module.smart_grid_search(
                        target_rgb, material=mat, substrate=sub,
                        angle_deg=0.0, polarization="TE (s-pol)",
                        # Keep the benchmark path consistent with the interactive
                        # route; the registered ONNX exports accept batch=1, so
                        # the larger legacy grid multiplies inference calls.
                        coarse_n=8, top_k=1, fine_steps=3, fine_range=8.0,
                    )
                elapsed = time.perf_counter() - t0
                if not result:
                    raise ValueError("搜索未返回候选")
                rows.append(benchmark_row_from_rgb(
                    "smart", "智能网格", elapsed_s=elapsed,
                    predicted_rgb=result[0][2], target_rgb=target_rgb,
                    route="RCWA-trained ML surrogate",
                    detail=(
                        "两阶段本地代理搜索；本次未调用直接 RCWA，"
                        "结果只是对目标色的模型空间匹配。"
                    ),
                ))
            except Exception as exc:
                rows.append(BenchmarkRow.error(
                    "smart", "智能网格", route="RCWA-trained ML surrogate",
                    detail=f"搜索失败：{type(exc).__name__}。",
                ))

        # 2. RL q-table: run only with the exact local TiO2/SiO2 proxy binding.
        if not _rl_route_ready(mat, sub, "TE (s-pol)", 0.0):
            rows.append(BenchmarkRow.unavailable(
                "rl", "RL Q-learning", route="rl_qlearning",
                detail=(
                    "缺少本地 rl_qtable 或 TiO2/SiO2 精确 RCWA 代理；"
                    "未跨模型补数。"
                ),
            ))
        else:
            t0 = time.perf_counter()
            try:
                with _bound_runtime_context("single", "rcwa_surrogate", mat, sub):
                    rl_agent = rl_design.get_trained_rl()
                    d_rl, h_rl, p_rl, _rl_hex, _ = rl_agent.search(
                        "#7A4F22", steps=30, restarts=5)
                    predicted_rgb = ml_module.predict_rgb(
                        d_rl, h_rl, p_rl, 0.0, "TE (s-pol)", mat, sub)
                if predicted_rgb is None:
                    raise ValueError("RL 代理未返回候选颜色")
                rows.append(benchmark_row_from_rgb(
                    "rl", "RL Q-learning",
                    elapsed_s=time.perf_counter() - t0,
                    predicted_rgb=predicted_rgb, target_rgb=target_rgb,
                    route="rl_qlearning",
                    detail=(
                        "本地 q-table 离散探索；颜色由同一 TiO2/SiO2 "
                        "RCWA 代理重新计算。"
                    ),
                ))
            except Exception as exc:
                rows.append(BenchmarkRow.error(
                    "rl", "RL Q-learning", route="rl_qlearning",
                    detail=f"搜索失败：{type(exc).__name__}。",
                ))

        # 3. Single-pillar gradient: local model only, no download fallback.
        _single_benchmark_ready, _single_benchmark_reason = ml_module.single_inverse_model_status(mat, sub)
        if not _single_benchmark_ready:
            rows.append(BenchmarkRow.unavailable(
                "single", "单柱梯度", route="本地 PyTorch 代理/解析路线",
                detail=_single_benchmark_reason,
            ))
        else:
            t0 = time.perf_counter()
            try:
                result = ml_module._inverse_design_ml_serial(
                    target_rgb, n_steps=150, n_restarts=8,
                    material=mat, substrate=sub,
                )
                elapsed = time.perf_counter() - t0
                if result is None:
                    raise ValueError("优化未返回候选")
                if isinstance(result, dict):
                    # The registered RCWA route returns a named mapping while
                    # the legacy local route returns a tuple.  Normalize both
                    # forms here so a valid single-pillar run is not reported
                    # as an unsupported result by the benchmark UI.
                    method_name = str(result.get("method") or "RCWA")
                    predicted_rgb = result.get("pred_rgb")
                    if predicted_rgb is None:
                        raise ValueError("单柱优化结果缺少 pred_rgb")
                elif len(result) == 6 and isinstance(result[0], str):
                    method_name, predicted_rgb = result[0], result[4]
                elif len(result) == 5:
                    method_name, predicted_rgb = "legacy local route", result[3]
                else:
                    raise ValueError("单柱优化返回格式不受支持")
                rows.append(benchmark_row_from_rgb(
                    "single", "单柱梯度", elapsed_s=elapsed,
                    predicted_rgb=predicted_rgb, target_rgb=target_rgb,
                    route=f"本地代理/解析梯度路线（返回标识 {method_name}）",
                    detail="只评估当次返回的候选 sRGB；不宣称直接 RCWA 或全局最优。",
                ))
            except Exception as exc:
                rows.append(BenchmarkRow.error(
                    "single", "单柱梯度", route="本地 PyTorch 代理/解析路线",
                    detail=f"优化失败：{type(exc).__name__}: {exc}",
                ))

        # 4. Prefer the registered dual ML route.  When its ONNX/domain
        # evidence is absent, keep that route fail-closed but expose the
        # already-supported analytical dual baseline so the UI still gives
        # users a runnable two-pillar reference instead of a dead row.
        dual_context = InverseContext(
            structure_type="dual", material=mat, substrate=sub,
            polarization="TE (s-pol)", angle_deg=0.0,
            target_rgb=(122, 79, 34), target_hex="#7A4F22",
            preview_route_id="benchmark", preview_model_version=_DUAL_MODEL_VERSION,
            geometry_valid=True,
        )
        if not (_dual_ml_ready and _dual_domain_manifest_verified(dual_context)):
            rows.append(BenchmarkRow.unavailable(
                "dual", "双柱梯度", route="Dual-pillar ML surrogate",
                    detail=(
                        "未运行：缺少已验证的双柱 ONNX 模型或其哈希绑定训练域 manifest；"
                        "当前保留双柱正向解析/半解析路线，不把未注册模型当作逆设计候选。"
                    ),
            ))
            if importlib.util.find_spec("torch") is not None:
                t0 = time.perf_counter()
                try:
                    from torch_model import inverse_design_dual_v2
                    result = inverse_design_dual_v2(
                        target_rgb, n_restarts=8, steps=80, lr=0.05,
                        material=mat, substrate=sub, theta=0.0,
                        pol_TE=True, p_fixed=None, loss_type="de2000",
                    )
                    elapsed = time.perf_counter() - t0
                    if result is None or len(result) != 7:
                        raise ValueError("双柱解析基线未返回完整候选")
                    predicted_rgb = result[5]
                    rows.append(benchmark_row_from_rgb(
                        "dual_physical", "双柱解析基线", elapsed_s=elapsed,
                        predicted_rgb=predicted_rgb, target_rgb=target_rgb,
                        route="Lorentz/Fano analytical dual",
                        detail=(
                            "双柱解析近似基线；可用于交互演示与路线对照，"
                            "不代表双柱 ONNX/RCWA 精度。"
                        ),
                    ))
                except Exception as exc:
                    rows.append(BenchmarkRow.error(
                        "dual_physical", "双柱解析基线",
                        route="Lorentz/Fano analytical dual",
                        detail=f"解析基线失败：{type(exc).__name__}: {exc}",
                    ))
        else:
            t0 = time.perf_counter()
            try:
                result = ml_module._inverse_design_dual_numpy(
                    target_rgb, n_steps=150, n_restarts=8,
                    material=mat, substrate=sub, theta=0.0,
                )
                elapsed = time.perf_counter() - t0
                if result is None or len(result) != 7:
                    raise ValueError("双柱优化未返回完整候选")
                rows.append(benchmark_row_from_rgb(
                    "dual", "双柱梯度", elapsed_s=elapsed,
                    predicted_rgb=result[5], target_rgb=target_rgb,
                    route="Dual-pillar ML surrogate",
                    detail="双柱五参数候选；仅在已验证的本地模型域内报告。",
                ))
            except Exception as exc:
                rows.append(BenchmarkRow.error(
                    "dual", "双柱梯度", route="Dual-pillar ML surrogate",
                    detail=f"优化失败：{type(exc).__name__}。",
                ))

        # 5. Cross-structure comparison is numeric only with >=2 actual contributors.
        t0 = time.perf_counter()
        try:
            contributors = []
            candidates = []
            for material_name in (
                "TiO2 (anatase)", "a-Si (amorphous)",
                "Si3N4 (nitride)", "Al2O3 (sapphire)",
            ):
                if not (
                    _smart_grid_has_matching_rcwa_session(material_name, sub)
                    and _smart_grid_has_local_weights(material_name, sub)
                ):
                    continue
                try:
                    _ensure_rcwa_ml(material_name, sub)
                    with _bound_runtime_context(
                            "single", "rcwa_surrogate", material_name, sub):
                        result = ml_module.smart_grid_search(
                            target_rgb, material=material_name, substrate=sub,
                            angle_deg=0.0, polarization="TE (s-pol)",
                            coarse_n=8, top_k=1, fine_steps=3, fine_range=6.0,
                        )
                    if result:
                        rgb_candidate = np.asarray(result[0][2], dtype=float)
                        de_candidate = delta_e2000(
                            target_lab, rgb_to_lab(rgb_candidate))
                        candidates.append((de_candidate, rgb_candidate))
                        contributors.append(f"{material_name} smart surrogate")
                except Exception:
                    pass
            if not contributors:
                rows.append(BenchmarkRow.unavailable(
                    "compare", "跨结构候选比较", route="本地可用代理 + FP-TMM",
                    detail=(
                        "当前没有可运行的精确材料/衬底 smart 代理；"
                        "为避免把单独 FP 扫描冒充跨结构比较，本行未运行。"
                    ),
                ))
                return tuple(rows)
            best_fp = None
            for t_nm in range(50, 601, 20):
                wls, refl = fp_cavity_spectrum(t_nm, 0.0, True)
                rgb_fp = spectrum_to_srgb(wls, np.clip(refl, 0, None))
                de_fp = delta_e2000(target_lab, rgb_to_lab(rgb_fp))
                if best_fp is None or de_fp < best_fp[0]:
                    best_fp = (de_fp, rgb_fp)
            if best_fp is not None:
                candidates.append(best_fp)
                contributors.append("FP-Ag TMM coarse scan")
            if len(contributors) < 2 or not candidates:
                rows.append(BenchmarkRow.unavailable(
                    "compare", "跨结构候选比较", route="本地可用代理 + FP-TMM",
                    detail=(
                        "本次不足两个实际可运行贡献者，不伪造跨结构数值；"
                        f"实际贡献者：{', '.join(contributors) or '无'}。"
                    ),
                ))
            else:
                _, best_rgb = min(candidates, key=lambda item: item[0])
                rows.append(benchmark_row_from_rgb(
                    "compare", "跨结构候选比较",
                    elapsed_s=time.perf_counter() - t0,
                    predicted_rgb=best_rgb, target_rgb=target_rgb,
                    route="本地可用代理 + FP-TMM",
                    detail=(
                        "只比较当次实际运行的贡献者，不代表所有方法；"
                        f"实际贡献者：{', '.join(contributors)}。"
                    ),
                ))
        except Exception as exc:
            rows.append(BenchmarkRow.error(
                "compare", "跨结构候选比较", route="本地可用代理 + FP-TMM",
                detail=f"比较失败：{type(exc).__name__}。",
            ))

        return tuple(rows)

    # === Method timing comparison ===
    st.divider()
    with st.expander("⏱ 当前已注册方法的耗时与候选色差", expanded=False):
        st.caption(
            "只有当次实际运行成功并返回有限 sRGB 的方法才会报告耗时和 ΔE00；"
            "未运行、缺模型和异常行只显示状态，不进入柱图。"
        )
        run_bench = st.button("▶ 运行当前可用方法基准", use_container_width=True)
        if run_bench or "_bench_cache" in st.session_state:
            if run_bench:
                with st.spinner("正在运行当前可用的本地方法..."):
                    st.session_state["_bench_cache"] = _benchmark_methods()
            cached_benchmark_rows = st.session_state.get("_bench_cache")
            benchmark_rows = validate_benchmark_cache(cached_benchmark_rows)
            if benchmark_rows is None:
                st.session_state.pop("_bench_cache", None)
                st.warning("旧基准缓存已失效，请重新运行")
            elif not benchmark_rows:
                st.info("本次基准没有返回结果，请重新运行。")
            else:
                available_rows = [
                    row for row in benchmark_rows if row.status == "available"]

                if available_rows:
                    labels = [row.label for row in available_rows]
                    times = [row.elapsed_s for row in available_rows]
                    # Keep the chart and the table full-width.  A half-width
                    # column made the method/status cells wrap one character
                    # per line on ordinary laptop viewports.
                    fig_t, ax_t = _get_plt().subplots(figsize=(7, 3.2))
                    colors_t = [
                        "#007e97", "#ef7c45", "#568f4c", "#9b6bb3", "#9a7b32",
                    ]
                    bars = ax_t.barh(
                        labels[::-1], times[::-1],
                        color=colors_t[:len(labels)][::-1], edgecolor="white",
                    )
                    for bar, elapsed in zip(bars, times[::-1]):
                        ax_t.text(
                            bar.get_width(), bar.get_y() + bar.get_height() / 2,
                            f"{elapsed:.3f}s", va="center", fontsize=8,
                        )
                    ax_t.set_xlabel("当次方法调用耗时 (s)")
                    ax_t.set_title("实际成功方法（目标 #7A4F22）")
                    ax_t.grid(True, alpha=0.2, axis="x")
                    fig_t.tight_layout()
                    st.pyplot(fig_t)
                    _get_plt().close(fig_t)
                else:
                    st.info("本次没有方法返回可验收的有限 sRGB，因此不绘制柱图。")

                table = (
                    "| 方法 | 状态 | 耗时 | 候选 ΔE00 | 实际路线 |\n"
                    "|---|---|---:|---:|---|\n"
                )
                status_labels = {
                    "available": "可用", "unavailable": "未运行/不可用", "error": "错误",
                }
                for row in benchmark_rows:
                    elapsed = f"{row.elapsed_s:.3f}s" if row.status == "available" else "—"
                    metric = f"{row.delta_e2000:.3f}" if row.status == "available" else "—"
                    route = row.route.replace("|", "/")
                    table += (
                        f"| {row.label} | {status_labels[row.status]} | "
                        f"{elapsed} | {metric} | {route} |\n"
                    )
                st.markdown(table)
                st.markdown("**每行证据与边界**")
                for row in benchmark_rows:
                    st.markdown(
                        f"- **{row.label} [{row.status}]**：{row.detail}  \n"
                        f"  耗时来源：{row.timing_source}。  \n"
                        f"  指标来源：{row.metric_source}。"
                    )
        else:
            st.info("点击按钮后只运行当前本地证据允许的方法。")

    # === Historical Fano/FDTD comparison (limited comparability) ===
    st.divider()
    st.subheader("Fano 近似与历史 FDTD 图：有限可比性对照")
    st.warning(
        "以下材料不能作为全波验证：历史 FDTD 图记录透射振幅，而本页 Fano 路线描述近似反射响应；"
        "由于 1-T ≠ R，两者只用于小直径范围内的峰位和形状趋势对照。"
    )

    _fdtd_evidence = resolve_fdtd_evidence()
    col_v1, col_v2 = st.columns([3, 2])
    with col_v1:
        if _fdtd_evidence.asset.available:
            st.image(
                _fdtd_evidence.asset.image_bytes,
                caption=(
                    "历史透射振幅图，非当前 Fano 反射全波验证 · "
                    f"SHA256 {_fdtd_evidence.asset.sha256[:12].upper()}…"
                ),
                width="stretch",
            )
        else:
            st.info(
                f"历史 FDTD 本地证据不可用（{_fdtd_evidence.asset.detail}）。"
                "页面未联网，当前正向结果不受影响。"
            )
    with col_v2:
        if _fdtd_evidence.manifest is not None:
            _fdtd_manifest = _fdtd_evidence.manifest
            _fdtd_peak = _fdtd_manifest.metric("resonance_peak_offset_nm")
            _fdtd_corr = _fdtd_manifest.metric("spectral_shape_correlation")
            st.markdown(
                f"""
**对照物理量**：{_fdtd_manifest.quantity}

**几何范围**：{_fdtd_manifest.geometry}

**绑定指标**：
- {_fdtd_peak.label} {_fdtd_peak.minimum:g}-{_fdtd_peak.maximum:g} {_fdtd_peak.unit}
- {_fdtd_corr.label} {_fdtd_corr.minimum:.2f}-{_fdtd_corr.maximum:.2f}

**有限可比性说明**：
- Fano 模型是**半解析近似**，用于快速筛选和趋势预测
- FDTD 数据为透射振幅，大直径前向散射强，1-T ≠ R
- {_fdtd_corr.interpretation}
- 精确设计仍需在同一物理量、边界条件和几何定义下进行独立全波复核
"""
            )
            st.caption(
                f"证据合同 {_fdtd_manifest.schema_version} / {_fdtd_manifest.revision} · "
                f"source={_fdtd_manifest.source} · "
                f"SHA256 {_fdtd_manifest.asset_sha256[:12].upper()}…"
            )
        elif _fdtd_evidence.asset.available:
            st.info(
                "历史图已通过本地完整性校验，但指标合同不一致，已隐藏指标。"
                "页面未联网，当前正向结果不受影响。"
            )

    # === Frozen Fano vs generic-ONNX model difference ===
    st.divider()
    st.subheader("模型差异分析")
    _difference_contract = model_difference_contracts.GENERIC_ONNX_ROUTE
    _difference_context_issue = _difference_contract.context_issue(
        material, substrate, polarization, angle)
    _difference_unavailable_reason = ""
    _difference_structure_not_applicable = bool(is_fp or is_dual)
    if not st.session_state.get("ml_accel", False):
        _difference_unavailable_reason = "ML 加速已关闭；不会加载或调用 generic ONNX"
    elif _difference_structure_not_applicable:
        _difference_unavailable_reason = "该冻结差异合同仅适用于单柱结构"
    elif _difference_context_issue:
        _difference_unavailable_reason = _difference_context_issue
    _difference_required_paths = (
        _difference_contract.model_relative_path,
        _difference_contract.external_data_relative_path,
        _difference_contract.source_pt_relative_path,
        _difference_contract.conversion_protocol_relative_path,
        _difference_contract.conversion_result_relative_path,
    )
    (
        _difference_artifact_version, _difference_code_identity,
        _difference_bundle_identity,
    ) = _difference_analysis_artifact_identity(_difference_contract)
    if not _difference_unavailable_reason and not _difference_code_identity.available:
        _difference_unavailable_reason = "分析源码身份缺失或不可读"
    if not _difference_unavailable_reason and not _difference_bundle_identity.available:
        _difference_unavailable_reason = (
            _difference_bundle_identity.reason or "模型文件或版本证据未通过校验")
    if not _difference_unavailable_reason and not all(
        _local_model_exists(path) if path.startswith("models/")
        else os.path.isfile(os.path.join(os.path.dirname(__file__), path))
        for path in _difference_required_paths
    ):
        _difference_unavailable_reason = "本地 generic ONNX bundle 或版本化证据文件缺失"
    _difference_ready = not _difference_unavailable_reason
    _difference_model_path = os.path.join(
        os.path.dirname(__file__), _difference_contract.model_relative_path)
    _difference_source_pt_path = os.path.join(
        os.path.dirname(__file__), _difference_contract.source_pt_relative_path)
    st.caption("对比快速近似模型与 ML 模型的颜色输出，检查两条路线是否一致。")
    with st.expander("查看技术详情", expanded=False):
        st.caption(
            "左侧路线：torch_model.batch_lorentzian_spectrum（Lorentz/Fano 半解析近似）；"
            f"右侧路线：{_difference_contract.route_id}；"
            f"模型：{_difference_model_path}；版本：{_difference_contract.model_version}；"
            f"graph SHA-256={_difference_contract.model_sha256}；"
            f"external-data SHA-256={_difference_contract.external_data_sha256}。"
        )
        st.caption(
            f"源 PT：{_difference_source_pt_path}；source PT SHA-256={_difference_contract.source_pt_sha256}；"
            f"训练提交 {_difference_contract.training_commit}；"
            f"转换提交 {_difference_contract.conversion_commit}；"
            f"协议 canonical SHA-256={_difference_contract.conversion_protocol_sha256}；"
            f"结果 canonical SHA-256={_difference_contract.conversion_result_sha256}。"
            "这些文件只在点击运行后加载并校验。"
        )
        st.caption(
            f"{_difference_contract.pair_evidence_text(material, substrate)}"
            f"{_difference_contract.boundary_text}"
            "指标仅为两条模型路线输出经同一 color_utils 色度管线得到的模型间 CIEDE2000 差异；"
            "不是 RCWA 精度、实验误差或人眼感知阈值。"
        )
    if _difference_unavailable_reason:
        if not st.session_state.get("ml_accel", False):
            _difference_notice = "ML 加速已关闭；开启后可运行模型差异分析。"
        else:
            _difference_notice = (
                f"当前模型差异分析不可用：{_difference_unavailable_reason}。")
        if _difference_structure_not_applicable:
            st.info(_difference_notice)
        else:
            st.warning(_difference_notice)

    run_model_difference = st.button(
        "▶ 运行模型间差异分析",
        use_container_width=True,
        disabled=not _difference_ready,
        help=(
            "仅在完整训练覆盖的 pair、TE 和冻结几何域内采样 400 组单柱参数；"
            "generic ONNX 使用独立固定 session，"
            "不会调用 predict_rgb、RCWA registry 或 Fano 回退。"
        ),
    )
    # Remove legacy dynamic keys; this analysis now owns exactly one fixed key.
    for _legacy_key in tuple(st.session_state.keys()):
        if str(_legacy_key).startswith("_model_diff_v2_"):
            st.session_state.pop(_legacy_key, None)
    _difference_context = _make_analysis_context(
        "difference",
        {
            "sample_count": 400, "rng": "numpy.RandomState", "seed": 42,
            "geometry_domain_nm": {
                "d": [50.0, 350.0], "h": [80.0, 600.0],
                "p": [200.0, 600.0], "period_rule": "P>=1.2D",
            },
            "routes": ["torch_model.batch_lorentzian_spectrum", _difference_contract.route_id],
        },
        route_id=_difference_contract.route_id,
        model_version=_difference_contract.model_version,
        artifact_version=_difference_artifact_version,
        registry_version=canonical_sha256({
            "contract": _difference_contract.route_id,
            "protocol": _difference_contract.conversion_protocol_sha256,
            "result": _difference_contract.conversion_result_sha256,
        }),
    )
    _difference_snapshot_load = load_analysis_snapshot(
        st.session_state, _difference_context)
    if _difference_snapshot_load.state == "stale":
        st.warning("模型差异快照已陈旧；当前结构、参数或证据身份已变化，旧结果已隐藏。")
    elif _difference_snapshot_load.state == "invalid":
        st.warning("模型差异快照未通过完整性校验，旧结果已隐藏。")

    def _run_model_difference(evaluator, n_samples=400):
        try:
            import torch as _torch_difference
            import torch_model as _torch_model_difference
        except ModuleNotFoundError as exc:
            return model_difference_contracts.ModelDifferenceResult(
                "unavailable", f"Fano 路线依赖缺失：{type(exc).__name__}")
        rng = np.random.RandomState(42)
        d_nm = rng.uniform(50.0, 350.0, n_samples)
        h_nm = rng.uniform(80.0, 600.0, n_samples)
        p_floor = np.maximum(200.0, 1.2 * d_nm)
        p_nm = p_floor + rng.random_sample(n_samples) * (600.0 - p_floor)
        geometries = np.column_stack((d_nm, h_nm, p_nm))
        try:
            fano_spectra = _torch_model_difference.batch_lorentzian_spectrum(
                _torch_difference.tensor(d_nm, dtype=_torch_difference.float32),
                _torch_difference.tensor(h_nm, dtype=_torch_difference.float32),
                _torch_difference.tensor(p_nm, dtype=_torch_difference.float32),
                _torch_difference.full(
                    (n_samples,), float(angle), dtype=_torch_difference.float32),
                polarization.startswith("TE"), material, substrate,
            ).detach().cpu().numpy()
        except Exception as exc:
            return model_difference_contracts.ModelDifferenceResult(
                "unavailable", f"Fano 路线调用失败：{type(exc).__name__}")
        return model_difference_contracts.evaluate_fano_vs_generic(
            evaluator, geometries, fano_spectra,
            material=material, substrate=substrate,
            polarization=polarization, angle_deg=float(angle),
        )

    if run_model_difference and _difference_snapshot_load.state != "fresh":
        with st.spinner("运行冻结的两路线模型间差异分析（400 组域内参数）..."):
            _store_difference_snapshot = True
            try:
                with analysis_engine_transaction(engine, st.session_state):
                    _difference_load = _load_model_difference_evaluator_cached(
                        _difference_context.artifact_version)
                    if not _difference_load.loaded or _difference_load.evaluator is None:
                        _difference_result = model_difference_contracts.ModelDifferenceResult(
                            "unavailable", _difference_load.reason or "generic ONNX 严格加载失败")
                    else:
                        _difference_result = _run_model_difference(_difference_load.evaluator)
            except EngineStateRestoreError as exc:
                logging.error("model difference engine restore failed: %s", exc)
                st.session_state.pop(_difference_context.session_key, None)
                st.error("模型差异分析会话引擎恢复失败；旧结果已清除，本次结果未保存。")
                _store_difference_snapshot = False
                _difference_result = model_difference_contracts.ModelDifferenceResult(
                    "unavailable", "会话引擎恢复失败；结果未保存")
            except EngineStateMutationError as exc:
                logging.error("model difference engine mutation restored: %s", exc)
                _difference_result = model_difference_contracts.ModelDifferenceResult(
                    "unavailable", "会话引擎状态被分析修改；已恢复原状态，本次结果因完整性失败而作废")
            except Exception as exc:
                logging.warning("model difference analysis failed: %s", exc)
                _difference_result = model_difference_contracts.ModelDifferenceResult(
                    "unavailable", f"模型差异分析失败：{type(exc).__name__}")
            _difference_payload = {
                "status": _difference_result.status,
                "reason": _difference_result.reason,
                "analysis_artifact_version": _difference_context.artifact_version,
                "model_artifact_version": _difference_bundle_identity.version,
            }
            if _difference_result.available:
                _difference_payload.update({
                    "delta_e2000": list(_difference_result.delta_e2000),
                    "fano_rgb": [list(row) for row in _difference_result.fano_rgb],
                    "generic_rgb": [list(row) for row in _difference_result.generic_rgb],
                })
                if _difference_load.evidence is not None:
                    _difference_payload["evidence"] = {
                        "protocol_sha256": _difference_load.evidence.protocol_sha256,
                        "result_sha256": _difference_load.evidence.result_sha256,
                        "source_pt_sha256": _difference_contract.source_pt_sha256,
                        "sample_count": _difference_load.evidence.sample_count,
                        "max_abs_diff": _difference_load.evidence.max_abs_diff,
                        "tolerance": _difference_load.evidence.tolerance,
                    }
            if _store_difference_snapshot:
                store_analysis_snapshot(
                    st.session_state,
                    AnalysisSnapshot.create(_difference_context, _difference_payload),
                )
            _difference_snapshot_load = load_analysis_snapshot(
                st.session_state, _difference_context)

    _difference_result = None
    if _difference_snapshot_load.state == "fresh":
        _difference_payload = _difference_snapshot_load.snapshot.payload
        _difference_result = model_difference_contracts.ModelDifferenceResult(
            _difference_payload["status"], _difference_payload.get("reason", ""),
            tuple(_difference_payload.get("delta_e2000", ())),
            tuple(tuple(row) for row in _difference_payload.get("fano_rgb", ())),
            tuple(tuple(row) for row in _difference_payload.get("generic_rgb", ())),
        )

    if _difference_result is None:
        if _difference_ready:
            st.info("点击上方按钮运行冻结路线；普通页面渲染不会加载 ONNX evaluator。")
    elif not _difference_result.available:
        st.warning(f"模型间差异结果不可用：{_difference_result.reason}。")
    else:
        _difference_evidence_payload = _difference_payload.get("evidence")
        if _difference_evidence_payload:
            st.caption(
                f"已验证快照证据：source PT SHA-256={_difference_evidence_payload['source_pt_sha256']}；"
                f"PT-vs-ONNX max|Δ|={_difference_evidence_payload['max_abs_diff']:.17g}；"
                f"容差={_difference_evidence_payload['tolerance']}；"
                f"N={_difference_evidence_payload['sample_count']}。")
        de2k_arr = np.asarray(_difference_result.delta_e2000, dtype=float)
        mean_de = float(np.mean(de2k_arr))
        median_de = float(np.median(de2k_arr))
        p95_de = float(np.percentile(de2k_arr, 95))
        min_de = float(np.min(de2k_arr))
        max_de = float(np.max(de2k_arr))

        col_s1, col_s2 = st.columns([1, 1])
        with col_s1:
            fig_ml, ax_ml = _get_plt().subplots(figsize=(6, 4))
            ax_ml.hist(de2k_arr, bins=40, color="#d86832", edgecolor="white", alpha=0.85)
            ax_ml.axvline(mean_de, color="#a32920", lw=1.5, ls="--", label=f"平均={mean_de:.1f}")
            ax_ml.axvline(median_de, color="#286f8e", lw=1.5, ls=":", label=f"中位数={median_de:.1f}")
            ax_ml.set_xlabel("模型间 CIEDE2000 差异")
            ax_ml.set_ylabel("样本数")
            ax_ml.set_title(f"Fano 近似 vs frozen generic ONNX（N={len(de2k_arr)}）")
            ax_ml.legend(fontsize=7)
            ax_ml.grid(True, alpha=0.2)
            ax_ml.text(
                0.98, 0.95, f"95% 分位：{p95_de:.1f}",
                transform=ax_ml.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
            )
            fig_ml.tight_layout()
            st.pyplot(fig_ml)
            _get_plt().close(fig_ml)
        with col_s2:
            st.markdown(f"""
**模型间差异统计**

| 指标 | 数值 |
|------|------|
| 平均 CIEDE2000 差异 | **{mean_de:.1f}** |
| 中位数 CIEDE2000 差异 | **{median_de:.1f}** |
| 95% 分位 | **{p95_de:.1f}** |
| 最小值 | **{min_de:.1f}** |
| 最大值 | **{max_de:.1f}** |

**证据边界**：这些数值只量化 Lorentz/Fano 半解析路线与指定 generic ONNX 路线
在冻结训练域内的输出差异。它们不是 RCWA 真值误差、实验误差、代理泛化精度或感知阈值。

右侧 ONNX 已独立加载并实际调用；本分析未调用 `ml_module.predict_rgb`、
RCWA surrogate registry，也没有用 Fano 数值填补 generic ONNX 失败样本。
""")

    # Angle scan: color vs incident angle
    st.divider()


    st.subheader("入射角扫描 (0° → 80°)")
    angles_scan = np.arange(0, 85, 5)
    # An angle sweep must use one forward route throughout.  Mixing the
    # near-normal RCWA ensemble with the generic ML model at 5 degrees creates
    # an artificial discontinuity that is not an angular-physics result.
    scan_route_target = None
    if is_fp:
        scan_route_target = "FP cavity TMM"
    elif is_dual:
        scan_route_target = "dual ML surrogate" if use_dual_ml else "dual physical fallback"
    elif use_ml and not _far_field_enabled:
        # The sweep is intentionally a single generic angle-conditioned route;
        # the near-normal RCWA surrogate is not valid for a mixed 0/5/10...
        # degree series and would create an artificial discontinuity.
        scan_route_target = "ML surrogate (generic angle-conditioned)"
    else:
        scan_route_target = "physical/far-field fallback"
    _angle_model_version = (
        model_difference_contracts.GENERIC_ONNX_ROUTE.model_version
        if scan_route_target == "ML surrogate (generic angle-conditioned)"
        else _route_model_version)
    _angle_artifact_version = (
        _analysis_artifact_version(
            "ML surrogate (generic angle-conditioned)", _angle_model_version,
            structure_type="single", material=_provenance_material,
            substrate=_provenance_substrate)
        if scan_route_target == "ML surrogate (generic angle-conditioned)"
        else _analysis_artifact_version(
            scan_route_target, _angle_model_version,
            structure_type=_structure_type, material=_provenance_material,
            substrate=_provenance_substrate))
    _angle_context = _make_analysis_context(
        "angle",
        {
            "angles_deg": angles_scan.tolist(), "start_deg": 0,
            "stop_deg": 80, "step_deg": 5, "sample_count": 17,
            "frozen_route": scan_route_target,
        },
        route_id=scan_route_target, model_version=_angle_model_version,
        artifact_version=_angle_artifact_version,
        registry_version="angle-scan-route-registry-v1",
    )
    _angle_artifact_available = _artifact_identity_available(
        _angle_context.artifact_version)
    _angle_runtime_issue = _analysis_runtime_identity_issue(_angle_context)
    _angle_forward_available = bool(
        _forward.spectrum_available and _route_id != "invalid_geometry")
    _angle_execution_available = bool(
        _angle_forward_available and _angle_artifact_available
        and not _angle_runtime_issue)
    _angle_load = load_analysis_snapshot(st.session_state, _angle_context)
    run_angle = st.button(
        "运行 / 加载当前入射角扫描", key="run_angle_analysis",
        use_container_width=True, disabled=not _angle_execution_available,
    )
    if not _angle_artifact_available:
        st.info("角扫所需文件或版本证据未通过校验，暂不能运行。")
        with st.expander("查看角扫校验详情", expanded=False):
            if scan_route_target == "ML surrogate (generic angle-conditioned)":
                st.caption(_generic_bundle_artifact_identity(
                    model_difference_contracts.GENERIC_ONNX_ROUTE).reason
                    or "当前模型会话或分析源码身份不可用。")
            else:
                st.caption("当前计算路线的源码或模型身份不可用，请检查运行文件完整性。")
    elif not _angle_forward_available:
        st.info("当前正向结果或几何不可用；旧角扫结果和导出已隐藏，当前不会运行 evaluator。")
    elif _angle_runtime_issue:
        st.info(
            f"角扫模型会话身份不可用：{_angle_runtime_issue}；"
            "旧结果和导出已隐藏，当前不会运行 evaluator。")
    if _angle_load.state == "stale":
        st.warning("角扫快照已陈旧；当前几何、路线或远场状态已变化，旧曲线和导出已隐藏。")
    elif _angle_load.state == "invalid":
        st.warning("角扫快照未通过完整性校验，旧曲线和导出已隐藏。")
    elif _angle_load.state == "missing":
        st.info("点击按钮后才计算 17 个角度；普通页面刷新不会运行角扫 evaluator。")

    if (
        run_angle and _angle_load.state != "fresh" and _angle_execution_available
        and not _analysis_runtime_identity_issue(_angle_context)
    ):
        _scan_material = str(material)
        _scan_substrate = str(substrate)
        _scan_polarization = str(polarization)
        _scan_period = float(period)
        _scan_far_field = bool(_angle_context.far_field_enabled)
        _scan_na = float(_angle_context.na)
        _scan_theta = float(_angle_context.theta_obs_deg)
        _scan_use_dual_ml = bool(use_dual_ml)
        _scan_use_generic = bool(use_ml and not _far_field_enabled and not is_dual and not is_fp)
        _scan_is_fp = bool(is_fp)
        _scan_is_dbr = bool(is_dbr_fp)
        _scan_is_dual = bool(is_dual)
        _scan_geometry = dict(_geometry)

        def _scan_at_angle(angle_deg):
            a = float(angle_deg)
            try:
                if _scan_is_fp:
                    if _scan_is_dbr:
                        aw, ar = fp_dielectric_spectrum(
                            _scan_geometry["T_nm"], _scan_geometry["center_wavelength_nm"],
                            3, 5, a, _scan_polarization.startswith("TE"))
                    else:
                        aw, ar = fp_cavity_spectrum(
                            _scan_geometry["T_nm"], a, _scan_polarization.startswith("TE"))
                    return _forward_result(aw, ar, {})
                if _scan_is_dual:
                    if _scan_use_dual_ml:
                        spec = ml_module.predict_dual_spectrum(
                            _scan_geometry["D1_nm"], _scan_geometry["H1_nm"],
                            _scan_geometry["D2_nm"], _scan_geometry["H2_nm"],
                            _scan_geometry["P_nm"], a, _scan_polarization,
                            _scan_material, _scan_substrate)
                        return _forward_result(ml_module.WL, spec, {}) if spec is not None else None
                    aw, ar = _cached_physical_forward(
                        _scan_geometry["D1_nm"], _scan_geometry["H1_nm"],
                        _scan_geometry["P_nm"], _scan_material, _scan_substrate,
                        _scan_polarization, a, _scan_geometry["D2_nm"],
                        _scan_geometry["H2_nm"], True, _scan_far_field,
                        _scan_na, _scan_theta)
                    return _forward_result(aw, ar, {})
                if _scan_use_generic:
                    spec = ml_module.predict_generic_spectrum(
                        _scan_geometry["D_nm"], _scan_geometry["H_nm"],
                        _scan_geometry["P_nm"], a, _scan_polarization,
                        _scan_material, _scan_substrate)
                    return _forward_result(ml_module.WL, spec, {}) if spec is not None else None
                aw, ar = _cached_physical_forward(
                    _scan_geometry["D_nm"], _scan_geometry["H_nm"],
                    _scan_geometry["P_nm"], _scan_material, _scan_substrate,
                    _scan_polarization, a, 0.0, 0.0, False,
                    _scan_far_field, _scan_na, _scan_theta)
                return _forward_result(aw, ar, {})
            except (ModelResourceDriftError, ModelResourceUnavailable):
                raise
            except Exception as exc:
                logging.debug("angle scan at %s failed: %s", a, exc)
                return None

        _store_angle_snapshot = True
        try:
            with (
                analysis_engine_transaction(engine, st.session_state),
                _bound_runtime_context(
                    _angle_context.structure_type, _angle_context.route_id,
                    _angle_context.material, _angle_context.substrate),
            ):
                _angle_route = FrozenSpectrumRoute(
                    scan_route_target, scan_route_target,
                    lambda angle_deg: _scan_at_angle(angle_deg))
                _routed_scan = evaluate_frozen_series(
                    _angle_route, [{"angle_deg": float(a)} for a in angles_scan])
                _route_consistent = route_results_consistent(scan_route_target, _routed_scan)
                raw_scan_rgb = np.full((len(angles_scan), 3), np.nan, dtype=float)
                available_mask = np.zeros(len(angles_scan), dtype=bool)
                route_ids = []
                if not _route_consistent:
                    route_ids = ["route_mismatch"] * len(angles_scan)
                else:
                    for index, (route_id, scan_result) in enumerate(_routed_scan):
                        route_ids.append(str(route_id))
                        if (
                            scan_result is not None and scan_result.spectrum_available
                            and scan_result.rgb is not None
                        ):
                            raw_scan_rgb[index] = np.asarray(scan_result.rgb, dtype=float)
                            available_mask[index] = True
                _angle_payload = build_angle_payload(
                    angles_scan, raw_scan_rgb, available_mask, route_ids)
                _angle_payload["analysis_artifact_version"] = (
                    _angle_context.artifact_version)
                _angle_payload["model_artifact_version"] = (
                    _analysis_model_artifact_version(_angle_context))
        except (ModelResourceDriftError, ModelResourceUnavailable) as exc:
            logging.error("angle scan model resource drift: %s", exc)
            st.session_state.pop(_angle_context.session_key, None)
            st.error("角扫模型会话身份发生漂移；结果与导出已清除，本次未保存。")
            _store_angle_snapshot = False
        except EngineStateRestoreError as exc:
            logging.error("angle scan engine restore failed: %s", exc)
            st.session_state.pop(_angle_context.session_key, None)
            st.error("角扫会话引擎恢复失败；旧结果已清除，本次结果未保存。")
            _store_angle_snapshot = False
        except EngineStateMutationError as exc:
            logging.error("angle scan engine mutation restored: %s", exc)
            _angle_payload = {
                "status": "unavailable",
                "reason": "会话引擎状态被分析修改；已恢复原状态，本次结果因完整性失败而作废",
            }
        except Exception as exc:
            if exception_has_model_resource_drift(exc):
                logging.error("angle business error with model drift: %s", exc)
                st.session_state.pop(_angle_context.session_key, None)
                st.error("角扫模型会话身份发生漂移；结果与导出已清除，本次未保存。")
                _store_angle_snapshot = False
            else:
                logging.warning("angle scan analysis failed: %s", exc)
                _angle_payload = {
                    "status": "unavailable", "reason": f"角扫执行失败：{type(exc).__name__}",
                }
        _post_angle_issue = _analysis_runtime_identity_issue(_angle_context)
        if _post_angle_issue:
            st.session_state.pop(_angle_context.session_key, None)
            st.error("角扫期间模型/源码身份发生变化；本次结果未保存。")
            _store_angle_snapshot = False
        if _store_angle_snapshot:
            store_analysis_snapshot(
                st.session_state, AnalysisSnapshot.create(_angle_context, _angle_payload))
        _angle_load = load_analysis_snapshot(st.session_state, _angle_context)

    if _angle_load.state == "fresh" and _angle_execution_available:
        _angle_payload = _angle_load.snapshot.payload
        if _angle_payload["status"] == "unavailable":
            st.warning(f"入射角扫描不可用：{_angle_payload['reason']}。未跨路线补数。")
        else:
            scan_angles, scan_rgbs, available_mask, scan_routes = angle_payload_arrays(
                _angle_payload)
            unavailable_angles = [
                int(value) for value, available in zip(scan_angles, available_mask)
                if not available]
            if unavailable_angles:
                st.warning(
                    f"部分角度没有可用光谱：{unavailable_angles}。"
                    "原始数值为 null/NaN；灰色斜纹仅用于展示。")
            st.caption(
                f"扫描固定路线：{scan_route_target}；fingerprint={_angle_context.fingerprint[:12]}…；"
                "颜色由同一路线各角度光谱计算，未做逐光谱最大值归一化。")
            fig_ang, (ax1, ax2) = _get_plt().subplots(1, 2, figsize=(10, 3))
            ax1.plot(scan_angles, scan_rgbs[:, 0], "r-", lw=1.5, label="R")
            ax1.plot(scan_angles, scan_rgbs[:, 1], "g-", lw=1.5, label="G")
            ax1.plot(scan_angles, scan_rgbs[:, 2], "b-", lw=1.5, label="B")
            ax1.set_xlabel("Incident Angle (deg)")
            ax1.set_ylabel("sRGB")
            ax1.set_ylim(0, 1.05)
            ax1.legend(fontsize=7)
            ax1.grid(True, alpha=0.3)
            ax1.set_title("RGB vs Incident Angle")
            for index, available in enumerate(available_mask):
                color = rgb_to_hex(scan_rgbs[index]) if available else "#9ca3af"
                patch = _get_plt().Rectangle(
                    (index, 0), 1, 1, facecolor=color, edgecolor="white", lw=0.3,
                    hatch=None if available else "////")
                ax2.add_patch(patch)
            ax2.set_xlim(0, len(scan_angles))
            ax2.set_ylim(0, 1)
            ax2.set_xticks(np.arange(len(scan_angles)) + 0.5)
            ax2.set_xticklabels([f"{int(a)}" for a in scan_angles], fontsize=6)
            ax2.set_yticks([])
            ax2.set_title("Color vs Incident Angle (deg)")
            fig_ang.tight_layout()
            st.pyplot(fig_ang)
            _get_plt().close(fig_ang)
            _angle_export = {
                "schema_version": 1,
                "context_fingerprint": _angle_context.fingerprint,
                "context": _angle_context.to_dict(),
                "angle_scan": _angle_payload,
            }
            st.download_button(
                "下载角扫 JSON",
                json.dumps(_angle_export, ensure_ascii=False, indent=2, allow_nan=False),
                file_name=f"angle_scan_{_angle_context.fingerprint[:12]}.json",
                mime="application/json", use_container_width=True,
            )

st.sidebar.markdown("---")
st.sidebar.subheader("导出")
# Spectrum CSV export: consume exactly the same forward result shown above.
if _forward.spectrum_available:
    if is_fp:
        _export_structure = "fp"
        _export_parameters = {
            "mirror": st.session_state.fp_mirror_type,
            "t": st.session_state.fp_t_val,
            "center_wavelength": (
                st.session_state.fp_target_wl if is_dbr_fp else None),
        }
    elif is_dual:
        _export_structure = "dual"
        _export_parameters = {
            "d1": st.session_state.d1_val, "h1": st.session_state.h1_val,
            "d2": st.session_state.d2_val, "h2": st.session_state.h2_val,
            "p": period,
        }
    else:
        _export_structure = "single"
        _export_parameters = {"d": diameter, "h": height, "p": period}
    _export_basename = forward_export_basename(
        _export_structure, _export_parameters)
    _forward_exports = build_forward_exports(
        _forward.wavelengths_nm, _forward.reflectance, _forward.rgb, _forward.provenance)
    st.sidebar.download_button(
        "下载光谱 CSV", _forward_exports.csv_text,
        file_name=f"spectrum_{_export_basename}.csv",
        mime="text/csv", use_container_width=True
    )
    st.sidebar.download_button(
        "下载当前结果 JSON", _forward_exports.json_text,
        file_name=f"forward_{_export_basename}.json",
        mime="application/json", use_container_width=True
    )
else:
    st.sidebar.warning(f"当前光谱不可导出：{_forward.error}")

# Color swatch PNG export
try:
    if _forward.spectrum_available:
        swatch_size = 100
        swatch = np.ones((swatch_size, swatch_size, 3), dtype=np.uint8)
        r255, g255, b255 = rgb_255(rgb)
        swatch[:,:,0] = r255; swatch[:,:,1] = g255; swatch[:,:,2] = b255
        img = Image.fromarray(swatch)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        st.sidebar.download_button(
            "下载色板 PNG", buf.getvalue(),
            file_name=f"swatch_{hex_color.lstrip('#')}.png",
            mime="image/png", use_container_width=True
        )
    else:
        st.sidebar.warning("当前无可用颜色，已禁用色板 PNG 导出。")
except Exception as e:
    logging.warning(f"swatch export: {e}")
    pass
