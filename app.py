# ===================== Streamlit 版本：超表面结构色设计系统 =====================
from __future__ import annotations

import io, os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from PIL import Image
import streamlit as st
import logging
import ml_module
import rl_design  # RL agent for inverse design
from color_utils import (
    CIE_WAVELENGTHS as _CIE_WAVELENGTHS, CIE_X as _CIE_X, CIE_Y as _CIE_Y, CIE_Z as _CIE_Z,
    WL as _WL, CIE_NORM as _CIE_NORM, D65, SRGB_M as _SRGB_M_NP,
    spectrum_to_xyz, xyz_to_srgb, spectrum_to_srgb, clamp01,
    srgb_to_linear, rgb_to_xyz, xyz_to_xy, rgb_to_xy,
    xyz_to_lab, rgb_to_lab, rgb_to_hex, rgb_255,
    delta_e76, delta_e2000,
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
    except Exception:
        pass
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
from dataclasses import dataclass
from typing import Tuple, List

st.set_page_config(page_title="AI超表面结构色设计", layout="wide")


# FP cavity apply-params callback (runs before rerun)
def _apply_fp_params(wl, t):
    st.session_state.fp_target_wl = wl
    st.session_state.fp_t_val = t

# ===================== Constants & Helpers =====================
# D65 imported from color_utils

# CIE 1931 data imported from color_utils



# Engine module imported from engine.py
from engine import (
    MaterialLibrary, MetaSurfaceParam, DualPillarParam,
    _single_pillar_complex, MetaSurfaceColorEngine,
)

# ===================== Streamlit UI =====================
@st.cache_resource
def get_engine(_cache_key="v17_angular"):
    return MetaSurfaceColorEngine()

try:
    engine = get_engine()
    engine._enable_far_field = st.session_state.get('far_field', False)
    engine._na = st.session_state.get('na_val', 0.1)
    engine._theta_obs_deg = st.session_state.get('theta_obs', 0.0)
except Exception as e:
    st.error(f"Engine init failed: {e}")
    import traceback; st.code(traceback.format_exc())
    st.stop()

@st.cache_resource
def get_ml_ready():
    ok = ml_module.init_ml()
    return (ok, ml_module._ORT_IS_V8 if ok else False)

_ml_ready, _ml_is_v8 = get_ml_ready()

@st.cache_resource
def get_dual_ml_ready():
    return ml_module.init_dual_ml()
_dual_ml_ready = get_dual_ml_ready()



st.title("🎨 AI超表面结构色设计助手")
st.caption("v3.0 | 光谱管线 | CIE 1931 色度学")
st.caption("TiO₂ 纳米柱 Lorentzian 共振 + CIE 1931 光谱色彩管线")

# Sidebar controls
with st.sidebar:
    st.header('⚙️ 参数控制')
    material = st.selectbox('材料 (Pillar)', MaterialLibrary.pillar_materials(), index=1)
    substrate = st.selectbox('衬底 (Substrate)', MaterialLibrary.substrate_materials(), index=0)
    polarization = st.selectbox('偏振', ['TE (s-pol)', 'TM (p-pol)'], index=0)
    if 'a_val' not in st.session_state:
        st.session_state.a_val = 0.0
    col_a1, col_a2 = st.columns([3, 1])
    with col_a1:
        st.session_state.a_val = st.slider('入射角 (°)', 0.0, 80.0, st.session_state.a_val, 0.1)
    with col_a2:
        st.session_state.a_val = st.number_input('精确输入 角度', 0.0, 80.0, st.session_state.a_val, 0.1)
    angle = st.session_state.a_val

    st.divider()
    st.header('🔭 远场传播 (角谱理论)')
    if 'far_field' not in st.session_state:
        st.session_state.far_field = False
    _fp_active = st.session_state.get('structure_type', 'single') == 'fp'
    if _fp_active:
        st.session_state.far_field = False
    st.session_state.far_field = st.checkbox(
        '启用角谱远场传播 (Angular Spectrum)',
        value=st.session_state.far_field,
        disabled=_fp_active,
        help='FP腔模式不需要角谱（平面薄膜无衍射）' if _fp_active else 'N×N超表面阵列FFT角谱 + NA锥积分，计算探测器实际接收光谱'
    )
    if 'theta_obs' not in st.session_state:
        st.session_state.theta_obs = 0.0
    if 'na_val' not in st.session_state:
        st.session_state.na_val = 0.1
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        st.session_state.theta_obs = st.slider(
            '观察角度 θ (°)', 0.0, 80.0, st.session_state.theta_obs, 1.0,
            disabled=not st.session_state.far_field,
            help='观察方向偏离法线的角度, 影响角谱中心位置'
        )
    with col_t2:
        st.session_state.theta_obs = st.number_input(
            '精确 θ', 0.0, 80.0, st.session_state.theta_obs, 1.0,
            disabled=not st.session_state.far_field
        )
    col_n1, col_n2 = st.columns([3, 1])
    with col_n1:
        st.session_state.na_val = st.slider(
            '收集数值孔径 NA', 0.05, 0.95, st.session_state.na_val, 0.01,
            disabled=not st.session_state.far_field,
            help='NA=0.1人眼瞳孔, NA=0.5显微镜20×, NA=0.95油镜100×'
        )
    with col_n2:
        st.session_state.na_val = st.number_input(
            '精确 NA', 0.05, 0.95, st.session_state.na_val, 0.01,
            disabled=not st.session_state.far_field
        )
    st.caption('👁 NA=0.1人眼 | 🔬 NA=0.5显微镜 | 🔍 NA=0.95油镜')

    st.divider()
    st.header("ML 加速")
    if 'ml_accel' not in st.session_state:
        st.session_state.ml_accel = _ml_ready
    st.session_state.ml_accel = st.checkbox(
        '启用 ML 代理模型 (秒级预测)',
        value=st.session_state.ml_accel,
        disabled=not _ml_ready,
        help='使用神经网络代替 Lorentzian 物理模型'
    )
    try:
        if _ml_ready:
            if _ml_is_v8:
                st.caption("模型: v8 Substrate | 7维输入(含衬底) | 256x4残差块 | 4种材料+3种衬底")
            else:
                st.caption("模型: v7 Multi | 6维输入 | 256x4残差块 | 4种材料")
        else:
            st.caption("模型: 未加载 (缺少onnxruntime或ONNX模型文件)")
    except Exception as e:
        st.caption(f"模型: 错误 - {e}")
    if _dual_ml_ready:
        st.caption("双柱 ML: DualResMLP v3 (Multi) 可用")

    if _ml_ready and st.session_state.get("ml_accel", False) and material not in ml_module.MATERIAL_CODES:
        st.warning(f"⚠️ 「{material}」不在 ML 训练数据中，ML 已自动禁用（金属/空气材料物理模型仅供参考）")
    st.divider()
    st.header('📏 纳米柱尺寸')

    if 'structure_type' not in st.session_state:
        st.session_state.structure_type = 'single'
    st.session_state.structure_type = st.radio(
        '📏 结构类型',
        ['单柱 (Single)', '双柱 (Dual)', 'FP腔 (Fabry-Perot)'],
        index=0 if st.session_state.structure_type == 'single' else (1 if st.session_state.structure_type == 'dual' else 2),
        horizontal=True,
        help='单柱/双柱纳米柱或法布里-珀罗腔'
    )
    _struct_map = {'单柱 (Single)': 'single', '双柱 (Dual)': 'dual', 'FP腔 (Fabry-Perot)': 'fp'}
    st.session_state.structure_type = _struct_map[st.session_state.structure_type]
    is_fp = st.session_state.structure_type == 'fp'
    is_dual = st.session_state.structure_type == 'dual'
    st.session_state.dual_pillar = is_dual
    st.caption('单柱/双柱纳米柱或FP腔 | 双柱模式搜索空间大')

    if 'd_val' not in st.session_state:
        st.session_state.d_val = 180.0
    if 'h_val' not in st.session_state:
        st.session_state.h_val = 300.0
    if 'p_val' not in st.session_state:
        st.session_state.p_val = 400.0
    # Dual-pillar state
    if 'd1_val' not in st.session_state:
        st.session_state.d1_val = 120.0
    if 'h1_val' not in st.session_state:
        st.session_state.h1_val = 250.0
    if 'd2_val' not in st.session_state:
        st.session_state.d2_val = 200.0
    if 'h2_val' not in st.session_state:
        st.session_state.h2_val = 350.0
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
                st.session_state.p_val = pre.period_nm
                st.session_state.d1_val = pre.d1_nm
                st.session_state.d2_val = pre.d2_nm
                st.session_state._dual_correction = pre._correction_msg
                st.session_state._prev_dual_params = (pre.d1_nm, pre.d2_nm, pre.period_nm)
        except Exception:
            pass

        # --- Dual-Pillar Controls ---
        # Safety clamp all values before rendering widgets
        st.session_state.d1_val = max(50.0, min(350.0, st.session_state.d1_val))
        st.session_state.h1_val = max(80.0, min(600.0, st.session_state.h1_val))
        st.session_state.d2_val = max(50.0, min(350.0, st.session_state.d2_val))
        st.session_state.h2_val = max(80.0, min(600.0, st.session_state.h2_val))
        st.session_state.p_val = max(200.0, min(600.0, st.session_state.p_val))
        col_d1, col_d2 = st.columns([3, 1])
        with col_d1:
            st.session_state.d1_val = st.slider('柱1直径 D1 (nm)', 50.0, 350.0, st.session_state.d1_val, 0.1)
        with col_d2:
            st.session_state.d1_val = st.number_input('精确输入 D1', 50.0, 350.0, st.session_state.d1_val, 0.1)

        col_h1, col_h2 = st.columns([3, 1])
        with col_h1:
            st.session_state.h1_val = st.slider('柱1高度 H1 (nm)', 80.0, 600.0, st.session_state.h1_val, 0.1)
        with col_h2:
            st.session_state.h1_val = st.number_input('精确输入 H1', 80.0, 600.0, st.session_state.h1_val, 0.1)

        col_d3, col_d4 = st.columns([3, 1])
        with col_d3:
            st.session_state.d2_val = st.slider('柱2直径 D2 (nm)', 50.0, 350.0, st.session_state.d2_val, 0.1)
        with col_d4:
            st.session_state.d2_val = st.number_input('精确输入 D2', 50.0, 350.0, st.session_state.d2_val, 0.1)

        col_h3, col_h4 = st.columns([3, 1])
        with col_h3:
            st.session_state.h2_val = st.slider('柱2高度 H2 (nm)', 80.0, 600.0, st.session_state.h2_val, 0.1)
        with col_h4:
            st.session_state.h2_val = st.number_input('精确输入 H2', 80.0, 600.0, st.session_state.h2_val, 0.1)

        col_p1, col_p2 = st.columns([3, 1])
        with col_p1:
            st.session_state.p_val = st.slider('周期 P (nm)', 200.0, 600.0, st.session_state.p_val, 0.1)
        with col_p2:
            st.session_state.p_val = st.number_input('精确输入 P', 200.0, 600.0, st.session_state.p_val, 0.1)

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
        if 'fp_t_val' not in st.session_state:
            st.session_state.fp_t_val = 200.0
        if 'fp_mirror_type' not in st.session_state:
            st.session_state.fp_mirror_type = '介质 DBR (TiO2/SiO2)'
        if 'fp_target_wl' not in st.session_state:
            st.session_state.fp_target_wl = 450.0

        st.session_state.fp_mirror_type = st.selectbox(
            '反射镜类型',
            ['介质 DBR (TiO2/SiO2)', '金属 Ag (减色)'],
            index=0 if st.session_state.fp_mirror_type.startswith('介质') else 1,
        )

        if st.session_state.fp_mirror_type.startswith('介质'):
            st.session_state.fp_target_wl = st.slider('DBR 中心波长 (nm)', 380.0, 780.0, st.session_state.fp_target_wl, 5.0)

        col_t1, col_t2 = st.columns([3, 1])
        with col_t1:
            st.session_state.fp_t_val = st.slider('腔长 T (nm)', 50.0, 600.0, st.session_state.fp_t_val, 1.0)
        with col_t2:
            st.session_state.fp_t_val = st.number_input('腔长 T', 50.0, 600.0, st.session_state.fp_t_val, 1.0)
        diameter = 0; height = st.session_state.fp_t_val; period = 0
        if st.session_state.fp_mirror_type.startswith('介质'):
            st.caption('FP腔 (DBR): (TiO2/SiO2)3 / TiO2(T) / (SiO2/TiO2)5 | 高饱和度')
        else:
            st.caption('FP腔 (Ag): Ag(30nm) / TiO2(T) / Ag(bulk) | 减色型 | 颜色偏淡')
    else:
        # --- Single-Pillar Controls (original) ---
        col_d1, col_d2 = st.columns([3, 1])
        with col_d1:
            st.session_state.d_val = st.slider('直径 D (nm)', 50.0, 350.0, st.session_state.d_val, 0.1)
        with col_d2:
            st.session_state.d_val = st.number_input('精确输入 D', 50.0, 350.0, st.session_state.d_val, 0.1)
        diameter = st.session_state.d_val

        col_h1, col_h2 = st.columns([3, 1])
        with col_h1:
            st.session_state.h_val = st.slider('高度 H (nm)', 80.0, 600.0, st.session_state.h_val, 0.1)
        with col_h2:
            st.session_state.h_val = st.number_input('精确输入 H', 80.0, 600.0, st.session_state.h_val, 0.1)
        height = st.session_state.h_val

        col_p1, col_p2 = st.columns([3, 1])
        with col_p1:
            st.session_state.p_val = st.slider('周期 P (nm)', 200.0, 600.0, st.session_state.p_val, 0.1)
        with col_p2:
            st.session_state.p_val = st.number_input('精确输入 P', 200.0, 600.0, st.session_state.p_val, 0.1)
        period = st.session_state.p_val

        if diameter > period:
            st.warning('⚠️ D > P：纳米柱会重叠，请调整')

    st.divider()
    presets = {
        '紫罗兰': (150, 100), '蓝色': (80, 250), '青色': (140, 200),
        '翠绿': (180, 250), '黄色': (250, 220), '橙色': (290, 200), '红色': (310, 160),
    }
    # Quick presets removed: dual-Lorentzian model limits prevent accurate preset colors.
    # Use the inverse design tab for precise color matching.
    st.caption('精准颜色请用「逆设计」标签页搜索匹配')
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
elif is_fp:
    param = MetaSurfaceParam(0, st.session_state.fp_t_val, 0, 'TiO2 (anatase)', 'SiO2 (fused silica)', polarization, angle)  # FP dummy
else:
    param = MetaSurfaceParam(diameter, height, period, material, substrate, polarization, angle)
# Cached color lookup: avoid recomputing for same parameters
@st.cache_data
def _cached_physical_color(d_nm, h_nm, p_nm, mat, sub, pol, ang, d2_nm, h2_nm, dual, far_field, na, theta_obs):
    engine._enable_far_field = far_field
    engine._na = na
    engine._theta_obs_deg = theta_obs
    if dual:
        p = DualPillarParam(d1_nm=d_nm, h1_nm=h_nm, d2_nm=d2_nm, h2_nm=h2_nm,
                            period_nm=p_nm, material=mat, substrate=sub,
                            polarization=pol, angle_deg=ang)
    else:
        p = MetaSurfaceParam(d_nm, h_nm, p_nm, mat, sub, pol, ang)
    return engine.physical_color(p)


# FP cavity module imported from fp_cavity.py
from fp_cavity import (
    _AG_NK_TABLE, _ag_nk, _ag_nk_vec, _n_sio2_sellmeier,
    fp_cavity_spectrum, fp_dielectric_spectrum,
)

use_ml = st.session_state.get('ml_accel', False) and _ml_ready and not st.session_state.get('far_field', False) and material in ml_module.MATERIAL_CODES
use_dual_ml = use_ml and st.session_state.get('dual_pillar', False) and _dual_ml_ready

# v7 multi-material ML supports all materials
if use_dual_ml:
    d1v = st.session_state.get('d1_val', diameter)
    h1v = st.session_state.get('h1_val', height)
    d2v = st.session_state.get('d2_val', diameter)
    h2v = st.session_state.get('h2_val', height)
    ml_rgb = ml_module.predict_dual_rgb(d1v, h1v, d2v, h2v, period, angle, polarization, material)
    if ml_rgb is not None:
        rgb = ml_rgb
    else:
        rgb = _cached_physical_color(
            round(st.session_state.d1_val, 1), round(st.session_state.h1_val, 1), round(st.session_state.p_val, 1),
            material, substrate, polarization, round(angle, 1),
            round(st.session_state.d2_val, 1), round(st.session_state.h2_val, 1), True,
            st.session_state.get('far_field', False),
            round(st.session_state.get('na_val', 0.1), 2),
            round(st.session_state.get('theta_obs', 0.0), 1))
elif use_ml and not st.session_state.get('dual_pillar', False):
    ml_rgb = ml_module.predict_rgb(diameter, height, period, angle, polarization, material, substrate)
    if ml_rgb is not None:
        rgb = ml_rgb
    else:
        rgb = _cached_physical_color(
            round(diameter, 1), round(height, 1), round(period, 1),
            material, substrate, polarization, round(angle, 1),
            0.0, 0.0, False,
            st.session_state.get('far_field', False),
            round(st.session_state.get('na_val', 0.1), 2),
            round(st.session_state.get('theta_obs', 0.0), 1))
else:
    if st.session_state.get('dual_pillar', False):
        rgb = _cached_physical_color(
            round(st.session_state.d1_val, 1), round(st.session_state.h1_val, 1), round(st.session_state.p_val, 1),
            material, substrate, polarization, round(angle, 1),
            round(st.session_state.d2_val, 1), round(st.session_state.h2_val, 1), True,
            st.session_state.get('far_field', False),
            round(st.session_state.get('na_val', 0.1), 2),
            round(st.session_state.get('theta_obs', 0.0), 1))
    else:
        rgb = _cached_physical_color(
            round(diameter, 1), round(height, 1), round(period, 1),
            material, substrate, polarization, round(angle, 1),
            0.0, 0.0, False,
            st.session_state.get('far_field', False),
            round(st.session_state.get('na_val', 0.1), 2),
            round(st.session_state.get('theta_obs', 0.0), 1))


# FP mode: compute cavity spectrum color (after ML/physical fallback to prevent overwrite)
is_dbr_fp = st.session_state.get('fp_mirror_type', '').startswith('介质')
_fp_done = (st.session_state.structure_type == 'fp')
if _fp_done:
    if is_dbr_fp:
        _twl = st.session_state.get('fp_target_wl', 450.0)
        _wls, _refl = fp_dielectric_spectrum(st.session_state.fp_t_val, _twl, 3, 5, angle, polarization.startswith('TE'))
    else:
        _wls, _refl = fp_cavity_spectrum(st.session_state.fp_t_val, angle, polarization.startswith('TE'))
    rgb = spectrum_to_srgb(_wls, _refl)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔬 实时预览", "🎯 逆设计", "🖼️ 图案生成",
    "📊 颜色映射", "🌈 光谱"
])

# Tab 1: 实时预览
with tab1:
    hex_color = rgb_to_hex(rgb)
    r255, g255, b255 = rgb_255(rgb)

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
    <div style="display:flex;align-items:center;gap:24px;padding:20px;
                background:linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                border-radius:16px;margin-bottom:20px;">
      <div style="width:130px;height:130px;background:{hex_color};
                  border-radius:16px;box-shadow:0 8px 32px {hex_color}66,
                  inset 0 1px 0 rgba(255,255,255,0.3);flex-shrink:0;"></div>
      <div style="color:#e0e0e0;">
        <div style="font-size:24px;font-weight:700;margin-bottom:6px;">{hex_color}</div>
        <div style="font-size:14px;opacity:0.85;">RGB({r255}, {g255}, {b255})</div>
        <div style="margin-top:10px;font-size:13px;opacity:0.6;line-height:1.6;">
          {st.session_state.get('fp_mirror_type', material) if is_fp else material} on {substrate}<br>
          {param_info}<br>
          {polarization} &nbsp; &theta;={angle:.0f}&deg;
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # --- AI Analysis (DeepSeek LLM) ---
    col_ai1, col_ai2 = st.columns([3, 1])
    with col_ai2:
        ai_clicked = st.button(u"🤖 AI 分析", key="ai_analyze_color", use_container_width=True,
                     help=u"使用 DeepSeek 大模型分析当前颜色结果")
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
        st.info(result)

    # --- Color gamut notice (non-FP only) ---
    if not is_fp and "TiO2" in material:
        st.info(
        "TiO2 纳米柱在当前参数范围（D 60-267nm, H 80-600nm）内无法产生高饱和青蓝色或纯红色。"
        "这是 Lorentzian 模型和 RCWA 严格仿真共同验证的物理限制。"
        "提示：1) 切换到 a-Si/Si3N4 材料获得更宽色域  2) 或使用下方 FP 腔模式。"
    )

    # --- Pillar visualization with pure CSS (non-FP only) ---
    if not is_fp:
        scale = 160.0 / max(height, 100)
        pw = max(diameter * scale * 0.45, 20)
        ph = height * scale * 0.45
        sh = 45
        period_w = period * scale * 0.45

        st.markdown(f"""
        <div style="background:#1a1a2e;border-radius:16px;padding:24px 24px 16px 24px;">
          <div style="text-align:center;color:#888;font-size:12px;margin-bottom:16px;
                      letter-spacing:0.5px;">
            CROSS-SECTION &nbsp;&middot;&nbsp; {param_info}
          </div>
          <div style="display:flex;justify-content:center;align-items:flex-end;
                      height:230px;position:relative;">
            <div style="position:absolute;bottom:0;left:50%;transform:translateX(-50%);
                        width:{period_w*2.2:.0f}px;height:{sh}px;
                        background:linear-gradient(180deg, #3a3a5c, #252540);
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
                        color:#666;font-size:10px;">{height:.0f}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Parameter sensitivity: +/-5nm tolerance (non-FP only)
    if not is_fp:
        st.divider()
        st.subheader("参数灵敏度 (工艺容差 +/-5nm)")
        try:
            import torch_model as _tm_sens
            import torch as _torch_sens
            tol = 5.0
            params = [
                ("D", diameter, height, period, "diameter"),
                ("H", diameter, height, period, "height"),
                ("P", diameter, height, period, "period"),
            ]
            cols = st.columns(4)
            cols[0].markdown("**参数**")
            cols[1].markdown(f"**-{tol:.0f}nm**")
            cols[2].markdown("**当前**")
            cols[3].markdown(f"**+{tol:.0f}nm**")
        
            for label, d_val, h_val, p_val, which in params:
                if which == "diameter":
                    d_lo, d_hi = max(50, d_val-tol), min(350, d_val+tol)
                    sp_lo = _tm_sens.batch_lorentzian_spectrum(_torch_sens.tensor([d_lo]), _torch_sens.tensor([h_val]), _torch_sens.tensor([p_val]), material=material)
                    sp_hi = _tm_sens.batch_lorentzian_spectrum(_torch_sens.tensor([d_hi]), _torch_sens.tensor([h_val]), _torch_sens.tensor([p_val]), material=material)
                elif which == "height":
                    h_lo, h_hi = max(80, h_val-tol), min(600, h_val+tol)
                    sp_lo = _tm_sens.batch_lorentzian_spectrum(_torch_sens.tensor([d_val]), _torch_sens.tensor([h_lo]), _torch_sens.tensor([p_val]), material=material)
                    sp_hi = _tm_sens.batch_lorentzian_spectrum(_torch_sens.tensor([d_val]), _torch_sens.tensor([h_hi]), _torch_sens.tensor([p_val]), material=material)
                else:
                    d_min = max(d_val, 50); p_lo = max(d_min*1.2, p_val-tol); p_hi = min(600, p_val+tol)
                    sp_lo = _tm_sens.batch_lorentzian_spectrum(_torch_sens.tensor([d_val]), _torch_sens.tensor([h_val]), _torch_sens.tensor([p_lo]), material=material)
                    sp_hi = _tm_sens.batch_lorentzian_spectrum(_torch_sens.tensor([d_val]), _torch_sens.tensor([h_val]), _torch_sens.tensor([p_hi]), material=material)
            
                rgb_lo = _tm_sens.batch_spectrum_to_rgb(sp_lo).squeeze().numpy()
                rgb_hi = _tm_sens.batch_spectrum_to_rgb(sp_hi).squeeze().numpy()
                hex_lo = rgb_to_hex(rgb_lo); hex_hi = rgb_to_hex(rgb_hi)
                de_lo = np.sqrt(np.sum((rgb - rgb_lo)**2))
                de_hi = np.sqrt(np.sum((rgb - rgb_hi)**2))
            
                c1, c2, c3, c4 = st.columns(4)
                c1.markdown(f"**{label}**")
                c2.markdown(f'<div style="width:40px;height:24px;background:{hex_lo};border-radius:4px;"></div><small>ΔE={de_lo:.3f}</small>', unsafe_allow_html=True)
                c3.markdown(f'<div style="width:40px;height:24px;background:{hex_color};border-radius:4px;border:2px solid white;"></div>', unsafe_allow_html=True)
                c4.markdown(f'<div style="width:40px;height:24px;background:{hex_hi};border-radius:4px;"></div><small>ΔE={de_hi:.3f}</small>', unsafe_allow_html=True)
            st.caption("工艺容差 ±5nm 下的颜色偏差 (ΔE < 0.02 肉眼不可分辨)")
        except Exception as e:
            st.caption(f"灵敏度分析不可用: {e}")

# Tab 2: Inverse Design
with tab2:
    st.subheader("选择目标颜色，自动匹配最优纳米柱参数")
    st.caption("侧边栏的 D/H/P 不影响逆设计，仅材料、衬底、偏振、入射角有效 | 网格搜索仅优化单柱 (D,H,P)，双柱请手动微调")

    dual_gd_btn = False
    col_pick, col_btn = st.columns([3, 1])
    with col_pick:
        picker_hex = st.color_picker("目标颜色", "#80c8ff")
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button('网格搜索', use_container_width=True, help='网格搜索: 精度高')
        rl_btn = st.button('🎮 RL智能搜索', use_container_width=True, help='强化学习 Q-learning 逆设计, 约3秒')
        gd_btn = st.button('🎯 梯度优化', use_container_width=True, help='PyTorch批量梯度下降, ~3-5秒, 需torch')
        dual_gd_btn = st.button('🎯 双柱梯度', use_container_width=True, help='PyTorch批量双柱梯度下降, ~3-5秒, 需torch')
    target_r = int(picker_hex[1:3], 16)
    target_g = int(picker_hex[3:5], 16)
    target_b = int(picker_hex[5:7], 16)
    st.caption(f"RGB({target_r}, {target_g}, {target_b})  |  {picker_hex}")
    if "TiO2" in material and target_b > 150 and target_b > target_r + 20 and target_b > target_g + 20:
        st.caption("💡 TiO₂ 做不出高饱和蓝/青色，建议切换到 **a-Si** 材料或使用 **FP 腔模式**")
    elif "TiO2" in material and target_r > 180 and target_r > target_g + 30 and target_r > target_b + 30:
        st.caption("💡 TiO₂ 做不出纯红色，建议切换到 **a-Si** + Si₃N₄ 衬底")

    target_rgb_norm = np.array([target_r, target_g, target_b]) / 255.0

    if rl_btn:
        with st.spinner("🎮 RL智能体搜索中 (Q-learning, 约3秒)..."):
            try:
                rl = rl_design.get_trained_rl()
                d_rl, h_rl, p_rl, hex_rl, de_rl = rl.search(picker_hex)
                r255_rl = int(hex_rl[1:3], 16)
                g255_rl = int(hex_rl[3:5], 16)
                b255_rl = int(hex_rl[5:7], 16)
                
                # Store RL results in session_state so "apply" button survives rerun
                st.session_state._rl_d = float(d_rl)
                st.session_state._rl_h = float(h_rl)
                st.session_state._rl_p = float(p_rl)
                st.success(f"🎮 RL搜索完成! {hex_rl} | ΔE2000={de_rl:.1f}")
                c1rl, c2rl, c3rl = st.columns([1, 3, 1])
                with c1rl:
                    st.markdown(f'<div style="width:64px;height:64px;background:{hex_rl};border-radius:12px;"></div>', unsafe_allow_html=True)
                with c2rl:
                    st.markdown(f"**{hex_rl}**  RGB({r255_rl}, {g255_rl}, {b255_rl})  \nD={d_rl:.1f}nm  H={h_rl:.1f}nm  P={p_rl:.1f}nm  \nΔE2000 = {de_rl:.1f} (RL智能体)")
                
                # Apply button using on_click callback (reliable)
                def _apply_rl_cb():
                    st.session_state.d_val = float(st.session_state._rl_d)
                    st.session_state.h_val = float(st.session_state._rl_h)
                    st.session_state.p_val = float(st.session_state._rl_p)
                st.button("应用RL参数", on_click=_apply_rl_cb, key="apply_rl_result", use_container_width=True)
                st.caption("强化学习通过试错学习参数调整方向，速度快但精度低于网格搜索。建议先用RL快速定位，再网格精搜。")
            except Exception as e:
                st.warning(f"RL搜索不可用: {e}")

    if gd_btn:
        with st.spinner("🎯 梯度优化中 (PyTorch批量Adam, ~3-5秒)..."):
            try:
                import torch_model as _tm_batch
                result = _tm_batch.inverse_design_ml_batch(
                    target_rgb_norm, n_steps=300, n_restarts=20,
                    material=material, substrate=substrate
                )
                if result is None:
                    st.warning("梯度优化不可用: 需要安装PyTorch")
                else:
                    d_gd, h_gd, p_gd, pred_rgb, loss = result
                    rc = [max(0, min(255, int(c * 255))) for c in pred_rgb]
                    hex_gd = f"#{rc[0]:02x}{rc[1]:02x}{rc[2]:02x}"
                    from color_utils import rgb_to_lab_scalar, delta_e2000_scalar
                    de_gd = delta_e2000_scalar(rgb_to_lab_scalar(pred_rgb), rgb_to_lab_scalar(target_rgb_norm))
                    st.session_state._gd_d = float(d_gd)
                    st.session_state._gd_h = float(h_gd)
                    st.session_state._gd_p = float(p_gd)
                    st.success(f"🎯 梯度优化完成! {hex_gd} | ΔE2000={de_gd:.1f}")
                    c1gd, c2gd = st.columns([1, 3])
                    with c1gd:
                        st.markdown(f'<div style="width:64px;height:64px;background:{hex_gd};border-radius:12px;"></div>', unsafe_allow_html=True)
                    with c2gd:
                        st.markdown(f"**{hex_gd}**  RGB({rc[0]}, {rc[1]}, {rc[2]})  \nD={d_gd:.1f}nm  H={h_gd:.1f}nm  P={p_gd:.1f}nm  \nΔE2000 = {de_gd:.1f} (梯度优化)")
                    def _apply_gd_cb():
                        st.session_state.d_val = float(st.session_state._gd_d)
                        st.session_state.h_val = float(st.session_state._gd_h)
                        st.session_state.p_val = float(st.session_state._gd_p)
                    st.button("应用梯度参数", on_click=_apply_gd_cb, key="apply_gd_result", use_container_width=True)
                    st.caption("梯度下降直接优化物理模型，精度高于RL，速度低于网格搜索。推荐在RL定位后用梯度精调。")
            except Exception as e:
                logging.warning(f"app fallback: {e}")
                st.warning(f"梯度优化不可用: {e}")


    if st.session_state.get('dual_pillar', False) and dual_gd_btn:
        with st.spinner("🎯 双柱梯度优化中 (PyTorch批量Adam, ~3-5秒)..."):
            try:
                import torch_model as _tm_gd
                result = _tm_gd.inverse_design_dual(
                    target_rgb_norm, n_steps=300, n_restarts=20,
                    material=material, substrate=substrate
                )
                d1_gd, h1_gd, d2_gd, h2_gd, p_gd, pred_rgb, loss = result
                rc = [max(0, min(255, int(c * 255))) for c in pred_rgb]
                hex_gd = f"#{rc[0]:02x}{rc[1]:02x}{rc[2]:02x}"
                from color_utils import rgb_to_lab_scalar, delta_e2000_scalar
                de_gd = delta_e2000_scalar(rgb_to_lab_scalar(pred_rgb), rgb_to_lab_scalar(target_rgb_norm))
                st.session_state._dual_gd_d1 = float(d1_gd)
                st.session_state._dual_gd_h1 = float(h1_gd)
                st.session_state._dual_gd_d2 = float(d2_gd)
                st.session_state._dual_gd_h2 = float(h2_gd)
                st.session_state._dual_gd_p = float(p_gd)
                st.success(f"🎯 双柱梯度优化完成! {hex_gd} | ΔE2000={de_gd:.1f}")
                c1gd, c2gd = st.columns([1, 3])
                with c1gd:
                    st.markdown(f'<div style="width:64px;height:64px;background:{hex_gd};border-radius:12px;"></div>', unsafe_allow_html=True)
                with c2gd:
                    st.markdown(f"**{hex_gd}**  RGB({rc[0]}, {rc[1]}, {rc[2]})  \nD1={d1_gd:.1f}nm H1={h1_gd:.1f}nm D2={d2_gd:.1f}nm H2={h2_gd:.1f}nm P={p_gd:.1f}nm  \nΔE2000 = {de_gd:.1f} (双柱梯度优化)")
                def _apply_dual_gd_cb():
                    st.session_state.d_val = float(st.session_state._dual_gd_d1)
                    st.session_state.h_val = float(st.session_state._dual_gd_h1)
                    st.session_state.d1_val = float(st.session_state._dual_gd_d1)
                    st.session_state.h1_val = float(st.session_state._dual_gd_h1)
                    st.session_state.d2_val = float(st.session_state._dual_gd_d2)
                    st.session_state.h2_val = float(st.session_state._dual_gd_h2)
                    st.session_state.p_val = float(st.session_state._dual_gd_p)
                st.button("应用双柱梯度参数", on_click=_apply_dual_gd_cb, key="apply_dual_gd_result", use_container_width=True)
                st.caption("双柱梯度下降同时优化5个参数(D1,H1,D2,H2,P)，搜索空间更大，耗时更长但能找到更优解。")
            except Exception as e:
                logging.warning(f"app fallback: {e}")
                st.warning(f"双柱梯度优化不可用: {e}")
    if run_btn:
        # Result cache: skip search for previously-searched colors
        cache_key = (target_r, target_g, target_b, material, substrate, polarization, angle)
        if "search_cache" not in st.session_state:
            st.session_state.search_cache = {}

        if cache_key in st.session_state.search_cache:
            st.session_state.top3_results = st.session_state.search_cache[cache_key]
            st.success("从缓存加载，瞬间完成!")
        else:
            engine.rebuild_library(material, substrate, polarization, angle)
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(current, total, label):
                pct = min(current / max(total, 1), 1.0)
                progress_bar.progress(pct)
                status_text.caption(f"{label}: {current}/{total}")

            st.session_state.top3_results = engine.inverse_design(target_rgb_norm, update_progress)
            st.session_state.search_cache[cache_key] = st.session_state.top3_results
            # Save to history
            if "search_history" not in st.session_state:
                st.session_state.search_history = []
            best = st.session_state.top3_results[0]
            entry = {
                "target_hex": picker_hex,
                "target_rgb": (target_r, target_g, target_b),
                "matched_hex": rgb_to_hex(best[2]),
                "matched_rgb": rgb_255(best[2]),
                "D": best[1].diameter_nm,
                "H": best[1].height_nm,
                "P": best[1].period_nm,
                "dE": best[4],
            }
            st.session_state.search_history.insert(0, entry)
            st.session_state.search_history = st.session_state.search_history[:10]
            progress_bar.progress(1.0)
            status_text.caption("搜索完成!")
    if 'top3_results' in st.session_state:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**🎯 目标颜色**")
            hex_t = rgb_to_hex(target_rgb_norm)
            st.markdown(f"""
            <div style="width:100px;height:100px;background:{hex_t};
                        border-radius:12px;box-shadow:0 4px 16px {hex_t}44;
                        border:2px solid rgba(255,255,255,0.1);margin:0 auto;"></div>
            <p style="text-align:center;margin-top:6px;font-size:13px;">{hex_t}<br>RGB({target_r}, {target_g}, {target_b})</p>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown("**✅ Top3 匹配结果**")
            # Build radio options
            options = []
            for i, (_, bp, brgb, bde, bde2k) in enumerate(st.session_state.top3_results):
                hx = rgb_to_hex(brgb)
                r, g, b = rgb_255(brgb)
                lbl = f"#{i+1} {hx} | D={bp.diameter_nm:.0f} H={bp.height_nm:.0f} P={bp.period_nm:.0f} | ΔE={bde2k:.1f}"
                options.append(lbl)
            choice = st.radio("选择结果", options, horizontal=False, index=0,
                              key=f'result_choice_{picker_hex}')
            idx = options.index(choice)
            best_param, matched_rgb, de_val, de2k_val = st.session_state.top3_results[idx][1], st.session_state.top3_results[idx][2], st.session_state.top3_results[idx][3], st.session_state.top3_results[idx][4]
            hex_m = rgb_to_hex(matched_rgb)
            mr, mg, mb = rgb_255(matched_rgb)

        st.markdown(f"""
        <div style="display:flex;gap:16px;align-items:center;margin-bottom:12px;">
          <div style="width:60px;height:60px;background:{hex_m};border-radius:8px;
                      box-shadow:0 3px 12px {hex_m}44;border:1px solid rgba(255,255,255,0.1);"></div>
          <div style="font-size:14px;">
            <b>{hex_m}</b> &nbsp; RGB({mr}, {mg}, {mb})<br>
            <span style="font-size:12px;opacity:0.7;">D={best_param.diameter_nm:.1f}nm H={best_param.height_nm:.1f}nm P={best_param.period_nm:.1f}nm</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"ΔE2000 = {de2k_val:.1f} (主要指标，<2 人眼不可分辨)  |  dE76 = {de_val:.1f}")
        # --- 智能标注：大色差原因分析 ---
        if de2k_val > 10:
            target_b_hint = target_b / 255.0
            target_is_blue = target_b > 180 and target_b > target_r + 30 and target_b > target_g + 30
            if "TiO2" in material and target_is_blue:
                st.warning(
                    f"⚠️ ΔE={de2k_val:.0f} 色差很大，因为 TiO₂ 纳米柱在当前参数范围"
                    "内做不出高饱和蓝/青色。建议：1) 切换到 **a-Si (amorphous)** 材料 "
                    "2) 或使用下方 **FP 腔模式** 来实现蓝色。"
                )
            elif "TiO2" in material and target_r > 180 and target_r > target_g + 30 and target_r > target_b + 30:
                st.warning(
                    f"⚠️ ΔE={de2k_val:.0f} 色差很大，因为 TiO₂ 纳米柱做不出纯红色。"
                    "建议：切换到 **a-Si (amorphous)** + Si₃N₄ 衬底获得更宽色域。"
                )
            elif de2k_val > 30:
                st.warning(
                    f"⚠️ ΔE={de2k_val:.0f} 色差很大，该目标颜色可能超出当前材料色域。"
                    "尝试：1) 换材料 (a-Si 色域更宽)  2) 换 FP 腔模式  3) 选色域内的目标色。"
                )

        # Copyable parameters
        param_text = f"D={best_param.diameter_nm:.1f}nm  H={best_param.height_nm:.1f}nm  P={best_param.period_nm:.1f}nm"
        st.code(param_text, language=None)

        # --- Spectral comparison chart ---
        st.markdown("📊 光谱对比")
        wls_m, refl_m = engine.compute_spectrum(best_param)
        target_xy = rgb_to_xy(target_rgb_norm)
        locus_xy = np.array([xyz_to_xy(np.array([_CIE_X[i], _CIE_Y[i], _CIE_Z[i]])) for i in range(81)])
        dists_locus = np.sum((locus_xy - target_xy)**2, axis=1)
        dominant_idx = int(np.argmin(dists_locus))
        target_peak_wl = float(_CIE_WAVELENGTHS[dominant_idx])
        sigma_ideal = 15.0
        wls_ideal = np.linspace(380, 780, 200)
        refl_ideal = 1.0 / (1.0 + ((wls_ideal - target_peak_wl) / sigma_ideal)**2)
        refl_m_norm = refl_m / (refl_m.max() if refl_m.max() > 1e-12 else 1.0)

        fig_spec, ax_spec = _get_plt().subplots(figsize=(6, 3))
        ax_spec.plot(wls_ideal, refl_ideal, "#80c8ff", linewidth=2, label="理想目标光谱")
        ax_spec.plot(wls_m, refl_m_norm, "#007e97", linewidth=2, label="匹配计算光谱")
        ax_spec.axvline(target_peak_wl, color="#80c8ff", linestyle="--", alpha=0.5)
        ax_spec.axvline(wls_m[np.argmax(refl_m_norm)], color="#007e97", linestyle="--", alpha=0.5)
        ax_spec.annotate(f"目标峰值 {target_peak_wl:.0f}nm", xy=(target_peak_wl, 0.95),
                         fontsize=8, color="#80c8ff", ha="center")
        ax_spec.annotate(f"匹配峰值 {wls_m[np.argmax(refl_m_norm)]:.0f}nm",
                         xy=(wls_m[np.argmax(refl_m_norm)], 0.85),
                         fontsize=8, color="#007e97", ha="center")
        ax_spec.set_xlabel("波长 (nm)")
        ax_spec.set_ylabel("归一化反射率")
        ax_spec.set_xlim(380, 780)
        ax_spec.set_ylim(0, 1.1)
        ax_spec.legend(fontsize=8, loc="upper right")
        ax_spec.grid(True, alpha=0.3)
        fig_spec.tight_layout()
        st.pyplot(fig_spec)
        _get_plt().close(fig_spec)

    # Search history
    if "search_history" in st.session_state and st.session_state.search_history:
        st.divider()
        st.caption("📋 搜索历史 (最近10次)")
        cols_h = st.columns([1, 1, 2, 3, 1, 1])
        cols_h[0].caption("目标")
        cols_h[1].caption("匹配")
        cols_h[2].caption("参数")
        cols_h[3].caption("")
        cols_h[4].caption("ΔE")
        cols_h[5].caption("")
        for hi, h in enumerate(st.session_state.search_history):
            c0, c1, c2, c3, c4, c5 = st.columns([1, 1, 2, 3, 1, 1])
            with c0:
                st.markdown(f'<div style="width:24px;height:24px;background:{h["target_hex"]};border-radius:4px;border:1px solid #fff3;"></div>', unsafe_allow_html=True)
            with c1:
                st.markdown(f'<div style="width:24px;height:24px;background:{h["matched_hex"]};border-radius:4px;border:1px solid #fff3;"></div>', unsafe_allow_html=True)
            with c2:
                st.caption(f"D={h['D']:.0f} H={h['H']:.0f} P={h['P']:.0f}")
            with c3:
                st.caption(f"{h['target_hex']}  {h['matched_hex']}")
            with c4:
                st.caption(f"{h['dE']:.1f}")
            with c5:
                if st.button("查看", key=f"hist_{hi}"):
                    st.session_state.history_view = h
                    st.rerun()

    # Show history detail if selected
    if "history_view" in st.session_state and st.session_state.history_view:
        hv = st.session_state.history_view
        st.divider()
        st.markdown(f"**历史回看: {hv['target_hex']} → {hv['matched_hex']}**")
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            st.markdown(f'<div style="width:80px;height:80px;background:{hv["target_hex"]};border-radius:12px;"></div>', unsafe_allow_html=True)
            st.caption(f"目标 {hv['target_hex']}")
        with col_h2:
            st.markdown(f'<div style="width:80px;height:80px;background:{hv["matched_hex"]};border-radius:12px;"></div>', unsafe_allow_html=True)
            st.caption(f"匹配 {hv['matched_hex']}")
        st.code(f"D={hv['D']:.1f}nm  H={hv['H']:.1f}nm  P={hv['P']:.1f}nm")
        st.caption(f"ΔE2000 = {hv['dE']:.1f}")
        if st.button("关闭回看"):
            st.session_state.pop("history_view", None)
            st.rerun()

    # --- FP Cavity Inverse Search ---
    if is_fp:
        st.divider()
        st.subheader("FP腔 自动寻色")
        st.caption("扫描 DBR 中心波长 + 腔长 T，匹配目标颜色")
        col_fp1, col_fp2 = st.columns([3, 1])
        with col_fp1:
            fp_target_hex = st.color_picker("FP目标颜色", "#80c8ff", key="fp_target_picker")
        with col_fp2:
            st.markdown("<br>", unsafe_allow_html=True)
            fp_search_btn = st.button("FP腔搜索", use_container_width=True)
            st.caption("预计搜索时间约 30 秒（粗扫 550 组 + 精细 300 组）")

        if fp_search_btn:
            fp_tr = int(fp_target_hex[1:3], 16)
            fp_tg = int(fp_target_hex[3:5], 16)
            fp_tb = int(fp_target_hex[5:7], 16)

            # --- Cache check ---
            cache_key = (fp_tr, fp_tg, fp_tb, int(angle), polarization)
            if "fp_search_cache" not in st.session_state:
                st.session_state.fp_search_cache = {}
            if cache_key in st.session_state.fp_search_cache:
                top3 = st.session_state.fp_search_cache[cache_key]
                st.success(f"从缓存加载，瞬间完成! 共 {len(st.session_state.fp_search_cache)} 组缓存")
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

            for rank, (de, wl, t, rgb) in enumerate(top3):
                hex_c = rgb_to_hex(rgb)
                r255, g255, b255 = rgb_255(rgb)
                with st.container():
                    c1, c2 = st.columns([1, 4])
                    with c1:
                        st.markdown(f'<div style="width:50px;height:50px;background:{hex_c};border-radius:8px;"></div>', unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"**#{rank+1} {hex_c}** | ΔE2000={de:.1f} | λ₀={wl:.0f}nm T={t:.0f}nm | RGB({r255},{g255},{b255})")
                        st.button(f"应用此参数 #{rank+1}", key=f"fp_apply_{rank}", on_click=_apply_fp_params, args=(wl, t))


# Tab 3: Pattern Generation
with tab3:
    st.subheader("上传图片，生成超表面纳米柱图案")
    uploaded = st.file_uploader("选择图片", type=["png", "jpg", "jpeg", "bmp"])

    if uploaded:
        engine.rebuild_library(material, substrate, polarization, angle)
        image = Image.open(uploaded)
        max_s = st.slider("最大分辨率", 20, 120, 60)

        if st.button("🎨 生成图案", use_container_width=True):
            with st.spinner("逐像素匹配最优纳米柱参数..."):
                orig, mapped, params_arr = engine.image_to_metasurface_map(image, max_s)

            fig3, (ax_o, ax_m, ax_d) = _get_plt().subplots(1, 3, figsize=(14, 4))
            ax_o.imshow(orig); ax_o.set_title("原图"); ax_o.axis("off")
            ax_m.imshow(mapped); ax_m.set_title("颜色映射图"); ax_m.axis("off")
            im = ax_d.imshow(params_arr[:,:,0], cmap="viridis")
            ax_d.set_title("直径 (nm)"); ax_d.axis("off")
            _get_plt().colorbar(im, ax=ax_d, fraction=0.046)
            fig3.tight_layout()
            st.pyplot(fig3); _get_plt().close(fig3)

            mean_err = float(np.mean(np.linalg.norm(orig - mapped, axis=2)))
            st.info(f"图案: {params_arr.shape[1]}×{params_arr.shape[0]} 像素 | 平均 RGB 误差: {mean_err:.4f}")

# Tab 4: Color Palette
with tab4:
    st.subheader("D-H 颜色映射")

    # Sample a grid of D/H values for the current material
    d_sample = np.linspace(80, 300, 8)
    h_sample = np.linspace(200, 600, 6)

    # Build HTML color grid
    rows_html = '<table style="border-collapse:collapse;width:100%;">'
    rows_html += '<tr><th style="padding:4px 8px;color:#888;font-size:11px;">D/H</th>'
    for h in h_sample:
        rows_html += f'<th style="padding:4px 8px;color:#888;font-size:11px;">{h:.0f}</th>'
    rows_html += '</tr>'

    # Try PyTorch batch acceleration for color map
    try:
        import torch_model as _tm
        _D_grid = torch_model.torch.tensor(d_sample, dtype=torch_model.torch.float32)
        _H_grid = torch_model.torch.tensor(h_sample, dtype=torch_model.torch.float32)
        _rgb_grid = _tm.batch_single_pillar_rgb_norm(_D_grid, _H_grid, period)
        _use_torch = True
    except Exception:
        _use_torch = False

    for di, d in enumerate(d_sample):
        rows_html += f'<tr><td style="padding:4px 8px;color:#888;font-size:11px;font-weight:600;">{d:.0f}</td>'
        for hi, h in enumerate(h_sample):
            if _use_torch:
                test_rgb = _rgb_grid[di, hi].numpy()
            else:
                test_param = MetaSurfaceParam(d, h, period, material, substrate, polarization, angle)
                test_rgb = engine.physical_color(test_param)
            hex_t = rgb_to_hex(test_rgb)
            tr, tg, tb = rgb_255(test_rgb)
            rows_html += f'<td style="padding:2px;"><div title="D={d:.0f}nm H={h:.0f}nm RGB({tr},{tg},{tb})" style="width:40px;height:32px;background:{hex_t};border-radius:4px;border:1px solid rgba(255,255,255,0.08);"></div></td>'
        rows_html += '</tr>'
    rows_html += '</table>'

    st.markdown(rows_html, unsafe_allow_html=True)
    st.caption(f"🔴: {material} | ⚪: {substrate} | 周期 P={period:.0f}nm")
    st.caption("⬅️: 高度 H (nm) | ⬇️: 直径 D (nm)")
    # CIE 1931 chromaticity diagram
    st.divider()
    st.subheader("CIE 1931 色度图")
    try:
        plt = _get_plt()
        fig, ax = plt.subplots(figsize=(5, 4.5))
        # CIE 1931 spectrum locus (standard data)
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
        ax.fill(cx, cy, color="#e8e8e8", alpha=0.5)
        ax.plot(cx, cy, "k-", linewidth=0.8)
        # sRGB gamut triangle
        sR, sG, sB = (0.640,0.330), (0.300,0.600), (0.150,0.060)
        ax.plot([sR[0],sG[0]],[sR[1],sG[1]],"k--",linewidth=0.5)
        ax.plot([sG[0],sB[0]],[sG[1],sB[1]],"k--",linewidth=0.5)
        ax.plot([sB[0],sR[0]],[sB[1],sR[1]],"k--",linewidth=0.5)
        ax.text(sR[0],sR[1]+0.02,"R",fontsize=8,ha="center")
        ax.text(sG[0]-0.02,sG[1]+0.02,"G",fontsize=8,ha="center")
        ax.text(sB[0],sB[1]-0.03,"B",fontsize=8,ha="center")
        # Compute xy from spectrum
        try:
            import torch_model as _tm2
            import torch as _torch
            # Local CIE data for numpy
            from color_utils import CIE_X as _cie_x_full, CIE_Y as _cie_y_full, CIE_Z as _cie_z_full
            _cie_x = _cie_x_full
            _cie_y = _cie_y_full
            _cie_z = _cie_z_full
            _wl = np.linspace(380, 780, 81)
            if st.session_state.get("dual_pillar", False):
                sp = _tm2.batch_dual_pillar_spectrum(
                    _torch.tensor([st.session_state.d1_val]), _torch.tensor([st.session_state.h1_val]),
                    _torch.tensor([st.session_state.d2_val]), _torch.tensor([st.session_state.h2_val]),
                    _torch.tensor([st.session_state.p_val]),
                    material=material
                )
            else:
                sp = _tm2.batch_lorentzian_spectrum(
                    _torch.tensor([diameter]), _torch.tensor([height]), _torch.tensor([period]),
                    material=material
                )
            sp_np = sp.squeeze().detach().numpy()
            Xv = np.trapezoid(sp_np * _cie_x, _wl)
            Yv = np.trapezoid(sp_np * _cie_y, _wl)
            Zv = np.trapezoid(sp_np * _cie_z, _wl)
            total = Xv + Yv + Zv
            if total > 0:
                px, py = Xv/total, Yv/total
                ax.plot(px, py, "ro", markersize=8, markeredgecolor="white", markeredgewidth=1.5)
                ax.annotate(f"({px:.3f},{py:.3f})", (px, py), textcoords="offset points",
                           xytext=(10,10), fontsize=9, color="#333")
        except Exception:
            pass
        ax.set_xlim(0, 0.8); ax.set_ylim(0, 0.9)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_title(f"{material}  |  D={diameter:.0f}nm H={height:.0f}nm P={period:.0f}nm", fontsize=9)
        ax.set_aspect("equal")
        st.pyplot(fig)
        plt.close(fig)
        st.caption("灰色区域: CIE 1931 色度图马蹄轨迹 | 虚线三角: sRGB 色域 | 红点: 当前预测色坐标")
    except Exception as e:
        st.caption(f"色度图渲染失败: {e}")

# Tab 5: Spectrum & CIE Chromaticity
with tab5:
    col_spec, col_cie = st.columns([3, 2])

    with col_spec:
        st.subheader("反射光谱 (380-780 nm)")
        if is_fp:
            if is_dbr_fp:
                wls, refl = fp_dielectric_spectrum(st.session_state.fp_t_val, st.session_state.get("fp_target_wl", 450.0), 3, 5, angle, polarization.startswith("TE"))
            else:
                wls, refl = fp_cavity_spectrum(st.session_state.fp_t_val, angle, polarization.startswith("TE"))
        else:
            wls, refl = engine.compute_spectrum(param, 380, 780, 81)

        fig5, ax5 = _get_plt().subplots(figsize=(10, 4))
        # Color the spectrum curve with the actual computed color
        hex_c = rgb_to_hex(rgb)
        ax5.plot(wls, refl, color="#333", lw=2.5,
                 label=f"D={diameter:.0f} H={height:.0f}nm")
        ax5.fill_between(wls, 0, refl, alpha=0.12, color=hex_c)
        ax5.set_xlabel("波长 (nm)")
        ax5.set_ylabel("Reflectance")
        ax5.set_title(f"Spectrum: {material} on {substrate}")
        ax5.set_xlim(380, 780)
        ax5.set_ylim(0, 1.08)
        ax5.grid(True, alpha=0.25)
        ax5.legend(loc='upper right')
        fig5.tight_layout()
        st.pyplot(fig5); _get_plt().close(fig5)

    with col_cie:
        st.subheader("CIE 1931 色度图")
        # Draw CIE 1931 chromaticity diagram with current color point
        fig_cie, ax_cie = _get_plt().subplots(figsize=(5, 5))

        # Spectrum locus (from CIE data)
        x_xy = _CIE_X / (_CIE_X + _CIE_Y + _CIE_Z + 1e-12)
        y_xy = _CIE_Y / (_CIE_X + _CIE_Y + _CIE_Z + 1e-12)
        ax_cie.plot(x_xy, y_xy, 'k-', lw=1.2, alpha=0.8)
        ax_cie.fill(x_xy, y_xy, alpha=0.05, color='gray')

        # sRGB gamut triangle
        srgb_primaries_xy = np.array([[0.64, 0.33], [0.30, 0.60], [0.15, 0.06], [0.64, 0.33]])
        ax_cie.plot(srgb_primaries_xy[:, 0], srgb_primaries_xy[:, 1],
                    'k--', lw=0.8, alpha=0.5, label='sRGB gamut')

        # TiO2 metasurface gamut (sampled from grid)
        if len(engine.grid_xy) > 0:
            sample_step = max(1, len(engine.grid_xy) // 2000)
            gx = engine.grid_xy[::sample_step, 0]
            gy = engine.grid_xy[::sample_step, 1]
            ax_cie.scatter(gx, gy, c='#ff6b35', s=1, alpha=0.15, label='TiO2 gamut')
        ax_cie.legend(fontsize=6, loc='lower left')

        # D65 white point
        ax_cie.plot(0.3127, 0.3290, 'k+', ms=8, alpha=0.5)

        # Current color point
        xy = rgb_to_xy(rgb)
        ax_cie.plot(xy[0], xy[1], 'o', color=hex_c, ms=10,
                    markeredgecolor='white', markeredgewidth=1.5)
        ax_cie.plot(xy[0], xy[1], 'o', color=hex_c, ms=14, alpha=0.3)

        ax_cie.set_xlabel('x')
        ax_cie.set_ylabel('y')
        ax_cie.set_title(f'CIE 1931 xy: ({xy[0]:.4f}, {xy[1]:.4f})')
        ax_cie.set_xlim(0, 0.75)
        ax_cie.set_ylim(0, 0.85)
        ax_cie.set_aspect('equal')
        ax_cie.grid(True, alpha=0.2)
        fig_cie.tight_layout()
        st.pyplot(fig_cie)
        _get_plt().close(fig_cie)

    # Angle scan: color vs incident angle
    st.divider()
    st.subheader("入射角扫描 (0° → 80°)")
    angles_scan = np.arange(0, 85, 5)
    try:
        import torch_model as _tm2
        _ang_t = torch_model.torch.tensor(angles_scan, dtype=torch_model.torch.float32)
        _scan_rgb = _tm2.batch_single_pillar_rgb_norm(
            torch_model.torch.tensor([diameter]*len(angles_scan)),
            torch_model.torch.tensor([height]*len(angles_scan)),
            torch_model.torch.tensor([period]*len(angles_scan)),
            _ang_t)
        scan_rgbs = _scan_rgb.numpy()
    except Exception:
        scan_rgbs = []
        for a in angles_scan:
            param_a = MetaSurfaceParam(diameter, height, period, material, substrate, polarization, float(a))
            scan_rgbs.append(engine.physical_color(param_a))
        scan_rgbs = np.array(scan_rgbs)
    scan_hex = [rgb_to_hex(c) for c in scan_rgbs]

    fig_ang, (ax1, ax2) = _get_plt().subplots(1, 2, figsize=(10, 3))
    ax1.plot(angles_scan, scan_rgbs[:, 0], "r-", lw=1.5, label="R")
    ax1.plot(angles_scan, scan_rgbs[:, 1], "g-", lw=1.5, label="G")
    ax1.plot(angles_scan, scan_rgbs[:, 2], "b-", lw=1.5, label="B")
    ax1.set_xlabel("入射角 (°)")
    ax1.set_ylabel("sRGB")
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("RGB分量 vs 角度")
    for i, a in enumerate(angles_scan):
        ax2.add_patch(_get_plt().Rectangle((i, 0), 1, 1, facecolor=scan_hex[i], edgecolor="white", lw=0.3))
    ax2.set_xlim(0, len(angles_scan))
    ax2.set_ylim(0, 1)
    ax2.set_xticks(np.arange(len(angles_scan)) + 0.5)
    ax2.set_xticklabels([f"{int(a)}" for a in angles_scan], fontsize=6)
    ax2.set_yticks([])
    ax2.set_title("色块 vs 角度 (°)")
    fig_ang.tight_layout()
    st.pyplot(fig_ang)
    _get_plt().close(fig_ang)

st.sidebar.markdown("---")
st.sidebar.subheader("导出")
# Spectrum CSV export
wl = np.linspace(380, 780, 81)
if st.session_state.get("dual_pillar", False):
    try:
        import torch_model as _tm_exp
        import torch as _torch_exp
        sp_dual = _tm_exp.batch_dual_pillar_spectrum(
            _torch_exp.tensor([st.session_state.d1_val]), _torch_exp.tensor([st.session_state.h1_val]),
            _torch_exp.tensor([st.session_state.d2_val]), _torch_exp.tensor([st.session_state.h2_val]),
            _torch_exp.tensor([st.session_state.p_val]),
            material=material
        )
        spec_export = sp_dual.squeeze().detach().numpy()
    except Exception as e:
        spec_export = np.zeros(81)
else:
    try:
        import torch_model as _tm_exp
        import torch as _torch_exp
        sp_single = _tm_exp.batch_lorentzian_spectrum(
            _torch_exp.tensor([diameter]), _torch_exp.tensor([height]), _torch_exp.tensor([period]),
            material=material
        )
        spec_export = sp_single.squeeze().detach().numpy()
    except Exception as e:
        spec_export = np.zeros(81)

csv_data = "Wavelength_nm,Reflectance\n"
for i in range(81):
    csv_data += f"{wl[i]:.0f},{spec_export[i]:.6f}\n"
st.sidebar.download_button(
    "下载光谱 CSV", csv_data,
    file_name=f"spectrum_D{diameter:.0f}_H{height:.0f}_P{period:.0f}.csv",
    mime="text/csv", use_container_width=True
)

# Color swatch PNG export
try:
    swatch_size = 100
    swatch = np.ones((swatch_size, swatch_size, 3), dtype=np.uint8)
    r255, g255, b255 = int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)
    swatch[:,:,0] = r255; swatch[:,:,1] = g255; swatch[:,:,2] = b255
    img = Image.fromarray(swatch)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    st.sidebar.download_button(
        "下载色板 PNG", buf.getvalue(),
        file_name=f"swatch_{hex_color.lstrip('#')}.png",
        mime="image/png", use_container_width=True
    )
except:
    pass

st.sidebar.markdown("---")
st.sidebar.caption("AI超表面结构色设计 v5.0 (MultiMaterial)")
st.sidebar.caption("物理模型: Fano 共振 + CIE 1931 光谱管线")
st.sidebar.markdown("---")
st.sidebar.caption("长沙理工大学 物理与电子科学学院")
st.sidebar.caption("光电2501 乔安琪")

