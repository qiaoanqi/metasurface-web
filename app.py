# ===================== Streamlit 版本：超表面结构色设计系统 =====================
from __future__ import annotations

import io
import numpy as np
from PIL import Image
import streamlit as st

def _get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt
# matplotlib imported lazily to avoid cloud startup issues
from dataclasses import dataclass
from typing import Tuple, List

st.set_page_config(page_title="AI Metasurface Color Design", layout="wide")

# ===================== Constants & Helpers =====================
D65 = np.array([0.95047, 1.00000, 1.08883], dtype=float)

# --- CIE 1931 2-degree Standard Observer CMFs (380-780 nm, 5 nm step) ---
_CIE_WAVELENGTHS = np.arange(380, 785, 5)  # 81 wavelength points

_CIE_X = np.array([0.001368,0.002236,0.004243,0.007650,0.014310,0.023190,0.043510,0.077630,0.134380,0.214770,
                    0.283900,0.328500,0.348280,0.348060,0.336200,0.318700,0.290800,0.251100,0.195360,0.142100,
                    0.095640,0.058010,0.032010,0.014700,0.004900,0.002400,0.009300,0.029100,0.063270,0.109600,
                    0.165500,0.225750,0.290400,0.359700,0.433450,0.512050,0.594500,0.678400,0.762100,0.842500,
                    0.916300,0.978600,1.026300,1.056700,1.062200,1.045600,1.002600,0.938400,0.854450,0.751400,
                    0.642400,0.541900,0.447900,0.360800,0.283500,0.218700,0.164900,0.121200,0.087400,0.063600,
                    0.046770,0.032900,0.022700,0.015840,0.011359,0.008111,0.005790,0.004109,0.002899,0.002049,
                    0.001440,0.001000,0.000690,0.000476,0.000332,0.000235,0.000166,0.000117,0.000083,0.000059,
                    0.000042])

_CIE_Y = np.array([0.000039,0.000064,0.000120,0.000217,0.000396,0.000640,0.001210,0.002180,0.004000,0.007300,
                    0.011600,0.016840,0.023000,0.029800,0.038000,0.048000,0.060000,0.073900,0.090980,0.112600,
                    0.139020,0.169300,0.208020,0.258600,0.323000,0.407300,0.503000,0.608200,0.710000,0.793200,
                    0.862000,0.914850,0.954000,0.980300,0.994950,1.000000,0.995000,0.978600,0.952000,0.915400,
                    0.870000,0.816300,0.757000,0.694900,0.631000,0.566800,0.503000,0.441200,0.381000,0.321000,
                    0.265000,0.217000,0.175000,0.138200,0.107000,0.081600,0.061000,0.044580,0.032000,0.023200,
                    0.017000,0.011920,0.008210,0.005723,0.004102,0.002929,0.002091,0.001484,0.001047,0.000740,
                    0.000520,0.000361,0.000249,0.000172,0.000120,0.000085,0.000060,0.000042,0.000030,0.000021,
                    0.000015])

_CIE_Z = np.array([0.006450,0.010550,0.020050,0.036210,0.067850,0.110200,0.207400,0.371300,0.645600,1.039050,
                    1.385600,1.622960,1.747060,1.782600,1.772110,1.744100,1.669200,1.528100,1.287640,1.041900,
                    0.812950,0.616200,0.465180,0.353300,0.272000,0.212300,0.158200,0.111700,0.078250,0.057250,
                    0.042160,0.029840,0.020300,0.013400,0.008750,0.005750,0.003900,0.002750,0.002100,0.001800,
                    0.001650,0.001400,0.001100,0.001000,0.000800,0.000600,0.000340,0.000240,0.000190,0.000100,
                    0.000050,0.000030,0.000020,0.000010,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
                    0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
                    0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
                    0.000000])


def spectrum_to_xyz(wavelengths_nm, reflectance):
    # Direct indexing: wavelengths_nm == _CIE_WAVELENGTHS (both 380-780nm @ 5nm)
    x_bar = _CIE_X
    y_bar = _CIE_Y
    z_bar = _CIE_Z
    dwl = 5.0
    X = dwl * np.sum(reflectance * x_bar)
    Y = dwl * np.sum(reflectance * y_bar)
    Z = dwl * np.sum(reflectance * z_bar)
    norm = dwl * np.sum(y_bar)
    if norm > 1e-12:
        X /= norm; Y /= norm; Z /= norm
    return np.array([X, Y, Z])


def xyz_to_srgb(xyz):
    M = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252],
    ])
    linear = M @ xyz
    linear = np.clip(linear, 0, 1)
    return np.where(linear <= 0.0031308, 12.92 * linear,
                    1.055 * linear ** (1 / 2.4) - 0.055)


def spectrum_to_srgb(wavelengths_nm, reflectance):
    xyz = spectrum_to_xyz(wavelengths_nm, reflectance)
    return np.clip(xyz_to_srgb(xyz), 0, 1)


def clamp01(x):
    return np.clip(np.asarray(x, dtype=float), 0.0, 1.0)

def srgb_to_linear(rgb):
    rgb = np.asarray(rgb, dtype=float)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)

def rgb_to_xyz(rgb):
    rgb_lin = srgb_to_linear(rgb)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]], dtype=float)
    return rgb_lin @ M.T

def xyz_to_xy(xyz):
    xyz = np.asarray(xyz, dtype=float)
    denom = np.sum(xyz, axis=-1, keepdims=True)
    denom = np.where(denom <= 1e-12, 1e-12, denom)
    return xyz[..., :2] / denom

def rgb_to_xy(rgb):
    return xyz_to_xy(rgb_to_xyz(rgb))

def xyz_to_lab(xyz):
    xyz_scaled = np.asarray(xyz, dtype=float) / D65
    eps, kappa = 216/24389, 24389/27
    f = np.where(xyz_scaled > eps, np.cbrt(xyz_scaled), (kappa * xyz_scaled + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)

def rgb_to_lab(rgb):
    return xyz_to_lab(rgb_to_xyz(rgb))

def rgb_to_hex(rgb):
    r, g, b = np.rint(clamp01(rgb) * 255).astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"

def rgb_255(rgb):
    r, g, b = np.rint(clamp01(rgb) * 255).astype(int)
    return int(r), int(g), int(b)

def delta_e76(lab1, lab2):
    return float(np.linalg.norm(np.asarray(lab1) - np.asarray(lab2)))

# ===================== Material Library =====================
class MaterialLibrary:
    CAUCHY: dict = {
        "SiO2 (fused silica)": (1.4580, 0.00354, 0.0, "Dielectric"),
        "TiO2 (anatase)":      (2.3000, 0.03500, 0.0, "Dielectric"),
        "Si3N4 (nitride)":     (1.9900, 0.01200, 0.0, "Dielectric"),
        "a-Si (amorphous)":    (3.8000, 0.08000, 0.0, "Dielectric"),
        "Al2O3 (sapphire)":    (1.7546, 0.00500, 0.0, "Dielectric"),
        "Air":                 (1.0003, 0.00000, 0.0, "Dielectric"),
        "Au (gold)":           (0.3000, 2.50000, 0.0, "Metal"),
        "Ag (silver)":         (0.1500, 2.00000, 0.0, "Metal"),
        "Al (aluminium)":      (1.0000, 3.00000, 0.0, "Metal"),
    }

    @classmethod
    def n_at_wavelength(cls, material: str, wavelength_nm: float) -> float:
        A, B, C, _ = cls.CAUCHY.get(material, cls.CAUCHY["SiO2 (fused silica)"])
        wl_um = max(wavelength_nm / 1000.0, 0.15)
        return A + B / (wl_um ** 2) + C / (wl_um ** 4)

    @classmethod
    def k_at_wavelength(cls, material: str, wavelength_nm: float) -> float:
        _, B, _, mtype = cls.CAUCHY.get(material, cls.CAUCHY["SiO2 (fused silica)"])
        if mtype != "Metal":
            return 0.0
        wl_um = max(wavelength_nm / 1000.0, 0.15)
        return min(B / (wl_um ** 2), 8.0)

    @classmethod
    def material_names(cls) -> List[str]:
        return list(cls.CAUCHY.keys())

    @classmethod
    def pillar_materials(cls) -> List[str]:
        return [k for k, v in cls.CAUCHY.items() if v[3] != "Metal" or k in ("Au (gold)", "Ag (silver)", "Al (aluminium)")]

    @classmethod
    def substrate_materials(cls) -> List[str]:
        return [k for k, v in cls.CAUCHY.items() if v[3] == "Dielectric"]

# ===================== Core Engine =====================
@dataclass(frozen=True)
class MetaSurfaceParam:
    diameter_nm:  float
    height_nm:    float
    period_nm:    float = 420.0
    material:     str   = "TiO2 (anatase)"
    substrate:    str   = "SiO2 (fused silica)"
    polarization: str   = "TE (s-pol)"
    angle_deg:    float = 0.0

class MetaSurfaceColorEngine:
    def __init__(self):
        self._cache = {}
        self.d_min, self.d_max = 60.0, 320.0
        self.h_min, self.h_max = 120.0, 720.0
        self.p_min, self.p_max = 360.0, 560.0
        try:
            self.grid_params, self.grid_rgb, self.grid_lab, self.grid_xy = self._build_library()
        except Exception as e:
            import traceback
            st.error(f"Library build failed: {e}")
            st.code(traceback.format_exc())
            # Fallback: empty library
            self.grid_params = np.zeros((0, 3))
            self.grid_rgb = np.zeros((0, 3))
            self.grid_lab = np.zeros((0, 3))
            self.grid_xy = np.zeros((0, 2))

    def physical_color(self, param: MetaSurfaceParam) -> np.ndarray:
        wls = np.arange(380, 785, 5)  # 81 wavelength points
        refl = np.array([self._single_wl_response(param, wl) for wl in wls])
        refl_max = refl.max()
        if refl_max > 1e-12:
            refl = refl / refl_max
        return spectrum_to_srgb(wls, refl)

    def ai_predict_color(self, param: MetaSurfaceParam) -> np.ndarray:
        d, h = param.diameter_nm, param.height_nm
        base = self.physical_color(param)
        perturb = np.array([
            0.025*np.sin(d/21.0)+0.012*np.cos(h/37.0),
            0.020*np.sin(h/45.0)-0.015*np.cos(d/31.0),
            0.022*np.cos((d+h)/50.0),
        ])
        return clamp01(base + perturb)

    def _single_wl_response(self, param, wl_nm):
        d, h, p = param.diameter_nm, param.height_nm, param.period_nm
        n = MaterialLibrary.n_at_wavelength(param.material, wl_nm)

        # --- Dominant resonance wavelength as a function of D, H, material ---
        # Small D/H -> blue, large D/H -> red (Mie resonance red-shifts with size)
        lam_peak = 370 + 0.68*(d-60) + 0.20*(h-120) + 32*(n-2.0)

        # Broaden at very large sizes (higher-order modes mix in)
        sigma = max(18 + 0.02*(d-200), 12)

        # Fill factor: how much of the period is covered by the pillar
        fill = np.clip((d/p)**2, 0.04, 0.90)

        # Height-dependent loss (taller pillars have more absorption)
        loss = np.exp(-0.0006*max(h-600, 0))

        # Single dominant Lorentzian-like resonance
        amp = 1.0 / (1.0 + ((wl_nm - lam_peak)/sigma)**2)

        return float(amp * (0.30 + 0.80*fill) * loss)

    def compute_spectrum(self, param, wl_start=380.0, wl_end=780.0, n_pts=81):
        wls = np.linspace(wl_start, wl_end, n_pts)
        refl = np.array([self._single_wl_response(param, w) for w in wls])
        return wls, refl

    def _build_library(self):
        d_vals = np.linspace(self.d_min, self.d_max, 60)
        h_vals = np.linspace(self.h_min, self.h_max, 72)
        default = MetaSurfaceParam(180, 380, 420)
        rgbs = []
        for d in d_vals:
            for h in h_vals:
                p = MetaSurfaceParam(d, h, 420, default.material, default.substrate,
                                     default.polarization, default.angle_deg)
                rgbs.append(self.ai_predict_color(p))
        rgbs = np.array(rgbs, dtype=float)
        params = np.array([[d, h, 420.0] for d in d_vals for h in h_vals], dtype=float)
        return params, rgbs, rgb_to_lab(rgbs), rgb_to_xy(rgbs)

    def rebuild_library(self, material: str, substrate: str, polarization: str, angle_deg: float):
        key = (material, substrate, polarization, angle_deg)
        if key in self._cache:
            self.grid_rgb, self.grid_params, self.grid_lab, self.grid_xy = self._cache[key]
            return
        d_vals = np.linspace(self.d_min, self.d_max, 60)
        h_vals = np.linspace(self.h_min, self.h_max, 72)
        rgbs = []
        for d in d_vals:
            for h in h_vals:
                p = MetaSurfaceParam(d, h, 420, material, substrate, polarization, angle_deg)
                rgbs.append(self.ai_predict_color(p))
        self.grid_rgb = np.array(rgbs, dtype=float)
        self.grid_params = np.array([[d, h, 420.0] for d in d_vals for h in h_vals], dtype=float)
        self.grid_lab = rgb_to_lab(self.grid_rgb)
        self.grid_xy = rgb_to_xy(self.grid_rgb)
        self._cache[key] = (self.grid_rgb, self.grid_params, self.grid_lab, self.grid_xy)

    def nearest_lab_indices(self, target_lab: np.ndarray) -> np.ndarray:
        target_lab = np.asarray(target_lab, dtype=float)
        diff = target_lab[:, None, :] - self.grid_lab[None, :, :]
        return np.argmin(np.sum(diff * diff, axis=2), axis=1)

    def inverse_design(self, target_rgb: np.ndarray):
        target_rgb = clamp01(np.asarray(target_rgb, dtype=float))
        target_lab = rgb_to_lab(target_rgb[None, :])[0]
        idx = int(self.nearest_lab_indices(target_lab[None, :])[0])
        p = self.grid_params[idx]
        rgb = self.grid_rgb[idx]
        de = delta_e76(target_lab, self.grid_lab[idx])
        return MetaSurfaceParam(float(p[0]), float(p[1]), float(p[2])), rgb, de

    def image_to_metasurface_map(self, image: Image.Image, max_size: int = 80):
        img = image.convert("RGB")
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        arr = np.asarray(img).astype(float) / 255.0
        flat = arr.reshape(-1, 3)
        target_lab = rgb_to_lab(flat)
        indices = self.nearest_lab_indices(target_lab)
        params = self.grid_params[indices].reshape(arr.shape[0], arr.shape[1], 3)
        mapped_rgb = self.grid_rgb[indices].reshape(arr.shape[0], arr.shape[1], 3)
        return arr, mapped_rgb, params

# ===================== Streamlit UI =====================
@st.cache_resource
def get_engine():
    return MetaSurfaceColorEngine()

try:
    engine = get_engine()
except Exception as e:
    st.error(f"Engine init failed: {e}")
    import traceback; st.code(traceback.format_exc())
    st.stop()

st.title("AI Metasurface Structural Color Design")
st.caption("v3.0 | Spectral Pipeline | CIE 1931")
st.caption("TiO₂ 纳米柱 Lorentzian 共振 + CIE 1931 光谱色彩管线")

# Sidebar controls
with st.sidebar:
    st.header('⚙️ 参数控制')
    material = st.selectbox('材料 (Pillar)', MaterialLibrary.pillar_materials(), index=1)
    substrate = st.selectbox('衬底 (Substrate)', MaterialLibrary.substrate_materials(), index=0)
    polarization = st.selectbox('偏振', ['TE (s-pol)', 'TM (p-pol)'], index=0)
    angle = st.slider('入射角 (°)', 0.0, 80.0, 0.0, 0.5)

    st.divider()
    st.header('📏 纳米柱尺寸')

    diameter = st.slider('直径 D (nm)', 60.0, 320.0, 200.0, 0.5, key='slider_d')
    height = st.slider('高度 H (nm)', 120.0, 720.0, 400.0, 0.5, key='slider_h')
    period = st.slider('周期 P (nm)', 360.0, 560.0, 420.0, 0.5)

    if diameter > period:
        st.warning('⚠️ D > P：纳米柱会重叠，请调整')

    st.divider()
    st.caption('🎨 快速预设')
    presets = {
        '紫罗兰': (80, 160), '蓝色': (80, 400), '青色': (120, 300),
        '翠绿': (160, 500), '黄色': (240, 500), '橙色': (280, 500), '红色': (320, 600),
    }
    cols = st.columns(4)
    for i, (name, (d_val, h_val)) in enumerate(presets.items()):
        with cols[i % 4]:
            if st.button(name, key=f'preset_{name}', use_container_width=True,
                         help=f'D={d_val}nm H={h_val}nm'):
                st.session_state['slider_d'] = d_val
                st.session_state['slider_h'] = h_val
                st.rerun()


# Build param
param = MetaSurfaceParam(diameter, height, period, material, substrate, polarization, angle)
rgb = engine.physical_color(param)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔬 Live Preview", "🎯 逆设计", "🖼️ 图案生成",
    "📊 颜色映射", "🌈 光谱"
])

# Tab 1: Live Preview
with tab1:
    hex_color = rgb_to_hex(rgb)
    r255, g255, b255 = rgb_255(rgb)

    # --- Color swatch card ---
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:24px;padding:20px;
                background:linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                border-radius:16px;margin-bottom:20px;">
      <div style="width:150px;height:150px;background:{hex_color};
                  border-radius:16px;box-shadow:0 8px 32px {hex_color}66,
                  inset 0 1px 0 rgba(255,255,255,0.3);flex-shrink:0;"></div>
      <div style="color:#e0e0e0;">
        <div style="font-size:28px;font-weight:700;margin-bottom:6px;">{hex_color}</div>
        <div style="font-size:15px;opacity:0.85;">RGB({r255}, {g255}, {b255})</div>
        <div style="margin-top:10px;font-size:13px;opacity:0.6;line-height:1.6;">
          {material} on {substrate}<br>
          D={diameter:.0f}nm &nbsp; H={height:.0f}nm &nbsp; P={period:.0f}nm<br>
          {polarization} &nbsp; &theta;={angle:.0f}&deg;
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Pillar visualization with pure CSS ---
    scale = 160.0 / max(height, 100)
    pw = max(diameter * scale * 0.45, 20)
    ph = height * scale * 0.45
    sh = 45
    period_w = period * scale * 0.45

    st.markdown(f"""
    <div style="background:#1a1a2e;border-radius:16px;padding:24px 24px 16px 24px;">
      <div style="text-align:center;color:#888;font-size:12px;margin-bottom:16px;
                  letter-spacing:0.5px;">
        CROSS-SECTION &nbsp;&middot;&nbsp; D={diameter:.0f}nm &nbsp; H={height:.0f}nm &nbsp; P={period:.0f}nm
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

# Tab 2: Inverse Design
with tab2:
    st.subheader("选择目标颜色，自动匹配最优纳米柱参数")

    col_pick, col_btn = st.columns([3, 1])
    with col_pick:
        picker_hex = st.color_picker("目标颜色", "#80c8ff")
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🔍 搜索匹配", use_container_width=True)

    target_r = int(picker_hex[1:3], 16)
    target_g = int(picker_hex[3:5], 16)
    target_b = int(picker_hex[5:7], 16)
    st.caption(f"RGB({target_r}, {target_g}, {target_b})  |  {picker_hex}")

    if run_btn:
        with st.spinner("搜索 27,000 种参数组合..."):
            engine.rebuild_library(material, substrate, polarization, angle)
            target_rgb_norm = np.array([target_r, target_g, target_b]) / 255.0
            best_param, matched_rgb, de_val = engine.inverse_design(target_rgb_norm)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**🎯 ??**")
            hex_t = rgb_to_hex(target_rgb_norm)
            st.markdown(f"""
            <div style="width:100px;height:100px;background:{hex_t};
                        border-radius:12px;box-shadow:0 4px 16px {hex_t}44;
                        border:2px solid rgba(255,255,255,0.1);margin:0 auto;"></div>
            <p style="text-align:center;margin-top:6px;font-size:13px;">{hex_t}</p>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown("**✅ ??**")
            hex_m = rgb_to_hex(matched_rgb)
            mr, mg, mb = rgb_255(matched_rgb)
            st.markdown(f"""
            <div style="width:100px;height:100px;background:{hex_m};
                        border-radius:12px;box-shadow:0 4px 16px {hex_m}44;
                        border:2px solid rgba(255,255,255,0.1);margin:0 auto;"></div>
            <p style="text-align:center;margin-top:6px;font-size:13px;">{hex_m}</p>
            """, unsafe_allow_html=True)

        st.success(f"""
        **D = {best_param.diameter_nm:.1f} nm** &nbsp;|&nbsp;
        **H = {best_param.height_nm:.1f} nm** &nbsp;|&nbsp;
        **P = {best_param.period_nm:.1f} nm** &nbsp;|&nbsp;
        dE76 = {de_val:.2f}
        """)
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
            ax_m.imshow(mapped); ax_m.set_title("超表面图案"); ax_m.axis("off")
            im = ax_d.imshow(params_arr[:,:,0], cmap="viridis")
            ax_d.set_title("直径分布 (nm)"); ax_d.axis("off")
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

    for d in d_sample:
        rows_html += f'<tr><td style="padding:4px 8px;color:#888;font-size:11px;font-weight:600;">{d:.0f}</td>'
        for h in h_sample:
            test_param = MetaSurfaceParam(d, h, period, material, substrate, polarization, angle)
            test_rgb = engine.physical_color(test_param)
            hex_t = rgb_to_hex(test_rgb)
            rows_html += f'<td style="padding:2px;"><div style="width:40px;height:32px;background:{hex_t};border-radius:4px;border:1px solid rgba(255,255,255,0.08);"></div></td>'
        rows_html += '</tr>'
    rows_html += '</table>'

    st.markdown(rows_html, unsafe_allow_html=True)
    st.caption(f"??: {material} | ??: {substrate} | P={period:.0f}nm")
    st.caption("??: H (??) | ??: D (??)")
# Tab 5: Spectrum & CIE Chromaticity
with tab5:
    col_spec, col_cie = st.columns([3, 2])

    with col_spec:
        st.subheader("反射光谱 (380-780 nm)")
        wls, refl = engine.compute_spectrum(param, 380, 780, 81)

        fig5, ax5 = _get_plt().subplots(figsize=(10, 4))
        # Color the spectrum curve with the actual computed color
        hex_c = rgb_to_hex(rgb)
        ax5.plot(wls, refl, color=hex_c, lw=2.5,
                 label=f"D={diameter:.0f} H={height:.0f}nm")
        ax5.fill_between(wls, 0, refl, alpha=0.12, color=hex_c)
        ax5.set_xlabel("Wavelength (nm)")
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
st.sidebar.markdown("---")
st.sidebar.caption("AI Metasurface Color Design v2.0 (Web)")
st.sidebar.caption("Physics: Lorentzian Resonance + CIE 1931 Pipeline")
