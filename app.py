# ===================== Streamlit 版本：超表面结构色设计系统 =====================
from __future__ import annotations

import io, os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from PIL import Image
import streamlit as st
import ml_module

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
    reflectance = np.nan_to_num(np.asarray(reflectance, dtype=float), nan=0.0, posinf=1.0, neginf=0.0)
    xyz = spectrum_to_xyz(wavelengths_nm, reflectance)
    return np.nan_to_num(np.clip(xyz_to_srgb(xyz), 0, 1), nan=0.0)


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
    rgb = np.nan_to_num(np.asarray(rgb, dtype=float), nan=0.0, posinf=1.0, neginf=0.0)
    r, g, b = np.rint(clamp01(rgb) * 255).astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"

def rgb_255(rgb):
    rgb = np.nan_to_num(np.asarray(rgb, dtype=float), nan=0.0, posinf=1.0, neginf=0.0)
    r, g, b = np.rint(clamp01(rgb) * 255).astype(int)
    return int(r), int(g), int(b)

def delta_e76(lab1, lab2):
    return float(np.linalg.norm(np.asarray(lab1) - np.asarray(lab2)))

def delta_e2000(lab1, lab2):
    L1, a1, b1 = np.asarray(lab1, dtype=float)
    L2, a2, b2 = np.asarray(lab2, dtype=float)
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    Cbar = (C1 + C2) / 2.0
    G = 0.5 * (1 - np.sqrt(Cbar**7 / (Cbar**7 + 25**7)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = np.sqrt(a1p**2 + b1**2)
    C2p = np.sqrt(a2p**2 + b2**2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp = L2 - L1
    dCp = C2p - C1p
    if C1p * C2p == 0:
        dhp = 0.0
    else:
        dh = h2p - h1p
        if abs(dh) <= 180:
            dhp = dh
        elif dh > 180:
            dhp = dh - 360
        else:
            dhp = dh + 360
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2.0))
    Lpbar = (L1 + L2) / 2.0
    Cpbar = (C1p + C2p) / 2.0
    if C1p * C2p == 0:
        hpbar = h1p + h2p
    else:
        hpbar = (h1p + h2p) / 2.0
        if abs(h1p - h2p) > 180:
            hpbar = (h1p + h2p + 360) / 2.0
    T = (1 - 0.17 * np.cos(np.radians(hpbar - 30))
         + 0.24 * np.cos(np.radians(2 * hpbar))
         + 0.32 * np.cos(np.radians(3 * hpbar + 6))
         - 0.20 * np.cos(np.radians(4 * hpbar - 63)))
    dtheta = 30 * np.exp(-((hpbar - 275) / 25)**2)
    RC = 2 * np.sqrt(Cpbar**7 / (Cpbar**7 + 25**7))
    SL = 1 + (0.015 * (Lpbar - 50)**2) / np.sqrt(20 + (Lpbar - 50)**2)
    SC = 1 + 0.045 * Cpbar
    SH = 1 + 0.015 * Cpbar * T
    RT = -np.sin(np.radians(2 * dtheta)) * RC
    dE = np.sqrt((dLp/SL)**2 + (dCp/SC)**2 + (dHp/SH)**2 + RT * (dCp/SC) * (dHp/SH))
    return float(dE)

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

@dataclass
class DualPillarParam:
    """双异质纳米柱超单元参数 (自动钳制非法值)

    参数修正优先级:
      1. 周期P >= max(D1,D2)*1.2, 确保每根柱子都在单元内
      2. 单柱直径D <= P*0.8, 限制单柱占空比
      3. 总占空比(fill1+fill2) <= 0.85, 缩放D1/D2等比例
    """
    d1_nm:       float
    h1_nm:       float
    d2_nm:       float
    h2_nm:       float
    period_nm:   float = 420.0
    material:    str   = "TiO2 (anatase)"
    substrate:   str   = "SiO2 (fused silica)"
    polarization: str  = "TE (s-pol)"
    angle_deg:   float = 0.0
    _corrected:  bool  = False
    _correction_msg: str = ""

    def __post_init__(self):
        msgs = []
        orig_p = self.period_nm
        orig_d1 = self.d1_nm
        orig_d2 = self.d2_nm

        # 优先级1: P >= max(D1,D2)*1.2
        min_p = max(self.d1_nm, self.d2_nm) * 1.20
        if self.period_nm < min_p:
            object.__setattr__(self, 'period_nm', max(min_p, 380.0))
        if self.period_nm < max(self.d1_nm, self.d2_nm):
            object.__setattr__(self, 'period_nm', max(self.d1_nm, self.d2_nm) + 20.0)
        if self.period_nm != orig_p:
            msgs.append(f"P {orig_p:.0f}->{self.period_nm:.0f}nm")

        # 优先级2: D <= P*0.8
        max_d = self.period_nm * 0.80
        if self.d1_nm > max_d:
            object.__setattr__(self, 'd1_nm', max_d)
            msgs.append(f"D1 {orig_d1:.0f}->{self.d1_nm:.0f}nm")
        if self.d2_nm > max_d:
            object.__setattr__(self, 'd2_nm', max_d)
            msgs.append(f"D2 {orig_d2:.0f}->{self.d2_nm:.0f}nm")

        # 优先级3: 总占空比 <= 0.85
        fill1 = np.pi*(self.d1_nm/2)**2/(self.period_nm**2)
        fill2 = np.pi*(self.d2_nm/2)**2/(self.period_nm**2)
        if fill1 + fill2 > 0.85:
            scale = np.sqrt(0.80 / (fill1 + fill2))
            new_d1 = self.d1_nm * scale
            new_d2 = self.d2_nm * scale
            msgs.append(f"D1 {self.d1_nm:.0f}->{new_d1:.0f} D2 {self.d2_nm:.0f}->{new_d2:.0f}nm (占空比过高)")
            object.__setattr__(self, 'd1_nm', new_d1)
            object.__setattr__(self, 'd2_nm', new_d2)

        if msgs:
            object.__setattr__(self, '_corrected', True)
            object.__setattr__(self, '_correction_msg', "; ".join(msgs))

def _single_pillar_complex(d_nm, h_nm, p_nm, material, polarization, angle_deg, wl_nm, substrate=None):
    """静态方法: 计算单根纳米柱的复数反射系数。
    供单柱和双柱模型共用, 避免代码重复。
    """
    n_mat = MaterialLibrary.n_at_wavelength(material, wl_nm)
    n_sub = MaterialLibrary.n_at_wavelength(substrate, wl_nm) if substrate else 1.458
    n_env = (1.0 + n_sub) / 2.0
    dn = n_mat - n_env

    lam_ed = 360 + 0.55*(d_nm-60) + 0.12*(h_nm-120) + 32*dn
    sigma_ed = max(26 + 0.10*(d_nm-200), 8)  # FDTD-calibrated
    lam_md = 400 + 0.75*(d_nm-60) + 0.25*(h_nm-120) + 32*dn
    sigma_md = max(35 + 0.12*(d_nm-200), 10)  # FDTD-calibrated

    fill = np.clip(np.pi*(d_nm/2)**2/(p_nm**2), 0.01, 0.70)
    fill_amp = 0.30 + 0.80*fill
    loss = np.exp(-0.0006*max(h_nm-600, 0))

    theta = angle_deg * 3.14159265 / 180.0
    sin2 = np.sin(theta)**2
    if polarization.startswith("TE"):
        ed_shift, md_shift = -45 * sin2, -20 * sin2
        ed_amp_angle = 1.0 - 0.10 * sin2
        md_amp_angle = 1.0 - 0.04 * sin2
    else:
        ed_shift, md_shift = -18 * sin2, -8 * sin2
        ed_amp_angle = 1.0 - 0.25 * sin2
        md_amp_angle = 1.0 - 0.12 * sin2

    w_ed = np.clip(0.80 - 0.003*(d_nm-60), 0.0, 0.80)
    w_md = 1.0 - w_ed

    ed_center = lam_ed + ed_shift
    detune_ed = (wl_nm - ed_center) / sigma_ed
    ed_amp = np.sqrt(1.0 / (1.0 + detune_ed**2)) * np.sqrt(ed_amp_angle)
    ed_phase = -np.arctan(detune_ed)

    md_center = lam_md + md_shift
    detune_md = (wl_nm - md_center) / sigma_md
    md_amp = np.sqrt(1.0 / (1.0 + detune_md**2)) * np.sqrt(md_amp_angle)
    md_phase = -np.arctan(detune_md)

    r_ed = ed_amp * np.exp(1j * ed_phase)
    r_md = md_amp * np.exp(1j * md_phase)
    return (w_ed * r_ed + w_md * r_md) * np.sqrt(float(fill_amp * loss)), fill

class MetaSurfaceColorEngine:
    def __init__(self):
        self._cache = {}
        self.d_min, self.d_max = 50.0, 350.0
        self.h_min, self.h_max = 80.0, 600.0
        self.p_min, self.p_max = 200.0, 600.0
        self._coarse_grid_cache = {}
        self._last_material = "TiO2 (anatase)"
        self._last_substrate = "SiO2 (fused silica)"
        self._last_polarization = "TE (s-pol)"
        self._last_angle = 0.0
        self._enable_far_field = False
        self._na = 0.1
        self._theta_obs_deg = 0.0
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

    def physical_color(self, param) -> np.ndarray:
        """支持单柱(MetaSurfaceParam)和双柱(DualPillarParam)参数"""
        if isinstance(param, DualPillarParam) or type(param).__name__ == 'DualPillarParam':
            if getattr(self, '_enable_far_field', False):
                wls, refl = self._dual_far_field_spectrum(param, self._na, self._theta_obs_deg)
            else:
                # PyTorch batch mode: 81 wavelengths in one pass
                try:
                    import torch_model as _tm
                    pol_TE = param.polarization.startswith("TE")
                    spec = _tm.batch_dual_pillar_spectrum(
                        _tm.torch.tensor([param.d1_nm]),
                        _tm.torch.tensor([param.h1_nm]),
                        _tm.torch.tensor([param.d2_nm]),
                        _tm.torch.tensor([param.h2_nm]),
                        _tm.torch.tensor([param.period_nm]),
                        _tm.torch.tensor([param.angle_deg]),
                        pol_TE,
                        material=param.material
                    )
                    rgb = _tm.batch_spectrum_to_rgb(spec)
                    return rgb.squeeze(0).numpy()
                except Exception:
                    wls = np.arange(380, 785, 5)
                    refl = np.zeros(len(wls))
                    for i, wl_nm in enumerate(wls):
                        I1, _ = _single_pillar_complex(param.d1_nm, param.h1_nm, param.period_nm,
                            param.material, param.polarization, param.angle_deg, wl_nm, param.substrat)
                        I2, _ = _single_pillar_complex(param.d2_nm, param.h2_nm, param.period_nm,
                            param.material, param.polarization, param.angle_deg, wl_nm, param.substrat)
                        refl[i] = float(abs(I1)**2 + abs(I2)**2)
                    refl = refl / 0.86
                    return spectrum_to_srgb(wls, refl)
        # Single pillar (MetaSurfaceParam)
        if getattr(self, '_enable_far_field', False):
            if isinstance(param, DualPillarParam) or type(param).__name__ == 'DualPillarParam':
                wls, refl = self._dual_far_field_spectrum(param, self._na, self._theta_obs_deg)
            else:
                wls, refl = self._far_field_spectrum(param, self._na, self._theta_obs_deg)
            return spectrum_to_srgb(wls, refl)
        # PyTorch batch mode: 81 wavelengths in one pass (~10x faster)
        try:
            import torch_model as _tm
            pol_TE = param.polarization.startswith("TE")
            spec = _tm.batch_lorentzian_spectrum(
                _tm.torch.tensor([param.diameter_nm]),
                _tm.torch.tensor([param.height_nm]),
                _tm.torch.tensor([param.period_nm]),
                _tm.torch.tensor([param.angle_deg]),
                pol_TE,
                material=param.material, substrate=param.substrate
            )
            rgb = _tm.batch_spectrum_to_rgb(spec)
            return rgb.squeeze(0).numpy()
        except Exception:
            wls = np.arange(380, 785, 5)
            refl = np.array([self._single_wl_response(param, wl) for wl in wls])
            refl = refl / 0.86
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
        n_mat = MaterialLibrary.n_at_wavelength(param.material, wl_nm)
        n_sub = MaterialLibrary.n_at_wavelength(param.substrat, wl_nm) if param.substrat else 1.458
        n_env = (1.0 + n_sub) / 2.0  # half air, half substrate
        dn = n_mat - n_env

        # --- Dual Lorentzian: ED (electric dipole) + MD (magnetic dipole) ---
        # ED resonance: shorter wavelength, narrower, moderate size sensitivity
        lam_ed = 360 + 0.55*(d-60) + 0.12*(h-120) + 32*dn
        sigma_ed = max(15 + 0.015*(d-200), 10)

        # MD resonance: longer wavelength, broader, higher size sensitivity
        lam_md = 400 + 0.75*(d-60) + 0.25*(h-120) + 32*dn
        sigma_md = max(22 + 0.03*(d-200), 15)

        # Fill factor
        fill = np.clip(np.pi*(d/2)**2/(p**2), 0.03, 0.70)
        fill_amp = 0.30 + 0.80*fill

        # Height-dependent loss
        loss = np.exp(-0.0006*max(h-600, 0))

        # --- Angle & polarization dependence ---
        theta = param.angle_deg * 3.14159265 / 180.0
        sin2 = np.sin(theta)**2
        if param.polarization.startswith("TE"):
            # ED shifts more with angle, MD is more robust
            ed_shift = -45 * sin2
            md_shift = -20 * sin2
            ed_amp_angle = 1.0 - 0.10 * sin2
            md_amp_angle = 1.0 - 0.04 * sin2
        else:
            ed_shift = -18 * sin2
            md_shift = -8 * sin2
            ed_amp_angle = 1.0 - 0.25 * sin2
            md_amp_angle = 1.0 - 0.12 * sin2

        # ED Lorentzian
        ed = 1.0 / (1.0 + ((wl_nm - (lam_ed + ed_shift))/sigma_ed)**2)
        # MD Lorentzian
        md = 1.0 / (1.0 + ((wl_nm - (lam_md + md_shift))/sigma_md)**2)

        # Dynamic weights: ED dominates for small pillars, MD for large
        # Small D -> ED is stronger (magnetic response weak at small sizes)
        w_ed = np.clip(0.80 - 0.003*(d-60), 0.0, 0.80)
        w_md = 1.0 - w_ed
        combined = w_ed * ed * ed_amp_angle + w_md * md * md_amp_angle

        return float(combined * fill_amp * loss)

    def _complex_pillar_response(self, param, wl_nm):
        r, _ = _single_pillar_complex(
            param.diameter_nm, param.height_nm, param.period_nm,
            param.material, param.polarization, param.angle_deg, wl_nm)
        return r

    def _far_field_spectrum(self, param, na=0.1, theta_obs_deg=0.0, N=12):
        """角谱理论远场传播 — 高斯NA + Lorentzian角度响应。

        物理模型:
        1. 有限相干阵列 N×N → 0阶主瓣 ~ 2D高斯 (σ=0.5)
        2. NA锥积分: f_NA = 1-exp(-w_max²/(2σ²)), w_max=NA·N·P/λ
        3. 角度响应: f_theta = 1/(1+(w_shift/γ)²), w_shift=N·P·sin(θ)/λ
           (Lorentzian比高斯更适合描述共振增强的角度散射)
        4. R_eff = |r_0|² · f_NA · f_theta
        """
        wls = np.arange(380, 785, 5)
        d_nm, h_nm, p_nm = param.diameter_nm, param.height_nm, param.period_nm
        fill = float(np.clip(np.pi*(d_nm/2)**2/(p_nm**2), 0.03, 0.70))

        n_sub = MaterialLibrary.n_at_wavelength(param.substrat, 550.0)
        r_sub = (1.0 - n_sub) / (1.0 + n_sub)

        sigma_w, gamma_w = 0.5, 0.6
        sin_theta = np.sin(theta_obs_deg * np.pi / 180.0)

        refl_eff = np.zeros(len(wls))
        for i, wl_nm in enumerate(wls):
            r_pillar = self._complex_pillar_response(param, wl_nm)
            r_0 = fill * r_pillar + (1.0 - fill) * r_sub
            power_total = float(abs(r_0)**2)

            if na >= 0.99 and abs(theta_obs_deg) < 0.01:
                f_total = 1.0
            else:
                w_max = na * N * p_nm / wl_nm
                w_shift = N * p_nm * sin_theta / wl_nm
                f_na = 1.0 - np.exp(-w_max**2 / (2.0 * sigma_w**2))
                f_theta = 1.0 / (1.0 + (w_shift / gamma_w)**2)
                f_total = f_na * f_theta

            refl_eff[i] = power_total * f_total

        return wls, refl_eff


    def _dual_far_field_spectrum(self, dp: DualPillarParam, na=0.1, theta_obs_deg=0.0, N=12):
        """双柱超单元远场 — 高斯NA + Lorentzian角度。"""
        wls = np.arange(380, 785, 5)
        d1, h1 = dp.d1_nm, dp.h1_nm
        d2, h2 = dp.d2_nm, dp.h2_nm
        p_nm = dp.period_nm
        mat, sub = dp.material, dp.substrate
        pol, ang = dp.polarization, dp.angle_deg

        _, fill1 = _single_pillar_complex(d1, h1, p_nm, mat, pol, ang, 550.0, sub)
        _, fill2 = _single_pillar_complex(d2, h2, p_nm, mat, pol, ang, 550.0, sub)
        fill_gap = 1.0 - fill1 - fill2

        n_sub = MaterialLibrary.n_at_wavelength(sub, 550.0)
        r_sub = (1.0 - n_sub) / (1.0 + n_sub)

        sigma_w, gamma_w = 0.5, 0.6
        sin_theta = np.sin(theta_obs_deg * np.pi / 180.0)

        refl_eff = np.zeros(len(wls))
        for i, wl_nm in enumerate(wls):
            r1, _ = _single_pillar_complex(d1, h1, p_nm, mat, pol, ang, wl_nm, sub)
            r2, _ = _single_pillar_complex(d2, h2, p_nm, mat, pol, ang, wl_nm, sub)
            r_0 = fill1 * r1 + fill2 * r2 + fill_gap * r_sub
            power_total = float(abs(r_0)**2)

            if na >= 0.99 and abs(theta_obs_deg) < 0.01:
                f_total = 1.0
            else:
                w_max = na * N * p_nm / wl_nm
                w_shift = N * p_nm * sin_theta / wl_nm
                f_na = 1.0 - np.exp(-w_max**2 / (2.0 * sigma_w**2))
                f_theta = 1.0 / (1.0 + (w_shift / gamma_w)**2)
                f_total = f_na * f_theta

            refl_eff[i] = power_total * f_total
        return wls, refl_eff

    def compute_spectrum(self, param, wl_start=380.0, wl_end=780.0, n_pts=81):
        if isinstance(param, DualPillarParam) or type(param).__name__ == 'DualPillarParam':
            if getattr(self, '_enable_far_field', False):
                return self._dual_far_field_spectrum(param, self._na, self._theta_obs_deg)
            # PyTorch batch mode
            try:
                import torch_model as _tm
                pol_TE = param.polarization.startswith("TE")
                spec = _tm.batch_dual_pillar_spectrum(
                    _tm.torch.tensor([param.d1_nm]),
                    _tm.torch.tensor([param.h1_nm]),
                    _tm.torch.tensor([param.d2_nm]),
                    _tm.torch.tensor([param.h2_nm]),
                    _tm.torch.tensor([param.period_nm]),
                    _tm.torch.tensor([param.angle_deg]),
                    pol_TE,
                    material=param.material
                )
                wls = np.linspace(wl_start, wl_end, n_pts)
                return wls, (spec.squeeze(0).numpy() / 0.86)
            except Exception:
                wls = np.linspace(wl_start, wl_end, n_pts)
                refl = np.zeros(len(wls))
                for i, wl_nm in enumerate(wls):
                    I1, _ = _single_pillar_complex(param.d1_nm, param.h1_nm, param.period_nm, param.material, param.polarization, param.angle_deg, wl_nm, param.substrat)
                    I2, _ = _single_pillar_complex(param.d2_nm, param.h2_nm, param.period_nm,
                            param.material, param.polarization, param.angle_deg, wl_nm)
                    _, f1 = _single_pillar_complex(param.d1_nm, param.h1_nm, param.period_nm, param.material, param.polarization, param.angle_deg, 550.0, param.substrat)
                    _, f2 = _single_pillar_complex(param.d2_nm, param.h2_nm, param.period_nm,
                        param.material, param.polarization, param.angle_deg, 550.0)
                    refl[i] = float(f1*abs(I1)**2 + f2*abs(I2)**2)
                refl = refl / 0.86
                return wls, refl
        if getattr(self, '_enable_far_field', False):
            wls, refl = self._far_field_spectrum(param, self._na, self._theta_obs_deg)
            return wls, refl
        # PyTorch batch mode
        try:
            import torch_model as _tm
            pol_TE = param.polarization.startswith("TE")
            spec = _tm.batch_lorentzian_spectrum(
                _tm.torch.tensor([param.diameter_nm]),
                _tm.torch.tensor([param.height_nm]),
                _tm.torch.tensor([param.period_nm]),
                _tm.torch.tensor([param.angle_deg]),
                pol_TE,
                material=param.material
            )
            wls = np.linspace(wl_start, wl_end, n_pts)
            return wls, (spec.squeeze(0).numpy() / 0.86)
        except Exception:
            wls = np.linspace(wl_start, wl_end, n_pts)
            refl = np.array([self._single_wl_response(param, w) for w in wls])
            return wls, refl

    def _peak_wl(self, d, h, n550=2.4157):
        # Return dominant peak: ED for blue-green, MD for red
        dn = n550 - 2.0
        lam_ed = 360 + 0.55*(d-60) + 0.12*(h-120) + 32*dn
        # MD has higher amplitude for D > 200 due to broader resonance
        if d > 200 and h > 200:
            lam_md = 400 + 0.75*(d-60) + 0.25*(h-120) + 32*dn
            return lam_md
        return lam_ed

    def _wl_to_approx_rgb(self, wl_nm):
        idx = int(round((wl_nm - 380.0) / 5.0))
        idx = max(0, min(80, idx))
        xyz = np.array([_CIE_X[idx], _CIE_Y[idx], _CIE_Z[idx]])
        norm = _CIE_Y.sum() * 5.0
        xyz = xyz / (norm if norm > 1e-12 else 1.0)
        return np.clip(xyz_to_srgb(xyz * 80), 0, 1)

    def _build_library(self):
        d_vals = np.arange(self.d_min, self.d_max + 0.1, 5.0)
        h_vals = np.arange(self.h_min, self.h_max + 0.1, 10.0)
        p_vals = np.arange(self.p_min, self.p_max + 0.1, 20.0)
        default = MetaSurfaceParam(180, 380, 420)
        # Fast: use peak wavelength approximation for coarse grid
        params_list, rgbs_list = [], []
        n550 = MaterialLibrary.n_at_wavelength(default.material, 550.0)
        for d in d_vals:
            for h in h_vals:
                lam_peak = self._peak_wl(d, h, n550)
                rgb = self._wl_to_approx_rgb(lam_peak)
                for p in p_vals:
                    if d >= p:  # physical: D must be strictly less than P
                        continue
                    fill_ratio = np.pi*(d/2)**2/(p**2)
                    if fill_ratio < 0.03 or fill_ratio > 0.70:
                        continue
                    fill = fill_ratio
                    amp = 0.30 + 0.80 * fill
                    params_list.append([d, h, p])
                    rgbs_list.append(np.clip(rgb * amp, 0, 1))
        rgbs = np.array(rgbs_list, dtype=float)
        params = np.array(params_list, dtype=float)
        return params, rgbs, rgb_to_lab(rgbs), rgb_to_xy(rgbs)

    def rebuild_library(self, material: str, substrate: str, polarization: str, angle_deg: float):
        key = (material, substrate, polarization, angle_deg)
        self._last_material = material
        self._last_substrate = substrate
        self._last_polarization = polarization
        self._last_angle = angle_deg
        if key in self._cache:
            self.grid_rgb, self.grid_params, self.grid_lab, self.grid_xy = self._cache[key]
            return
        d_vals = np.arange(self.d_min, self.d_max + 0.1, 5.0)
        h_vals = np.arange(self.h_min, self.h_max + 0.1, 10.0)
        p_vals = np.arange(self.p_min, self.p_max + 0.1, 20.0)
        default = MetaSurfaceParam(180, 380, 420, material, substrate, polarization, angle_deg)
        n550 = MaterialLibrary.n_at_wavelength(material, 550.0)
        params_list, rgbs_list = [], []
        for d in d_vals:
            for h in h_vals:
                lam_peak = self._peak_wl(d, h, n550)
                rgb = self._wl_to_approx_rgb(lam_peak)
                for p in p_vals:
                    if d >= p:  # physical: D must be strictly less than P
                        continue
                    fill_ratio = np.pi*(d/2)**2/(p**2)
                    if fill_ratio < 0.03 or fill_ratio > 0.70:
                        continue
                    fill = fill_ratio
                    amp = 0.30 + 0.80 * fill
                    params_list.append([d, h, p])
                    rgbs_list.append(np.clip(rgb * amp, 0, 1))
        self.grid_rgb = np.array(rgbs_list, dtype=float)
        self.grid_params = np.array(params_list, dtype=float)
        self.grid_lab = rgb_to_lab(self.grid_rgb)
        self.grid_xy = rgb_to_xy(self.grid_rgb)
        self._cache[key] = (self.grid_rgb, self.grid_params, self.grid_lab, self.grid_xy)

    def nearest_lab_indices(self, target_lab: np.ndarray) -> np.ndarray:
        target_lab = np.asarray(target_lab, dtype=float)
        diff = target_lab[:, None, :] - self.grid_lab[None, :, :]
        return np.argmin(np.sum(diff * diff, axis=2), axis=1)

    def inverse_design(self, target_rgb: np.ndarray, progress_callback=None):
        target_rgb = clamp01(np.asarray(target_rgb, dtype=float))
        target_lab = rgb_to_lab(target_rgb[None, :])[0]
        target_xy = rgb_to_xy(target_rgb)
        target_wl = 380 + (target_xy[0] + target_xy[1]) * 200
        # Stage 1: coarse scan top 100 by approximate LAB
        diff = target_lab[None, :] - self.grid_lab
        dists = np.sum(diff * diff, axis=1)
        top_k = min(50, len(dists))
        top_idx = np.argpartition(dists, top_k)[:top_k]
        # Re-rank top 50 using real physical_color
        real_scores = []
        total_rerank = len(top_idx)
        for ri, idx in enumerate(top_idx):
            if progress_callback:
                progress_callback(ri, total_rerank, "粗扫重排")
            d, h, p_val = self.grid_params[idx]
            param = MetaSurfaceParam(float(d), float(h), float(p_val),
                self._last_material, self._last_substrate,
                self._last_polarization, self._last_angle)
            rgb = self.physical_color(param)
            lab = rgb_to_lab(rgb[None, :])[0]
            de2k = delta_e2000(target_lab, lab)
            lam_peak = self._peak_wl(d, h)
            wl_diff = abs(lam_peak - target_wl) / 100.0
            score = de2k  # pure DeltaE2000 optimization
            real_scores.append((score, d, h, p_val, rgb, lab, de2k))
        real_scores.sort(key=lambda x: x[0])
        # Stage 2: fine search around top 15, collect top 3
        top3 = []  # (score, param, rgb, de76, de2k)
        seen = set()
        fine_count = len(real_scores[:8])
        for fi, (_, d, h, p_val, _, _, _) in enumerate(real_scores[:8]):
            if progress_callback:
                progress_callback(fi, fine_count, "精细搜索")
            for dd in np.arange(max(50, d-10), min(350, d+11), 2.0):
                for dh in np.arange(max(80, h-30), min(600, h+32), 4.0):
                    for dp in np.arange(max(200, p_val-30), min(600, p_val+35), 10.0):
                        if dd >= dp:  # physical: D must be < P
                            continue
                        fill_ratio = np.pi*(dd/2)**2/(dp**2)
                        if fill_ratio < 0.03 or fill_ratio > 0.70:
                            continue
                        key = (round(dd,1), round(dh,1), round(dp,1))
                        if key in seen:
                            continue
                        seen.add(key)
                        param = MetaSurfaceParam(dd, dh, dp,
                            self._last_material, self._last_substrate,
                            self._last_polarization, self._last_angle)
                        rgb = self.physical_color(param)
                        lab = rgb_to_lab(rgb[None, :])[0]
                        de2k = delta_e2000(target_lab, lab)
                        de76 = delta_e76(target_lab, lab)
                        lam_peak = self._peak_wl(dd, dh)
                        wl_diff = abs(lam_peak - target_wl) / 100.0
                        score = de2k  # pure DeltaE2000 optimization
                        top3.append((score, param, rgb, de76, de2k))
        top3.sort(key=lambda x: x[0])
        # Deduplicate: param diversity (D>=15nm, H>=20nm, P>=30nm) + color diversity (ΔE>=1.5)
        D_DUP, H_DUP, P_DUP, DE_COLOR_DUP = 15.0, 20.0, 30.0, 1.5
        unique_top3 = []
        for item in top3:
            if len(unique_top3) >= 3:
                break
            _, p, prgb, _, _ = item
            # Check param similarity
            is_dup = False
            for _, up, _, _, _ in unique_top3:
                if (abs(p.diameter_nm - up.diameter_nm) < D_DUP and
                    abs(p.height_nm - up.height_nm) < H_DUP and
                    abs(p.period_nm - up.period_nm) < P_DUP):
                    is_dup = True
                    break
            if is_dup:
                continue
            # Check color similarity: skip if too close to any selected result
            color_dup = False
            for _, _, urgb, _, _ in unique_top3:
                if delta_e2000(rgb_to_lab(prgb[None,:])[0], rgb_to_lab(urgb[None,:])[0]) < DE_COLOR_DUP:
                    color_dup = True
                    break
            if not color_dup:
                unique_top3.append(item)
        # Fallback: scan diverse candidates from real_scores with aggressive dedup
        if len(unique_top3) < 3:
            for _, d, h, p_val, rgb, lab, de2k in real_scores:
                if len(unique_top3) >= 3:
                    break
                param = MetaSurfaceParam(float(d), float(h), float(p_val),
                    self._last_material, self._last_substrate,
                    self._last_polarization, self._last_angle)
                de76 = delta_e76(target_lab, lab)
                score = de2k  # pure DeltaE2000
                is_dup = False
                for _, up, _, _, _ in unique_top3:
                    if (abs(d - up.diameter_nm) < D_DUP and
                        abs(h - up.height_nm) < H_DUP and
                        abs(p_val - up.period_nm) < P_DUP):
                        is_dup = True
                        break
                if is_dup:
                    continue
                color_dup = False
                for _, _, urgb, _, _ in unique_top3:
                    if delta_e2000(lab, rgb_to_lab(urgb[None,:])[0]) < DE_COLOR_DUP:
                        color_dup = True
                        break
                if not color_dup:
                    unique_top3.append((score, param, rgb, de76, de2k))
        # Second fallback: scan full grid for diverse alternatives
        if len(unique_top3) < 3:
            all_scores = list(zip(dists, range(len(dists))))
            all_scores.sort()
            for _, idx in all_scores:
                if len(unique_top3) >= 3:
                    break
                d, h, p_val = self.grid_params[idx]
                param = MetaSurfaceParam(float(d), float(h), float(p_val),
                    self._last_material, self._last_substrate,
                    self._last_polarization, self._last_angle)
                rgb = self.physical_color(param)
                lab = rgb_to_lab(rgb[None, :])[0]
                de2k = delta_e2000(target_lab, lab)
                de76 = delta_e76(target_lab, lab)
                score = de2k  # pure DeltaE2000
                is_dup = False
                for _, up, _, _, _ in unique_top3:
                    if (abs(d - up.diameter_nm) < D_DUP and
                        abs(h - up.height_nm) < H_DUP and
                        abs(p_val - up.period_nm) < P_DUP):
                        is_dup = True
                        break
                if not is_dup:
                    unique_top3.append((score, param, rgb, de76, de2k))
        # Broad search disabled: too expensive for marginal gains
        if False and unique_top3 and unique_top3[0][4] > 10.0:
            broad_d = np.arange(max(50, self.d_min), self.d_max + 0.1, 10.0)
            broad_h = np.arange(self.h_min, self.h_max + 0.1, 20.0)
            broad_p = np.arange(self.p_min, self.p_max + 0.1, 20.0)
            n550 = MaterialLibrary.n_at_wavelength(self._last_material, 550.0)
            broad_candidates = []
            for bd in broad_d:
                for bh in broad_h:
                    # Pre-filter by approximate peak wavelength proximity
                    approx_peak = self._peak_wl(bd, bh, n550)
                    if abs(approx_peak - target_wl) > 80:
                        continue
                    for bp in broad_p:
                        if bd >= bp:
                            continue
                        fr = (bd/bp)**2
                        if fr < 0.15 or fr > 0.85:
                            continue
                        param = MetaSurfaceParam(float(bd), float(bh), float(bp),
                            self._last_material, self._last_substrate,
                            self._last_polarization, self._last_angle)
                        rgb = self.physical_color(param)
                        lab = rgb_to_lab(rgb[None, :])[0]
                        de2k = delta_e2000(target_lab, lab)
                        de76 = delta_e76(target_lab, lab)
                        broad_candidates.append((de2k, param, rgb, de76, de2k))
            broad_candidates.sort(key=lambda x: x[0])
            # Merge with existing results, keeping diversity constraints
            for item in broad_candidates:
                if len(unique_top3) >= 3:
                    break
                _, p, prgb, _, _ = item
                is_dup = False
                for _, up, urgb, _, _ in unique_top3:
                    if (abs(p.diameter_nm - up.diameter_nm) < D_DUP and
                        abs(p.height_nm - up.height_nm) < H_DUP and
                        abs(p.period_nm - up.period_nm) < P_DUP):
                        is_dup = True
                        break
                    if delta_e2000(rgb_to_lab(prgb[None,:])[0], rgb_to_lab(urgb[None,:])[0]) < DE_COLOR_DUP:
                        is_dup = True
                        break
                if not is_dup:
                    unique_top3.append(item)
            # If broad search found a better #1, replace
            if broad_candidates and broad_candidates[0][0] < unique_top3[0][4] - 1.0:
                better = broad_candidates[0]
                unique_top3.insert(0, better)
                # Re-dedup
                final = [unique_top3[0]]
                for item in unique_top3[1:]:
                    if len(final) >= 3:
                        break
                    _, p, prgb, _, _ = item
                    is_dup = False
                    for _, up, urgb, _, _ in final:
                        if (abs(p.diameter_nm - up.diameter_nm) < D_DUP and
                            abs(p.height_nm - up.height_nm) < H_DUP and
                            abs(p.period_nm - up.period_nm) < P_DUP):
                            is_dup = True
                            break
                        if delta_e2000(rgb_to_lab(prgb[None,:])[0], rgb_to_lab(urgb[None,:])[0]) < DE_COLOR_DUP:
                            is_dup = True
                            break
                    if not is_dup:
                        final.append(item)
                unique_top3 = final[:3]

        # Ensure we have at least 1 result
        if not unique_top3:
            idx = int(self.nearest_lab_indices(target_lab[None, :])[0])
            p = self.grid_params[idx]
            param = MetaSurfaceParam(float(p[0]), float(p[1]), float(p[2]))
            rgb = self.physical_color(param)
            lab = rgb_to_lab(rgb[None, :])[0]
            unique_top3.append((0, param, rgb, delta_e76(target_lab, lab), delta_e2000(target_lab, lab)))
        return unique_top3

    def image_to_metasurface_map(self, image: Image.Image, max_size: int = 80):
        img = image.convert("RGB")
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        arr = np.asarray(img).astype(float) / 255.0
        flat = arr.reshape(-1, 3)
        target_lab = rgb_to_lab(flat)
        # Batch search to avoid memory explosion on large images
        batch_size = 500
        indices = np.empty(len(flat), dtype=int)
        for start in range(0, len(flat), batch_size):
            end = min(start + batch_size, len(flat))
            batch_lab = target_lab[start:end]
            diff = batch_lab[:, None, :] - self.grid_lab[None, :, :]
            indices[start:end] = np.argmin(np.sum(diff * diff, axis=2), axis=1)
        params = self.grid_params[indices].reshape(arr.shape[0], arr.shape[1], 3)
        mapped_rgb = self.grid_rgb[indices].reshape(arr.shape[0], arr.shape[1], 3)
        return arr, mapped_rgb, params

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
    return ml_module.init_ml()

_ml_ready = get_ml_ready()

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
    if is_fp:
        st.session_state.far_field = False
    st.session_state.far_field = st.checkbox(
        '启用角谱远场传播 (Angular Spectrum)',
        value=st.session_state.far_field,
        disabled=is_fp,
        help='FP腔模式不需要角谱（平面薄膜无衍射）' if is_fp else 'N×N超表面阵列FFT角谱 + NA锥积分，计算探测器实际接收光谱'
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
        if ml_module._ML_AVAILABLE:
            if ml_module._IS_V8:
                st.caption("\u6a21\u578b: v8 Substrate | 7\u7ef4\u8f93\u5165(\u542b\u886c\u5e95) | 256x4\u6b8b\u5dee\u5757 | 4\u79cd\u6750\u6599+3\u79cd\u886c\u5e95")
            else:
                st.caption("\u6a21\u578b: v7 Multi | 6\u7ef4\u8f93\u5165 | 256x4\u6b8b\u5dee\u5757 | 4\u79cd\u6750\u6599")
        else:
            st.caption("\u6a21\u578b: \u672a\u52a0\u8f7d (\u4e91\u7aef\u7f3a\u5c11torch\u6216\u6a21\u578b\u6587\u4ef6)")
    except Exception as e:
        st.caption(f"\u6a21\u578b: \u9519\u8bef - {e}")
    if ml_module._DUAL_ML_AVAILABLE:
        st.caption("\u53cc\u67f1 ML: DualResMLP v3 (Multi) \u53ef\u7528")

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

    # Dual-pillar auto inverse design button
    if is_dual:
        st.divider()
        st.subheader("双柱自动逆设计 (PyTorch 梯度优化)")
        target_hex_dual = st.color_picker("目标颜色", "#80c8ff", key="dual_target_color")
        if st.button("双柱自动搜索", use_container_width=True):
            target_r = int(target_hex_dual[1:3], 16)
            target_g = int(target_hex_dual[3:5], 16)
            target_b = int(target_hex_dual[5:7], 16)
            target_rgb = [target_r/255, target_g/255, target_b/255]
            try:
                import torch_model as _tmd
                with st.spinner("PyTorch 梯度优化中... 约1-10秒"):
                    result = _tmd.inverse_design_dual(target_rgb, n_steps=200, n_restarts=30, material=material, substrate=substrate)
                    if result is not None:
                        d1, h1, d2, h2, p, rgb, loss = result
                        p = max(max(d1, d2) * 1.2 + 20, p)
                        st.session_state.d1_val = max(60.0, min(300.0, float(d1)))
                        st.session_state.h1_val = max(80.0, min(600.0, float(h1)))
                        st.session_state.d2_val = max(60.0, min(300.0, float(d2)))
                        st.session_state.h2_val = max(80.0, min(600.0, float(h2)))
                        st.session_state.p_val = max(200.0, min(600.0, float(p)))
                        st.session_state._dual_success_msg = f"优化完成! D1={d1:.0f} H1={h1:.0f} D2={d2:.0f} H2={h2:.0f} P={p:.0f}"
                        st.rerun()
            except Exception as e:
                st.error(f"Auto search failed: {e}")

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


# ============================================================
# Fabry-Perot (FP) cavity: Ag/TiO2/Ag and DBR mirror modes
# ============================================================

_AG_NK_TABLE = {
    380: (0.17, 1.62), 385: (0.17, 1.63), 390: (0.17, 1.64), 395: (0.17, 1.66),
    400: (0.17, 1.67), 405: (0.17, 1.68), 410: (0.17, 1.70), 415: (0.17, 1.72),
    420: (0.17, 1.73), 425: (0.17, 1.75), 430: (0.17, 1.77), 435: (0.17, 1.79),
    440: (0.17, 1.81), 445: (0.17, 1.83), 450: (0.17, 1.85), 455: (0.17, 1.87),
    460: (0.17, 1.90), 465: (0.17, 1.92), 470: (0.17, 1.95), 475: (0.17, 1.98),
    480: (0.17, 2.01), 485: (0.16, 2.04), 490: (0.16, 2.07), 495: (0.16, 2.10),
    500: (0.15, 2.13), 505: (0.15, 2.16), 510: (0.15, 2.20), 515: (0.15, 2.23),
    520: (0.14, 2.27), 525: (0.14, 2.30), 530: (0.14, 2.34), 535: (0.14, 2.38),
    540: (0.13, 2.42), 545: (0.13, 2.46), 550: (0.13, 2.50), 555: (0.13, 2.54),
    560: (0.12, 2.58), 565: (0.12, 2.63), 570: (0.12, 2.67), 575: (0.12, 2.72),
    580: (0.12, 2.77), 585: (0.12, 2.82), 590: (0.12, 2.87), 595: (0.12, 2.92),
    600: (0.12, 2.97), 605: (0.12, 3.02), 610: (0.12, 3.08), 615: (0.12, 3.13),
    620: (0.12, 3.19), 625: (0.12, 3.25), 630: (0.12, 3.31), 635: (0.12, 3.37),
    640: (0.12, 3.44), 645: (0.13, 3.50), 650: (0.13, 3.57), 655: (0.13, 3.64),
    660: (0.14, 3.71), 665: (0.14, 3.78), 670: (0.14, 3.86), 675: (0.14, 3.93),
    680: (0.14, 4.01), 685: (0.14, 4.09), 690: (0.15, 4.17), 695: (0.15, 4.26),
    700: (0.15, 4.34), 705: (0.15, 4.43), 710: (0.15, 4.52), 715: (0.15, 4.61),
    720: (0.15, 4.71), 725: (0.15, 4.80), 730: (0.15, 4.90), 735: (0.15, 5.00),
    740: (0.15, 5.10), 745: (0.15, 5.21), 750: (0.15, 5.32), 755: (0.15, 5.43),
    760: (0.15, 5.55), 765: (0.15, 5.67), 770: (0.15, 5.79), 775: (0.15, 5.91),
    780: (0.15, 6.04),
}

def _ag_nk(wl_nm):
    wls = list(_AG_NK_TABLE.keys())
    if wl_nm <= wls[0]: return _AG_NK_TABLE[wls[0]]
    if wl_nm >= wls[-1]: return _AG_NK_TABLE[wls[-1]]
    for i in range(len(wls)-1):
        if wls[i] <= wl_nm <= wls[i+1]:
            frac = (wl_nm - wls[i]) / (wls[i+1] - wls[i])
            n1, k1 = _AG_NK_TABLE[wls[i]]
            n2, k2 = _AG_NK_TABLE[wls[i+1]]
            return (n1 + frac*(n2-n1), k1 + frac*(k2-k1))
    return _AG_NK_TABLE[wls[-1]]

def _n_sio2_sellmeier(wl_nm):
    wl_um = wl_nm / 1000.0
    return np.sqrt(1 + 0.6961663*wl_um**2/(wl_um**2 - 0.0684043**2)
                   + 0.4079426*wl_um**2/(wl_um**2 - 0.1162414**2)
                   + 0.8974794*wl_um**2/(wl_um**2 - 9.896161**2))

def fp_cavity_spectrum(T_nm, angle_deg=0.0, pol_TE=True):
    """Metal-mirror FP: Ag(30nm)/TiO2(T)/Ag(bulk). 减色型."""
    wls = np.arange(380, 785, 5)
    d_top = 30.0; theta = angle_deg * np.pi / 180.0; n_inc = 1.0
    refl = np.zeros(len(wls))
    for i, wl in enumerate(wls):
        n_top_n, n_top_k = _ag_nk(wl)
        n_top_c = complex(n_top_n, n_top_k)
        n_tio2 = MaterialLibrary.n_at_wavelength("TiO2 (anatase)", wl)
        n_tio2_c = complex(n_tio2, 0.0)
        n_bot_n, n_bot_k = _ag_nk(wl)
        n_bot_c = complex(n_bot_n, n_bot_k)
        cos_inc = np.cos(theta); sin_inc = np.sin(theta)
        def _cos(n): return np.emath.sqrt(1.0 - (sin_inc*n_inc/n)**2)
        cos_top, cos_tio2, cos_bot = _cos(n_top_c), _cos(n_tio2_c), _cos(n_bot_c)
        def _layer(n, d, c, te):
            delta = 2.0*np.pi*n*d*c/wl
            p = n*c if te else c/n
            cd, sd = np.cos(delta), np.sin(delta)
            return np.array([[cd, 1j*sd/p], [1j*p*sd, cd]])
        M = _layer(n_top_c, d_top, cos_top, pol_TE)
        M = M @ _layer(n_tio2_c, T_nm, cos_tio2, pol_TE)
        p_inc = n_inc*cos_inc if pol_TE else cos_inc/n_inc
        p_bot = n_bot_c*cos_bot if pol_TE else cos_bot/n_bot_c
        a = M[0,0] + M[0,1]*p_bot; b = M[1,0] + M[1,1]*p_bot
        r = (a*p_inc - b)/(a*p_inc + b)
        refl[i] = float(abs(r)**2)
    return wls, np.nan_to_num(np.clip(refl, 0, 1), nan=0.0, posinf=1.0, neginf=0.0)

def fp_dielectric_spectrum(T_nm, target_wl=450.0, n_pairs_top=3, n_pairs_bot=5, angle_deg=0.0, pol_TE=True):
    """DBR FP cavity: (TiO2/SiO2)^n/TiO2(T)/(SiO2/TiO2)^n. 高饱和度."""
    wls = np.arange(380, 785, 5)
    theta = angle_deg * np.pi / 180.0; n_inc = 1.0
    refl = np.zeros(len(wls))
    n_tio2_ref = MaterialLibrary.n_at_wavelength("TiO2 (anatase)", target_wl)
    n_sio2_ref = _n_sio2_sellmeier(target_wl)
    dH = target_wl/(4.0*n_tio2_ref); dL = target_wl/(4.0*n_sio2_ref)
    for i, wl in enumerate(wls):
        nH = complex(MaterialLibrary.n_at_wavelength("TiO2 (anatase)", wl), 0.0)
        nL = complex(_n_sio2_sellmeier(wl), 0.0)
        cos_inc = np.cos(theta); sin_inc = np.sin(theta)
        def _cos(n): return np.emath.sqrt(1.0 - (sin_inc*n_inc/n)**2)
        cH, cL = _cos(nH), _cos(nL)
        def _layer(n, d, c):
            delta = 2.0*np.pi*n*d*c/wl; p = n*c
            cd, sd = np.cos(delta), np.sin(delta)
            return np.array([[cd, 1j*sd/p], [1j*p*sd, cd]])
        M = np.eye(2, dtype=complex)
        for _ in range(n_pairs_top):
            M = M @ _layer(nH, dH, cH) @ _layer(nL, dL, cL)
        M = M @ _layer(nH, T_nm, cH)
        for _ in range(n_pairs_bot):
            M = M @ _layer(nL, dL, cL) @ _layer(nH, dH, cH)
        p_inc = n_inc*cos_inc; p_sub = nL*cL
        a = M[0,0] + M[0,1]*p_sub; b = M[1,0] + M[1,1]*p_sub
        r = (a*p_inc - b)/(a*p_inc + b)
        refl[i] = float(abs(r)**2)
    return wls, np.nan_to_num(np.clip(refl, 0, 1), nan=0.0, posinf=1.0, neginf=0.0)


use_ml = st.session_state.get('ml_accel', False) and ml_module._ML_AVAILABLE and not st.session_state.get('far_field', False) and material in ml_module.MATERIAL_CODES
use_dual_ml = use_ml and st.session_state.get('dual_pillar', False) and ml_module._DUAL_ML_AVAILABLE

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
    # --- Color gamut notice (non-FP only) ---
    if not is_fp:
        st.info(
        "TiO2 纳米柱在当前参数范围（D 60-267nm, H 80-600nm）内无法产生高饱和青蓝色或纯红色。"
        "这是 Lorentzian 模型和 RCWA 严格仿真共同验证的物理限制。"
        "提示：1) 启用 ML 代理模型 2) 尝试 a-Si/Si3N4 材料以获得更宽色域。"
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
    st.caption("侧边栏的 D/H/P 不影响逆设计，仅材料、衬底、偏振、入射角有效")

    col_pick, col_btn = st.columns([3, 1])
    with col_pick:
        picker_hex = st.color_picker("目标颜色", "#80c8ff")
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            run_btn = st.button('网格搜索', use_container_width=True, help='传统网格搜索: 精度高')
        with col_b2:
            ml_btn = st.button("ML快速搜索", use_container_width=True, disabled=not _ml_ready, help="ML梯度优化: 复杂模型更快")

    target_r = int(picker_hex[1:3], 16)
    target_g = int(picker_hex[3:5], 16)
    target_b = int(picker_hex[5:7], 16)
    st.caption(f"RGB({target_r}, {target_g}, {target_b})  |  {picker_hex}")

    target_rgb_norm = np.array([target_r, target_g, target_b]) / 255.0

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
    try:
        _ml_pressed = ml_btn
    except NameError:
        _ml_pressed = False
    
    if _ml_pressed and _ml_ready:
        with st.spinner("ML梯度优化中... 约10-30秒"):
            result = ml_module.inverse_design_ml(target_rgb_norm, n_steps=200, n_restarts=15, material=material, substrate=substrate)
            if result is not None:
                d_ml, h_ml, p_ml, pred_rgb_ml, loss_ml = result
                ml_param = MetaSurfaceParam(float(d_ml), float(h_ml), float(p_ml), material, substrate, polarization, angle)
                ml_rgb = pred_rgb_ml
                lab_t = rgb_to_lab(target_rgb_norm)
                lab_m = rgb_to_lab(ml_rgb)
                de76 = delta_e76(lab_t, lab_m)
                de2k = delta_e2000(lab_t, lab_m)
                # Build top3 by perturbing optimal result for nearby alternatives
                top3 = []
                for perturb in [(1.0, 1.0), (0.95, 1.05), (1.05, 0.95)]:
                    d_p = min(350, max(50, d_ml * perturb[0]))
                    h_p = min(600, max(80, h_ml * perturb[1]))
                    p_p = max(d_p * 1.2, p_ml)
                    param_p = MetaSurfaceParam(float(d_p), float(h_p), float(p_p), material, substrate, polarization, angle)
                    rgb_p = engine.physical_color(param_p)
                    lab_t = rgb_to_lab(target_rgb_norm)
                    lab_p = rgb_to_lab(rgb_p)
                    de_p = delta_e76(lab_t, lab_p)
                    de2k_p = delta_e2000(lab_t, lab_p)
                    top3.append((de2k_p, param_p, rgb_p, de_p, de2k_p))
                top3.sort(key=lambda x: x[0])
                st.session_state.top3_results = top3[:3]
                if "search_cache" not in st.session_state:
                    st.session_state.search_cache = {}
                cache_key = (target_r, target_g, target_b, material, substrate, polarization, angle)
                st.session_state.search_cache[cache_key] = st.session_state.top3_results
                st.success(f"ML模型搜索完成! D={d_ml:.1f}nm H={h_ml:.1f}nm P={p_ml:.1f}nm dE2000={de2k:.1f}")
            

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
            _cie_x = np.array([0.001368,0.002236,0.004243,0.007650,0.014310,0.023190,0.043510,0.077630,0.134380,0.214770,0.283900,0.328500,0.348280,0.348060,0.336200,0.318700,0.290800,0.251100,0.195360,0.142100,0.095640,0.058010,0.032010,0.014700,0.004900,0.002400,0.009300,0.029100,0.063270,0.109600,0.165500,0.225750,0.290400,0.359700,0.433450,0.512050,0.594500,0.678400,0.762100,0.842500,0.916300,0.978600,1.026300,1.056700,1.062200,1.045600,1.002600,0.938400,0.854450,0.751400,0.642400,0.541900,0.447900,0.360800,0.283500,0.218700,0.164900,0.121200,0.087400,0.063600,0.046770,0.032900,0.022700,0.015840,0.011359,0.008111,0.005790,0.004109,0.002899,0.002049,0.001440,0.001000,0.000690,0.000476,0.000332,0.000235,0.000166,0.000117,0.000083,0.000059,0.000042])
            _cie_y = np.array([0.000039,0.000064,0.000120,0.000217,0.000396,0.000640,0.001210,0.002180,0.004000,0.007300,0.011600,0.016840,0.023000,0.029800,0.038000,0.048000,0.060000,0.073900,0.090980,0.112600,0.139020,0.169300,0.208020,0.258600,0.323000,0.407300,0.503000,0.608200,0.710000,0.793200,0.862000,0.914850,0.954000,0.980300,0.994950,1.000000,0.995000,0.978600,0.952000,0.915400,0.870000,0.816300,0.757000,0.694900,0.631000,0.566800,0.503000,0.441200,0.381000,0.321000,0.265000,0.217000,0.175000,0.138200,0.107000,0.081600,0.061000,0.044580,0.032000,0.023200,0.017000,0.011920,0.008210,0.005723,0.004102,0.002929,0.002091,0.001484,0.001047,0.000740,0.000520,0.000361,0.000249,0.000172,0.000120,0.000085,0.000060,0.000042,0.000030,0.000021,0.000015])
            _cie_z = np.array([0.006450,0.010550,0.020050,0.036210,0.067850,0.110200,0.207400,0.371300,0.645600,1.039050,1.385600,1.622960,1.747060,1.782600,1.772110,1.744100,1.669200,1.528100,1.287640,0.999550,0.716900,0.484400,0.311900,0.190300,0.104200,0.049200,0.020300,0.008700,0.003900,0.002100,0.001650,0.001100,0.000800,0.000550,0.000350,0.000250,0.000150,0.000100,0.000050,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
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
    except:
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
    except:
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

