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
    import matplotlib.font_manager as fm
    _CONFIGURED = getattr(_get_plt, '_cfg', False)
    if not _CONFIGURED:
        # Rebuild font cache (picks up newly installed fonts)
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
        _get_plt._cfg = True
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
        dn = n - 2.0

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

    def compute_spectrum(self, param, wl_start=380.0, wl_end=780.0, n_pts=81):
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
        indices = self.nearest_lab_indices(target_lab)
        params = self.grid_params[indices].reshape(arr.shape[0], arr.shape[1], 3)
        mapped_rgb = self.grid_rgb[indices].reshape(arr.shape[0], arr.shape[1], 3)
        return arr, mapped_rgb, params

# ===================== Streamlit UI =====================
@st.cache_resource
def get_engine(_cache_key="v15_progress"):
    return MetaSurfaceColorEngine()

try:
    engine = get_engine()
except Exception as e:
    st.error(f"Engine init failed: {e}")
    import traceback; st.code(traceback.format_exc())
    st.stop()

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
    st.header('📏 纳米柱尺寸')

    if 'd_val' not in st.session_state:
        st.session_state.d_val = 180.0
    if 'h_val' not in st.session_state:
        st.session_state.h_val = 300.0
    if 'p_val' not in st.session_state:
        st.session_state.p_val = 400.0

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
param = MetaSurfaceParam(diameter, height, period, material, substrate, polarization, angle)
rgb = engine.physical_color(param)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔬 实时预览", "🎯 逆设计", "🖼️ 图案生成",
    "📊 颜色映射", "🌈 光谱"
])

# Tab 1: 实时预览
with tab1:
    hex_color = rgb_to_hex(rgb)
    r255, g255, b255 = rgb_255(rgb)

    # --- Color swatch card ---
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:24px;padding:20px;
                background:linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                border-radius:16px;margin-bottom:20px;">
      <div style="width:220px;height:220px;background:{hex_color};
                  border-radius:16px;box-shadow:0 8px 32px {hex_color}66,
                  inset 0 1px 0 rgba(255,255,255,0.3);flex-shrink:0;"></div>
      <div style="color:#e0e0e0;">
        <div style="font-size:36px;font-weight:700;margin-bottom:6px;">{hex_color}</div>
        <div style="font-size:18px;opacity:0.85;">RGB({r255}, {g255}, {b255})</div>
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
        run_btn = st.button("🔍 搜索匹配", use_container_width=True, help="全参数空间扫描约需10-30秒，请耐心等待")

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
        cols_h = st.columns([1, 1, 2, 4, 2])
        cols_h[0].caption("目标")
        cols_h[1].caption("匹配")
        cols_h[2].caption("参数")
        cols_h[3].caption("")
        cols_h[4].caption("ΔE2000")
        for h in st.session_state.search_history:
            c0, c1, c2, c3, c4 = st.columns([1, 1, 2, 4, 2])
            with c0:
                st.markdown(f'<div style="width:24px;height:24px;background:{h["target_hex"]};border-radius:4px;border:1px solid #fff3;"></div>', unsafe_allow_html=True)
            with c1:
                st.markdown(f'<div style="width:24px;height:24px;background:{h["matched_hex"]};border-radius:4px;border:1px solid #fff3;"></div>', unsafe_allow_html=True)
            with c2:
                st.caption(f"D={h['D']:.0f} H={h['H']:.0f} P={h['P']:.0f}")
            with c3:
                st.caption(f"{h['target_hex']} → {h['matched_hex']}")
            with c4:
                st.caption(f"{h['dE']:.1f}")

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

    for d in d_sample:
        rows_html += f'<tr><td style="padding:4px 8px;color:#888;font-size:11px;font-weight:600;">{d:.0f}</td>'
        for h in h_sample:
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
st.sidebar.caption("AI超表面结构色设计 v3.0")
st.sidebar.caption("物理模型: Lorentzian 共振 + CIE 1931 光谱管线")
