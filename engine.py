# engine.py - Core metasurface color engine
# Extracted from app.py for modularity.
from __future__ import annotations
import numpy as np; import logging
from PIL import Image
from dataclasses import dataclass
from typing import List
from color_utils import (
    CIE_X as _CIE_X, CIE_Y as _CIE_Y, CIE_Z as _CIE_Z,
    spectrum_to_srgb, clamp01, rgb_to_lab, rgb_to_xy,
    xyz_to_srgb, delta_e76, delta_e2000,
)
import os, hashlib, pickle as _pickle
_GRID_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_grid_cache")
_GRID_CACHE_VERSION = "v1"
os.makedirs(_GRID_CACHE_DIR, exist_ok=True)


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
    """Dual-pillar meta-atom parameters with auto-clamping."""

    # Parameter correction priority:
    #  1. P >= max(D1,D2)*1.2, ensure both pillars fit in unit cell
    #  2. D <= P*0.8 to limit fill factor
    #  3. Total fill (fill1+fill2) <= 0.85, prevent D1/D2 overlap

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

# Priority: P >= max(D1,D2)*1.2
        min_p = max(self.d1_nm, self.d2_nm) * 1.20
        if self.period_nm < min_p:
            object.__setattr__(self, 'period_nm', max(min_p, 380.0))
        if self.period_nm < max(self.d1_nm, self.d2_nm):
            object.__setattr__(self, 'period_nm', max(self.d1_nm, self.d2_nm) + 20.0)
        if self.period_nm != orig_p:
            msgs.append(f"P {orig_p:.0f}->{self.period_nm:.0f}nm")

# Priority: D <= P*0.8
        max_d = self.period_nm * 0.80
        if self.d1_nm > max_d:
            object.__setattr__(self, 'd1_nm', max_d)
            msgs.append(f"D1 {orig_d1:.0f}->{self.d1_nm:.0f}nm")
        if self.d2_nm > max_d:
            object.__setattr__(self, 'd2_nm', max_d)
            msgs.append(f"D2 {orig_d2:.0f}->{self.d2_nm:.0f}nm")

# Priority: total fill ratio <= 0.85
        fill1 = np.pi*(self.d1_nm/2)**2/(self.period_nm**2)
        fill2 = np.pi*(self.d2_nm/2)**2/(self.period_nm**2)
        if fill1 + fill2 > 0.85:
            scale = np.sqrt(0.80 / (fill1 + fill2))
            new_d1 = self.d1_nm * scale
            new_d2 = self.d2_nm * scale
            msgs.append(f"D1 {self.d1_nm:.0f}->{new_d1:.0f} D2 {self.d2_nm:.0f}->{new_d2:.0f}nm (fill ratio too high)")
            object.__setattr__(self, "d1_nm", new_d1)
            object.__setattr__(self, "d2_nm", new_d2)

        if msgs:
            object.__setattr__(self, '_corrected', True)
            object.__setattr__(self, '_correction_msg', "; ".join(msgs))

def _single_pillar_complex(d_nm, h_nm, p_nm, material, polarization, angle_deg, wl_nm, substrate=None):
    """Compute complex reflection coefficient for a single nanopillar. Shared by single-pillar and dual-pillar models."""


    n_mat = MaterialLibrary.n_at_wavelength(material, wl_nm)
    n_sub = MaterialLibrary.n_at_wavelength(substrate, wl_nm) if substrate else 1.458
    n_env = (1.0 + n_sub) / 2.0
    dn = n_mat - n_env

    lam_ed = 360 + 0.55*(d_nm-60) + 0.12*(h_nm-120) + 32*dn
    sigma_ed = max(26 + 0.10*(d_nm-200), 8)  # FDTD-calibrated
    lam_md = 400 + 0.75*(d_nm-60) + 0.25*(h_nm-120) + 32*dn
    sigma_md = max(35 + 0.12*(d_nm-200), 10)  # FDTD-calibrated

    p_safe = max(p_nm, 200.0)
    fill = np.clip(np.pi*(d_nm/2)**2/(p_safe**2), 0.01, 0.70)
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
            default_key = (self._last_material, self._last_substrate, self._last_polarization, self._last_angle)
            # Try loading from on-disk cache first (survives app restarts)
            loaded = self._load_grid_disk(default_key)
            if loaded is not None:
                self.grid_params, self.grid_rgb, self.grid_lab, self.grid_xy = loaded
            else:
                self.grid_params, self.grid_rgb, self.grid_lab, self.grid_xy = self._build_library()
                self._save_grid_disk(default_key, (self.grid_params, self.grid_rgb, self.grid_lab, self.grid_xy))
        except Exception as e:
            import traceback
            logging.error(f"Library build failed: {e}")
            logging.error(traceback.format_exc())
            # Fallback: empty library
            self.grid_params = np.zeros((0, 3))
            self.grid_rgb = np.zeros((0, 3))
            self.grid_lab = np.zeros((0, 3))
            self.grid_xy = np.zeros((0, 2))

    def physical_color(self, param) -> np.ndarray:
        """Compute sRGB color from MetaSurfaceParam or DualPillarParam."""
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
                        material=param.material,
                        substrate=param.substrate
                    )
                    rgb = _tm.batch_spectrum_to_rgb(spec)
                    return np.nan_to_num(rgb.squeeze(0).numpy(), nan=0.5)
                except Exception as e:
                    logging.warning(f"engine fallback: {e}")
                    wls = np.arange(380, 785, 5)
                    refl = np.zeros(len(wls))
                    for i, wl_nm in enumerate(wls):
                        I1, _ = _single_pillar_complex(param.d1_nm, param.h1_nm, param.period_nm,
                            param.material, param.polarization, param.angle_deg, wl_nm, param.substrate)
                        I2, _ = _single_pillar_complex(param.d2_nm, param.h2_nm, param.period_nm,
                            param.material, param.polarization, param.angle_deg, wl_nm, param.substrate)
                        refl[i] = float(abs(I1)**2 + abs(I2)**2)
                    refl = refl / 0.86
                    return np.nan_to_num(spectrum_to_srgb(wls, refl), nan=0.5)
        # Single pillar (MetaSurfaceParam) — far-field
        if getattr(self, '_enable_far_field', False):
            wls, refl = self._far_field_spectrum(param, self._na, self._theta_obs_deg)
            return np.nan_to_num(spectrum_to_srgb(wls, refl), nan=0.5)



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
            return np.nan_to_num(rgb.squeeze(0).numpy(), nan=0.5)
        except Exception as e:
            logging.warning(f"engine fallback: {e}")
            wls = np.arange(380, 785, 5)
            refl = np.array([self._single_wl_response(param, wl) for wl in wls])
            refl = refl / 0.86
            return np.nan_to_num(spectrum_to_srgb(wls, refl), nan=0.5)



    def _batch_compute_rgb(self, d_arr, h_arr, p_arr):
        """Batch compute RGB from arrays of (D, H, P) using torch_model."""
        import torch_model as _tm
        _t = _tm.torch.tensor
        pol_TE = self._last_polarization.startswith("TE")
        spec = _tm.batch_lorentzian_spectrum(
            _t(d_arr, dtype=_tm.torch.float32),
            _t(h_arr, dtype=_tm.torch.float32),
            _t(p_arr, dtype=_tm.torch.float32),
            _t([self._last_angle] * len(d_arr), dtype=_tm.torch.float32),
            pol_TE,
            material=self._last_material, substrate=self._last_substrate
        )
        rgb = _tm.batch_spectrum_to_rgb(spec)
        return np.nan_to_num(rgb.numpy(), nan=0.5)

    def _single_wl_response(self, param, wl_nm):
        d, h, p = param.diameter_nm, param.height_nm, param.period_nm
        n_mat = MaterialLibrary.n_at_wavelength(param.material, wl_nm)
        n_sub = MaterialLibrary.n_at_wavelength(param.substrate, wl_nm) if param.substrate else 1.458
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
        d_s = max(getattr(param, "diameter_nm", 180), 10.0)
        h_s = max(getattr(param, "height_nm", 300), 10.0)
        p_s = max(getattr(param, "period_nm", 400), 100.0)
        r, _ = _single_pillar_complex(
            d_s, h_s, p_s,
            param.material, param.polarization, param.angle_deg, wl_nm)
        return r

    def _far_field_spectrum(self, param, na=0.1, theta_obs_deg=0.0, N=12):
        """Far-field propagation: Gaussian NA + Lorentzian angular response."""

        # Physical model:
        # 1. Finite coherent array NxN -> 0th order lobe ~ 2D Gaussian (sigma=0.5)
        # 2. NA cone integration: f_NA = 1-exp(-w_max^2/(2*sigma^2)), w_max=NA*N*P/lambda
        # 3. Angular response: f_theta = 1/(1+(w_shift/gamma)^2), w_shift=N*P*sin(theta)/lambda
        # (Lorentzian better than Gaussian at modeling resonance angular dispersion)
        # 4. R_eff = |r_0|^2 * f_NA * f_theta

        wls = np.arange(380, 785, 5)
        d_nm = max(getattr(param, "diameter_nm", 180), 10.0)
        h_nm = max(getattr(param, "height_nm", 300), 10.0)
        p_nm = max(getattr(param, "period_nm", 400), 100.0)
        fill = float(np.clip(np.pi*(d_nm/2)**2/(p_nm**2), 0.03, 0.70))

        n_sub = MaterialLibrary.n_at_wavelength(param.substrate, 550.0)
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
        """Dual-pillar far-field spectrum (Gaussian NA + Lorentzian angle)."""
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
                    material=param.material,
                    substrate=param.substrate
                )
                wls = np.linspace(wl_start, wl_end, n_pts)
                return wls, (spec.squeeze(0).numpy() / 0.86)
            except Exception as e:
                logging.warning(f"engine fallback: {e}")
                wls = np.linspace(wl_start, wl_end, n_pts)
                refl = np.zeros(len(wls))
                for i, wl_nm in enumerate(wls):
                    I1, _ = _single_pillar_complex(param.d1_nm, param.h1_nm, param.period_nm, param.material, param.polarization, param.angle_deg, wl_nm, param.substrate)
                    I2, _ = _single_pillar_complex(param.d2_nm, param.h2_nm, param.period_nm,
                            param.material, param.polarization, param.angle_deg, wl_nm, param.substrate)
                    _, f1 = _single_pillar_complex(param.d1_nm, param.h1_nm, param.period_nm, param.material, param.polarization, param.angle_deg, 550.0, param.substrate)
                    _, f2 = _single_pillar_complex(param.d2_nm, param.h2_nm, param.period_nm,
                        param.material, param.polarization, param.angle_deg, 550.0, param.substrate)
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
                material=param.material,
                substrate=param.substrate
            )
            wls = np.linspace(wl_start, wl_end, n_pts)
            return wls, (spec.squeeze(0).numpy() / 0.86)
        except Exception as e:
            logging.warning(f"engine fallback: {e}")
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
        # Try on-disk cache (persists across app restarts)
        loaded = self._load_grid_disk(key)
        if loaded is not None:
            self.grid_params, self.grid_rgb, self.grid_lab, self.grid_xy = loaded
            self._cache[key] = (self.grid_rgb, self.grid_params, self.grid_lab, self.grid_xy)
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
        self._save_grid_disk(key, (self.grid_params, self.grid_rgb, self.grid_lab, self.grid_xy))

    @staticmethod
    def _grid_cache_path(key):
        h = hashlib.sha256((str(key) + _GRID_CACHE_VERSION).encode()).hexdigest()[:16]
        return os.path.join(_GRID_CACHE_DIR, f"grid_{h}.pkl")

    @staticmethod
    def _load_grid_disk(key):
        path = MetaSurfaceColorEngine._grid_cache_path(key)
        try:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    return _pickle.load(f)
        except Exception:
            pass
        return None

    @staticmethod
    def _save_grid_disk(key, data):
        path = MetaSurfaceColorEngine._grid_cache_path(key)
        try:
            with open(path + ".tmp", "wb") as f:
                _pickle.dump(data, f, protocol=_pickle.HIGHEST_PROTOCOL)
            os.replace(path + ".tmp", path)
        except Exception:
            pass

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
        # Re-rank top 50 using batched physical_color (~50x faster)
        real_scores = []
        total_rerank = len(top_idx)
        try:
            _ds = np.array([self.grid_params[i][0] for i in top_idx])
            _hs = np.array([self.grid_params[i][1] for i in top_idx])
            _ps = np.array([self.grid_params[i][2] for i in top_idx])
            _rgbs = self._batch_compute_rgb(_ds, _hs, _ps)
            _labs = rgb_to_lab(_rgbs)
            for ri, idx in enumerate(top_idx):
                if progress_callback:
                    progress_callback(ri, total_rerank, "Coarse rerank")
                d, h, p_val = self.grid_params[idx]
                rgb = _rgbs[ri]
                lab = _labs[ri]
                de2k = delta_e2000(target_lab, lab)
                real_scores.append((de2k, d, h, p_val, rgb, lab, de2k))
        except Exception as e:
            logging.warning(f"engine fallback: {e}")
            for ri, idx in enumerate(top_idx):
                if progress_callback:
                    progress_callback(ri, total_rerank, "Coarse rerank")
                d, h, p_val = self.grid_params[idx]
                param = MetaSurfaceParam(float(d), float(h), float(p_val),
                    self._last_material, self._last_substrate,
                    self._last_polarization, self._last_angle)
                rgb = self.physical_color(param)
                lab = rgb_to_lab(rgb[None, :])[0]
                de2k = delta_e2000(target_lab, lab)
                real_scores.append((de2k, d, h, p_val, rgb, lab, de2k))
        real_scores.sort(key=lambda x: x[0])
        # Stage 2: fine search - collect candidates then batch infer (~50-100x faster)
        top3 = []  # (score, param, rgb, de76, de2k)
        seen = set()
        fine_candidates = []  # (dd, dh, dp)
        fine_count = len(real_scores[:8])
        for fi, (_, d, h, p_val, _, _, _) in enumerate(real_scores[:8]):
            if progress_callback:
                progress_callback(fi, fine_count, "Fine search")
            for dd in np.arange(max(50, d-10), min(350, d+11), 2.0):
                for dh in np.arange(max(80, h-30), min(600, h+32), 4.0):
                    for dp in np.arange(max(200, p_val-30), min(600, p_val+35), 10.0):
                        if dd >= dp:
                            continue
                        fill_ratio = np.pi*(dd/2)**2/(dp**2)
                        if fill_ratio < 0.03 or fill_ratio > 0.70:
                            continue
                        key = (round(dd,1), round(dh,1), round(dp,1))
                        if key in seen:
                            continue
                        seen.add(key)
                        fine_candidates.append((dd, dh, dp))
        # Batch inference for all fine candidates at once
        try:
            _fds = np.array([c[0] for c in fine_candidates], dtype=np.float32)
            _fhs = np.array([c[1] for c in fine_candidates], dtype=np.float32)
            _fps = np.array([c[2] for c in fine_candidates], dtype=np.float32)
            _frgbs = self._batch_compute_rgb(_fds, _fhs, _fps)
            _flabs = rgb_to_lab(_frgbs)
            for ci, (dd, dh, dp) in enumerate(fine_candidates):
                rgb = _frgbs[ci]
                lab = _flabs[ci]
                de2k = delta_e2000(target_lab, lab)
                de76 = delta_e76(target_lab, lab)
                param = MetaSurfaceParam(dd, dh, dp,
                    self._last_material, self._last_substrate,
                    self._last_polarization, self._last_angle)
                top3.append((de2k, param, rgb, de76, de2k))
        except Exception as e:
            logging.warning(f"engine fallback: {e}")
            # Fallback: per-candidate inference
            for dd, dh, dp in fine_candidates:
                param = MetaSurfaceParam(dd, dh, dp,
                    self._last_material, self._last_substrate,
                    self._last_polarization, self._last_angle)
                rgb = self.physical_color(param)
                lab = rgb_to_lab(rgb[None, :])[0]
                de2k = delta_e2000(target_lab, lab)
                de76 = delta_e76(target_lab, lab)
                top3.append((de2k, param, rgb, de76, de2k))
        top3.sort(key=lambda x: x[0])
        # Deduplicate: param diversity (D>=15nm, H>=20nm, P>=30nm) + color diversity (dE>=1.5)
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


