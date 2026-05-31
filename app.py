# ===================== Streamlit 版本：超表面结构色设计系统 =====================
from __future__ import annotations

import io
import numpy as np
from PIL import Image
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
from dataclasses import dataclass
from typing import Tuple, List

st.set_page_config(page_title="AI Metasurface Color Design", layout="wide")

# ===================== Constants & Helpers =====================
D65 = np.array([0.95047, 1.00000, 1.08883], dtype=float)

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
        self.grid_params, self.grid_rgb, self.grid_lab, self.grid_xy = self._build_library()

    def physical_color(self, param: MetaSurfaceParam) -> np.ndarray:
        d, h = param.diameter_nm, param.height_nm
        angle = param.angle_deg
        pol = param.polarization
        mat_pillar = param.material
        mat_sub = param.substrate

        wl_r, wl_g, wl_b = 620.0, 540.0, 460.0
        n_r = MaterialLibrary.n_at_wavelength(mat_pillar, wl_r)
        n_g = MaterialLibrary.n_at_wavelength(mat_pillar, wl_g)
        n_b = MaterialLibrary.n_at_wavelength(mat_pillar, wl_b)
        n_sub_r = MaterialLibrary.n_at_wavelength(mat_sub, wl_r)
        n_sub_g = MaterialLibrary.n_at_wavelength(mat_sub, wl_g)
        n_sub_b = MaterialLibrary.n_at_wavelength(mat_sub, wl_b)

        lambda_r = 610 + 0.42*(d-180) - 0.09*(h-380) + 18*np.sin(h/95) + 20*(n_r-2.3)
        lambda_g = 535 - 0.26*(d-180) + 0.14*(h-380) + 13*np.sin(d/70) + 18*(n_g-2.3)
        lambda_b = 455 + 0.17*(d-180) - 0.11*(h-380) + 9*np.cos((d+h)/100) + 15*(n_b-2.3)

        r = np.exp(-0.5 * ((lambda_r - wl_r) / 65) ** 2)
        g = np.exp(-0.5 * ((lambda_g - wl_g) / 55) ** 2)
        b = np.exp(-0.5 * ((lambda_b - wl_b) / 50) ** 2)

        theta_rad = np.radians(angle)
        theta_t_r = np.arcsin(np.clip(np.sin(theta_rad) / max(n_r, 1.0001), -1, 1))
        theta_t_g = np.arcsin(np.clip(np.sin(theta_rad) / max(n_g, 1.0001), -1, 1))
        theta_t_b = np.arcsin(np.clip(np.sin(theta_rad) / max(n_b, 1.0001), -1, 1))

        if angle > 0.5:
            if pol == "TE (s-pol)":
                amp_r = (np.cos(theta_rad) - n_r*np.cos(theta_t_r)) / (np.cos(theta_rad) + n_r*np.cos(theta_t_r) + 1e-9)
                amp_g = (np.cos(theta_rad) - n_g*np.cos(theta_t_g)) / (np.cos(theta_rad) + n_g*np.cos(theta_t_g) + 1e-9)
                amp_b = (np.cos(theta_rad) - n_b*np.cos(theta_t_b)) / (np.cos(theta_rad) + n_b*np.cos(theta_t_b) + 1e-9)
            else:
                amp_r = (n_r*np.cos(theta_rad) - np.cos(theta_t_r)) / (n_r*np.cos(theta_rad) + np.cos(theta_t_r) + 1e-9)
                amp_g = (n_g*np.cos(theta_rad) - np.cos(theta_t_g)) / (n_g*np.cos(theta_rad) + np.cos(theta_t_g) + 1e-9)
                amp_b = (n_b*np.cos(theta_rad) - np.cos(theta_t_b)) / (n_b*np.cos(theta_rad) + np.cos(theta_t_b) + 1e-9)
            refl_r = 1.0 - abs(amp_r)**2
            refl_g = 1.0 - abs(amp_g)**2
            refl_b = 1.0 - abs(amp_b)**2
            if pol == "TM (p-pol)":
                brew_r = np.arctan(n_r); brew_g = np.arctan(n_g); brew_b = np.arctan(n_b)
                refl_r *= 1.0 - 0.5*np.exp(-((theta_rad-brew_r)**2)/0.03)
                refl_g *= 1.0 - 0.5*np.exp(-((theta_rad-brew_g)**2)/0.03)
                refl_b *= 1.0 - 0.5*np.exp(-((theta_rad-brew_b)**2)/0.03)
        else:
            refl_r = refl_g = refl_b = 1.0

        r_bot_r = (n_sub_r - n_r) / (n_sub_r + n_r + 1e-9)
        r_bot_g = (n_sub_g - n_g) / (n_sub_g + n_g + 1e-9)
        r_bot_b = (n_sub_b - n_b) / (n_sub_b + n_b + 1e-9)

        sub_factor_r = 1.0 - 0.3 * abs(r_bot_r)
        sub_factor_g = 1.0 - 0.3 * abs(r_bot_g)
        sub_factor_b = 1.0 - 0.3 * abs(r_bot_b)

        ff = np.clip(np.pi * (d/2)**2 / (param.period_nm**2), 0.01, 0.85)
        aspect = d / max(h, 1)
        loss = 1.0 - 0.12 * (aspect - 0.45)**2

        rgb = np.array([
            r * refl_r * sub_factor_r * ff * loss,
            g * refl_g * sub_factor_g * ff * loss,
            b * refl_b * sub_factor_b * ff * loss,
        ])
        return clamp01(0.25 + 0.75 * rgb)

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
        d, h = param.diameter_nm, param.height_nm
        n_p = MaterialLibrary.n_at_wavelength(param.material, wl_nm)
        n_s = MaterialLibrary.n_at_wavelength(param.substrate, wl_nm)
        lambda_res = 580 + 0.38*(d-180) - 0.12*(h-380) + 15*np.sin(h/90) + 18*(n_p-2.3)
        r0 = np.exp(-0.5 * ((lambda_res - wl_nm) / 60) ** 2)
        theta_rad = np.radians(param.angle_deg)
        theta_t = np.arcsin(np.clip(np.sin(theta_rad) / max(n_p, 1.0001), -1, 1))
        if param.angle_deg > 0.5:
            if param.polarization == "TE (s-pol)":
                amp = (np.cos(theta_rad) - n_p*np.cos(theta_t)) / (np.cos(theta_rad) + n_p*np.cos(theta_t) + 1e-9)
            else:
                amp = (n_p*np.cos(theta_rad) - np.cos(theta_t)) / (n_p*np.cos(theta_rad) + np.cos(theta_t) + 1e-9)
            refl = 1.0 - abs(amp)**2
        else:
            refl = 1.0
        r_bot = (n_s - n_p) / (n_s + n_p + 1e-9)
        sub_f = 1.0 - 0.3 * abs(r_bot)
        ff = np.clip(np.pi * (d/2)**2 / (param.period_nm**2), 0.01, 0.85)
        return clamp01(0.2 + 0.8 * r0 * refl * sub_f * ff)

    def compute_spectrum(self, param, wl_start=400.0, wl_end=700.0, n_pts=61):
        wls = np.linspace(wl_start, wl_end, n_pts)
        refl = np.array([self._single_wl_response(param, w) for w in wls])
        return wls, refl

    def _build_library(self):
        d_vals = np.linspace(self.d_min, self.d_max, 150)
        h_vals = np.linspace(self.h_min, self.h_max, 180)
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
        d_vals = np.linspace(self.d_min, self.d_max, 150)
        h_vals = np.linspace(self.h_min, self.h_max, 180)
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

engine = get_engine()

st.title("AI Metasurface Structural Color Design")
st.caption("基于 TiO₂ 纳米柱的 Mie 散射 + 法珀腔干涉模型")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ 参数控制")
    material = st.selectbox("材料 (Pillar)", MaterialLibrary.pillar_materials(), index=1)
    substrate = st.selectbox("衬底 (Substrate)", MaterialLibrary.substrate_materials(), index=0)
    polarization = st.selectbox("偏振", ["TE (s-pol)", "TM (p-pol)"], index=0)
    angle = st.slider("入射角 (°)", 0.0, 80.0, 0.0, 0.5)

    st.divider()
    st.header("📏 纳米柱尺寸")

    diameter = st.slider("直径 D (nm)", 60.0, 320.0, 200.0, 0.5)
    height = st.slider("高度 H (nm)", 120.0, 720.0, 400.0, 0.5)
    period = st.slider("周期 P (nm)", 360.0, 560.0, 420.0, 0.5)

    if diameter > period:
        st.warning("⚠️ D > P：纳米柱会重叠，请调整")

    st.divider()
    if st.button("🔄 重建材料库", use_container_width=True):
        with st.spinner("正在重建 27,000 色库..."):
            engine.rebuild_library(material, substrate, polarization, angle)
        st.success("库重建完成！")

# Build param
param = MetaSurfaceParam(diameter, height, period, material, substrate, polarization, angle)
rgb = engine.physical_color(param)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔬 Live Preview", "🎯 逆设计", "🖼️ 图案生成",
    "📊 FDTD 仿真", "🌈 光谱"
])

# Tab 1: Live Preview
with tab1:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("纳米柱结构")
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
        # Side view
        ax1.add_patch(Rectangle((-diameter/2, 0), diameter, height,
                                 facecolor=rgb, edgecolor='black', lw=0.8))
        ax1.set_xlim(-period, period)
        ax1.set_ylim(-50, height + 100)
        ax1.set_title(f"D={diameter:.0f} H={height:.0f} P={period:.0f} nm")
        ax1.set_xlabel("x (nm)"); ax1.set_ylabel("z (nm)")
        ax1.set_aspect("equal")
        # Top view
        circ = plt.Circle((0, 0), diameter/2, facecolor=rgb, edgecolor='black', lw=0.8)
        ax2.add_patch(circ)
        ax2.set_xlim(-period, period); ax2.set_ylim(-period, period)
        ax2.set_title("Top View"); ax2.set_aspect("equal")
        ax2.axis("off")
        fig1.tight_layout()
        st.pyplot(fig1); plt.close(fig1)

    with col2:
        st.subheader("结构色")
        hex_color = rgb_to_hex(rgb)
        r255, g255, b255 = rgb_255(rgb)
        st.markdown(f"""
        <div style="width:200px;height:200px;background:{hex_color};
                    border-radius:10px;border:3px solid #333;margin:10px auto;"></div>
        <p style="text-align:center;font-size:18px;"><b>{hex_color}</b></p>
        <p style="text-align:center;">RGB({r255}, {g255}, {b255})</p>
        """, unsafe_allow_html=True)

        st.caption(f"材料: {material} | 衬底: {substrate}")
        st.caption(f"偏振: {polarization} | 入射角: {angle:.1f}°")

# Tab 2: Inverse Design
with tab2:
    st.subheader("输入目标颜色，自动寻找最优纳米柱参数")
    col_r, col_g, col_b = st.columns(3)
    with col_r:
        target_r = st.number_input("R", 0, 255, 128)
    with col_g:
        target_g = st.number_input("G", 0, 255, 120)
    with col_b:
        target_b = st.number_input("B", 0, 255, 200)

    if st.button("🔍 开始逆设计", use_container_width=True):
        with st.spinner("搜索 27,000 种参数组合..."):
            engine.rebuild_library(material, substrate, polarization, angle)
            target_rgb_norm = np.array([target_r, target_g, target_b]) / 255.0
            best_param, matched_rgb, de_val = engine.inverse_design(target_rgb_norm)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**🎯 目标颜色**")
            hex_t = rgb_to_hex(target_rgb_norm)
            st.markdown(f"""
            <div style="width:120px;height:120px;background:{hex_t};border-radius:8px;border:3px solid #333;"></div>
            <p>RGB({target_r}, {target_g}, {target_b})</p>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown("**✅ 匹配结果**")
            hex_m = rgb_to_hex(matched_rgb)
            mr, mg, mb = rgb_255(matched_rgb)
            st.markdown(f"""
            <div style="width:120px;height:120px;background:{hex_m};border-radius:8px;border:3px solid #333;"></div>
            <p>RGB({mr}, {mg}, {mb})</p>
            """, unsafe_allow_html=True)

        st.success(f"""
        **最优参数**: D = {best_param.diameter_nm:.1f} nm | H = {best_param.height_nm:.1f} nm | P = {best_param.period_nm:.1f} nm
        **色差 dE76** = {de_val:.2f}
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

            fig3, (ax_o, ax_m, ax_d) = plt.subplots(1, 3, figsize=(14, 4))
            ax_o.imshow(orig); ax_o.set_title("原图"); ax_o.axis("off")
            ax_m.imshow(mapped); ax_m.set_title("超表面图案"); ax_m.axis("off")
            im = ax_d.imshow(params_arr[:,:,0], cmap="viridis")
            ax_d.set_title("直径分布 (nm)"); ax_d.axis("off")
            plt.colorbar(im, ax=ax_d, fraction=0.046)
            fig3.tight_layout()
            st.pyplot(fig3); plt.close(fig3)

            mean_err = float(np.mean(np.linalg.norm(orig - mapped, axis=2)))
            st.info(f"图案: {params_arr.shape[1]}×{params_arr.shape[0]} 像素 | 平均 RGB 误差: {mean_err:.4f}")

# Tab 4: FDTD Simulation
with tab4:
    st.subheader("FDTD 仿真颜色 (简化为物理模型变体)")
    fdtd_rgb = engine.ai_predict_color(param)
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.markdown("**物理模型**")
        hex_p = rgb_to_hex(rgb)
        st.markdown(f"""
        <div style="width:120px;height:120px;background:{hex_p};border-radius:8px;border:3px solid #333;"></div>
        <p>{hex_p}</p>
        """, unsafe_allow_html=True)
    with col_f2:
        st.markdown("**FDTD 预测**")
        hex_f = rgb_to_hex(fdtd_rgb)
        st.markdown(f"""
        <div style="width:120px;height:120px;background:{hex_f};border-radius:8px;border:3px solid #333;"></div>
        <p>{hex_f}</p>
        """, unsafe_allow_html=True)

    diff = np.linalg.norm(rgb - fdtd_rgb)
    st.caption(f"两者 RGB 差异: {diff:.4f}")

# Tab 5: Spectrum
with tab5:
    st.subheader("反射光谱")
    wls, refl = engine.compute_spectrum(param, 400, 700, 81)

    fig5, ax5 = plt.subplots(figsize=(10, 4))
    ax5.plot(wls, refl, "b-", lw=2, label=f"D={diameter:.0f} H={height:.0f}nm")
    ax5.fill_between(wls, 0, refl, alpha=0.15, color="blue")
    ax5.set_xlabel("Wavelength (nm)"); ax5.set_ylabel("Reflectance")
    ax5.set_title(f"Spectrum: {material} on {substrate}")
    ax5.set_xlim(400, 700); ax5.set_ylim(0, 1.05)
    ax5.grid(True, alpha=0.3); ax5.legend()
    fig5.tight_layout()
    st.pyplot(fig5); plt.close(fig5)

    # CIE 1931
    xy = rgb_to_xy(rgb)
    st.caption(f"CIE 1931 xy: ({xy[0]:.4f}, {xy[1]:.4f})")

st.sidebar.markdown("---")
st.sidebar.caption("AI Metasurface Color Design v2.0 (Web)")
st.sidebar.caption("Physics: Mie + Fabry-Perot + Cauchy")
