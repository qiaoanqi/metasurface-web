# ml_module.py - ML acceleration for metasurface color engine (ONNX Runtime + optional PyTorch)
import os
import numpy as np
import glob

# --- Auto-download models from HF Hub if not present ---
_MODEL_REPO = 'qiaoanqi/metasurface-models'

def _ensure_model_file(rel_path):
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), rel_path)
    if os.path.exists(local):
        return local
    try:
        from huggingface_hub import hf_hub_download
        os.makedirs(os.path.dirname(local), exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=_MODEL_REPO, filename=rel_path,
            cache_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), '.hf_cache'),
            local_dir=os.path.dirname(os.path.abspath(__file__)),
            local_dir_use_symlinks=False,
            timeout=5)
        return downloaded
    except Exception:
        return local

from color_utils import CIE_X, CIE_Y, CIE_Z, WL, CIE_NORM, SRGB_M, spectrum_to_srgb

# ---- globals ----
_ORT_AVAILABLE = False
_ORT_SESSION = None
_ORT_IS_V8 = False
_DUAL_ORT_AVAILABLE = False
_DUAL_ORT_SESSION = None
_DUAL_IS_V3 = False
_TORCH_FWD = None      # PyTorch model for gradient-based inverse design
_TORCH_IS_V8 = False
# _RCWA_SESSION replaced by _RCWA_SESSIONS dict (ensemble)
_RCWA_AVAILABLE = False

MATERIAL_CODES = {"TiO2 (anatase)": 0, "a-Si (amorphous)": 1, "Si3N4 (nitride)": 2, "Al2O3 (sapphire)": 3}
SUBSTRATE_CODES = {"SiO2 (fused silica)": 0, "Si3N4 (nitride)": 1, "Al2O3 (sapphire)": 2}

def _spectrum_to_rgb(spec: np.ndarray) -> np.ndarray:
    return spectrum_to_srgb(WL, np.clip(spec, 0, None))

# ---- ONNX init ----
def init_ml():
    global _ORT_AVAILABLE, _ORT_SESSION, _ORT_IS_V8
    try:
        import onnxruntime as ort
        path_v8 = _ensure_model_file("models/forward_mlp_v8_sub.onnx")
        if os.path.exists(path_v8):
            _ORT_SESSION = ort.InferenceSession(path_v8, providers=["CPUExecutionProvider"])
            _ORT_IS_V8 = True
        else:
            return False
        _ORT_AVAILABLE = True
        return True
    except Exception:
        return False

# RCWA model registry: material -> onnx file
_RCWA_MODELS = {
    "TiO2 (anatase)": ["forward_mlp_rcwa_TiO2_s?.onnx"],
    "a-Si (amorphous)": ["forward_mlp_rcwa_aSi_k_s?.onnx"],  # complex-k (Green&Keevers 1995), ΔE=2.68
    "Si3N4 (nitride)": ["forward_mlp_rcwa_Si3N4_s?.onnx"],
    "Al2O3 (sapphire)": ["forward_mlp_rcwa_Al2O3_s1.onnx"],  # v2 multi-substrate, ΔE=1.67
}
# Substrate-specific overrides (tuple key: (material, substrate))
_RCWA_SUBSTRATE_MODELS = {
    # a-Si now uses multi-substrate ensemble (80/20 retrained 2026-07-21)
}
_RCWA_SESSIONS = {}  # material -> list of ort sessions (ensemble)

# Wavelength-conditioned RCWA models (7D: D,H,P,n,k,sub,wl -> scalar R)
_RCWA_WL_MODELS = {
    }
_RCWA_WL_SESSIONS = {}  # material -> ort session

def init_rcwa_ml():
    """加载所有可用的RCWA模型 (支持glob ensemble)"""
    global _RCWA_AVAILABLE, _RCWA_SESSIONS
    _RCWA_SESSIONS = {}
    try:
        import onnxruntime as ort
        base = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base, "models")
        def _load_patterns(patterns):
            sessions = []
            for pat in patterns:
                if "*" in pat or "?" in pat or "[" in pat:
                    files = sorted(glob.glob(os.path.join(models_dir, pat)))
                else:
                    fpath = os.path.join(models_dir, pat)
                    files = [fpath] if os.path.exists(fpath) else []
                for fpath in files:
                    try:
                        sessions.append(ort.InferenceSession(fpath, providers=["CPUExecutionProvider"]))
                    except Exception:
                        pass
            return sessions

        for mat, patterns in _RCWA_MODELS.items():
            sessions = _load_patterns(patterns)
            if sessions:
                _RCWA_SESSIONS[mat] = sessions
        # Load substrate-specific models
        for (mat, sub), patterns in _RCWA_SUBSTRATE_MODELS.items():
            sessions = _load_patterns(patterns)
            if sessions:
                _RCWA_SESSIONS[(mat, sub)] = sessions
        # Load wavelength-conditioned models
        _RCWA_WL_SESSIONS.clear()
        for mat, fname in _RCWA_WL_MODELS.items():
            fpath = os.path.join(models_dir, fname)
            if os.path.exists(fpath):
                try:
                    _RCWA_WL_SESSIONS[mat] = ort.InferenceSession(fpath, providers=['CPUExecutionProvider'])
                except Exception:
                    pass
        _RCWA_AVAILABLE = len(_RCWA_SESSIONS) > 0 or len(_RCWA_WL_SESSIONS) > 0
        return _RCWA_AVAILABLE
    except Exception:
        _RCWA_AVAILABLE = False
        return False

def _get_rcwa_sessions(material, substrate=None):
    """Get list of RCWA ONNX sessions (supports substrate-specific override)."""
    if substrate is not None:
        key = (material, substrate)
        if key in _RCWA_SESSIONS:
            return _RCWA_SESSIONS[key]
    return _RCWA_SESSIONS.get(material, [])

def _ensemble_predict(material, substrate, x):
    """Run ensemble prediction across all sessions for a material, return averaged spectrum."""
    sessions = _get_rcwa_sessions(material, substrate)
    if not sessions:
        return None
    specs = []
    for sess in sessions:
        input_name = sess.get_inputs()[0].name
        spec = sess.run(None, {input_name: x})[0][0]
        specs.append(spec)
    return np.mean(specs, axis=0)

# Cauchy refractive index model for wavelength-conditioned models
_CAUCHY_COEFFS = {
    'TiO2 (anatase)': (2.3, 0.035, 0.0),
    'a-Si (amorphous)': (3.8, 0.08, 0.0),
    'Si3N4 (nitride)': (1.99, 0.012, 0.0),
    'Al2O3 (sapphire)': (1.7546, 0.005, 0.0),
    'SiO2 (fused silica)': (1.458, 0.00354, 0.0),
}

def _cauchy_n(material, wavelength_nm):
    A, B, C = _CAUCHY_COEFFS.get(material, (1.5, 0.0, 0.0))
    wl_um = wavelength_nm / 1000.0
    return A + B / (wl_um ** 2) + C / (wl_um ** 4)

def _predict_rcwa_wavelength(material, substrate, d_nm, h_nm, p_nm):
    sess = _RCWA_WL_SESSIONS.get(material)
    if sess is None:
        return None
    WLS = np.linspace(380, 780, 81)
    sub_code = float(SUBSTRATE_CODES.get(substrate, 0))
    d_norm = (d_nm - 50) / 300
    h_norm = (h_nm - 80) / 520
    p_norm = (p_nm - 200) / 400
    batch = np.zeros((81, 7), dtype=np.float32)
    for i, wl in enumerate(WLS):
        n_val = _cauchy_n(material, wl)
        n_norm = (n_val - 1.0) / 4.0
        wl_norm = (wl - 380) / 400
        batch[i] = [d_norm, h_norm, p_norm, n_norm, 0.0, sub_code, wl_norm]
    input_name = sess.get_inputs()[0].name
    result = sess.run(None, {input_name: batch})[0]
    return result.flatten().astype(np.float64)

def init_dual_ml():
    global _DUAL_ORT_AVAILABLE, _DUAL_ORT_SESSION, _DUAL_IS_V3
    try:
        import onnxruntime as ort
        path_v3 = _ensure_model_file("models/dual_mlp_v3_multi.onnx")
        if not os.path.exists(path_v3):
            return False
        _DUAL_ORT_SESSION = ort.InferenceSession(path_v3, providers=["CPUExecutionProvider"])
        _DUAL_IS_V3 = True
        _DUAL_ORT_AVAILABLE = True
        return True
    except Exception:
        return False

# ---- init PyTorch model for gradient-based inverse design ----
def _init_torch_for_inverse():
    global _TORCH_FWD, _TORCH_IS_V8
    if _TORCH_FWD is not None:
        return _TORCH_FWD is not False
    try:
        import torch, torch.nn as nn
        path_v8 = _ensure_model_file("models/forward_mlp_v8_sub.pt")
        path = path_v8
        is_v8 = os.path.exists(path_v8)
        if not os.path.exists(path):
            _TORCH_FWD = False
            return False

        class ResidualBlock(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim), nn.BatchNorm1d(dim))
            def forward(self, x):
                return nn.functional.relu(self.net(x) + x)

        class DeepResMLP_Multi(nn.Module):
            def __init__(self, in_dim=6, hidden=256, out_dim=81, n_blocks=4):
                super().__init__()
                self.input_proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.BatchNorm1d(hidden))
                self.blocks = nn.Sequential(*[ResidualBlock(hidden) for _ in range(n_blocks)])
                self.output = nn.Sequential(nn.Linear(hidden, out_dim), nn.Sigmoid())
            def forward(self, x):
                return self.output(self.blocks(self.input_proj(x)))

        _TORCH_FWD = DeepResMLP_Multi(in_dim=7 if is_v8 else 6)
        _TORCH_FWD.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        _TORCH_FWD.eval()
        _TORCH_IS_V8 = is_v8
        return True
    except Exception:
        _TORCH_FWD = False
        return False

def _build_input(d_nm, h_nm, p_nm, angle_deg=0.0, polarization="TE", material="TiO2 (anatase)", substrate="SiO2 (fused silica)"):
    """Shared input normalization helper."""
    pol_code = 0.0 if polarization.startswith("TE") else 1.0
    mat_code = float(MATERIAL_CODES.get(material, 0))
    sub_code = float(SUBSTRATE_CODES.get(substrate, 0))
    if _ORT_IS_V8:
        return np.array([[(d_nm - 50) / 300, (h_nm - 80) / 520, (p_nm - 200) / 400,
                           angle_deg / 80, pol_code, mat_code, sub_code]], dtype=np.float32)
    else:
        return np.array([[(d_nm - 50) / 300, (h_nm - 80) / 520, (p_nm - 200) / 400,
                           angle_deg / 80, pol_code, mat_code]], dtype=np.float32)

def _build_rcwa_input(d_nm, h_nm, p_nm, angle_deg=0.0, polarization="TE",
                      material="TiO2 (anatase)", substrate="SiO2 (fused silica)"):
    """RCWA模型输入 (固定7维，和v8归一化一致)"""
    pol_code = 0.0 if polarization.startswith("TE") else 1.0
    mat_code = float(MATERIAL_CODES.get(material, 0))
    sub_code = float(SUBSTRATE_CODES.get(substrate, 0))
    return np.array([[(d_nm - 50) / 300, (h_nm - 80) / 520, (p_nm - 200) / 400,
                       angle_deg / 80, pol_code, mat_code, sub_code]], dtype=np.float32)

def _should_use_rcwa(material, substrate, angle_deg):
    """判断是否应该使用RCWA模型 (支持衬底特定路由)"""
    if not _RCWA_AVAILABLE or abs(angle_deg) >= 5:
        return False
    # Substrate-specific models take priority
    key = (material, substrate)
    if key in _RCWA_SESSIONS:
        return True
    # Fall back to material-level model
    if material in _RCWA_SESSIONS:
        return True
    # Check wavelength-conditioned models
    return material in _RCWA_WL_SESSIONS

# ---- predict (ONNX) ----
def _run_onnx(session, x):
    """自动检测ONNX输入名并运行推理"""
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: x})[0][0]

def predict_rgb(d_nm, h_nm, p_nm, angle_deg=0.0, polarization="TE", material="TiO2 (anatase)", substrate="SiO2 (fused silica)"):
    if material not in MATERIAL_CODES:
        return None
    if substrate not in SUBSTRATE_CODES:
        return None
    # RCWA优先 (ensemble)
    if _should_use_rcwa(material, substrate, angle_deg):
        x = _build_rcwa_input(d_nm, h_nm, p_nm, angle_deg, polarization, material, substrate)
        spec = _ensemble_predict(material, substrate, x)
        if spec is not None:
            return _spectrum_to_rgb(spec)
    # 回退Fano模型
    if not _ORT_AVAILABLE:
        return None
    x = _build_input(d_nm, h_nm, p_nm, angle_deg, polarization, material, substrate)
    spec = _run_onnx(_ORT_SESSION, x)
    return _spectrum_to_rgb(spec)

def predict_spectrum(d_nm, h_nm, p_nm, angle_deg=0.0, polarization="TE", material="TiO2 (anatase)", substrate="SiO2 (fused silica)"):
    if material not in MATERIAL_CODES:
        return None
    if substrate not in SUBSTRATE_CODES:
        return None
    # RCWA wavelength-conditioned model (priority 1)
    if material in _RCWA_WL_SESSIONS and abs(angle_deg) < 5:
        spec = _predict_rcwa_wavelength(material, substrate, d_nm, h_nm, p_nm)
        if spec is not None:
            return np.clip(spec, 0, None)
    # RCWA ensemble model (priority 2)
    if _should_use_rcwa(material, substrate, angle_deg):
        x = _build_rcwa_input(d_nm, h_nm, p_nm, angle_deg, polarization, material, substrate)
        spec = _ensemble_predict(material, substrate, x)
        if spec is not None:
            return np.clip(spec, 0, None)
    # 回退Fano模型
    if not _ORT_AVAILABLE:
        return None
    x = _build_input(d_nm, h_nm, p_nm, angle_deg, polarization, material, substrate)
    spec = _run_onnx(_ORT_SESSION, x)
    return np.clip(spec, 0, None)


def predict_with_uncertainty(d_nm, h_nm, p_nm, angle_deg=0.0, polarization="TE",
                              material="TiO2 (anatase)", substrate="SiO2 (fused silica)", n_samples=20):
    """Input-perturbation uncertainty: add small noise to D/H/P, run N forward passes.
    Returns (mean_spec, std_spec, mean_rgb, std_rgb, mean_de_vs_nominal)."""
    specs = []
    for _ in range(n_samples):
        d2 = d_nm + np.random.normal(0, 2.0)  # sigma=2nm
        h2 = h_nm + np.random.normal(0, 2.0)
        p2 = p_nm + np.random.normal(0, 2.0)
        spec = predict_spectrum(d2, h2, p2, angle_deg, polarization, material, substrate)
        if spec is not None:
            specs.append(spec)
    if not specs:
        return None, None, None, None, None

    specs = np.array(specs)
    mean_spec = np.mean(specs, axis=0)
    std_spec = np.std(specs, axis=0)
    rgbs = np.array([_spectrum_to_rgb(s) for s in specs])
    mean_rgb = np.mean(rgbs, axis=0)
    std_rgb = np.std(rgbs, axis=0)

    # CIEDE2000 between nominal and mean prediction
    nominal_rgb = predict_rgb(d_nm, h_nm, p_nm, angle_deg, polarization, material, substrate)
    if nominal_rgb is not None:
        from color_utils import delta_e2000, rgb_to_lab
        de = delta_e2000(rgb_to_lab(nominal_rgb), rgb_to_lab(mean_rgb))
    else:
        de = 0.0

    return mean_spec, std_spec, mean_rgb, std_rgb, de

def predict_dual_spectrum(d1_nm, h1_nm, d2_nm, h2_nm, p_nm, angle_deg=0.0, polarization="TE", material="TiO2 (anatase)", substrate="SiO2 (fused silica)"):
    if substrate != "SiO2 (fused silica)":
        return None  # dual ML model does not support non-SiO2 substrate
    if not _DUAL_ORT_AVAILABLE:
        return None
    if material not in MATERIAL_CODES:
        return None
    pol_code = 0.0 if polarization.startswith("TE") else 1.0
    mat_code = float(MATERIAL_CODES.get(material, 0))
    if _DUAL_IS_V3:
        x = np.array([[(d1_nm - 60) / 247, (h1_nm - 80) / 520, (d2_nm - 60) / 247, (h2_nm - 80) / 520,
                        (p_nm - 200) / 400, angle_deg / 60, pol_code, mat_code]], dtype=np.float32)
    else:
        x = np.array([[(d1_nm - 60) / 247, (h1_nm - 80) / 520, (d2_nm - 60) / 247, (h2_nm - 80) / 520,
                        (p_nm - 200) / 400, angle_deg / 60, pol_code]], dtype=np.float32)
    spec = _DUAL_ORT_SESSION.run(None, {"input": x})[0][0]
    return np.clip(spec, 0, None)

def predict_dual_rgb(d1_nm, h1_nm, d2_nm, h2_nm, p_nm, angle_deg=0.0, polarization="TE", material="TiO2 (anatase)", substrate="SiO2 (fused silica)"):
    if substrate != "SiO2 (fused silica)":
        return None  # dual ML model does not support non-SiO2 substrate
    spec = predict_dual_spectrum(d1_nm, h1_nm, d2_nm, h2_nm, p_nm, angle_deg, polarization, material)
    if spec is None:
        return None
    return _spectrum_to_rgb(spec)

# ---- inverse design (PyTorch gradient-based, requires torch) ----
def _inverse_design_ml_serial(target_rgb, n_steps=300, n_restarts=40, material="TiO2 (anatase)", substrate="SiO2 (fused silica)"):
    # RCWA route: use differentiable ResMLP for TiO2/SiO2
    if _should_use_rcwa(material, substrate, 0.0):
        from torch_model import inverse_design_rcwa
        result = inverse_design_rcwa(target_rgb, n_restarts=min(n_restarts, 24), steps=100, p_fixed=None, material=material, substrate=substrate)
        if result is not None:
            return ("RCWA", result["D"], result["H"], result["P"],
                    result["pred_rgb"], result["de2000"])
    
    if not _init_torch_for_inverse():
        return None
    import torch
    if material not in MATERIAL_CODES:
        return None
    mat_code = float(MATERIAL_CODES.get(material, 0))
    sub_code = float(SUBSTRATE_CODES.get(substrate, 0))
    target = torch.tensor(list(target_rgb), dtype=torch.float32).unsqueeze(0)

    cie_x = torch.from_numpy(CIE_X.astype(np.float32))
    cie_y = torch.from_numpy(CIE_Y.astype(np.float32))
    cie_z = torch.from_numpy(CIE_Z.astype(np.float32))
    cie_norm = torch.tensor(CIE_NORM, dtype=torch.float32)
    srgb_m = torch.from_numpy(SRGB_M.astype(np.float32))
    wl = torch.linspace(380, 780, 81)

    best_loss, best_result = 1e9, None
    for _ in range(n_restarts):
        d = torch.tensor(float(np.random.uniform(50, 350)), dtype=torch.float32, requires_grad=True)
        h = torch.tensor(float(np.random.uniform(80, 600)), dtype=torch.float32, requires_grad=True)
        p = torch.tensor(float(np.random.uniform(200, 600)), dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam([d, h, p], lr=8.0)
        for step in range(n_steps):
            opt.zero_grad()
            d_c = torch.clamp(d, 50.0, 350.0)
            h_c = torch.clamp(h, 80.0, 600.0)
            d_min_p = (d_c.detach() * 1.2).clamp(200.0, 600.0)
            p_c = torch.max(p, d_min_p)
            p_c = torch.clamp(p_c, 200.0, 600.0)
            if _TORCH_IS_V8:
                x = torch.stack([(d_c - 50) / 300, (h_c - 80) / 520, (p_c - 200) / 400,
                                 torch.tensor(0.0), torch.tensor(0.0), torch.tensor(mat_code), torch.tensor(sub_code)]).unsqueeze(0)
            else:
                x = torch.stack([(d_c - 50) / 300, (h_c - 80) / 520, (p_c - 200) / 400,
                                 torch.tensor(0.0), torch.tensor(0.0), torch.tensor(mat_code)]).unsqueeze(0)
            spec = _TORCH_FWD(x)
            X = torch.trapz(spec * cie_x.unsqueeze(0), wl, dim=1)
            Y = torch.trapz(spec * cie_y.unsqueeze(0), wl, dim=1)
            Z = torch.trapz(spec * cie_z.unsqueeze(0), wl, dim=1)
            xyz = torch.stack([X / cie_norm, Y / cie_norm, Z / cie_norm], dim=1).float()
            rgb_lin = xyz @ srgb_m.T
            rgb = torch.where(rgb_lin <= 0.0031308, 12.92 * rgb_lin, 1.055 * torch.clamp(rgb_lin, min=0.0).pow(1 / 2.4) - 0.055)
            rgb = torch.clamp(rgb, 0, 1)
            loss = ((rgb - target) ** 2).sum()
            loss.backward()
            opt.step()

        d_f = float(np.clip(d.item(), 50, 350))
        h_f = float(np.clip(h.item(), 80, 600))
        p_f = float(max(d_f * 1.2, np.clip(p.item(), 200, 600)))

        with torch.no_grad():
            if _TORCH_IS_V8:
                x_f = torch.tensor([[(d_f - 50) / 300, (h_f - 80) / 520, (p_f - 200) / 400, 0.0, 0.0, mat_code, sub_code]], dtype=torch.float32)
            else:
                x_f = torch.tensor([[(d_f - 50) / 300, (h_f - 80) / 520, (p_f - 200) / 400, 0.0, 0.0, mat_code]], dtype=torch.float32)
            spec_f = _TORCH_FWD(x_f)
            X = torch.trapz(spec_f * cie_x.unsqueeze(0), wl, dim=1)
            Y = torch.trapz(spec_f * cie_y.unsqueeze(0), wl, dim=1)
            Z = torch.trapz(spec_f * cie_z.unsqueeze(0), wl, dim=1)
            xyz = torch.stack([X / cie_norm, Y / cie_norm, Z / cie_norm], dim=1).float()
            rgb_lin = xyz @ srgb_m.T
            rgb = torch.where(rgb_lin <= 0.0031308, 12.92 * rgb_lin, 1.055 * torch.clamp(rgb_lin, min=0.0).pow(1 / 2.4) - 0.055)
            pred = torch.clamp(rgb, 0, 1).squeeze().numpy()
            fl = float(((pred - target.squeeze().numpy()) ** 2).sum())
        if fl < best_loss:
            best_loss = fl
            best_result = ("Fano", d_f, h_f, p_f, pred, fl)
    return best_result


# ---- inverse design (numpy finite-difference, no torch) ----

def ml_grid_search(target_rgb, material="TiO2 (anatase)", substrate="SiO2 (fused silica)", 
                   top_n=3, d_range=(80,350,28), h_range=(80,600,27), p_range=(200,600,21)):
    """Fast ML-based grid search for inverse design. Returns list of (MetaSurfaceParam, rgb, de76, de2000)."""
    import numpy as np
    from color_utils import rgb_to_lab, delta_e2000
    from engine import MetaSurfaceParam
    
    target = np.array(target_rgb, dtype=float)
    target_lab = rgb_to_lab(target)
    
    D_vals = np.linspace(*d_range)
    H_vals = np.linspace(*h_range)
    P_vals = np.linspace(*p_range)
    
    all_results = []
    for D in D_vals:
        for H in H_vals:
            for P in P_vals:
                rgb = predict_rgb(float(D), float(H), float(P), 0.0, "TE", material, substrate)
                if rgb is None:
                    continue
                de2000 = delta_e2000(rgb_to_lab(rgb), target_lab)
                # Simple dE76 fallback
                de76 = float(np.sqrt(np.sum((rgb - target)**2)) * 30)
                all_results.append((D, H, P, rgb, de76, de2000))
    
    all_results.sort(key=lambda x: x[5])  # sort by de2000
    top = all_results[:top_n]
    
    result = []
    for D, H, P, rgb, de76, de2000 in top:
        bp = MetaSurfaceParam(diameter_nm=round(D,1), height_nm=round(H,1), period_nm=round(P,1),
                              material=material, substrate=substrate)
        result.append((None, bp, rgb, de76, de2000))
    return result


def ml_grid_search_refined(target_rgb, material="TiO2 (anatase)", substrate="SiO2 (fused silica)"):
    """ML grid search + gradient refinement for top-3 candidates."""
    import numpy as np
    from color_utils import rgb_to_lab, delta_e2000
    from engine import MetaSurfaceParam
    
    # Step 1: Coarse grid search (faster: fewer points)
    target_lab = rgb_to_lab(np.array(target_rgb, dtype=float))
    
    D_vals = np.linspace(80, 350, 15)
    H_vals = np.linspace(80, 600, 14)
    P_vals = np.linspace(200, 600, 11)
    
    all_results = []
    for D in D_vals:
        for H in H_vals:
            for P in P_vals:
                rgb = predict_rgb(float(D), float(H), float(P), 0.0, "TE", material, substrate)
                if rgb is None:
                    continue
                de2000 = delta_e2000(rgb_to_lab(rgb), target_lab)
                all_results.append((D, H, P, rgb, de2000))
    
    all_results.sort(key=lambda x: x[4])
    top3 = all_results[:3]
    
    # Step 2: Gradient refinement from each top-3 candidate
    from torch_model import inverse_design_rcwa
    best_result = None
    best_de = 999
    
    for D0, H0, P0, rgb0, de0 in top3:
        try:
            result = inverse_design_rcwa(
                target_rgb, n_restarts=1, steps=200, lr=0.02,
                p_fixed=None, device=None, material=material,
                init_D=float(D0), init_H=float(H0), init_P=float(P0)
            )
            if result and result["de2000"] < best_de:
                best_de = result["de2000"]
                best_result = result
        except Exception:
            pass
    
    # Fallback: use best grid result if all refinements fail
    if best_result is None:
        D0, H0, P0, rgb0, de0 = top3[0]
        bp = MetaSurfaceParam(diameter_nm=round(D0,1), height_nm=round(H0,1), period_nm=round(P0,1),
                              material=material, substrate=substrate)
        return [(None, bp, rgb0, float(de0*0.8), de0)]
    
    # Step 3: Build full result
    from engine import MetaSurfaceParam
    bp = MetaSurfaceParam(
        diameter_nm=round(best_result["D"],1),
        height_nm=round(best_result["H"],1),
        period_nm=round(best_result["P"],1),
        material=material, substrate=substrate
    )
    de2000 = best_result["de2000"]
    de76 = float(np.sqrt(np.sum((np.array(best_result["pred_rgb"]) - np.array(target_rgb))**2)) * 30)
    return [(None, bp, best_result["pred_rgb"], de76, de2000)]


def smart_grid_search(target_rgb, material="TiO2 (anatase)", substrate="SiO2 (fused silica)",
                      angle_deg=0.0, polarization="TE", coarse_n=12, top_k=5,
                      fine_steps=5, fine_range=6.0, max_results=3):
    """
    Two-stage smart grid search using RCWA/ML ensemble with PyTorch batch inference.
    Stage 1: Coarse grid (coarse_n^3 points) -> top-K
    Stage 2: Fine grid around each candidate -> best result
    Returns: list of (None, MetaSurfaceParam, rgb, de76, de2000)
    """
    import numpy as np
    from color_utils import rgb_to_lab, delta_e2000, spectrum_to_srgb
    from engine import MetaSurfaceParam

    if not _should_use_rcwa(material, substrate, angle_deg):
        return None

    target = np.array(target_rgb, dtype=float)
    target_lab = rgb_to_lab(target)
    mat_code = float(MATERIAL_CODES.get(material, 0))
    sub_code = float(SUBSTRATE_CODES.get(substrate, 0))
    pol_code = 0.0 if polarization.startswith("TE") else 1.0

    D_range = (50.0, 350.0)
    H_range = (80.0, 600.0)
    P_range = (200.0, 600.0)

    WL = np.linspace(380, 780, 81)

    # ---- Load PyTorch models for batch inference ----
    import torch
    from torch_model import _RCWA_ResMLP
    import os as _os
    base = _os.path.dirname(_os.path.abspath(__file__))
    models_dir = _os.path.join(base, "models")

    # Find .pt files for this material+substrate
    key = (material, substrate)
    if key in _RCWA_SUBSTRATE_MODELS:
        onnx_patterns = _RCWA_SUBSTRATE_MODELS[key]
    elif material in _RCWA_MODELS:
        onnx_patterns = _RCWA_MODELS[material]
    else:
        return None

    # Convert .onnx glob patterns to .pt
    pt_patterns = [p.replace('.onnx', '.pt') for p in onnx_patterns]

    pt_models = []
    import glob as _glob
    for pat in pt_patterns:
        full = _os.path.join(models_dir, pat)
        matches = sorted(_glob.glob(full))
        for m in matches:
            state = torch.load(m, map_location='cpu', weights_only=False)
            # Infer model dimensions from state_dict
            head_weight = state.get('head.weight')
            if head_weight is not None:
                hidden = head_weight.shape[1]
            else:
                hidden = 256
            # Count ResBlocks
            n_blocks = 0
            while f'blocks.{n_blocks}.net.0.weight' in state:
                n_blocks += 1
            if n_blocks == 0:
                n_blocks = 4
            model = _RCWA_ResMLP(in_dim=7, hidden=hidden, out_dim=81, n_blocks=n_blocks)
            model.load_state_dict(state)
            model.eval()
            pt_models.append(model)

    if not pt_models:
        return None

    def _batch_predict(x_batch):
        """x_batch: (N, 7) numpy -> averaged spectrum (N, 81)"""
        x_t = torch.from_numpy(x_batch).float()
        with torch.no_grad():
            specs = [m(x_t).numpy() for m in pt_models]
        return np.mean(specs, axis=0)

    # === Stage 1: Coarse grid ===
    D_vals = np.linspace(D_range[0], D_range[1], coarse_n)
    H_vals = np.linspace(H_range[0], H_range[1], coarse_n)
    P_vals = np.linspace(P_range[0], P_range[1], coarse_n)

    batch_inputs = []
    batch_params = []
    for D in D_vals:
        for H in H_vals:
            for P in P_vals:
                if P < D * 1.2:
                    continue
                batch_inputs.append([
                    (D - 50) / 300, (H - 80) / 520, (P - 200) / 400,
                    angle_deg / 80, pol_code, mat_code, sub_code
                ])
                batch_params.append((D, H, P))

    if not batch_inputs:
        return None

    x_batch = np.array(batch_inputs, dtype=np.float32)
    all_specs = _batch_predict(x_batch)
    all_rgbs = np.array([spectrum_to_srgb(WL, np.clip(s, 0, None)) for s in all_specs])

    de2000s = []
    for rgb in all_rgbs:
        try:
            de = delta_e2000(rgb_to_lab(rgb), target_lab)
        except Exception:
            de = 99.0
        de2000s.append(de)

    ranked = sorted(zip(de2000s, batch_params, all_rgbs), key=lambda x: x[0])
    top_candidates = ranked[:top_k]

    # === Stage 2: Fine grid ===
    fine_offsets = np.linspace(-fine_range, fine_range, fine_steps)
    fine_inputs = []
    fine_params = []
    for _, (D0, H0, P0), _ in top_candidates:
        for dD in fine_offsets:
            for dH in fine_offsets:
                for dP in fine_offsets:
                    D = float(np.clip(D0 + dD, D_range[0], D_range[1]))
                    H = float(np.clip(H0 + dH, H_range[0], H_range[1]))
                    P = float(np.clip(P0 + dP, P_range[0], P_range[1]))
                    if P < D * 1.2:
                        continue
                    fine_inputs.append([
                        (D - 50) / 300, (H - 80) / 520, (P - 200) / 400,
                        angle_deg / 80, pol_code, mat_code, sub_code
                    ])
                    fine_params.append((D, H, P))

    if fine_inputs:
        x_fine = np.array(fine_inputs, dtype=np.float32)
        fine_specs = _batch_predict(x_fine)
        fine_rgbs = np.array([spectrum_to_srgb(WL, np.clip(s, 0, None)) for s in fine_specs])

        for (D, H, P), rgb in zip(fine_params, fine_rgbs):
            try:
                de = delta_e2000(rgb_to_lab(rgb), target_lab)
            except Exception:
                de = 99.0
            ranked.append((de, (D, H, P), rgb))

    ranked.sort(key=lambda x: x[0])
    result = []
    accepted_params = []  # store (D,H,P) of accepted candidates
    for de, (D, H, P), rgb in ranked:
        # Enforce minimum parameter distance between candidates (diverse Top-K)
        too_close = False
        for (aD, aH, aP) in accepted_params:
            if abs(D - aD) < 15 and abs(H - aH) < 15 and abs(P - aP) < 15:
                too_close = True
                break
        if too_close:
            continue
        accepted_params.append((D, H, P))
        bp = MetaSurfaceParam(diameter_nm=round(D, 1), height_nm=round(H, 1), period_nm=round(P, 1),
                              material=material, substrate=substrate)
        de76 = float(np.sqrt(np.sum((np.array(rgb) - target)**2)) * 30)
        result.append((None, bp, list(rgb), de76, de))
        if len(result) >= max_results:
            break

    return result if result else None

def _inverse_design_numpy(target_rgb, n_steps=300, n_restarts=20, material="TiO2 (anatase)", substrate="SiO2 (fused silica)", theta=0.0):
    """Gradient-based inverse design using numpy finite differences (no PyTorch needed).

    Uses central finite differences to approximate gradients through the ONNX model.
    Each step: 6 ONNX forward passes (for dD, dH, dP) at ~0.5ms each.
    Total: ~1s per restart, comparable to PyTorch version.
    """
    if not _ORT_AVAILABLE:
        return None
    if material not in MATERIAL_CODES:
        return None
    if substrate not in SUBSTRATE_CODES:
        return None

    target = np.array(target_rgb, dtype=np.float32)
    eps = 1.0  # finite difference step (nm)
    best_loss = 1e9
    best_result = None
    rng = np.random.RandomState(42)

    # Adam optimizer state (numpy)
    beta1, beta2 = 0.9, 0.999
    lr = 0.5

    for restart in range(n_restarts):
        d = np.clip(rng.uniform(50, 350), 50, 350)
        h = np.clip(rng.uniform(80, 600), 80, 600)
        p = np.clip(rng.uniform(200, 600), 200, 600)
        # Adam moments
        m_d, m_h, m_p = 0.0, 0.0, 0.0
        v_d, v_h, v_p = 0.0, 0.0, 0.0

        for step in range(n_steps):
            # Forward at current point
            rgb_c = predict_rgb(d, h, p, theta, "TE", material, substrate)
            if rgb_c is None:
                break
            loss_c = float(np.mean((target - rgb_c) ** 2))

            # Finite difference: D
            rgb_dp = predict_rgb(d + eps, h, p, theta, "TE", material, substrate)
            rgb_dm = predict_rgb(d - eps, h, p, theta, "TE", material, substrate)
            if rgb_dp is None or rgb_dm is None:
                break
            loss_dp = float(np.mean((target - rgb_dp) ** 2))
            loss_dm = float(np.mean((target - rgb_dm) ** 2))
            grad_d = (loss_dp - loss_dm) / (2 * eps)

            # Finite difference: H
            rgb_hp = predict_rgb(d, h + eps, p, theta, "TE", material, substrate)
            rgb_hm = predict_rgb(d, h - eps, p, theta, "TE", material, substrate)
            if rgb_hp is None or rgb_hm is None:
                break
            loss_hp = float(np.mean((target - rgb_hp) ** 2))
            loss_hm = float(np.mean((target - rgb_hm) ** 2))
            grad_h = (loss_hp - loss_hm) / (2 * eps)

            # Finite difference: P
            rgb_pp = predict_rgb(d, h, p + eps, theta, "TE", material, substrate)
            rgb_pm = predict_rgb(d, h, p - eps, theta, "TE", material, substrate)
            if rgb_pp is None or rgb_pm is None:
                break
            loss_pp = float(np.mean((target - rgb_pp) ** 2))
            loss_pm = float(np.mean((target - rgb_pm) ** 2))
            grad_p = (loss_pp - loss_pm) / (2 * eps)

            # Adam update
            m_d = beta1 * m_d + (1 - beta1) * grad_d
            m_h = beta1 * m_h + (1 - beta1) * grad_h
            m_p = beta1 * m_p + (1 - beta1) * grad_p
            v_d = beta2 * v_d + (1 - beta2) * grad_d ** 2
            v_h = beta2 * v_h + (1 - beta2) * grad_h ** 2
            v_p = beta2 * v_p + (1 - beta2) * grad_p ** 2

            t = step + 1
            m_d_hat = m_d / (1 - beta1 ** t)
            m_h_hat = m_h / (1 - beta1 ** t)
            m_p_hat = m_p / (1 - beta1 ** t)
            v_d_hat = v_d / (1 - beta2 ** t)
            v_h_hat = v_h / (1 - beta2 ** t)
            v_p_hat = v_p / (1 - beta2 ** t)

            d -= lr * m_d_hat / (np.sqrt(v_d_hat) + 1e-8)
            h -= lr * m_h_hat / (np.sqrt(v_h_hat) + 1e-8)
            p -= lr * m_p_hat / (np.sqrt(v_p_hat) + 1e-8)

            # Clamp
            d = np.clip(d, 50, 350)
            h = np.clip(h, 80, 600)
            p = np.clip(p, max(d * 1.2, 200), 600)

            if loss_c < best_loss:
                best_loss = loss_c
                best_result = (float(d), float(h), float(p), [float(x) for x in rgb_c], float(loss_c))

    if best_result is None:
        return None
    return best_result


# ---- numpy-based dual pillar inverse design ----
def _inverse_design_dual_numpy(target_rgb, n_steps=300, n_restarts=30, material="TiO2 (anatase)", substrate="SiO2 (fused silica)", theta=0.0):
    """Dual-pillar gradient-based inverse design using numpy finite differences."""
    if not _DUAL_ORT_AVAILABLE:
        return None
    if material not in MATERIAL_CODES:
        return None
    if substrate not in SUBSTRATE_CODES:
        # Dual ML only supports SiO2, fall through to physical model in app.py
        return None

    target = np.array(target_rgb, dtype=np.float32)
    eps = 1.0
    best_loss = 1e9
    best_result = None
    rng = np.random.RandomState(42)
    beta1, beta2 = 0.9, 0.999
    lr = 0.5

    for restart in range(n_restarts):
        d1 = np.clip(rng.uniform(60, 267), 60, 267)
        h1 = np.clip(rng.uniform(80, 600), 80, 600)
        d2 = np.clip(rng.uniform(60, 267), 60, 267)
        h2 = np.clip(rng.uniform(80, 600), 80, 600)
        p = np.clip(rng.uniform(200, 600), 200, 600)

        m = np.zeros(5)
        v = np.zeros(5)

        for step in range(n_steps):
            params = np.array([d1, h1, d2, h2, p])
            rgb_c = predict_dual_rgb(d1, h1, d2, h2, p, theta, "TE", material, substrate)
            if rgb_c is None:
                break
            loss_c = float(np.mean((target - rgb_c) ** 2))

            grads = np.zeros(5)
            for i in range(5):
                p_plus = params.copy(); p_plus[i] += eps
                p_minus = params.copy(); p_minus[i] -= eps
                r_plus = predict_dual_rgb(p_plus[0], p_plus[1], p_plus[2], p_plus[3], p_plus[4], theta, "TE", material, substrate)
                r_minus = predict_dual_rgb(p_minus[0], p_minus[1], p_minus[2], p_minus[3], p_minus[4], theta, "TE", material, substrate)
                if r_plus is None or r_minus is None:
                    grads[i] = 0
                else:
                    grads[i] = (float(np.mean((target - r_plus) ** 2)) - float(np.mean((target - r_minus) ** 2))) / (2 * eps)

            m = beta1 * m + (1 - beta1) * grads
            v = beta2 * v + (1 - beta2) * grads ** 2
            t = step + 1
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            update = lr * m_hat / (np.sqrt(v_hat) + 1e-8)

            d1 -= update[0]; h1 -= update[1]; d2 -= update[2]; h2 -= update[3]; p -= update[4]
            d1 = np.clip(d1, 60, 267); h1 = np.clip(h1, 80, 600)
            d2 = np.clip(d2, 60, 267); h2 = np.clip(h2, 80, 600)
            min_p = max(d1, d2) * 1.2 + 20
            p = np.clip(p, min_p, 600)

            if loss_c < best_loss:
                best_loss = loss_c
                best_result = (float(d1), float(h1), float(d2), float(h2), float(p),
                               [float(x) for x in rgb_c], float(loss_c))

    return best_result
