# ml_module.py - ML acceleration for metasurface color engine (ONNX Runtime + optional PyTorch)
import os
import numpy as np
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

MATERIAL_CODES = {"TiO2 (anatase)": 0, "a-Si (amorphous)": 1, "Si3N4 (nitride)": 2, "Al2O3 (sapphire)": 3}
SUBSTRATE_CODES = {"SiO2 (fused silica)": 0, "Si3N4 (nitride)": 1, "Al2O3 (sapphire)": 2}

def _spectrum_to_rgb(spec: np.ndarray) -> np.ndarray:
    return spectrum_to_srgb(WL, np.clip(spec, 0, None))

# ---- ONNX init ----
def init_ml():
    global _ORT_AVAILABLE, _ORT_SESSION, _ORT_IS_V8
    try:
        import onnxruntime as ort
        _dir = os.path.dirname(os.path.abspath(__file__))
        path_v8 = os.path.join(_dir, "models", "forward_mlp_v8_sub.onnx")
        path_v7 = os.path.join(_dir, "models", "forward_mlp_v7_multi.onnx")
        if os.path.exists(path_v8):
            _ORT_SESSION = ort.InferenceSession(path_v8, providers=["CPUExecutionProvider"])
            _ORT_IS_V8 = True
        elif os.path.exists(path_v7):
            _ORT_SESSION = ort.InferenceSession(path_v7, providers=["CPUExecutionProvider"])
            _ORT_IS_V8 = False
        else:
            return False
        _ORT_AVAILABLE = True
        return True
    except Exception:
        return False

def init_dual_ml():
    global _DUAL_ORT_AVAILABLE, _DUAL_ORT_SESSION, _DUAL_IS_V3
    try:
        import onnxruntime as ort
        _dir = os.path.dirname(os.path.abspath(__file__))
        path_v3 = os.path.join(_dir, "models", "dual_mlp_v3_multi.onnx")
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
        _dir = os.path.dirname(os.path.abspath(__file__))
        path_v8 = os.path.join(_dir, "models", "forward_mlp_v8_sub.pt")
        path_v7 = os.path.join(_dir, "models", "forward_mlp_v7_multi.pt")
        path = path_v8 if os.path.exists(path_v8) else path_v7
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

# ---- predict (ONNX) ----
def predict_rgb(d_nm, h_nm, p_nm, angle_deg=0.0, polarization="TE", material="TiO2 (anatase)", substrate="SiO2 (fused silica)"):
    if not _ORT_AVAILABLE:
        return None
    if material not in MATERIAL_CODES:
        return None
    x = _build_input(d_nm, h_nm, p_nm, angle_deg, polarization, material, substrate)
    spec = _ORT_SESSION.run(None, {"input": x})[0][0]
    return _spectrum_to_rgb(spec)

def predict_spectrum(d_nm, h_nm, p_nm, angle_deg=0.0, polarization="TE", material="TiO2 (anatase)", substrate="SiO2 (fused silica)"):
    if not _ORT_AVAILABLE:
        return None
    if material not in MATERIAL_CODES:
        return None
    x = _build_input(d_nm, h_nm, p_nm, angle_deg, polarization, material, substrate)
    spec = _ORT_SESSION.run(None, {"input": x})[0][0]
    return np.clip(spec, 0, None)

def predict_dual_spectrum(d1_nm, h1_nm, d2_nm, h2_nm, p_nm, angle_deg=0.0, polarization="TE", material="TiO2 (anatase)"):
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

def predict_dual_rgb(d1_nm, h1_nm, d2_nm, h2_nm, p_nm, angle_deg=0.0, polarization="TE", material="TiO2 (anatase)"):
    spec = predict_dual_spectrum(d1_nm, h1_nm, d2_nm, h2_nm, p_nm, angle_deg, polarization, material)
    if spec is None:
        return None
    return _spectrum_to_rgb(spec)

# ---- inverse design (PyTorch gradient-based, requires torch) ----
def inverse_design_ml(target_rgb, n_steps=300, n_restarts=40, material="TiO2 (anatase)", substrate="SiO2 (fused silica)"):
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
            best_result = (d_f, h_f, p_f, pred, fl)
    return best_result
