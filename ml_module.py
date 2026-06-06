# ml_module.py - ML acceleration for metasurface color engine (ONNX Runtime + optional PyTorch)
import os, numpy as np

# ---- globals ----
_ORT_AVAILABLE = False
_ORT_SESSION = None
_ORT_IS_V8 = False
_DUAL_ORT_AVAILABLE = False
_DUAL_ORT_SESSION = None
_DUAL_IS_V3 = False
_TORCH_FWD = None      # PyTorch model for gradient-based inverse design
_TORCH_IS_V8 = False

# CIE 1931 color matching functions (81 points, 380-780 nm)
_CIE_X = np.array([0.001368,0.002236,0.004243,0.007650,0.014310,0.023190,0.043510,0.077630,0.134380,0.214770,0.283900,0.328500,0.348280,0.348060,0.336200,0.318700,0.290800,0.251100,0.195360,0.142100,0.095640,0.058010,0.032010,0.014700,0.004900,0.002400,0.009300,0.029100,0.063270,0.109600,0.165500,0.225750,0.290400,0.359700,0.433450,0.512050,0.594500,0.678400,0.762100,0.842500,0.916300,0.978600,1.026300,1.056700,1.062200,1.045600,1.002600,0.938400,0.854450,0.751400,0.642400,0.541900,0.447900,0.360800,0.283500,0.218700,0.164900,0.121200,0.087400,0.063600,0.046770,0.032900,0.022700,0.015840,0.011359,0.008111,0.005790,0.004109,0.002899,0.002049,0.001440,0.001000,0.000690,0.000476,0.000332,0.000235,0.000166,0.000117,0.000083,0.000059,0.000042], dtype=np.float32)
_CIE_Y = np.array([0.000039,0.000064,0.000120,0.000217,0.000396,0.000640,0.001210,0.002180,0.004000,0.007300,0.011600,0.016840,0.023000,0.029800,0.038000,0.048000,0.060000,0.073900,0.090980,0.112600,0.139020,0.169300,0.208020,0.258600,0.323000,0.407300,0.503000,0.608200,0.710000,0.793200,0.862000,0.914850,0.954000,0.980300,0.994950,1.000000,0.995000,0.978600,0.952000,0.915400,0.870000,0.816300,0.757000,0.694900,0.631000,0.566800,0.503000,0.441200,0.381000,0.321000,0.265000,0.217000,0.175000,0.138200,0.107000,0.081600,0.061000,0.044580,0.032000,0.023200,0.017000,0.011920,0.008210,0.005723,0.004102,0.002929,0.002091,0.001484,0.001047,0.000740,0.000520,0.000361,0.000249,0.000172,0.000120,0.000085,0.000060,0.000042,0.000030,0.000021,0.000015], dtype=np.float32)
_CIE_Z = np.array([0.006450,0.010550,0.020050,0.036210,0.067850,0.110200,0.207400,0.371300,0.645600,1.039050,1.385600,1.622960,1.747060,1.782600,1.772110,1.744100,1.669200,1.528100,1.287640,0.999550,0.716900,0.484400,0.311900,0.190300,0.104200,0.049200,0.020300,0.008700,0.003900,0.002100,0.001650,0.001100,0.000800,0.000550,0.000350,0.000250,0.000150,0.000100,0.000050,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000], dtype=np.float32)
_WL = np.linspace(380, 780, 81, dtype=np.float32)
_CIE_NORM = np.trapezoid(_CIE_Y, _WL)
_SRGB_M = np.array([[3.2406,-1.5372,-0.4986],[-0.9689,1.8758,0.0415],[0.0557,-0.2040,1.0570]], dtype=np.float32)

MATERIAL_CODES = {"TiO2 (anatase)": 0, "a-Si (amorphous)": 1, "Si3N4 (nitride)": 2, "Al2O3 (sapphire)": 3}
SUBSTRATE_CODES = {"SiO2 (fused silica)": 0, "Si3N4 (nitride)": 1, "Al2O3 (sapphire)": 2}

def _spectrum_to_rgb(spec: np.ndarray) -> np.ndarray:
    X = np.trapezoid(spec * _CIE_X, _WL)
    Y = np.trapezoid(spec * _CIE_Y, _WL)
    Z = np.trapezoid(spec * _CIE_Z, _WL)
    xyz = np.array([X / _CIE_NORM, Y / _CIE_NORM, Z / _CIE_NORM])
    rgb_lin = _SRGB_M @ xyz
    rgb = np.where(rgb_lin <= 0.0031308, 12.92 * rgb_lin, 1.055 * np.power(np.maximum(rgb_lin, 0), 1 / 2.4) - 0.055)
    return np.clip(rgb, 0, 1)

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

# ---- predict (ONNX) ----
def predict_rgb(d_nm, h_nm, p_nm, angle_deg=0.0, polarization="TE", material="TiO2 (anatase)", substrate="SiO2 (fused silica)"):
    if not _ORT_AVAILABLE:
        return None
    if material not in MATERIAL_CODES:
        return None
    pol_code = 0.0 if polarization.startswith("TE") else 1.0
    mat_code = float(MATERIAL_CODES.get(material, 0))
    sub_code = float(SUBSTRATE_CODES.get(substrate, 0))
    if _ORT_IS_V8:
        x = np.array([[(d_nm - 50) / 300, (h_nm - 80) / 520, (p_nm - 200) / 400, angle_deg / 80, pol_code, mat_code, sub_code]], dtype=np.float32)
    else:
        x = np.array([[(d_nm - 50) / 300, (h_nm - 80) / 520, (p_nm - 200) / 400, angle_deg / 80, pol_code, mat_code]], dtype=np.float32)
    spec = _ORT_SESSION.run(None, {"input": x})[0][0]
    return _spectrum_to_rgb(np.clip(spec, 0, None))

def predict_spectrum(d_nm, h_nm, p_nm, angle_deg=0.0, polarization="TE", material="TiO2 (anatase)", substrate="SiO2 (fused silica)"):
    if not _ORT_AVAILABLE:
        return None
    if material not in MATERIAL_CODES:
        return None
    pol_code = 0.0 if polarization.startswith("TE") else 1.0
    mat_code = float(MATERIAL_CODES.get(material, 0))
    sub_code = float(SUBSTRATE_CODES.get(substrate, 0))
    if _ORT_IS_V8:
        x = np.array([[(d_nm - 50) / 300, (h_nm - 80) / 520, (p_nm - 200) / 400, angle_deg / 80, pol_code, mat_code, sub_code]], dtype=np.float32)
    else:
        x = np.array([[(d_nm - 50) / 300, (h_nm - 80) / 520, (p_nm - 200) / 400, angle_deg / 80, pol_code, mat_code]], dtype=np.float32)
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
        x = np.array([[(d1_nm - 60) / 247, (h1_nm - 80) / 520, (d2_nm - 60) / 247, (h2_nm - 80) / 520, (p_nm - 200) / 400, angle_deg / 60, pol_code, mat_code]], dtype=np.float32)
    else:
        x = np.array([[(d1_nm - 60) / 247, (h1_nm - 80) / 520, (d2_nm - 60) / 247, (h2_nm - 80) / 520, (p_nm - 200) / 400, angle_deg / 60, pol_code]], dtype=np.float32)
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

    cie_x = torch.from_numpy(_CIE_X).float()
    cie_y = torch.from_numpy(_CIE_Y).float()
    cie_z = torch.from_numpy(_CIE_Z).float()
    cie_norm = torch.tensor(_CIE_NORM, dtype=torch.float32)
    srgb_m = torch.from_numpy(_SRGB_M).float()
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
