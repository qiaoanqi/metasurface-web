# ml_module.py - ML acceleration for metasurface color engine (ONNX Runtime + optional PyTorch)
import os
import numpy as np

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
            local_dir_use_symlinks=False)
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
        path_v7 = _ensure_model_file("models/forward_mlp_v7_multi.onnx")
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
        path_v7 = _ensure_model_file("models/forward_mlp_v7_multi.pt")
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
    if substrate not in SUBSTRATE_CODES:
        return None
    x = _build_input(d_nm, h_nm, p_nm, angle_deg, polarization, material, substrate)
    spec = _ORT_SESSION.run(None, {"input": x})[0][0]
    return _spectrum_to_rgb(spec)

def predict_spectrum(d_nm, h_nm, p_nm, angle_deg=0.0, polarization="TE", material="TiO2 (anatase)", substrate="SiO2 (fused silica)"):
    if not _ORT_AVAILABLE:
        return None
    if material not in MATERIAL_CODES:
        return None
    if substrate not in SUBSTRATE_CODES:
        return None
    x = _build_input(d_nm, h_nm, p_nm, angle_deg, polarization, material, substrate)
    spec = _ORT_SESSION.run(None, {"input": x})[0][0]
    return np.clip(spec, 0, None)

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


# ---- inverse design (numpy finite-difference, no torch) ----
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
