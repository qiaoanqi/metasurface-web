# torch_model.py - PyTorch batch Lorentzian/Fano model (v2 - full Cauchy + coherent)
import torch
import numpy as np
from ccm import get_ccm
from color_utils import CIE_X as _CIE_X_NP, CIE_Y as _CIE_Y_NP, CIE_Z as _CIE_Z_NP
from color_utils import SRGB_M as _SRGB_M_NP

# ============================================================
# CIE 1931 (81 points, 380-780nm step 5nm) from color_utils
# ============================================================
CIE_X = torch.from_numpy(_CIE_X_NP.astype(np.float32))
CIE_Y = torch.from_numpy(_CIE_Y_NP.astype(np.float32))
CIE_Z = torch.from_numpy(_CIE_Z_NP.astype(np.float32))
WL = torch.linspace(380, 780, 81)
# torch 1.x/2.x compatibility
try:
    CIE_NORM = torch.trapezoid(CIE_Y, WL)
except AttributeError:
    CIE_NORM = torch.trapz(CIE_Y, WL)
SRGB_M = torch.from_numpy(_SRGB_M_NP.astype(np.float32))

# ============================================================
# Cauchy dispersion model
# ============================================================
CAUCHY = {
    "SiO2 (fused silica)": (1.4580, 0.00354, 0.0),
    "TiO2 (anatase)":      (2.3000, 0.03500, 0.0),
    "Si3N4":               (1.9900, 0.01200, 0.0),
    "a-Si (amorphous)":    (3.8000, 0.08000, 0.0),
    "Al2O3 (alumina)":     (1.7546, 0.00500, 0.0),
    "Air":                 (1.0003, 0.00000, 0.0),
}


def _cauchy_n(wl, material):
    """Cauchy dispersion: n(lambda) = A + B/lambda^2 + C/lambda^4"""
    if material is None:
        material = "TiO2 (anatase)"
    A, B, C = CAUCHY.get(material, CAUCHY["TiO2 (anatase)"])
    wl_um = wl / 1000.0
    wl2 = wl_um ** 2
    return A + B / wl2 + C / (wl2 ** 2)



def _ccm_fill_torch(D, P, coeff):
    """PyTorch-native CCM fill factor (differentiable).  For circular pillars L=W=D."""
    P_safe = torch.clamp(P, min=200.0)
    f0 = torch.pi * (D / 2.0) ** 2 / P_safe ** 2
    lp = D / P_safe
    # Circular symmetry: l_w = w_l = 0, alpha_4 and alpha_5 terms vanish
    delta_f = (
        (coeff.alpha_1 + coeff.alpha_2 + coeff.alpha_3) * lp ** 2 +
        coeff.beta_1 * (1.0 - lp) ** 2 +
        coeff.gamma
    )
    return torch.clamp(f0 + delta_f, 0.01, 0.75).unsqueeze(1)
def batch_lorentzian_spectrum(D, H, P, theta=0.0, pol_TE=True, material=None, substrate=None):
    """
    Batch Fano resonance spectrum with full Cauchy dispersion and coherent ED+MD addition.
    Fano lineshape: R = (q+eps)^2 / ((1+q^2)(1+eps^2)) replaces Lorentzian for better asymmetry.
    Supports multiple materials via Cauchy coefficients.
    """
    if not isinstance(D, torch.Tensor): D = torch.tensor(D, dtype=torch.float32)
    if not isinstance(H, torch.Tensor): H = torch.tensor(H, dtype=torch.float32)
    if not isinstance(P, torch.Tensor): P = torch.tensor(P, dtype=torch.float32)
    if not isinstance(theta, torch.Tensor): theta = torch.tensor(theta, dtype=torch.float32)

    D = D.view(-1); H = H.view(-1); P = P.view(-1); theta = theta.view(-1)
    batch = D.shape[0]
    wl = WL.unsqueeze(0)  # (1, 81)

    # Wavelength-dependent n (batch, 81)
    n_mat = _cauchy_n(wl, material)  # (1, 81) -> broadcast to (batch, 81) automatically

    # Peak wavelengths (batch, 1) - use n@550nm as reference
    n550 = _cauchy_n(torch.tensor(550.0), material).item()
    n_sub550 = _cauchy_n(torch.tensor(550.0), substrate).item() if substrate else 1.4580
    n_env550 = (1.0 + n_sub550) / 2.0  # effective environment: half air, half substrate
    dn550 = n550 - n_env550

    lam_ed = (360 + 0.55*(D-60) + 0.12*(H-120) + 32*dn550).unsqueeze(1)  # (batch, 1)
    sigma_ed = torch.clamp(26 + 0.10*(D-200), min=8).unsqueeze(1)  # FDTD-calibrated
    lam_md = (400 + 0.75*(D-60) + 0.25*(H-120) + 32*dn550).unsqueeze(1)
    sigma_md = torch.clamp(35 + 0.12*(D-200), min=10).unsqueeze(1)  # FDTD-calibrated

    # CCM effective fill factor: f_eff = f0 + Delta-f (PyTorch-native, differentiable)
    ccm = get_ccm(material or "TiO2 (anatase)", substrate or "SiO2 (fused silica)")
    fill = _ccm_fill_torch(D, P, ccm.coeff)
    fill_amp = (0.30 + 0.80 * fill)
    loss = torch.exp(-0.0006*torch.clamp(H-600, min=0)).unsqueeze(1)

    # Angle
    theta_rad = theta * np.pi / 180.0
    sin2 = torch.sin(theta_rad)**2
    sin2 = sin2.unsqueeze(1)

    ed_shift = -45*sin2; md_shift = -20*sin2
    ed_amp_a = 1.0 - 0.10*sin2; md_amp_a = 1.0 - 0.04*sin2
    if not pol_TE:
        ed_shift = -18*sin2; md_shift = -8*sin2
        ed_amp_a = 1.0 - 0.25*sin2; md_amp_a = 1.0 - 0.12*sin2

    # Dynamic weight
    w_ed = torch.clamp(0.80 - 0.003*(D-60), 0.0, 0.80).unsqueeze(1)
    w_md = 1.0 - w_ed

    # Fano asymmetry parameters
    aspect = H.unsqueeze(1) / D.unsqueeze(1).clamp(min=50.0)
    q_ed = torch.clamp(2.5 + 0.5 * (aspect - 1.0), 1.5, 6.0)
    q_md = torch.clamp(4.0 + 0.3 * (aspect - 1.0), 2.5, 8.0)
    inv_norm_ed = 1.0 / torch.sqrt(1.0 + q_ed**2)
    inv_norm_md = 1.0 / torch.sqrt(1.0 + q_md**2)

    ed_center = lam_ed + ed_shift
    detune_ed = (wl - ed_center) / sigma_ed
    fano_num_ed = q_ed + detune_ed
    denom_ed = 1.0 + detune_ed**2
    r_ed_real = fano_num_ed * inv_norm_ed / denom_ed * torch.sqrt(ed_amp_a)
    r_ed_imag = -fano_num_ed * detune_ed * inv_norm_ed / denom_ed * torch.sqrt(ed_amp_a)

    md_center = lam_md + md_shift
    detune_md = (wl - md_center) / sigma_md
    fano_num_md = q_md + detune_md
    denom_md = 1.0 + detune_md**2
    r_md_real = fano_num_md * inv_norm_md / denom_md * torch.sqrt(md_amp_a)
    r_md_imag = -fano_num_md * detune_md * inv_norm_md / denom_md * torch.sqrt(md_amp_a)

    r_real = w_ed*r_ed_real + w_md*r_md_real
    r_imag = w_ed*r_ed_imag + w_md*r_md_imag
    intensity = (r_real**2 + r_imag**2) * fill_amp * loss

    return torch.clamp(intensity, 0.0, 1.0)


def batch_spectrum_to_rgb(spectrum):
    """Batch spectrum (B,81) -> sRGB (B,3)"""
    wl = WL.unsqueeze(0)
    X = torch.trapezoid(spectrum * CIE_X.unsqueeze(0), wl, dim=1)
    Y = torch.trapezoid(spectrum * CIE_Y.unsqueeze(0), wl, dim=1)
    Z = torch.trapezoid(spectrum * CIE_Z.unsqueeze(0), wl, dim=1)
    xyz = torch.stack([X/CIE_NORM, Y/CIE_NORM, Z/CIE_NORM], dim=1).float()
    rgb_lin = xyz @ SRGB_M.T
    rgb = torch.where(rgb_lin <= 0.0031308, 12.92*rgb_lin, 1.055*torch.clamp(rgb_lin, min=0.0).pow(1/2.4)-0.055)
    return torch.clamp(rgb, 0.0, 1.0)


def batch_single_pillar_rgb(D, H, P, theta=0.0, pol_TE=True, material=None, substrate=None):
    """End-to-end batch: D,H,P -> spectrum -> RGB"""
    spec = batch_lorentzian_spectrum(D, H, P, theta, pol_TE, material, substrate)
    return batch_spectrum_to_rgb(spec)


def batch_color_map_grid(D_grid, H_grid, P_val=400.0):
    """Batch color map grid computation."""
    if D_grid.ndim == 1 and H_grid.ndim == 1:
        D_mesh, H_mesh = torch.meshgrid(torch.as_tensor(D_grid, dtype=torch.float32),
                                         torch.as_tensor(H_grid, dtype=torch.float32), indexing='ij')
    else:
        D_mesh = torch.as_tensor(D_grid, dtype=torch.float32)
        H_mesh = torch.as_tensor(H_grid, dtype=torch.float32)
    n_total = D_mesh.numel()
    D_flat = D_mesh.flatten(); H_flat = H_mesh.flatten()
    P_flat = torch.full_like(D_flat, float(P_val))
    rgb_flat = batch_single_pillar_rgb(D_flat, H_flat, P_flat)
    return rgb_flat.reshape(D_mesh.shape[0], D_mesh.shape[1], 3)


# ============================================================
# Dual-pillar batch functions
# ============================================================

def batch_dual_pillar_spectrum(D1, H1, D2, H2, P, theta=0.0, pol_TE=True, material=None, substrate=None):
    """Incoherent sum of two pillar spectra."""
    spec1 = batch_lorentzian_spectrum(D1, H1, P, theta, pol_TE, material, substrate)
    spec2 = batch_lorentzian_spectrum(D2, H2, P, theta, pol_TE, material, substrate)
    return torch.clamp(spec1 + spec2, 0.0, 1.0)


def batch_dual_pillar_rgb(D1, H1, D2, H2, P, theta=0.0, pol_TE=True, material=None, substrate=None):
    """Dual pillar -> RGB."""
    spec = batch_dual_pillar_spectrum(D1, H1, D2, H2, P, theta, pol_TE, material, substrate)
    return batch_spectrum_to_rgb(spec)


def inverse_design_dual(target_rgb, n_restarts=10, steps=50, lr=0.05,
                         material=None, substrate=None, theta=0.0, pol_TE=True, p_fixed=400.0):
    """
    Gradient-based inverse design for dual-pillar structures.
    Uses Adam optimizer with random restarts.
    Returns (best_d1, best_h1, best_d2, best_h2, best_p, best_rgb, best_loss).
    """
    if not isinstance(target_rgb, torch.Tensor):
        target = torch.tensor(target_rgb, dtype=torch.float32).unsqueeze(0)
    else:
        target = target_rgb.clone().detach().unsqueeze(0)

    device = target.device
    n = n_restarts

    # Random initial parameters
    d1 = (torch.rand(n, device=device) * 207 + 60).requires_grad_(True)
    h1 = (torch.rand(n, device=device) * 520 + 80).requires_grad_(True)
    d2 = (torch.rand(n, device=device) * 207 + 60).requires_grad_(True)
    h2 = (torch.rand(n, device=device) * 520 + 80).requires_grad_(True)
    if p_fixed is not None:
        p = torch.full((n,), float(p_fixed), device=device)
    else:
        p = (torch.rand(n, device=device) * 400 + 200).requires_grad_(True)

    # Optimize all restarts together
    params = [d1, h1, d2, h2]
    if p_fixed is None:
        params.append(p)
    optimizer = torch.optim.Adam(params, lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        d1_c = torch.clamp(d1, 60, 267)
        h1_c = torch.clamp(h1, 80, 600)
        d2_c = torch.clamp(d2, 60, 267)
        h2_c = torch.clamp(h2, 80, 600)
        p_c = torch.clamp(p, 200, 600)
        spec = batch_dual_pillar_spectrum(d1_c, h1_c, d2_c, h2_c, p_c, theta, pol_TE, material, substrate)
        pred = batch_spectrum_to_rgb(spec)
        loss = ((pred - target) ** 2).sum(dim=1).mean()
        loss.backward()
        optimizer.step()

    # Final evaluation to find best restart
    with torch.no_grad():
        d1_f = torch.clamp(d1, 60, 267)
        h1_f = torch.clamp(h1, 80, 600)
        d2_f = torch.clamp(d2, 60, 267)
        h2_f = torch.clamp(h2, 80, 600)
        p_f = torch.clamp(p, 200, 600)
        spec = batch_dual_pillar_spectrum(d1_f, h1_f, d2_f, h2_f, p_f, theta, pol_TE, material, substrate)
        pred = batch_spectrum_to_rgb(spec)
        losses = ((pred - target) ** 2).sum(dim=1)
        best_idx = torch.argmin(losses).item()
        best_d1 = float(d1_f[best_idx])
        best_h1 = float(h1_f[best_idx])
        best_d2 = float(d2_f[best_idx])
        best_h2 = float(h2_f[best_idx])
        best_p = float(p_f[best_idx])
        best_rgb = pred[best_idx].cpu().numpy()
        best_loss = float(losses[best_idx])

    return best_d1, best_h1, best_d2, best_h2, best_p, best_rgb, best_loss


def batch_spectrum_to_lab(spec):
    """Batch spectrum (N,81) -> CIE L*a*b* (N,3). Differentiable."""
    cw = WL.to(spec.device)
    # XYZ from spectrum
    X = torch.trapezoid(spec * CIE_X.to(spec.device).unsqueeze(0), cw, dim=1)
    Y = torch.trapezoid(spec * CIE_Y.to(spec.device).unsqueeze(0), cw, dim=1)
    Z = torch.trapezoid(spec * CIE_Z.to(spec.device).unsqueeze(0), cw, dim=1)
    norm = CIE_NORM.to(spec.device)
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883  # D65 white point (CIE_NORM-scaled)
    # X,Y,Z are raw integrals; divide by CIE_NORM then by D65 white point
    xr = X / (Xn * norm)
    yr = Y / (Yn * norm)
    zr = Z / (Zn * norm)
    delta = 6.0 / 29.0
    delta3 = delta ** 3
    # f(t) = t^(1/3) for t > delta3, else t/(3*delta^2) + 4/29
    fx = torch.where(xr > delta3, xr ** (1.0/3.0), xr / (3.0 * delta**2) + 4.0/29.0)
    fy = torch.where(yr > delta3, yr ** (1.0/3.0), yr / (3.0 * delta**2) + 4.0/29.0)
    fz = torch.where(zr > delta3, zr ** (1.0/3.0), zr / (3.0 * delta**2) + 4.0/29.0)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return torch.stack([L, a, b], dim=1)


def delta_e2000_torch(lab1, lab2, kL=1.0, kC=1.0, kH=1.0):
    """CIEDE2000 colour difference between two LAB tensors (N,3) and (N,3).
    Differentiable approximation using PyTorch operations."""
    L1, a1, b1 = lab1[:, 0], lab1[:, 1], lab1[:, 2]
    L2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]

    C1 = torch.sqrt(a1**2 + b1**2)
    C2 = torch.sqrt(a2**2 + b2**2)
    Cbar = (C1 + C2) / 2.0
    C7 = Cbar ** 7
    G = 0.5 * (1.0 - torch.sqrt(C7 / (C7 + 25.0**7 + 1e-12)))
    a1p = a1 * (1.0 + G)
    a2p = a2 * (1.0 + G)

    C1p = torch.sqrt(a1p**2 + b1**2)
    C2p = torch.sqrt(a2p**2 + b2**2)
    Cbarp = (C1p + C2p) / 2.0

    h1p = torch.atan2(b1, a1p + 1e-12) * 180.0 / torch.pi
    h2p = torch.atan2(b2, a2p + 1e-12) * 180.0 / torch.pi
    h1p = torch.where(h1p < 0, h1p + 360.0, h1p)
    h2p = torch.where(h2p < 0, h2p + 360.0, h2p)

    dh = h2p - h1p
    dh = torch.where(torch.abs(dh) > 180.0, dh - torch.sign(dh) * 360.0, dh)
    dHp = 2.0 * torch.sqrt(C1p * C2p + 1e-12) * torch.sin(dh * torch.pi / 360.0 / 2.0)

    dLp = L2 - L1
    dCp = C2p - C1p

    Lbarp = (L1 + L2) / 2.0
    hbarp = (h1p + h2p) / 2.0
    hbarp = torch.where(torch.abs(dh) > 180.0, hbarp - torch.sign(hbarp - 180.0) * 180.0, hbarp)

    T = (1.0 - 0.17 * torch.cos((hbarp - 30.0) * torch.pi / 180.0)
         + 0.24 * torch.cos(2.0 * hbarp * torch.pi / 180.0)
         + 0.32 * torch.cos((3.0 * hbarp + 6.0) * torch.pi / 180.0)
         - 0.20 * torch.cos((4.0 * hbarp - 63.0) * torch.pi / 180.0))

    SL = 1.0 + (0.015 * (Lbarp - 50.0)**2) / torch.sqrt(20.0 + (Lbarp - 50.0)**2 + 1e-12)
    SC = 1.0 + 0.045 * Cbarp
    SH = 1.0 + 0.015 * Cbarp * T

    dtheta = 30.0 * torch.exp(-((hbarp - 275.0) / 25.0)**2)
    RC = 2.0 * torch.sqrt(Cbarp**7 / (Cbarp**7 + 25.0**7 + 1e-12))
    RT = -torch.sin(2.0 * dtheta * torch.pi / 180.0) * RC

    dE = torch.sqrt(
        (dLp / (kL * SL + 1e-12))**2 +
        (dCp / (kC * SC + 1e-12))**2 +
        (dHp / (kH * SH + 1e-12))**2 +
        RT * (dCp / (kC * SC + 1e-12)) * (dHp / (kH * SH + 1e-12)) + 1e-12
    )
    return dE


def inverse_design_dual_v2(target_rgb, n_restarts=24, steps=100, lr=0.05,
                             material=None, substrate=None, theta=0.0, pol_TE=True,
                             p_fixed=None, loss_type='de2000'):
    """
    Gradient-based inverse design for dual-pillar (v2: CIEDE2000 loss + cosine annealing).
    Uses Adam with cosine annealing LR schedule for better convergence.
    loss_type: 'mse' (RGB MSE) or 'de2000' (CIEDE2000 perceptual).
    """
    if not isinstance(target_rgb, torch.Tensor):
        target = torch.tensor(target_rgb, dtype=torch.float32).unsqueeze(0)
    else:
        target = target_rgb.clone().detach().unsqueeze(0)

    device = target.device
    n = n_restarts

    # Pre-compute target LAB once for CIEDE2000 loss
    if loss_type == 'de2000':
        target_rgb_np = target.cpu().numpy()
        from color_utils import rgb_to_lab as _r2l
        target_lab_np = _r2l(target_rgb_np)
        target_lab_t = torch.from_numpy(target_lab_np.astype(np.float32)).to(device)
    else:
        target_lab_t = None

    # Random initial parameters
    d1 = (torch.rand(n, device=device) * 207 + 60).requires_grad_(True)
    h1 = (torch.rand(n, device=device) * 520 + 80).requires_grad_(True)
    d2 = (torch.rand(n, device=device) * 207 + 60).requires_grad_(True)
    h2 = (torch.rand(n, device=device) * 520 + 80).requires_grad_(True)
    if p_fixed is not None:
        p = torch.full((n,), float(p_fixed), device=device)
    else:
        p = (torch.rand(n, device=device) * 400 + 200).requires_grad_(True)

    params = [d1, h1, d2, h2]
    if p_fixed is None:
        params.append(p)
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.01)

    for _ in range(steps):
        optimizer.zero_grad()
        d1_c = torch.clamp(d1, 60, 267)
        h1_c = torch.clamp(h1, 80, 600)
        d2_c = torch.clamp(d2, 60, 267)
        h2_c = torch.clamp(h2, 80, 600)
        p_c = torch.clamp(p, 200, 600)
        spec = batch_dual_pillar_spectrum(d1_c, h1_c, d2_c, h2_c, p_c, theta, pol_TE, material, substrate)

        if loss_type == 'de2000':
            pred_rgb = batch_spectrum_to_rgb(spec)
            pred_lab = batch_spectrum_to_lab(spec)
            mse_loss = ((pred_rgb - target) ** 2).sum(dim=1).mean()
            de2000_loss = delta_e2000_torch(pred_lab, target_lab_t).mean()
            loss = mse_loss * 0.3 + de2000_loss * 0.7
        else:
            pred = batch_spectrum_to_rgb(spec)
            loss = ((pred - target) ** 2).sum(dim=1).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()

    # Final evaluation (use same loss metric for selection as for training)
    with torch.no_grad():
        d1_f = torch.clamp(d1, 60, 267)
        h1_f = torch.clamp(h1, 80, 600)
        d2_f = torch.clamp(d2, 60, 267)
        h2_f = torch.clamp(h2, 80, 600)
        p_f = torch.clamp(p, 200, 600)
        spec = batch_dual_pillar_spectrum(d1_f, h1_f, d2_f, h2_f, p_f, theta, pol_TE, material, substrate)
        pred = batch_spectrum_to_rgb(spec)
        if loss_type == 'de2000' and target_lab_t is not None:
            pred_lab = batch_spectrum_to_lab(spec)
            losses = delta_e2000_torch(pred_lab, target_lab_t)
        else:
            losses = ((pred - target) ** 2).sum(dim=1)
        best_idx = torch.argmin(losses).item()
        best_d1 = float(d1_f[best_idx])
        best_h1 = float(h1_f[best_idx])
        best_d2 = float(d2_f[best_idx])
        best_h2 = float(h2_f[best_idx])
        best_p = float(p_f[best_idx])
        best_rgb = pred[best_idx].cpu().numpy()
        best_loss = float(losses[best_idx])

    return best_d1, best_h1, best_d2, best_h2, best_p, best_rgb, best_loss


def inverse_design_single_v2(target_rgb, n_restarts=24, steps=100, lr=0.05,
                               material=None, substrate=None, theta=0.0, pol_TE=True,
                               p_fixed=400.0, loss_type='de2000'):
    """
    Gradient-based inverse design for single-pillar (v2: CIEDE2000 + cosine annealing).
    """
    if not isinstance(target_rgb, torch.Tensor):
        target = torch.tensor(target_rgb, dtype=torch.float32).unsqueeze(0)
    else:
        target = target_rgb.clone().detach().unsqueeze(0)

    device = target.device
    n = n_restarts

    # Pre-compute target LAB once for CIEDE2000 loss
    if loss_type == 'de2000':
        target_rgb_np = target.cpu().numpy()
        from color_utils import rgb_to_lab as _r2l
        target_lab_np = _r2l(target_rgb_np)
        target_lab_t = torch.from_numpy(target_lab_np.astype(np.float32)).to(device)
    else:
        target_lab_t = None

    d = (torch.rand(n, device=device) * 300 + 50).requires_grad_(True)
    h = (torch.rand(n, device=device) * 520 + 80).requires_grad_(True)
    if p_fixed is not None:
        p = torch.full((n,), float(p_fixed), device=device)
    else:
        p = (torch.rand(n, device=device) * 400 + 200).requires_grad_(True)

    params = [d, h]
    if p_fixed is None:
        params.append(p)
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.01)

    for _ in range(steps):
        optimizer.zero_grad()
        d_c = torch.clamp(d, 50, 350)
        h_c = torch.clamp(h, 80, 600)
        p_c = torch.clamp(p, 200, 600)
        spec = batch_lorentzian_spectrum(d_c, h_c, p_c, theta, pol_TE, material, substrate)

        if loss_type == 'de2000':
            pred_rgb = batch_spectrum_to_rgb(spec)
            pred_lab = batch_spectrum_to_lab(spec)
            mse_loss = ((pred_rgb - target) ** 2).sum(dim=1).mean()
            de2000_loss = delta_e2000_torch(pred_lab, target_lab_t).mean()
            loss = mse_loss * 0.3 + de2000_loss * 0.7
        else:
            pred = batch_spectrum_to_rgb(spec)
            loss = ((pred - target) ** 2).sum(dim=1).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()

    with torch.no_grad():
        d_f = torch.clamp(d, 50, 350)
        h_f = torch.clamp(h, 80, 600)
        p_f = torch.clamp(p, 200, 600)
        spec = batch_lorentzian_spectrum(d_f, h_f, p_f, theta, pol_TE, material, substrate)
        pred = batch_spectrum_to_rgb(spec)
        if loss_type == 'de2000' and target_lab_t is not None:
            pred_lab = batch_spectrum_to_lab(spec)
            losses = delta_e2000_torch(pred_lab, target_lab_t)
        else:
            losses = ((pred - target) ** 2).sum(dim=1)
        best_idx = torch.argmin(losses).item()
        best_d = float(d_f[best_idx])
        best_h = float(h_f[best_idx])
        best_p = float(p_f[best_idx])
        best_rgb = pred[best_idx].cpu().numpy()
        best_loss = float(losses[best_idx])

    return best_d, best_h, best_p, best_rgb, best_loss


def batch_single_pillar_rgb_norm(D, H, P, theta=0.0, pol_TE=True):
    """Single pillar -> normalized RGB (divides by per-sample max)."""
    spec = batch_lorentzian_spectrum(D, H, P, theta, pol_TE)
    mx = spec.max(dim=1, keepdim=True).values
    mx = torch.where(mx > 1e-12, mx, torch.ones_like(mx))
    return batch_spectrum_to_rgb(spec / mx)


def inverse_design_ml_batch(target_rgb, n_restarts=20, steps=50, lr=0.05,
                              material=None, substrate=None, theta=0.0, pol_TE=True, p_fixed=400.0):
    """
    Gradient-based inverse design for single-pillar structures using ML proxy.
    Uses Adam optimizer with random restarts.
    Returns (best_d, best_h, best_p, best_rgb, best_loss).
    """
    if not isinstance(target_rgb, torch.Tensor):
        target = torch.tensor(target_rgb, dtype=torch.float32).unsqueeze(0)
    else:
        target = target_rgb.clone().detach().unsqueeze(0)

    device = target.device
    n = n_restarts

    d = (torch.rand(n, device=device) * 300 + 50).requires_grad_(True)
    h = (torch.rand(n, device=device) * 520 + 80).requires_grad_(True)
    if p_fixed is not None:
        p = torch.full((n,), float(p_fixed), device=device)
    else:
        p = (torch.rand(n, device=device) * 400 + 200).requires_grad_(True)

    params = [d, h]
    if p_fixed is None:
        params.append(p)
    optimizer = torch.optim.Adam(params, lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        d_c = torch.clamp(d, 50, 350)
        h_c = torch.clamp(h, 80, 600)
        p_c = torch.clamp(p, 200, 600)
        spec = batch_lorentzian_spectrum(d_c, h_c, p_c, theta, pol_TE, material, substrate)
        pred = batch_spectrum_to_rgb(spec)
        loss = ((pred - target) ** 2).sum(dim=1).mean()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        d_f = torch.clamp(d, 50, 350)
        h_f = torch.clamp(h, 80, 600)
        p_f = torch.clamp(p, 200, 600)
        spec = batch_lorentzian_spectrum(d_f, h_f, p_f, theta, pol_TE, material, substrate)
        pred = batch_spectrum_to_rgb(spec)
        losses = ((pred - target) ** 2).sum(dim=1)
        best_idx = torch.argmin(losses).item()
        best_d = float(d_f[best_idx])
        best_h = float(h_f[best_idx])
        best_p = float(p_f[best_idx])
        best_rgb = pred[best_idx].cpu().numpy()
        best_loss = float(losses[best_idx])

    return best_d, best_h, best_p, best_rgb, best_loss




# === RCWA Inverse Design (add to torch_model.py) ===


# ── RCWA ResMLP Model (self-contained, no dependency on train_rcwa) ──

class _ResBlock(torch.nn.Module):
    def __init__(self, dim=256, dropout=0.1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, dim), torch.nn.BatchNorm1d(dim), torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim, dim), torch.nn.BatchNorm1d(dim),
        )
    def forward(self, x):
        return torch.nn.functional.relu(self.net(x) + x)

class _RCWA_ResMLP(torch.nn.Module):
    """RCWA-trained ResMLP: 7-dim input -> 81-dim reflectance spectrum."""
    def __init__(self, in_dim=7, hidden=256, out_dim=81, n_blocks=4):
        super().__init__()
        self.input_proj = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(), torch.nn.BatchNorm1d(hidden)
        )
        self.blocks = torch.nn.Sequential(*[_ResBlock(hidden) for _ in range(n_blocks)])
        self.head = torch.nn.Linear(hidden, out_dim)
    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        return torch.sigmoid(self.head(x))


def _load_rcwa_torch_model(device="cpu", material="TiO2 (anatase)", substrate="SiO2 (fused silica)"):
    """Load RCWA-trained ResMLP weights."""
    import os as _os
    _RCWA_MODEL_FILES = {
        "TiO2 (anatase)": "forward_mlp_rcwa_TiO2_s1.pt",
        "a-Si (amorphous)": "forward_mlp_rcwa_aSi_s1.pt",
        "Si3N4 (nitride)": "forward_mlp_rcwa_Si3N4_s1.pt",
        "Al2O3 (sapphire)": "forward_mlp_rcwa_Al2O3_s1.pt",
    }
    # Substrate-specific routing for a-Si
    if material == "a-Si (amorphous)":
        _SUB_FILES = {
            "SiO2 (fused silica)": "forward_mlp_rcwa_aSi_s1.pt",
            "Si3N4 (nitride)": "forward_mlp_rcwa_aSi_Si3N4_s1.pt",
            "Al2O3 (sapphire)": "forward_mlp_rcwa_aSi_Al2O3_s1.pt",
        }
        _fname = _SUB_FILES.get(substrate, "forward_mlp_rcwa_aSi_s1.pt")
    else:
        _fname = _RCWA_MODEL_FILES.get(material, "forward_mlp_rcwa_TiO2_s1.pt")
    _path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "models", _fname)
    if not _os.path.exists(_path):
        raise FileNotFoundError(f"RCWA model not found: {_path}")
    model = _RCWA_ResMLP(in_dim=7, hidden=256, out_dim=81, n_blocks=4)
    state = torch.load(_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def inverse_design_rcwa(target_rgb, n_restarts=24, steps=100, lr=0.05,
                        p_fixed=400.0, device=None, material="TiO2 (anatase)",
                        substrate="SiO2 (fused silica)",
                        init_D=None, init_H=None, init_P=None):
    """
    RCWA ResMLP-based batched inverse design for TiO2/SiO2 single pillar.
    All restarts run in parallel (batched tensor), with D/H/P clamped to
    physical bounds at every step.  Uses CIEDE2000 loss.
    Returns: dict with D, H, P, pred_rgb, pred_lab, de2000, spectrum
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model (cached)
    _cache = getattr(inverse_design_rcwa, "_cache", None)
    if _cache is None or _cache[0] != device or _cache[1] != material or _cache[2] != substrate:
        model = _load_rcwa_torch_model(device, material, substrate)
        inverse_design_rcwa._cache = (device, material, substrate, model)
    else:
        model = _cache[2]
    # Map material/substrate to training codes (must match train_rcwa.py encoding)
    _MAT_MAP = {"TiO2 (anatase)": 0, "a-Si (amorphous)": 1,
                "Si3N4 (nitride)": 2, "Al2O3 (sapphire)": 3}
    _SUB_MAP = {"SiO2 (fused silica)": 0, "Si3N4 (nitride)": 1,
                "Al2O3 (sapphire)": 2}
    mc = _MAT_MAP.get(material, 0)
    sc = _SUB_MAP.get(substrate, 0)
    # Pre-compute target LAB
    if not isinstance(target_rgb, torch.Tensor):
        target = torch.tensor(target_rgb, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        target = target_rgb.clone().detach().to(device).unsqueeze(0)
    target_lab_t = _rgb_to_lab_tensor(target, device)
    n = n_restarts
    # Batched initialization (warm start if init values provided)
    if init_D is not None:
        D = (torch.randn(n, device=device) * 5 + float(init_D)).clamp(100, 320).requires_grad_(True)
    else:
        D = (torch.rand(n, device=device) * 220 + 100).requires_grad_(True)
    if init_H is not None:
        H = (torch.randn(n, device=device) * 5 + float(init_H)).clamp(150, 550).requires_grad_(True)
    else:
        H = (torch.rand(n, device=device) * 400 + 150).requires_grad_(True)
    if p_fixed is not None:
        P_batch = torch.full((n,), float(p_fixed), device=device)
    elif init_P is not None:
        P_batch = (torch.randn(n, device=device) * 5 + float(init_P)).clamp(220, 500).requires_grad_(True)
    else:
        P_batch = (torch.rand(n, device=device) * 280 + 220).requires_grad_(True)
    params = [D, H]
    if p_fixed is None:
        params.append(P_batch)
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.01)
    for _ in range(steps):
        opt.zero_grad()
        x = _rcwa_input_batch(D, H, P_batch if p_fixed is None else P_batch, mc, sc)
        spec = model(x)
        pred_lab = batch_spectrum_to_lab(spec)
        loss = delta_e2000_torch(pred_lab, target_lab_t.expand(n, -1)).mean()
        if torch.isfinite(loss):
            loss.backward()
            opt.step()
        sched.step()
        # Clamp to physical bounds
        with torch.no_grad():
            D.clamp_(100, 320)
            H.clamp_(150, 550)
            if p_fixed is None:
                P_batch.clamp_(220, 500)
    # Select best restart
    with torch.no_grad():
        x = _rcwa_input_batch(D, H, P_batch if p_fixed is None else P_batch, mc, sc)
        spec = model(x)
        pred_lab = batch_spectrum_to_lab(spec)
        de_all = delta_e2000_torch(pred_lab, target_lab_t.expand(n, -1))
        best_idx = torch.argmin(de_all)
        D_best = round(D[best_idx].item())
        H_best = round(H[best_idx].item())
        P_best = round(P_batch[best_idx].item()) if p_fixed is None else p_fixed
        de_best = de_all[best_idx].item()
        spec_best = spec[best_idx].cpu().numpy()
        lab_best = pred_lab[best_idx].cpu().numpy()
    # Clamp to safe ranges
    D_best = max(100, min(320, D_best))
    H_best = max(150, min(550, H_best))
    P_best = max(220, min(500, P_best))
    # Convert to RGB
    from color_utils import spectrum_to_xyz, xyz_to_srgb
    xyz_np = spectrum_to_xyz(WL.numpy(), spec_best)
    rgb_best = xyz_to_srgb(xyz_np).tolist()
    return {
        "D": D_best, "H": H_best, "P": P_best,
        "pred_rgb": rgb_best,
        "pred_lab": lab_best.tolist() if hasattr(lab_best, "tolist") else list(lab_best),
        "de2000": de_best,
        "spectrum": spec_best.tolist() if hasattr(spec_best, "tolist") else list(spec_best),
    }


def _rcwa_input_batch(D, H, P, mat_code=0, sub_code=0):
    """Build 7-dim normalized input tensor (N,7) for RCWA ResMLP."""
    device = D.device
    n = D.shape[0]
    d_norm = (D - 50.0) / 300.0
    h_norm = (H - 80.0) / 520.0
    p_norm = (P - 200.0) / 400.0
    zeros = torch.zeros(n, device=device)
    mc = torch.full((n,), float(mat_code), device=device)
    sc = torch.full((n,), float(sub_code), device=device)
    return torch.stack([d_norm, h_norm, p_norm, zeros, zeros, mc, sc], dim=1)


def _rgb_to_lab_tensor(rgb_t, device):
    """sRGB tensor (1,3) -> CIELAB tensor (1,3) via numpy bridge."""
    rgb_np = rgb_t.cpu().numpy()
    from color_utils import rgb_to_lab as _r2l
    lab_np = _r2l(rgb_np)
    return torch.from_numpy(lab_np.astype(np.float32)).to(device)
