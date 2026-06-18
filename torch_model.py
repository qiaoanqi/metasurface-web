# torch_model.py - PyTorch batch Lorentzian/Fano model (v2 - full Cauchy + coherent)
import torch
import numpy as np
from color_utils import CIE_X as _CIE_X_NP, CIE_Y as _CIE_Y_NP, CIE_Z as _CIE_Z_NP
from color_utils import SRGB_M as _SRGB_M_NP

# ============================================================
# CIE 1931 (81 points, 380-780nm step 5nm) — from color_utils
# ============================================================
CIE_X = torch.from_numpy(_CIE_X_NP.astype(np.float32))
CIE_Y = torch.from_numpy(_CIE_Y_NP.astype(np.float32))
CIE_Z = torch.from_numpy(_CIE_Z_NP.astype(np.float32))
WL = torch.linspace(380, 780, 81)
CIE_NORM = torch.trapezoid(CIE_Y, WL)
SRGB_M = torch.from_numpy(_SRGB_M_NP.astype(np.float32))

# Cauchy coefficients (A, B) for n(um) = A + B/um^2
CAUCHY_DB = {
    "TiO2 (anatase)":      (2.3000, 0.03500),
    "a-Si (amorphous)":    (3.8000, 0.08000),
    "Si3N4 (nitride)":     (1.9900, 0.01200),
    "Al2O3 (sapphire)":    (1.7546, 0.00500),
    "SiO2 (fused silica)": (1.4580, 0.00354),
    "Air":                 (1.0003, 0.00000),
}
DEFAULT_MATERIAL = "TiO2 (anatase)"


def _cauchy_n(wl_nm, material=None):
    """Refractive index via Cauchy: n = A + B/(wl_um)^2"""
    if material is None:
        material = DEFAULT_MATERIAL
    A, B = CAUCHY_DB.get(material, CAUCHY_DB[DEFAULT_MATERIAL])
    wl_um = wl_nm.clamp(min=350.0) / 1000.0
    return A + B / (wl_um ** 2)


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

    # Fill & loss
    fill = torch.clamp(np.pi*(D/2)**2/(P**2), 0.03, 0.70).unsqueeze(1)
    fill_amp = (0.30 + 0.80*fill)
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
    return spec1 + spec2


def batch_dual_pillar_rgb(D1, H1, D2, H2, P, theta=0.0, pol_TE=True, material=None, substrate=None):
    """Dual pillar -> RGB."""
    spec = batch_dual_pillar_spectrum(D1, H1, D2, H2, P, theta, pol_TE, material, substrate)
    return batch_spectrum_to_rgb(spec)


def inverse_design_dual(target_rgb, n_steps=300, n_restarts=30, material=None, substrate=None):
    """Batched gradient-based inverse design for dual-pillar."""
    if not isinstance(target_rgb, torch.Tensor):
        target = torch.tensor(target_rgb, dtype=torch.float32).unsqueeze(0)
    else:
        target = target_rgb.clone().detach().unsqueeze(0)

    device = target.device
    n = n_restarts

    d1 = (torch.rand(n, device=device) * 207 + 60).requires_grad_(True)
    h1 = (torch.rand(n, device=device) * 520 + 80).requires_grad_(True)
    d2 = (torch.rand(n, device=device) * 207 + 60).requires_grad_(True)
    h2 = (torch.rand(n, device=device) * 520 + 80).requires_grad_(True)
    p  = (torch.rand(n, device=device) * 400 + 200).requires_grad_(True)

    opt = torch.optim.Adam([d1, h1, d2, h2, p], lr=1.0)

    for step in range(n_steps):
        opt.zero_grad()
        d1_c = torch.clamp(d1, 60.0, 267.0)
        h1_c = torch.clamp(h1, 80.0, 600.0)
        d2_c = torch.clamp(d2, 60.0, 267.0)
        h2_c = torch.clamp(h2, 80.0, 600.0)
        min_p = torch.maximum(d1_c.detach(), d2_c.detach()) * 1.2 + 20
        p_c = torch.maximum(torch.minimum(p, torch.tensor(600.0, device=device)), min_p)

        spec = batch_dual_pillar_spectrum(d1_c, h1_c, d2_c, h2_c, p_c,
                                          torch.zeros(n, device=device), True, material, substrate)
        rgb = batch_spectrum_to_rgb(spec)
        loss = ((rgb - target)**2).mean(dim=1).sum()
        loss.backward()
        torch.nn.utils.clip_grad_value_([d1, h1, d2, h2, p], 1.0)
        opt.step()

    with torch.no_grad():
        d1_f = torch.clamp(d1, 60, 267)
        h1_f = torch.clamp(h1, 80, 600)
        d2_f = torch.clamp(d2, 60, 267)
        h2_f = torch.clamp(h2, 80, 600)
        min_p_f = torch.maximum(d1_f, d2_f) * 1.2 + 20
        p_f = torch.maximum(torch.minimum(p, torch.tensor(600.0, device=device)), min_p_f)

        spec = batch_dual_pillar_spectrum(d1_f, h1_f, d2_f, h2_f, p_f,
                                          torch.zeros(n, device=device), True, material, substrate)
        pred = batch_spectrum_to_rgb(spec)
        losses = ((pred - target)**2).sum(dim=1)
        best_idx = torch.argmin(losses).item()

        best_d1 = float(d1_f[best_idx])
        best_h1 = float(h1_f[best_idx])
        best_d2 = float(d2_f[best_idx])
        best_h2 = float(h2_f[best_idx])
        best_p  = float(p_f[best_idx])
        best_rgb = pred[best_idx].cpu().numpy()
        best_loss = float(losses[best_idx])

    return (best_d1, best_h1, best_d2, best_h2, best_p, best_rgb, best_loss)


def batch_single_pillar_rgb_norm(D, H, P, theta=0.0, pol_TE=True):
    spec = batch_lorentzian_spectrum(D, H, P, theta, pol_TE)
    mx = spec.max(dim=1, keepdim=True).values
    mx = torch.where(mx > 1e-12, mx, torch.ones_like(mx))
    return batch_spectrum_to_rgb(spec / mx)


if __name__ == "__main__":
    import time, sys
    sys.stdout.reconfigure(encoding="utf-8")
    print("PyTorch Batch Lorentzian v2")
    D_test = torch.tensor([310.0, 150.0, 80.0, 180.0])
    H_test = torch.tensor([160.0, 100.0, 250.0, 250.0])
    P_test = torch.tensor([400.0, 400.0, 400.0, 400.0])
    spec = batch_lorentzian_spectrum(D_test, H_test, P_test)
    rgb = batch_spectrum_to_rgb(spec)
    for j in range(4):
        print(f"  D={D_test[j]:.0f} H={H_test[j]:.0f} P={P_test[j]:.0f} -> RGB=({rgb[j,0]:.4f},{rgb[j,1]:.4f},{rgb[j,2]:.4f})")
    D_batch = torch.rand(1000)*300+50
    H_batch = torch.rand(1000)*520+80
    P_batch = torch.clamp(D_batch*1.2, min=200)
    t0 = time.time()
    batch_single_pillar_rgb(D_batch, H_batch, P_batch)
    print(f"\nBatch 1000 samples: {(time.time()-t0)*1000:.1f}ms")
    D_vals = torch.linspace(60, 300, 12)
    H_vals = torch.linspace(80, 500, 12)
    t0 = time.time()
    grid = batch_color_map_grid(D_vals, H_vals)
    print(f"Color map 12x12: {(time.time()-t0)*1000:.1f}ms")
