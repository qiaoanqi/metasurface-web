# torch_model.py - PyTorch batch Lorentzian model (v2 - full Cauchy + coherent)
import torch
import numpy as np

# ============================================================
# CIE 1931 (81 points, 380-780nm step 5nm)
# ============================================================
CIE_X = torch.tensor([0.001368,0.002236,0.004243,0.007650,0.014310,0.023190,0.043510,0.077630,0.134380,0.214770,0.283900,0.328500,0.348280,0.348060,0.336200,0.318700,0.290800,0.251100,0.195360,0.142100,0.095640,0.058010,0.032010,0.014700,0.004900,0.002400,0.009300,0.029100,0.063270,0.109600,0.165500,0.225750,0.290400,0.359700,0.433450,0.512050,0.594500,0.678400,0.762100,0.842500,0.916300,0.978600,1.026300,1.056700,1.062200,1.045600,1.002600,0.938400,0.854450,0.751400,0.642400,0.541900,0.447900,0.360800,0.283500,0.218700,0.164900,0.121200,0.087400,0.063600,0.046770,0.032900,0.022700,0.015840,0.011359,0.008111,0.005790,0.004109,0.002899,0.002049,0.001440,0.001000,0.000690,0.000476,0.000332,0.000235,0.000166,0.000117,0.000083,0.000059,0.000042], dtype=torch.float32)
CIE_Y = torch.tensor([0.000039,0.000064,0.000120,0.000217,0.000396,0.000640,0.001210,0.002180,0.004000,0.007300,0.011600,0.016840,0.023000,0.029800,0.038000,0.048000,0.060000,0.073900,0.090980,0.112600,0.139020,0.169300,0.208020,0.258600,0.323000,0.407300,0.503000,0.608200,0.710000,0.793200,0.862000,0.914850,0.954000,0.980300,0.994950,1.000000,0.995000,0.978600,0.952000,0.915400,0.870000,0.816300,0.757000,0.694900,0.631000,0.566800,0.503000,0.441200,0.381000,0.321000,0.265000,0.217000,0.175000,0.138200,0.107000,0.081600,0.061000,0.044580,0.032000,0.023200,0.017000,0.011920,0.008210,0.005723,0.004102,0.002929,0.002091,0.001484,0.001047,0.000740,0.000520,0.000361,0.000249,0.000172,0.000120,0.000085,0.000060,0.000042,0.000030,0.000021,0.000015], dtype=torch.float32)
CIE_Z = torch.tensor([0.006450,0.010550,0.020050,0.036210,0.067850,0.110200,0.207400,0.371300,0.645600,1.039050,1.385600,1.622960,1.747060,1.782600,1.772110,1.744100,1.669200,1.528100,1.287640,0.999550,0.716900,0.484400,0.311900,0.190300,0.104200,0.049200,0.020300,0.008700,0.003900,0.002100,0.001650,0.001100,0.000800,0.000550,0.000350,0.000250,0.000150,0.000100,0.000050]+ [0.0]*42, dtype=torch.float32)

WL = torch.linspace(380, 780, 81)
CIE_NORM = CIE_Y.sum()
SRGB_M = torch.tensor([[3.2406,-1.5372,-0.4986],[-0.9689,1.8758,0.0415],[0.0557,-0.2040,1.0570]], dtype=torch.float32)

# Cauchy: TiO2 anatase: n(um) = A + B/um^2
A_TIO2, B_TIO2 = 2.3000, 0.03500


def _cauchy_n(wl_nm):
    """TiO2 refractive index via Cauchy: n = A + B/(wl_um)^2"""
    wl_um = wl_nm.clamp(min=350.0) / 1000.0  # (batch,)
    return A_TIO2 + B_TIO2 / (wl_um ** 2)


def batch_lorentzian_spectrum(D, H, P, theta=0.0, pol_TE=True):
    """
    Batch Lorentzian spectrum with full Cauchy dispersion and coherent ED+MD addition.
    Matches the NumPy _single_pillar_complex model.
    """
    if not isinstance(D, torch.Tensor): D = torch.tensor(D, dtype=torch.float32)
    if not isinstance(H, torch.Tensor): H = torch.tensor(H, dtype=torch.float32)
    if not isinstance(P, torch.Tensor): P = torch.tensor(P, dtype=torch.float32)
    if not isinstance(theta, torch.Tensor): theta = torch.tensor(theta, dtype=torch.float32)

    D = D.view(-1); H = H.view(-1); P = P.view(-1); theta = theta.view(-1)
    batch = D.shape[0]
    wl = WL.unsqueeze(0)  # (1, 81)

    # Wavelength-dependent n (batch, 81)
    n_mat = _cauchy_n(wl)  # (1, 81) -> broadcast to (batch, 81) automatically
    dn = n_mat - 2.0  # (1, 81)

    # Peak wavelengths (batch, 1) - use n@550nm as reference
    n550 = _cauchy_n(torch.tensor(550.0)).item()
    dn550 = n550 - 2.0

    lam_ed = (360 + 0.55*(D-60) + 0.12*(H-120) + 32*dn550).unsqueeze(1)  # (batch, 1)
    sigma_ed = torch.clamp(15 + 0.015*(D-200), min=10).unsqueeze(1)
    lam_md = (400 + 0.75*(D-60) + 0.25*(H-120) + 32*dn550).unsqueeze(1)
    sigma_md = torch.clamp(22 + 0.03*(D-200), min=15).unsqueeze(1)

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

    # Lorentzian shapes (batch, 81)
    ed_center = lam_ed + ed_shift
    detune_ed = (wl - ed_center) / sigma_ed
    ed_amp = torch.sqrt(1.0/(1.0+detune_ed**2)) * torch.sqrt(ed_amp_a)
    ed_phase = -torch.atan(detune_ed)

    md_center = lam_md + md_shift
    detune_md = (wl - md_center) / sigma_md
    md_amp = torch.sqrt(1.0/(1.0+detune_md**2)) * torch.sqrt(md_amp_a)
    md_phase = -torch.atan(detune_md)

    # Coherent addition: r_total = w_ed*r_ed + w_md*r_md
    r_ed_real = ed_amp * torch.cos(ed_phase)
    r_ed_imag = ed_amp * torch.sin(ed_phase)
    r_md_real = md_amp * torch.cos(md_phase)
    r_md_imag = md_amp * torch.sin(md_phase)

    r_real = w_ed*r_ed_real + w_md*r_md_real
    r_imag = w_ed*r_ed_imag + w_md*r_md_imag
    intensity = (r_real**2 + r_imag**2) * fill_amp * loss

    return torch.clamp(intensity, 0.0, 1.0)


def batch_spectrum_to_rgb(spectrum):
    """Batch spectrum (B,81) -> sRGB (B,3)"""
    wl = WL.unsqueeze(0)
    X = torch.trapz(spectrum * CIE_X.unsqueeze(0), wl, dim=1)
    Y = torch.trapz(spectrum * CIE_Y.unsqueeze(0), wl, dim=1)
    Z = torch.trapz(spectrum * CIE_Z.unsqueeze(0), wl, dim=1)
    xyz = torch.stack([X/CIE_NORM, Y/CIE_NORM, Z/CIE_NORM], dim=1)
    rgb_lin = xyz @ SRGB_M.T
    rgb = torch.where(rgb_lin <= 0.0031308, 12.92*rgb_lin, 1.055*rgb_lin.pow(1/2.4)-0.055)
    return torch.clamp(rgb, 0.0, 1.0)


def batch_single_pillar_rgb(D, H, P, theta=0.0, pol_TE=True):
    """End-to-end batch: D,H,P -> spectrum -> RGB"""
    spec = batch_lorentzian_spectrum(D, H, P, theta, pol_TE)
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

def batch_dual_pillar_spectrum(D1, H1, D2, H2, P, theta=0.0, pol_TE=True):
    """Incoherent sum of two pillar spectra.
    Each spectrum already includes fill_amp weighting, so just add them."""
    spec1 = batch_lorentzian_spectrum(D1, H1, P, theta, pol_TE)
    spec2 = batch_lorentzian_spectrum(D2, H2, P, theta, pol_TE)
    return spec1 + spec2


def batch_dual_pillar_rgb(D1, H1, D2, H2, P, theta=0.0, pol_TE=True):
    """Dual pillar -> RGB."""
    spec = batch_dual_pillar_spectrum(D1, H1, D2, H2, P, theta, pol_TE)
    return batch_spectrum_to_rgb(spec)


def inverse_design_dual(target_rgb, n_steps=500, n_restarts=30):
    """Gradient-based inverse design for dual-pillar (5 params)."""
    if not isinstance(target_rgb, torch.Tensor):
        target = torch.tensor(target_rgb, dtype=torch.float32).unsqueeze(0)
    else:
        target = target_rgb.clone().detach().unsqueeze(0)

    best_loss = 1e9
    best_result = None

    for _ in range(n_restarts):
        d1 = torch.tensor(np.random.uniform(60, 267), dtype=torch.float32, requires_grad=True)
        h1 = torch.tensor(np.random.uniform(80, 600), dtype=torch.float32, requires_grad=True)
        d2 = torch.tensor(np.random.uniform(60, 267), dtype=torch.float32, requires_grad=True)
        h2 = torch.tensor(np.random.uniform(80, 600), dtype=torch.float32, requires_grad=True)
        p = torch.tensor(np.random.uniform(200, 600), dtype=torch.float32, requires_grad=True)

        opt = torch.optim.Adam([d1, h1, d2, h2, p], lr=5.0)
        for step in range(n_steps):
            opt.zero_grad()
            d1_c = torch.clamp(d1, 60.0, 267.0)
            h1_c = torch.clamp(h1, 80.0, 600.0)
            d2_c = torch.clamp(d2, 60.0, 267.0)
            h2_c = torch.clamp(h2, 80.0, 600.0)
            min_p = torch.max(d1_c.detach(), d2_c.detach()) * 1.2 + 20
            p_c = torch.clamp(p, min_p, 600.0)

            spec = batch_dual_pillar_spectrum(
                d1_c.unsqueeze(0), h1_c.unsqueeze(0),
                d2_c.unsqueeze(0), h2_c.unsqueeze(0),
                p_c.unsqueeze(0))
            rgb = batch_spectrum_to_rgb(spec)
            loss = ((rgb - target)**2).sum()
            loss.backward()
            opt.step()

        d1_f = float(np.clip(d1.item(), 60, 267))
        h1_f = float(np.clip(h1.item(), 80, 600))
        d2_f = float(np.clip(d2.item(), 60, 267))
        h2_f = float(np.clip(h2.item(), 80, 600))
        p_f = float(max(max(d1_f, d2_f) * 1.2 + 20, np.clip(p.item(), 200, 600)))

        with torch.no_grad():
            spec = batch_dual_pillar_spectrum(
                torch.tensor([d1_f]), torch.tensor([h1_f]),
                torch.tensor([d2_f]), torch.tensor([h2_f]),
                torch.tensor([p_f]))
            pred = batch_spectrum_to_rgb(spec).squeeze().numpy()
            fl = float(((pred - target.squeeze().numpy())**2).sum())
        if fl < best_loss:
            best_loss = fl
            best_result = (d1_f, h1_f, d2_f, h2_f, p_f, pred, fl)

    return best_result

# Quick test

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
