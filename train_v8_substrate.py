"""train_v8_substrate.py - Multi-material + substrate ResMLP (7 inputs, 180k samples)"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, pickle, os

CIE_X = np.array([0.001368,0.002236,0.004243,0.007650,0.014310,0.023190,0.043510,0.077630,0.134380,0.214770,0.283900,0.328500,0.348280,0.348060,0.336200,0.318700,0.290800,0.251100,0.195360,0.142100,0.095640,0.058010,0.032010,0.014700,0.004900,0.002400,0.009300,0.029100,0.063270,0.109600,0.165500,0.225750,0.290400,0.359700,0.433450,0.512050,0.594500,0.678400,0.762100,0.842500,0.916300,0.978600,1.026300,1.056700,1.062200,1.045600,1.002600,0.938400,0.854450,0.751400,0.642400,0.541900,0.447900,0.360800,0.283500,0.218700,0.164900,0.121200,0.087400,0.063600,0.046770,0.032900,0.022700,0.015840,0.011359,0.008111,0.005790,0.004109,0.002899,0.002049,0.001440,0.001000,0.000690,0.000476,0.000332,0.000235,0.000166,0.000117,0.000083,0.000059,0.000042])
CIE_Y = np.array([0.000039,0.000064,0.000120,0.000217,0.000396,0.000640,0.001210,0.002180,0.004000,0.007300,0.011600,0.016840,0.023000,0.029800,0.038000,0.048000,0.060000,0.073900,0.090980,0.112600,0.139020,0.169300,0.208020,0.258600,0.323000,0.407300,0.503000,0.608200,0.710000,0.793200,0.862000,0.914850,0.954000,0.980300,0.994950,1.000000,0.995000,0.978600,0.952000,0.915400,0.870000,0.816300,0.757000,0.694900,0.631000,0.566800,0.503000,0.441200,0.381000,0.321000,0.265000,0.217000,0.175000,0.138200,0.107000,0.081600,0.061000,0.044580,0.032000,0.023200,0.017000,0.011920,0.008210,0.005723,0.004102,0.002929,0.002091,0.001484,0.001047,0.000740,0.000520,0.000361,0.000249,0.000172,0.000120,0.000085,0.000060,0.000042,0.000030,0.000021,0.000015])
CIE_Z = np.array([0.006450,0.010550,0.020050,0.036210,0.067850,0.110200,0.207400,0.371300,0.645600,1.039050,1.385600,1.622960,1.747060,1.782600,1.772110,1.744100,1.669200,1.528100,1.287640,0.999550,0.716900,0.484400,0.311900,0.190300,0.104200,0.049200,0.020300,0.008700,0.003900,0.002100,0.001650,0.001100,0.000800,0.000550,0.000350,0.000250,0.000150,0.000100,0.000050]+[0.0]*42)
WL = np.linspace(380, 780, 81)
CIE_NORM = np.trapezoid(CIE_Y, WL)

PILLARS = {  # (A, B) for n = A + B/um^2
    "TiO2 (anatase)":   (2.3000, 0.03500),
    "a-Si (amorphous)": (3.8000, 0.08000),
    "Si3N4 (nitride)":  (1.9900, 0.01200),
    "Al2O3 (sapphire)": (1.7546, 0.00500),
}
SUBSTRATES = {
    "SiO2 (fused silica)": (1.4580, 0.00354),
    "Si3N4 (nitride)":     (1.9900, 0.01200),
    "Al2O3 (sapphire)":    (1.7546, 0.00500),
}
MAT_LIST = list(PILLARS.keys())
SUB_LIST = list(SUBSTRATES.keys())

def single_spectrum(d, h, p, angle_deg, mat_name, sub_name):
    A_mat, B_mat = PILLARS[mat_name]
    A_sub, B_sub = SUBSTRATES[sub_name]
    n_mat550 = A_mat + B_mat / (0.55**2)
    n_sub550 = A_sub + B_sub / (0.55**2)
    n_env550 = (1.0 + n_sub550) / 2.0
    dn_ref = n_mat550 - n_env550
    
    lam_ed = 360 + 0.55*(d-60) + 0.12*(h-120) + 32*dn_ref
    sigma_ed = max(26 + 0.10*(d-200), 8)  # FDTD-calibrated
    lam_md = 400 + 0.75*(d-60) + 0.25*(h-120) + 32*dn_ref
    sigma_md = max(35 + 0.12*(d-200), 10)  # FDTD-calibrated
    
    fill = np.clip(np.pi*(d/2)**2/(p**2), 0.01, 0.70)
    fill_amp = 0.30 + 0.80*fill
    loss_exp = np.exp(-0.0006*max(h-600, 0))
    theta = angle_deg*np.pi/180.0
    sin2 = np.sin(theta)**2
    ed_shift, md_shift = -45*sin2, -20*sin2
    eaa = 1.0-0.10*sin2; maa = 1.0-0.04*sin2
    w_ed = np.clip(0.80-0.003*(d-60), 0.0, 0.80); w_md = 1.0-w_ed
    aspect = h / max(d, 50.0)
    q_ed = np.clip(2.5 + 0.5*(aspect-1.0), 1.5, 6.0)
    q_md = np.clip(4.0 + 0.3*(aspect-1.0), 2.5, 8.0)
    ied = 1.0/np.sqrt(1.0+q_ed**2); imd = 1.0/np.sqrt(1.0+q_md**2)
    ec = lam_ed+ed_shift; de = (WL-ec)/sigma_ed
    fn = q_ed+de; den = 1.0+de**2
    rer = fn*ied/den*np.sqrt(eaa); rei = -fn*de*ied/den*np.sqrt(eaa)
    mc = lam_md+md_shift; dm = (WL-mc)/sigma_md
    fnm = q_md+dm; denm = 1.0+dm**2
    rmr = fnm*imd/denm*np.sqrt(maa); rmi = -fnm*dm*imd/denm*np.sqrt(maa)
    rr = w_ed*rer + w_md*rmr; ri = w_ed*rei + w_md*rmi
    return (rr**2 + ri**2) * fill_amp * loss_exp

def spec_to_rgb(spec):
    X = np.trapezoid(spec*CIE_X, WL); Y = np.trapezoid(spec*CIE_Y, WL); Z = np.trapezoid(spec*CIE_Z, WL)
    xyz = np.array([X/CIE_NORM, Y/CIE_NORM, Z/CIE_NORM])
    M = np.array([[3.2406,-1.5372,-0.4986],[-0.9689,1.8758,0.0415],[0.0557,-0.2040,1.0570]])
    rgb_lin = M @ xyz; rgb_lin = np.clip(rgb_lin, 0, None)
    rgb = np.where(rgb_lin <= 0.0031308, 12.92*rgb_lin, 1.055*rgb_lin**(1/2.4)-0.055)
    return np.clip(rgb, 0, 1)

N_PER_COMBO = 13000
N_SAMPLES = N_PER_COMBO * len(MAT_LIST) * len(SUB_LIST)
print(f"Generating {N_SAMPLES} samples ({N_PER_COMBO} per pillar-sub combo)...")
np.random.seed(42)
X_norm = np.zeros((N_SAMPLES, 7), dtype=np.float32)  # D,H,P,angle,pol,mat_code,sub_code
Y_spec = np.zeros((N_SAMPLES, 81), dtype=np.float32)
Y_rgb  = np.zeros((N_SAMPLES, 3), dtype=np.float32)

idx = 0
for mat_code, mat_name in enumerate(MAT_LIST):
    for sub_code, sub_name in enumerate(SUB_LIST):
        for _ in range(N_PER_COMBO):
            d = np.random.uniform(50, 350)
            h = np.random.uniform(80, 600)
            p = max(d*1.2, np.random.uniform(200, 600))
            angle = np.random.uniform(0, 80)
            pol = np.random.choice([0, 1])
            spec = single_spectrum(d, h, p, angle, mat_name, sub_name)
            rgb = spec_to_rgb(spec)
            X_norm[idx] = [(d-50)/300, (h-80)/520, (p-200)/400, angle/80, float(pol), float(mat_code), float(sub_code)]
            Y_spec[idx] = spec
            Y_rgb[idx] = rgb
            idx += 1

print(f"Generated {idx} samples")

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim), nn.BatchNorm1d(dim))
    def forward(self, x):
        return F.relu(self.net(x) + x)

class DeepResMLP_Sub(nn.Module):
    def __init__(self, in_dim=7, hidden=256, out_dim=81, n_blocks=4):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.BatchNorm1d(hidden))
        self.blocks = nn.Sequential(*[ResidualBlock(hidden) for _ in range(n_blocks)])
        self.output = nn.Sequential(nn.Linear(hidden, out_dim), nn.Sigmoid())
    def forward(self, x):
        x = self.input_proj(x); x = self.blocks(x); return self.output(x)

cie_x = torch.tensor(CIE_X, dtype=torch.float32)
cie_y = torch.tensor(CIE_Y, dtype=torch.float32)
cie_z = torch.tensor(CIE_Z, dtype=torch.float32)
cie_norm = torch.tensor(CIE_NORM, dtype=torch.float32)
srgb_m = torch.tensor([[3.2406,-1.5372,-0.4986],[-0.9689,1.8758,0.0415],[0.0557,-0.2040,1.0570]], dtype=torch.float32)

def batch_spec_to_rgb(spec):
    wl = torch.linspace(380, 780, 81, device=spec.device)
    X = torch.trapezoid(spec * cie_x.unsqueeze(0).to(spec.device), wl, dim=1)
    Y = torch.trapezoid(spec * cie_y.unsqueeze(0).to(spec.device), wl, dim=1)
    Z = torch.trapezoid(spec * cie_z.unsqueeze(0).to(spec.device), wl, dim=1)
    xyz = torch.stack([X/cie_norm.to(spec.device), Y/cie_norm.to(spec.device), Z/cie_norm.to(spec.device)], dim=1)
    rgb_lin = xyz @ srgb_m.to(spec.device).T
    rgb_lin = torch.clamp(rgb_lin, min=0.0)
    rgb = torch.where(rgb_lin <= 0.0031308, 12.92*rgb_lin, 1.055*rgb_lin.pow(1/2.4)-0.055)
    return torch.clamp(rgb, 0, 1)

X_t = torch.tensor(X_norm, dtype=torch.float32)
Y_spec_t = torch.tensor(Y_spec, dtype=torch.float32)
Y_rgb_t = torch.tensor(Y_rgb, dtype=torch.float32)

n = len(X_t); split = int(n * 0.85)
X_tr, X_val = X_t[:split], X_t[split:]
Y_spec_tr, Y_spec_val = Y_spec_t[:split], Y_spec_t[split:]
Y_rgb_tr, Y_rgb_val = Y_rgb_t[:split], Y_rgb_t[split:]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")
model = DeepResMLP_Sub().to(device)
X_tr, X_val = X_tr.to(device), X_val.to(device)
Y_spec_tr, Y_spec_val = Y_spec_tr.to(device), Y_spec_val.to(device)
Y_rgb_tr, Y_rgb_val = Y_rgb_tr.to(device), Y_rgb_val.to(device)

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80, eta_min=1e-5)
crit_spec = nn.MSELoss(); crit_rgb = nn.MSELoss()
ALPHA_RGB = 0.1

best_val = 1e9
save_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(save_dir, "models", "forward_mlp_v8_sub.pt")
os.makedirs(os.path.dirname(save_path), exist_ok=True)

BATCH = 512 if device.type == "cuda" else 128
print(f"Training v8 Substrate (80 epochs, {N_SAMPLES} samples, batch={BATCH})...")
for epoch in range(80):
    model.train()
    perm = torch.randperm(len(X_tr), device=device)
    for i in range(0, len(X_tr), BATCH):
        idx_b = perm[i:i+BATCH]
        pred_spec = model(X_tr[idx_b])
        pred_rgb = batch_spec_to_rgb(pred_spec)
        loss = crit_spec(pred_spec, Y_spec_tr[idx_b]) + ALPHA_RGB * crit_rgb(pred_rgb, Y_rgb_tr[idx_b])
        opt.zero_grad(); loss.backward(); opt.step()
    sched.step()
    model.eval()
    with torch.no_grad():
        pred_val = model(X_val)
        pred_rgb_val = batch_spec_to_rgb(pred_val)
        val_spec = crit_spec(pred_val, Y_spec_val).item()
        val_rgb = crit_rgb(pred_rgb_val, Y_rgb_val).item()
        val_total = val_spec + ALPHA_RGB * val_rgb
    if val_total < best_val:
        best_val = val_total
        torch.save(model.state_dict(), save_path)
    if epoch % 10 == 0:
        print(f"  Epoch {epoch:3d}: spec={val_spec:.6f} rgb={val_rgb:.6f} total={val_total:.6f}")

print(f"Best total loss: {best_val:.6f}")
model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
model.eval()

print("\nv8 Substrate ML vs Physics:")
test_cases = [
    (180,300,400,0,"TiO2 (anatase)","SiO2 (fused silica)"),
    (180,300,400,0,"TiO2 (anatase)","Si3N4 (nitride)"),
    (180,300,400,0,"a-Si (amorphous)","SiO2 (fused silica)"),
    (180,300,400,0,"Si3N4 (nitride)","Al2O3 (sapphire)"),
]
for d,h,p,ang,mat,sub in test_cases:
    mc = MAT_LIST.index(mat); sc = SUB_LIST.index(sub)
    x = torch.tensor([[(d-50)/300,(h-80)/520,(p-200)/400,ang/80,0.0,float(mc),float(sc)]], dtype=torch.float32).to(device)
    with torch.no_grad():
        spec_ml = model(x).cpu().squeeze().numpy()
        rgb_ml = spec_to_rgb(spec_ml)
    spec_phys = single_spectrum(d,h,p,ang,mat,sub)
    rgb_phys = spec_to_rgb(spec_phys)
    de = np.sqrt(np.sum((rgb_ml - rgb_phys)**2))
    print(f"  {mat[:6]}+{sub[:6]}: dE={de:.4f}")

print(f"\nDone! Model: {save_path}")
