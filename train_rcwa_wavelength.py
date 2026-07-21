# train_rcwa_wavelength.py - 波长条件ResMLP: 输入(D,H,P,n(λ),k(λ),sub,λ) → R(λ)
# 每个RCWA样本产生81个训练点，模型学到波长依赖的干涉物理

import numpy as np
import pickle, argparse, time, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# 波长条件ResMLP
# ============================================================
class ResBlock(nn.Module):
    def __init__(self, dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim)
        )
    def forward(self, x):
        return F.relu(self.net(x) + x)

class WavelengthResMLP(nn.Module):
    def __init__(self, in_dim=7, hidden=256, out_dim=1, n_blocks=4):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.BatchNorm1d(hidden)
        )
        self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        return torch.sigmoid(self.head(x))

# ============================================================
# Cauchy 折射率模型
# ============================================================
_CAUCHY = {
    'TiO2': (2.3, 0.035, 0.0),
    'TiO2 (anatase)': (2.3, 0.035, 0.0),
    'a-Si': (3.8, 0.08, 0.0),
    'a-Si (amorphous)': (3.8, 0.08, 0.0),
    'Si3N4': (1.99, 0.012, 0.0),
    'Si3N4 (nitride)': (1.99, 0.012, 0.0),
    'Al2O3': (1.7546, 0.005, 0.0),
    'Al2O3 (sapphire)': (1.7546, 0.005, 0.0),
    'SiO2': (1.458, 0.00354, 0.0),
    'SiO2 (fused silica)': (1.458, 0.00354, 0.0),
}

def cauchy_n(material_name, wavelength_nm):
    A, B, C = _CAUCHY.get(material_name, (1.5, 0.0, 0.0))
    wl_um = wavelength_nm / 1000.0
    return A + B / (wl_um ** 2) + C / (wl_um ** 4)

# ============================================================
# 材料/衬底编码
# ============================================================
_MAT_MAP = {'TiO2': 0, 'TiO2 (anatase)': 0, 'a-Si': 1, 'a-Si (amorphous)': 1,
             'Si3N4': 2, 'Si3N4 (nitride)': 2, 'Al2O3': 3, 'Al2O3 (sapphire)': 3}
_SUB_MAP = {'SiO2': 0, 'SiO2 (fused silica)': 0, 'Si3N4': 1, 'Si3N4 (nitride)': 1,
             'Al2O3': 2, 'Al2O3 (sapphire)': 2}

# ============================================================
def load_wavelength_data(pkl_paths, augment=True, quality=0.05):
    if isinstance(pkl_paths, str):
        pkl_paths = [p.strip() for p in pkl_paths.split(',')]

    all_samples = []
    for path in pkl_paths:
        with open(path, 'rb') as f:
            samples = pickle.load(f)
        print('  {}: {} samples'.format(path, len(samples)))
        all_samples.extend(samples)
    print('Total: {} RCWA samples'.format(len(all_samples)))

    # Per-substrate R+T means
    if quality > 0:
        rt_by_sub = {}
        for s in all_samples:
            sub = s.get('substrate', 'SiO2')
            rt_by_sub.setdefault(sub, []).append(s['R_plus_T_mean'])
        sub_means = {k: sum(v)/len(v) for k, v in rt_by_sub.items()}
        print('  R+T means by substrate:', {k: round(v,4) for k,v in sub_means.items()})

    WLS = np.linspace(380, 780, 81)
    X_list, Y_list = [], []
    skipped = 0
    mat_counts = {}

    for s in all_samples:
        if quality > 0:
            sub = s.get('substrate', 'SiO2')
            rt_mean_ref = sub_means.get(sub, 1.0)
            if abs(s['R_plus_T_mean'] - rt_mean_ref) > quality:
                skipped += 1
                continue

        D, H, P = float(s['D']), float(s['H']), float(s['P'])
        R = np.asarray(s['R'], dtype=np.float32)
        mat_name = s.get('material', 'TiO2')
        sub_name = s.get('substrate', 'SiO2')
        mat_code = float(_MAT_MAP.get(mat_name, 0))
        sub_code = float(_SUB_MAP.get(sub_name, 0))
        key = '{}/{}'.format(mat_name, sub_name)
        mat_counts[key] = mat_counts.get(key, 0) + 1

        # Normalize geometric params
        d_norm = (D - 50) / 300
        h_norm = (H - 80) / 520
        p_norm = (P - 200) / 400

        # Create 81 training points (one per wavelength)
        for i, wl in enumerate(WLS):
            n_val = cauchy_n(mat_name, wl)
            # Normalize n to [0,1] range (n ranges from ~1.5 to ~4.5)
            n_norm = (n_val - 1.0) / 4.0
            wl_norm = (wl - 380) / 400  # [0, 1]
            k_norm = 0.0  # dielectric, k=0

            x = [d_norm, h_norm, p_norm, n_norm, k_norm, sub_code, wl_norm]
            X_list.append(x)
            Y_list.append(R[i])

        # Data augmentation
        if augment:
            rng = np.random.default_rng(int(D*1000 + H*100 + P))
            for _ in range(2):
                d2 = np.clip(D + rng.uniform(-2, 2), 80, 350)
                h2 = np.clip(H + rng.uniform(-2, 2), 100, 600)
                p2 = np.clip(P + rng.uniform(-2, 2), max(200, d2+20), 500)
                d2_norm = (d2 - 50) / 300
                h2_norm = (h2 - 80) / 520
                p2_norm = (p2 - 200) / 400
                for i, wl in enumerate(WLS):
                    n_val = cauchy_n(mat_name, wl)
                    n_norm = (n_val - 1.0) / 4.0
                    wl_norm = (wl - 380) / 400
                    x = [d2_norm, h2_norm, p2_norm, n_norm, 0.0, sub_code, wl_norm]
                    X_list.append(x)
                    Y_list.append(R[i])

    print('Training points: {} (filtered: {})'.format(len(X_list), skipped))
    print('Material/substrate distribution:', mat_counts)
    if augment:
        print('With augmentation: {} raw x 81 wl x 3 = {}'.format(
            len(all_samples) - skipped, len(X_list)))

    X = torch.tensor(X_list, dtype=torch.float32)
    Y = torch.tensor(Y_list, dtype=torch.float32).unsqueeze(1)
    return X, Y

# ============================================================
def train(model, train_loader, val_X, val_Y, epochs, lr, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            pred = model(batch_X)
            loss = criterion(pred, batch_Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        scheduler.step()
        avg_train_loss = train_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            val_pred = model(val_X.to(device))
            val_loss = criterion(val_pred, val_Y.to(device)).item()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print('  Epoch {:3d}/{} | train_loss={:.6f} | val_loss={:.6f} | lr={:.6f}'.format(
                epoch+1, epochs, avg_train_loss, val_loss, scheduler.get_last_lr()[0]))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/forward_mlp_rcwa_wl.pt')

    print('Best val_loss: {:.6f}'.format(best_val_loss))
    return model

# ============================================================
def evaluate(model, val_X, val_Y, device):
    from color_utils import spectrum_to_xyz, xyz_to_srgb, delta_e2000_scalar, rgb_to_lab_scalar
    model.eval()
    with torch.no_grad():
        pred_all = model(val_X.to(device)).cpu().numpy().flatten()
    target_all = val_Y.numpy().flatten()

    # Reshape to (N_structures, 81) for color evaluation
    n_structures = len(pred_all) // 81
    de_list = []
    WLS = np.linspace(380, 780, 81)
    for i in range(n_structures):
        pred_spec = pred_all[i*81:(i+1)*81]
        targ_spec = target_all[i*81:(i+1)*81]
        pred_xyz = spectrum_to_xyz(WLS, pred_spec)
        pred_rgb = xyz_to_srgb(pred_xyz)
        targ_xyz = spectrum_to_xyz(WLS, targ_spec)
        targ_rgb = xyz_to_srgb(targ_xyz)
        pred_lab = rgb_to_lab_scalar(pred_rgb.tolist())
        targ_lab = rgb_to_lab_scalar(targ_rgb.tolist())
        de_list.append(delta_e2000_scalar(targ_lab, pred_lab))

    de_arr = np.array(de_list)
    print('=== Color Accuracy (N={}) ==='.format(len(de_arr)))
    print('  dE2000 mean:   {:.3f}'.format(np.mean(de_arr)))
    print('  dE2000 median: {:.3f}'.format(np.median(de_arr)))
    print('  dE2000 95%:    {:.3f}'.format(np.percentile(de_arr, 95)))
    print('  dE2000 < 1.0:  {:.1f}%'.format(np.sum(de_arr < 1.0)/len(de_arr)*100))
    print('  dE2000 < 2.3:  {:.1f}%'.format(np.sum(de_arr < 2.3)/len(de_arr)*100))
    print('  dE2000 < 5.0:  {:.1f}%'.format(np.sum(de_arr < 5.0)/len(de_arr)*100))
    return de_arr

# ============================================================
def export_onnx(model, output_path, device):
    model.eval()
    dummy = torch.randn(1, 7).to(device)
    torch.onnx.export(model, dummy, output_path,
        input_names=['params'], output_names=['reflectance'],
        opset_version=14, dynamo=False)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print('ONNX exported: {} ({:.1f} MB)'.format(output_path, size_mb))

# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/rcwa_new/rcwa_TiO2_SiO2.pkl')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--quality', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--blocks', type=int, default=4)
    parser.add_argument('--name', type=str, default='forward_mlp_rcwa_wl')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    print('Model: hidden={}, blocks={}'.format(args.hidden, args.blocks))

    X, Y = load_wavelength_data(args.data, quality=args.quality)

    # Split by structure (not by wavelength point)
    n_struct = len(X) // 81
    n_val_struct = int(n_struct * args.val_ratio)
    idx = np.random.permutation(n_struct)
    val_struct_idx = set(idx[:n_val_struct])

    # Create train/val masks
    train_mask = np.ones(n_struct * 81, dtype=bool)
    for vi in val_struct_idx:
        train_mask[vi*81:(vi+1)*81] = False

    train_X, train_Y = X[train_mask], Y[train_mask]
    val_X, val_Y = X[~train_mask], Y[~train_mask]
    print('Train points: {}, Val points: {} ({} structures)'.format(
        len(train_X), len(val_X), n_val_struct))

    train_dataset = TensorDataset(train_X, train_Y)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = WavelengthResMLP(in_dim=7, hidden=args.hidden, out_dim=1, n_blocks=args.blocks)
    print('Parameters: {:,}'.format(sum(p.numel() for p in model.parameters())))
    print('{} epochs, lr={}, batch_size={}'.format(args.epochs, args.lr, args.batch_size))
    print('=' * 60)

    t0 = time.time()
    train(model, train_loader, val_X, val_Y, args.epochs, args.lr, device)
    print('Training time: {:.0f}s'.format(time.time() - t0))

    model.load_state_dict(torch.load('models/forward_mlp_rcwa_wl.pt', weights_only=True))
    evaluate(model, val_X, val_Y, device)

    os.makedirs('models', exist_ok=True)
    export_onnx(model, 'models/{}.onnx'.format(args.name), device)
    # Also save pt with name
    torch.save(model.state_dict(), 'models/{}.pt'.format(args.name))

    print('Done! Models saved as models/{}.pt/onnx'.format(args.name))

if __name__ == '__main__':
    main()
