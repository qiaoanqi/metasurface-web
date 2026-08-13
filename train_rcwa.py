# train_rcwa.py — 用RCWA高保真数据训练ResMLP代理模型
# 用法: python train_rcwa.py --data data/rcwa_5k.pkl --epochs 300 --lr 1e-3
# 输出: models/forward_mlp_rcwa.pt (PyTorch) + models/forward_mlp_rcwa.onnx (ONNX)

import numpy as np
import pickle, argparse, time, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# ResMLP 架构 (和现有模型完全一致)
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

class DeepResMLP_Multi(nn.Module):
    def __init__(self, in_dim=7, hidden=256, out_dim=81, n_blocks=4):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.BatchNorm1d(hidden)
        )
        self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        return torch.sigmoid(self.head(x))

# ============================================================
# 数据加载与预处理
# ============================================================
# 材料/衬底编码映射 (和 ml_module.py 一致)
# ============================================================
_MAT_MAP = {'TiO2': 0, 'TiO2 (anatase)': 0, 'a-Si': 1, 'a-Si (amorphous)': 1,
             'Si3N4': 2, 'Si3N4 (nitride)': 2, 'Al2O3': 3, 'Al2O3 (sapphire)': 3,
             'GaN': 4, 'GaN (wurtzite)': 4, 'HfO2': 5, 'HfO2 (hafnia)': 5,
             'Ta2O5': 6, 'Ta2O5 (tantala)': 6}
_SUB_MAP = {'SiO2': 0, 'SiO2 (fused silica)': 0, 'Si3N4': 1, 'Si3N4 (nitride)': 1,
             'Al2O3': 2, 'Al2O3 (sapphire)': 2}

# ============================================================
def load_rcwa_data(pkl_paths, augment=True, quality=0.05):
    """加载RCWA数据并转换为训练格式，支持多个pkl合并"""
    if isinstance(pkl_paths, str):
        pkl_paths = [p.strip() for p in pkl_paths.split(',')]

    all_samples = []
    for path in pkl_paths:
        with open(path, 'rb') as f:
            samples = pickle.load(f)
        print(f"  {path}: {len(samples)} 组")
        all_samples.extend(samples)
    print(f"合计 {len(all_samples)} 组RCWA数据")

    X_list, Y_list = [], []
    skipped = 0
    mat_counts = {}

    # Compute per-substrate R+T means for relative filtering
    if quality > 0:
        rt_by_sub = {}
        for s in all_samples:
            sub = s.get('substrate', 'SiO2')
            rt_by_sub.setdefault(sub, []).append(s['R_plus_T_mean'])
        sub_means = {k: sum(v)/len(v) for k, v in rt_by_sub.items()}
        print(f"  R+T means by substrate: { {k: round(v,4) for k,v in sub_means.items()} }")

    for s in all_samples:
        # 质量过滤: lossless 用原相对判据 (|R+T - 衬底均值| > quality 跳过);
        # lossy (a-Si) 用物理判据: 仅滤能量异常 (R+T<=0 或 >1.05),
        # 因为强吸收下 R+T 分布宽 (0.1-1.0), 相对均值过滤会错误截断 (A1 审计 2026-08-07)
        if quality > 0:
            sub = s.get('substrate', 'SiO2')
            rt_mean_ref = sub_means.get(sub, 1.0)
            rt = s['R_plus_T_mean']
            mat_name0 = s.get('material', 'TiO2')
            if mat_name0 in ('a-Si', 'a-Si (amorphous)'):
                if rt <= 0.0 or rt > 1.05:
                    skipped += 1
                    continue
            else:
                if abs(rt - rt_mean_ref) > quality:
                    skipped += 1
                    continue

        D, H, P = float(s['D']), float(s['H']), float(s['P'])
        R = s['R']

        # 读取材料和衬底 (旧数据默认 TiO2/SiO2)
        mat_name = s.get('material', 'TiO2')
        sub_name = s.get('substrate', 'SiO2')
        mat_code = float(_MAT_MAP.get(mat_name, 0))
        sub_code = float(_SUB_MAP.get(sub_name, 0))
        key = f"{mat_name}/{sub_name}"
        mat_counts[key] = mat_counts.get(key, 0) + 1

        # 原始样本
        def add_sample(d, h, p, spectrum, mc, sc):
            d_norm = (d - 50) / 300
            h_norm = (h - 80) / 520
            p_norm = (p - 200) / 400
            x = [d_norm, h_norm, p_norm, 0.0, 0.0, mc, sc]
            X_list.append(x)
            Y_list.append(spectrum)

        add_sample(D, H, P, R, mat_code, sub_code)

        # 数据增强: ±2nm 随机抖动 (仅当augment=True)
        if augment:
            rng = np.random.default_rng(int(D*1000 + H*100 + P))
            for _ in range(2):  # 每个样本增强2次 → 数据量×3
                d2 = np.clip(D + rng.uniform(-2, 2), 80, 350)
                h2 = np.clip(H + rng.uniform(-2, 2), 100, 600)
                p2 = np.clip(P + rng.uniform(-2, 2), max(200, d2+20), 500)
                add_sample(d2, h2, p2, R, mat_code, sub_code)

    print(f"有效样本: {len(X_list)}, 跳过(质量差): {skipped}")
    print(f"材料/衬底分布: {mat_counts}")
    if augment:
        print(f"含数据增强: 原始{len(all_samples)-skipped} × 3 = {len(X_list)}")

    X = torch.tensor(X_list, dtype=torch.float32)
    Y = torch.tensor(Y_list, dtype=torch.float32)
    return X, Y

# ============================================================
# 训练循环
# ============================================================
def train(model, train_loader, val_X, val_Y, epochs, lr, device, name='forward_mlp_rcwa'):
    """训练ResMLP"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch_X, batch_Y in train_loader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)

            pred = model(batch_X)
            loss = criterion(pred, batch_Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = train_loss / max(n_batches, 1)

        # 验证
        model.eval()
        with torch.no_grad():
            val_pred = model(val_X.to(device))
            val_loss = criterion(val_pred, val_Y.to(device)).item()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"train_loss={avg_train_loss:.6f} | "
                  f"val_loss={val_loss:.6f} | "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f'models/{name}.pt')

    print(f"\n最佳验证损失: {best_val_loss:.6f} (epoch {best_epoch})")
    return model

# ============================================================
# 色度学评估 (ΔE2000)
# ============================================================
def evaluate_color_accuracy(model, val_X, val_Y, device):
    """评估ML模型的颜色预测精度"""
    from color_utils import spectrum_to_xyz, xyz_to_srgb, delta_e2000_scalar, rgb_to_lab_scalar

    model.eval()
    with torch.no_grad():
        pred = model(val_X.to(device)).cpu().numpy()

    target = val_Y.numpy()
    de_list = []
    WLS = np.linspace(380, 780, 81)

    for i in range(len(pred)):
        pred_xyz = spectrum_to_xyz(WLS, pred[i])
        pred_rgb = xyz_to_srgb(pred_xyz)
        targ_xyz = spectrum_to_xyz(WLS, target[i])
        targ_rgb = xyz_to_srgb(targ_xyz)

        pred_lab = rgb_to_lab_scalar(pred_rgb.tolist())
        targ_lab = rgb_to_lab_scalar(targ_rgb.tolist())
        de = delta_e2000_scalar(targ_lab, pred_lab)
        de_list.append(de)

    de_arr = np.array(de_list)
    print(f"\n=== 颜色精度评估 (N={len(de_arr)}) ===")
    print(f"  ΔE2000 平均:  {np.mean(de_arr):.3f}")
    print(f"  ΔE2000 中位数: {np.median(de_arr):.3f}")
    print(f"  ΔE2000 95%分位: {np.percentile(de_arr, 95):.3f}")
    print(f"  ΔE2000 < 1.0:  {np.sum(de_arr < 1.0)/len(de_arr)*100:.1f}%")
    print(f"  ΔE2000 < 2.3:  {np.sum(de_arr < 2.3)/len(de_arr)*100:.1f}% (人眼不可分辨)")
    print(f"  ΔE2000 < 5.0:  {np.sum(de_arr < 5.0)/len(de_arr)*100:.1f}%")
    return de_arr

# ============================================================
# ONNX导出
# ============================================================
def export_onnx(model, output_path, device):
    """导出ONNX格式"""
    model.eval()
    dummy = torch.randn(1, 7).to(device)
    torch.onnx.export(
        model, dummy, output_path,
        input_names=['params'],
        output_names=['spectrum'],
        opset_version=14,
        dynamo=False
    )
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"ONNX导出: {output_path} ({size_mb:.1f} MB)")

# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='RCWA数据训练ResMLP')
    parser.add_argument('--data', type=str, default='data/rcwa_5k.pkl')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--quality', type=float, default=0.05, help='R+T quality threshold (relative to substrate mean)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test-split', type=float, default=0.0,
                        help='Hold-out test set ratio (0=no test set, 0.2=80/20 split)')
    parser.add_argument('--test-file', type=str, default='',
                        help='Path to frozen test set pkl (auto-generated if test-split>0)')
    parser.add_argument('--split-seed', type=int, default=2026,
                        help='Fixed seed for train/test split (frozen for reproducibility)')
    parser.add_argument('--hidden', type=int, default=256, help='Hidden layer size')
    parser.add_argument('--blocks', type=int, default=4, help='Number of ResBlocks')
    parser.add_argument('--name', type=str, default='forward_mlp_rcwa',
                        help='Output model name (without extension)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 加载数据
    X, Y = load_rcwa_data(args.data, quality=args.quality)

    # === Hold-out test set (frozen, stratified by substrate) ===
    test_X, test_Y = None, None
    test_file = args.test_file
    if args.test_split > 0:
        if not test_file:
            test_file = f'data/{args.name}_test.pkl'
        import os as _os
        if _os.path.exists(test_file):
            print(f"Loading frozen test set: {test_file}")
            test_data = torch.load(test_file, weights_only=False)
            test_X, test_Y = test_data['X'], test_data['Y']
            # Use stored indices to exclude test samples (fast O(1) lookup)
            if 'indices' in test_data:
                test_indices = test_data['indices']
                train_indices = np.setdiff1d(np.arange(len(X)), test_indices)
                X, Y = X[train_indices], Y[train_indices]
                print(f"  Removed {len(test_indices)} test samples from training (frozen split, index-based)")
            else:
                print(f"  WARNING: test file missing indices, using full data")
        else:
            # First run: create stratified split
            rng = np.random.RandomState(args.split_seed)
            n_full = len(X)
            # Extract substrate codes for stratification (index 6 in 7-dim input)
            sub_codes = X[:, 6].numpy() if hasattr(X, 'numpy') else np.array(X[:, 6])
            unique_subs = np.unique(sub_codes)
            test_indices = []
            for sc in unique_subs:
                sc_idx = np.where(sub_codes == sc)[0]
                n_test_sub = max(1, int(len(sc_idx) * args.test_split))
                sc_perm = rng.permutation(sc_idx)
                test_indices.extend(sc_perm[:n_test_sub].tolist())
            test_indices = np.array(test_indices)
            train_indices = np.setdiff1d(np.arange(n_full), test_indices)

            test_X = X[test_indices]
            test_Y = Y[test_indices]
            X, Y = X[train_indices], Y[train_indices]

            torch.save({'X': test_X, 'Y': test_Y, 'indices': test_indices}, test_file)
            print(f"Frozen test set saved: {test_file} ({len(test_X)} samples)")
            print(f"  Stratified by substrate: { {float(sc): (sub_codes==sc).sum() for sc in unique_subs} }")

    # 划分训练/验证集
    n = len(X)
    n_val = int(n * args.val_ratio)
    rng_val = np.random.RandomState(args.seed)
    idx = rng_val.permutation(n)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    train_X, train_Y = X[train_idx], Y[train_idx]
    val_X, val_Y = X[val_idx], Y[val_idx]
    print(f"训练集: {len(train_X)}, 验证集: {len(val_X)}")

    train_dataset = TensorDataset(train_X, train_Y)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 训练
    model = DeepResMLP_Multi(in_dim=7, hidden=args.hidden, out_dim=81, n_blocks=args.blocks)
    print(f"\n模型参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"训练 {args.epochs} epochs, lr={args.lr}, batch_size={args.batch_size}")
    print("=" * 60)

    t0 = time.time()
    train(model, train_loader, val_X, val_Y, args.epochs, args.lr, device, args.name)
    print(f"训练耗时: {time.time()-t0:.0f}秒")

    # 加载最佳模型评估
    model.load_state_dict(torch.load(f'models/{args.name}.pt', weights_only=True))
    val_de = evaluate_color_accuracy(model, val_X, val_Y, device)

        # === Hold-out test evaluation (frozen split) ===
    if test_X is not None:
        print(f"\n{'='*60}")
        print(f"HOLD-OUT TEST SET EVALUATION (frozen, N={len(test_X)})")
        print(f"{'='*60}")
        test_de = evaluate_color_accuracy(model, test_X, test_Y, device)
        # Save test metrics
        test_metrics = {
            'mean': float(np.mean(test_de)), 'median': float(np.median(test_de)),
            'p95': float(np.percentile(test_de, 95)),
            'pct_lt_1': float(np.mean(test_de < 1.0) * 100),
            'pct_lt_2_3': float(np.mean(test_de < 2.3) * 100),
            'pct_lt_5': float(np.mean(test_de < 5.0) * 100),
            'val_mean': float(np.mean(val_de)),
        }
        import json
        metrics_file = f'models/{args.name}_test_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        print(f"Test metrics saved: {metrics_file}")

# 导出ONNX
    os.makedirs('models', exist_ok=True)
    export_onnx(model, f'models/{args.name}.onnx', device)

    print("\n完成! 模型已保存到:")
    print(f"  PyTorch: models/{args.name}.pt")
    print(f"  ONNX:    models/{args.name}.onnx")

if __name__ == '__main__':
    main()
