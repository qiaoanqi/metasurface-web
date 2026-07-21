"""
四材料折射率对比 vs 反射谱变化量 — 材料筛选判据数据
输出: 每对 (pillar, substrate) 的 Δn、单样本 R 范围 mean/std、样本间 std
用于论文 Fig: "折射率对比度 vs 共振幅度" 单调曲线
"""
import os, pickle
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Cauchy A 系数 (rcwa_batch.py line 25-31)
N_PILLAR = {"TiO2": 2.30, "a-Si": 3.80, "Si3N4": 1.99, "Al2O3": 1.7546}
N_SUB    = {"SiO2": 1.458, "Si3N4": 1.99, "Al2O3": 1.7546}

# 数据文件映射: (pillar, substrate) -> filename
FILES = {
    ("TiO2", "SiO2"):   "rcwa_5k.pkl",
    ("a-Si", "SiO2"):   "rcwa_a-Si_SiO2.pkl",
    ("a-Si", "Si3N4"):  "rcwa_a-Si_Si3N4.pkl",
    ("a-Si", "Al2O3"):  "rcwa_a-Si_Al2O3.pkl",
    ("Si3N4", "SiO2"):  "rcwa_Si3N4_SiO2.pkl",
    ("Si3N4", "Si3N4"): "rcwa_Si3N4_Si3N4.pkl",
    ("Si3N4", "Al2O3"): "rcwa_Si3N4_Al2O3.pkl",
    ("Al2O3", "Si3N4"): "rcwa_al2o3_sin4_5k.pkl",
    ("Al2O3", "Al2O3"): "rcwa_al2o3_al2o3_5k.pkl",
    # TiO2 on other substrates
    ("TiO2", "Si3N4"):  "rcwa_tio2_sin4_5k.pkl",
    ("TiO2", "Al2O3"):  "rcwa_tio2_al2o3_5k.pkl",
}

def load_R(path):
    """从 pkl 提取反射谱矩阵 [N, 81]."""
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, list) and obj and isinstance(obj[0], dict) and 'R' in obj[0]:
        return np.stack([np.asarray(d['R'], dtype=np.float64) for d in obj])
    if isinstance(obj, dict):
        for k in ('Y', 'y', 'spectra', 'R', 'reflectance'):
            if k in obj:
                return np.asarray(obj[k], dtype=np.float64)
    if isinstance(obj, (tuple, list)):
        for item in obj:
            a = np.asarray(item)
            if a.ndim == 2 and a.shape[1] == 81:
                return a.astype(np.float64)
    a = np.asarray(obj)
    if a.ndim == 2 and a.shape[1] == 81:
        return a.astype(np.float64)
    return None

print("=" * 80)
print("  折射率对比度 vs 反射谱变化量 — 四材料筛选判据")
print("=" * 80)
print(f"{'材料对':<20s} {'Δn':>6s} {'N':>6s} {'R范围mean':>10s} {'R范围std':>10s} "
      f"{'样本间std':>10s} {'R总mean':>8s} {'判定':<12s}")
print("-" * 80)

results = []
for (pillar, sub), fname in sorted(FILES.items(), key=lambda x: -(N_PILLAR[x[0][0]] - N_SUB[x[0][1]])):
    path = os.path.join(DATA, fname)
    dn = N_PILLAR[pillar] - N_SUB[sub]
    if not os.path.exists(path):
        print(f"  {pillar+'/'+sub:<18s} {dn:6.3f}  -- 文件缺失: {fname}")
        continue
    R = load_R(path)
    if R is None or R.ndim != 2:
        print(f"  {pillar+'/'+sub:<18s} {dn:6.3f}  -- 加载失败")
        continue
    per_sample_range = R.max(1) - R.min(1)  # 每样本共振幅度
    cross_std = R.std(0).mean()              # 样本间变异
    r_mean = R.mean()
    # 判定: R范围 mean < 0.05 → 平谱(不适用); 0.05-0.10 → 边缘; > 0.10 → 可用
    if per_sample_range.mean() < 0.05:
        verdict = "平谱(负对照)"
    elif per_sample_range.mean() < 0.10:
        verdict = "边缘"
    else:
        verdict = "可用(正例)"
    label = f"{pillar}/{sub}"
    print(f"  {label:<18s} {dn:6.3f} {R.shape[0]:6d} {per_sample_range.mean():10.4f} "
          f"{per_sample_range.std():10.4f} {cross_std:10.4f} {r_mean:8.4f} {verdict}")
    results.append({
        "pair": label, "pillar": pillar, "substrate": sub,
        "delta_n": dn, "N": R.shape[0],
        "R_range_mean": per_sample_range.mean(),
        "R_range_std": per_sample_range.std(),
        "cross_sample_std": cross_std,
        "R_global_mean": r_mean,
        "verdict": verdict,
    })

print("-" * 80)
print("\n  判据阈值建议: R范围 mean >= 0.10 → 可用; < 0.05 → 不适用(平谱)")
print(f"  数据点: {len(results)} 对")

# 额外: 聚焦 SiO2 衬底 (论文主场景)
print("\n" + "=" * 80)
print("  聚焦 SiO₂ 衬底 (论文主场景, 按 Δn 降序)")
print("=" * 80)
sio2 = [r for r in results if r["substrate"] == "SiO2"]
sio2.sort(key=lambda r: -r["delta_n"])
for r in sio2:
    print(f"  {r['pillar']:<8s}  Δn={r['delta_n']:.3f}  R范围={r['R_range_mean']:.4f}  "
          f"样本间std={r['cross_sample_std']:.4f}  → {r['verdict']}")

# 保存为 pickle 方便后续画图
out_path = os.path.join(DATA, "index_contrast_criterion.pkl")
with open(out_path, 'wb') as f:
    pickle.dump(results, f)
print(f"\n  数据已保存: {out_path}")
