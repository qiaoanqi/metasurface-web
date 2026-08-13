import pickle, os, numpy as np
os.chdir(r"C:\Users\A\Desktop\AI超表面结构色智能设计系统")
from color_utils import spectrum_to_srgb, rgb_to_lab, delta_e2000

# a-Si mean predictor on full clean dataset
with open("data/rcwa_aSi_k_SiO2_clean.pkl", "rb") as f:
    asi = pickle.load(f)
asi_rgb = np.array([r["rgb"] for r in asi])
mean_rgb = asi_rgb.mean(axis=0)
mean_lab = rgb_to_lab(mean_rgb)

de00s = []
for rgb in asi_rgb:
    de00s.append(delta_e2000(rgb_to_lab(rgb), mean_lab))
de00s = np.array(de00s)
print(f"a-Si mean predictor (full N={len(asi)}):")
print(f"  mean DE00: {np.mean(de00s):.2f}")
print(f"  median DE00: {np.median(de00s):.2f}")
print(f"  ML holdout DE00: 2.38")

# Also check Si3N4 for comparison
for fname in ["rcwa_Si3N4_SiO2.pkl"]:
    try:
        with open(os.path.join("data", fname), "rb") as f:
            sin = pickle.load(f)
        sin_rgb = np.array([r["rgb"] for r in sin])
        mean_rgb_sin = sin_rgb.mean(axis=0)
        mean_lab_sin = rgb_to_lab(mean_rgb_sin)
        de_sin = np.array([delta_e2000(rgb_to_lab(r), mean_lab_sin) for r in sin_rgb])
        print(f"\nSi3N4/SiO2 mean predictor (N={len(sin)}):")
        print(f"  mean DE00: {np.mean(de_sin):.2f}")
        print(f"  median DE00: {np.median(de_sin):.2f}")
        print(f"  (paper says Si3N4 ML is 1.57)")
    except Exception as e:
        print(f"Si3N4: {e}")

# Also on a-Si test sets if available
for seed in [1,2,3]:
    fname = f"data/forward_mlp_rcwa_aSi_k_s{seed}_test.pkl"
    if os.path.exists(fname):
        with open(fname, "rb") as f:
            test = pickle.load(f)
        test_rgb = np.array([r["rgb"] for r in test])
        de_test = np.array([delta_e2000(rgb_to_lab(r), mean_lab) for r in test_rgb])
        print(f"  a-Si test s{seed} (N={len(test)}): mean={np.mean(de_test):.2f}, med={np.median(de_test):.2f}")
