"""
Fig. 5 — Forward prediction accuracy: ML-predicted vs RCWA-ground-truth ΔE
Scatter plot on holdout test sets for TiO₂ and a-Si.
Each point = one structure; color = material; diagonal = perfect prediction.
Output: figs/predicted_vs_achieved.pdf + .png
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import ml_module
ml_module.init_rcwa_ml()  # Load ONNX ensemble sessions
from color_utils import spectrum_to_srgb, rgb_to_lab, delta_e2000

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(FIG_DIR, exist_ok=True)

WL = np.linspace(380, 780, 81)

# Normalization inverse (from ml_module)
def denorm(x):
    D = x[0] * 300 + 50
    H = x[1] * 520 + 80
    P = x[2] * 400 + 200
    angle = x[3] * 80
    pol = "TE" if x[4] < 0.5 else "TM"
    return D, H, P, angle, pol

INV_MAT = {v: k for k, v in ml_module.MATERIAL_CODES.items()}
INV_SUB = {v: k for k, v in ml_module.SUBSTRATE_CODES.items()}

def evaluate_holdout(test_file, mat_filter=None):
    """Load holdout, run ML prediction, compute ΔE vs ground truth."""
    data = torch.load(test_file, weights_only=False)
    X = data['X'] if isinstance(data, dict) else data[0]
    Y = data['Y'] if isinstance(data, dict) else data[1]
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.numpy()
    
    pred_des = []
    true_des = []
    for i in range(len(X)):
        x = X[i]
        D, H, P, angle, pol = denorm(x)
        mat = INV_MAT.get(int(round(x[5])), "TiO2 (anatase)")
        sub = INV_SUB.get(int(round(x[6])), "SiO2 (fused silica)")
        
        if mat_filter and mat != mat_filter:
            continue
        
        # ML predicted spectrum -> RGB -> Lab
        try:
            pred_spec = ml_module.predict_spectrum(D, H, P, angle_deg=angle, polarization=pol,
                                                    material=mat, substrate=sub)
            pred_rgb = spectrum_to_srgb(WL, np.clip(pred_spec, 0, None))
            pred_lab = rgb_to_lab(pred_rgb)
            
            # Ground truth spectrum -> RGB -> Lab
            true_rgb = spectrum_to_srgb(WL, np.clip(Y[i], 0, None))
            true_lab = rgb_to_lab(true_rgb)
            
            de = delta_e2000(pred_lab, true_lab)
            pred_des.append(de)
            true_des.append(de)  # same value, we plot ΔE distribution
        except Exception:
            continue
    
    return np.array(pred_des)

print("Evaluating TiO2 holdout...")
tio2_de = evaluate_holdout(os.path.join(DATA, "TiO2_holdout_test.pkl"), "TiO2 (anatase)")
print(f"  N={len(tio2_de)}, mean={tio2_de.mean():.2f}, median={np.median(tio2_de):.2f}")

print("Evaluating a-Si holdout...")
asi_de = evaluate_holdout(os.path.join(DATA, "aSi_holdout_test.pkl"), "a-Si (amorphous)")
print(f"  N={len(asi_de)}, mean={asi_de.mean():.2f}, median={np.median(asi_de):.2f}")

# === Plot: ΔE histogram with CDF ===
plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in',
})

fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), dpi=150)

# Panel (a): Histogram
ax = axes[0]
bins = np.linspace(0, 20, 41)
ax.hist(tio2_de, bins=bins, alpha=0.5, color='#2166AC', density=True,
        label=f'TiO₂ (mean={tio2_de.mean():.2f})')
ax.hist(asi_de, bins=bins, alpha=0.5, color='#B2182B', density=True,
        label=f'a-Si (mean={asi_de.mean():.2f})')
ax.axvline(2.3, color='black', ls='--', lw=1.2, alpha=0.7)
ax.text(2.5, ax.get_ylim()[1]*0.9 if ax.get_ylim()[1] > 0 else 0.1, 'JND = 2.3',
        fontsize=9, color='black')
ax.set_xlabel(r'Forward prediction $\Delta E_{00}$', fontsize=10)
ax.set_ylabel('Density', fontsize=10)
ax.set_title('(a) Prediction error distribution', fontsize=10)
ax.legend(fontsize=9)
ax.set_xlim(0, 20)

# Panel (b): CDF
ax = axes[1]
for de, label, color in [(tio2_de, 'TiO₂', '#2166AC'), (asi_de, 'a-Si', '#B2182B')]:
    sorted_de = np.sort(de)
    cdf = np.arange(1, len(sorted_de) + 1) / len(sorted_de)
    ax.plot(sorted_de, cdf, color=color, lw=2.0, label=label)

ax.axvline(2.3, color='black', ls='--', lw=1.0, alpha=0.6)
ax.axhline(0.5, color='gray', ls=':', lw=0.8, alpha=0.5)
# Annotate % < JND
tio2_pct = 100 * np.mean(tio2_de < 2.3)
asi_pct = 100 * np.mean(asi_de < 2.3)
ax.scatter([2.3], [tio2_pct/100], color='#2166AC', s=60, zorder=5, marker='o')
ax.scatter([2.3], [asi_pct/100], color='#B2182B', s=60, zorder=5, marker='s')
ax.text(3.0, tio2_pct/100 + 0.03, f'{tio2_pct:.0f}%', fontsize=9, color='#2166AC')
ax.text(3.0, asi_pct/100 - 0.05, f'{asi_pct:.0f}%', fontsize=9, color='#B2182B')

ax.set_xlabel(r'$\Delta E_{00}$ threshold', fontsize=10)
ax.set_ylabel('Cumulative fraction', fontsize=10)
ax.set_title('(b) CDF of prediction error', fontsize=10)
ax.legend(fontsize=9, loc='lower right')
ax.set_xlim(0, 15)
ax.set_ylim(0, 1.02)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "predicted_vs_achieved.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(FIG_DIR, "predicted_vs_achieved.png"), bbox_inches='tight', dpi=300)
print(f"\nSaved: figs/predicted_vs_achieved.pdf + .png")
plt.close()
