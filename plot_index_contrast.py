"""
Fig. X — 折射率对比度 vs 反射谱动态范围（材料筛选判据）
Publication-quality plot: 11 data points, threshold annotation, material color-coding.
Output: figs/index_contrast_criterion.pdf + .png
"""
import os, pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# === Load data ===
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
with open(os.path.join(DATA, "index_contrast_criterion.pkl"), 'rb') as f:
    results = pickle.load(f)

# === Style ===
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.linewidth': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 4,
    'ytick.major.size': 4,
})

# Material colors
MAT_COLORS = {
    "TiO2": "#2166AC",   # blue
    "a-Si": "#B2182B",   # red
    "Si3N4": "#4DAF4A",  # green
    "Al2O3": "#FF7F00",  # orange
}
MAT_MARKERS = {
    "TiO2": "o",
    "a-Si": "s",
    "Si3N4": "^",
    "Al2O3": "D",
}

fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.0), dpi=150)

# Plot each data point
for r in results:
    pillar = r["pillar"]
    color = MAT_COLORS.get(pillar, "#333333")
    marker = MAT_MARKERS.get(pillar, "o")
    ax.scatter(r["delta_n"], r["R_range_mean"],
               c=color, marker=marker, s=70, zorder=5,
               edgecolors='white', linewidth=0.5, alpha=0.9)

# Threshold region
ax.axvspan(0.45, 0.55, alpha=0.08, color='gray', zorder=1)
ax.axvline(0.5, color='gray', linestyle='--', linewidth=1.0, alpha=0.7, zorder=2)
ax.text(0.50, 0.55, r'$\Delta n_{\mathrm{th}} \approx 0.5$',
        fontsize=10, color='gray', ha='center', va='bottom')

# Annotate the 7x collapse pair
ax.annotate('', xy=(0.532, 0.043), xytext=(0.545, 0.309),
            arrowprops=dict(arrowstyle='<->', color='#666666', lw=1.2))
ax.text(0.56, 0.18, '7× collapse\n(Δn diff = 0.013)',
        fontsize=8.5, color='#444444', ha='left', va='center',
        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#cccccc', lw=0.5))

# Annotate a-Si impedance mismatch
ax.annotate('Impedance\nmismatch limit', xy=(2.342, 0.186), xytext=(1.8, 0.35),
            fontsize=8.5, color=MAT_COLORS["a-Si"], ha='center',
            arrowprops=dict(arrowstyle='->', color=MAT_COLORS["a-Si"], lw=1.0))

# Region labels
ax.text(0.25, 0.02, 'Flat spectrum\n(no structural color)', fontsize=9,
        color='#888888', ha='center', va='bottom', style='italic')
ax.text(1.5, 0.02, 'Viable resonance\n(structural color)', fontsize=9,
        color='#888888', ha='center', va='bottom', style='italic')

# Axes
ax.set_xlabel(r'Refractive index contrast $\Delta n = n_{\mathrm{pillar}} - n_{\mathrm{substrate}}$', fontsize=11)
ax.set_ylabel('Reflection dynamic range\n' + r'$\langle R_{\max} - R_{\min} \rangle$', fontsize=11)
ax.set_xlim(-0.4, 2.6)
ax.set_ylim(-0.02, 0.65)
ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])

# Legend
legend_elements = [
    Line2D([0], [0], marker=MAT_MARKERS[m], color='w', markerfacecolor=MAT_COLORS[m],
           markersize=8, markeredgecolor='white', markeredgewidth=0.5, label=m)
    for m in ["TiO2", "a-Si", "Si3N4", "Al2O3"]
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
          framealpha=0.9, edgecolor='#cccccc')

plt.tight_layout()

# Save
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(FIG_DIR, exist_ok=True)
fig.savefig(os.path.join(FIG_DIR, "index_contrast_criterion.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(FIG_DIR, "index_contrast_criterion.png"), bbox_inches='tight', dpi=300)
print(f"Saved: figs/index_contrast_criterion.pdf + .png")
plt.close()
