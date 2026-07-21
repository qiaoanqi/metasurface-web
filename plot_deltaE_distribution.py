"""
Fig. X — 闭环逆设计 ΔE 分布: naïve vs hybrid (TiO₂ + a-Si 双面板)
Left panel: TiO₂/SiO₂, Right panel: a-Si/SiO₂
Histogram + KDE overlay, JND threshold line, mean markers.
Output: figs/deltaE_distribution.pdf + .png
"""
import os, pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(FIG_DIR, exist_ok=True)

JND = 2.3

def load_hybrid(fname):
    with open(os.path.join(DATA, fname), 'rb') as f:
        records = pickle.load(f)
    naive = [r['naive_ach_de2000'] for r in records if r.get('naive_ach_de2000') is not None]
    hybrid = [r['ach_de2000'] for r in records if r.get('ach_de2000') is not None]
    return np.array(naive), np.array(hybrid)

tio2_naive, tio2_hybrid = load_hybrid("hybrid_TiO2_SiO2.pkl")
asi_naive, asi_hybrid = load_hybrid("hybrid_aSi_SiO2.pkl")

# === Style ===
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.linewidth': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})

fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), dpi=150, sharey=True)

def plot_panel(ax, naive, hybrid, title, xlim_max):
    bins = np.linspace(0, xlim_max, 16)
    
    # Histograms
    ax.hist(naive, bins=bins, alpha=0.35, color='#D6604D', density=True, label='Naïve (ML-only)')
    ax.hist(hybrid, bins=bins, alpha=0.35, color='#4393C3', density=True, label='Hybrid (RCWA re-rank)')
    
    # KDE
    if len(naive) > 3:
        kde_n = gaussian_kde(naive, bw_method=0.4)
        x_kde = np.linspace(0, xlim_max, 200)
        ax.plot(x_kde, kde_n(x_kde), color='#B2182B', lw=1.8, ls='-')
    if len(hybrid) > 3:
        kde_h = gaussian_kde(hybrid, bw_method=0.4)
        ax.plot(x_kde, kde_h(x_kde), color='#2166AC', lw=1.8, ls='-')
    
    # JND threshold
    ax.axvline(JND, color='black', ls='--', lw=1.2, alpha=0.7)
    ax.text(JND + 0.3, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 0.15,
            f'JND = {JND}', fontsize=9, color='black', va='top')
    
    # Mean markers
    mean_n = naive.mean()
    mean_h = hybrid.mean()
    ax.axvline(mean_n, color='#B2182B', ls=':', lw=1.5, alpha=0.8)
    ax.axvline(mean_h, color='#2166AC', ls=':', lw=1.5, alpha=0.8)
    
    # Stats text
    stats_text = (f"Naïve: {mean_n:.1f} ({100*np.mean(naive<JND):.0f}% < JND)\n"
                  f"Hybrid: {mean_h:.1f} ({100*np.mean(hybrid<JND):.0f}% < JND)\n"
                  f"Improve: +{mean_n - mean_h:.1f}")
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes,
            fontsize=8.5, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', lw=0.5, alpha=0.9))
    
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel(r'Achieved $\Delta E_{00}$ (RCWA verified)', fontsize=10)
    ax.set_xlim(0, xlim_max)

plot_panel(axes[0], tio2_naive, tio2_hybrid, r'TiO$_2$/SiO$_2$  ($n$ = 2.30)', 15)
plot_panel(axes[1], asi_naive, asi_hybrid, r'a-Si/SiO$_2$  ($n$ = 3.80)', 45)
axes[0].set_ylabel('Density', fontsize=10)

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=9.5,
           bbox_to_anchor=(0.5, -0.02), frameon=False)

plt.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(os.path.join(FIG_DIR, "deltaE_distribution.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(FIG_DIR, "deltaE_distribution.png"), bbox_inches='tight', dpi=300)
print(f"Saved: figs/deltaE_distribution.pdf + .png")
print(f"  TiO2: naive mean={tio2_naive.mean():.2f}, hybrid mean={tio2_hybrid.mean():.2f}, N={len(tio2_naive)}")
print(f"  a-Si: naive mean={asi_naive.mean():.2f}, hybrid mean={asi_hybrid.mean():.2f}, N={len(asi_naive)}")
plt.close()
