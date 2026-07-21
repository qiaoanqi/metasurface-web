"""
Fig. X — nG 收敛性汇总 (基于已验证数据, 不重跑 RCWA)
Bar chart: mean spread + % < 1 JND for TiO₂ vs a-Si
Data from: closed_loop_validate.py --verify-nG 65,101 results
Output: figs/nG_convergence_summary.pdf + .png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(FIG_DIR, exist_ok=True)

# Verified data from --verify-nG 65,101 runs
data = {
    'TiO₂/SiO₂\n(n=2.30, Δn=0.84)': {
        'mean_spread': 0.77, 'max_spread': 3.74, 'pct_1jnd': 90,
        'mean_de_65': 2.33, 'mean_de_101': 2.84, 'rt': 1.0250, 'n': 30,
    },
    'a-Si/SiO₂\n(n=3.80, Δn=2.34)': {
        'mean_spread': 5.02, 'max_spread': 18.39, 'pct_1jnd': 40,
        'mean_de_65': 12.15, 'mean_de_101': 14.84, 'rt': 1.0436, 'n': 30,
    },
}

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in',
})

fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.5), dpi=150)
materials = list(data.keys())
colors = ['#2166AC', '#B2182B']

# Panel (a): Mean ΔE at nG=65 vs nG=101
ax = axes[0]
x = np.arange(len(materials))
w = 0.35
de65 = [data[m]['mean_de_65'] for m in materials]
de101 = [data[m]['mean_de_101'] for m in materials]
bars1 = ax.bar(x - w/2, de65, w, color=colors, alpha=0.7, label='nG = 65')
bars2 = ax.bar(x + w/2, de101, w, color=colors, alpha=0.35, label='nG = 101',
               edgecolor=colors, linewidth=1.2, hatch='//')
ax.set_ylabel(r'Mean achieved $\Delta E_{00}$', fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(materials, fontsize=9)
ax.legend(fontsize=9)
ax.set_title('(a) ΔE vs Fourier order', fontsize=10)
ax.axhline(2.3, color='gray', ls='--', lw=1.0, alpha=0.6)
ax.text(1.45, 2.5, 'JND', fontsize=8, color='gray')
# Value labels
for bar, val in zip(bars1, de65):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.3, f'{val:.1f}',
            ha='center', fontsize=8.5, color=bar.get_facecolor())
for bar, val in zip(bars2, de101):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.3, f'{val:.1f}',
            ha='center', fontsize=8.5, color='#555555')

# Panel (b): Per-structure spread
ax = axes[1]
spreads = [data[m]['mean_spread'] for m in materials]
max_spreads = [data[m]['max_spread'] for m in materials]
bars = ax.bar(x, spreads, 0.5, color=colors, alpha=0.7)
ax.errorbar(x, spreads, yerr=[[0,0], [s-m for s,m in zip(max_spreads, spreads)]],
            fmt='none', color='black', capsize=5, lw=1.2)
ax.axhline(2.3, color='black', ls='--', lw=1.2, alpha=0.7)
ax.text(1.45, 2.5, '1 JND threshold', fontsize=8.5, color='black')
ax.set_ylabel(r'Mean $|\Delta E$(nG=101) $-$ ΔE(nG=65)$|$', fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(materials, fontsize=9)
ax.set_title('(b) Convergence spread', fontsize=10)
# Value labels
for bar, val, mx in zip(bars, spreads, max_spreads):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.3, f'{val:.2f}',
            ha='center', fontsize=9, fontweight='bold')
    ax.text(bar.get_x() + bar.get_width()/2, mx + 0.5, f'max={mx:.1f}',
            ha='center', fontsize=8, color='#666666')

# Panel (c): % within 1 JND
ax = axes[2]
pcts = [data[m]['pct_1jnd'] for m in materials]
bars = ax.bar(x, pcts, 0.5, color=colors, alpha=0.7)
ax.axhline(90, color='gray', ls=':', lw=1.0, alpha=0.5)
ax.text(1.45, 91, '90% target', fontsize=8, color='gray')
ax.set_ylabel('Structures within 1 JND (%)', fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(materials, fontsize=9)
ax.set_ylim(0, 105)
ax.set_title('(c) Convergence rate', fontsize=10)
for bar, val in zip(bars, pcts):
    ax.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val}%',
            ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "nG_convergence_summary.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(FIG_DIR, "nG_convergence_summary.png"), bbox_inches='tight', dpi=300)
print(f"Saved: figs/nG_convergence_summary.pdf + .png")
plt.close()
