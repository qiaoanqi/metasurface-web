"""
Fig. X — nG 收敛性: 逐结构 ΔE(nG=65) vs ΔE(nG=101) + 极差分布
Two panels: (a) scatter with 1:1 line, (b) histogram of per-structure spread.
Runs RCWA at both nG values for hybrid-selected structures.
Output: figs/nG_convergence.pdf + .png
"""
import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from color_utils import rgb_to_lab, delta_e2000
from closed_loop_validate import rcwa_verify

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(FIG_DIR, exist_ok=True)

def convergence_per_structure(pkl_path, nG_low=65, nG_high=101, Nxy=256):
    """Re-run RCWA at two nG values, return per-structure ΔE for each."""
    with open(pkl_path, 'rb') as f:
        records = pickle.load(f)
    ok = [r for r in records if r.get('status') == 'ok' and 'D' in r]
    
    results = []
    for i, r in enumerate(ok):
        D, H, P = r['D'], r['H'], r['P']
        target_lab = rgb_to_lab(np.array(r['target_rgb']))
        mat = r['material']
        sub = r['substrate']
        
        try:
            ver_low = rcwa_verify(D, H, P, mat, sub, nG=nG_low, Nxy=Nxy)
            ver_high = rcwa_verify(D, H, P, mat, sub, nG=nG_high, Nxy=Nxy)
            if ver_low is None or ver_high is None:
                continue
            rgb_low = ver_low[0]
            rgb_high = ver_high[0]
            de_low = delta_e2000(rgb_to_lab(rgb_low), target_lab)
            de_high = delta_e2000(rgb_to_lab(rgb_high), target_lab)
            results.append({'de_low': de_low, 'de_high': de_high, 'spread': abs(de_high - de_low)})
        except Exception as e:
            print(f"  [{i}] FAIL: {e}")
            continue
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(ok)} done")
    
    return results

print("=== TiO2/SiO2 convergence ===")
tio2 = convergence_per_structure(os.path.join(DATA, "hybrid_TiO2_SiO2.pkl"))
print(f"  N={len(tio2)}")

print("=== a-Si/SiO2 convergence ===")
asi = convergence_per_structure(os.path.join(DATA, "hybrid_aSi_SiO2.pkl"))
print(f"  N={len(asi)}")

# Save per-structure data
with open(os.path.join(DATA, "nG_convergence_per_structure.pkl"), 'wb') as f:
    pickle.dump({'TiO2': tio2, 'a-Si': asi}, f)

# === Plot ===
plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in',
})

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), dpi=150)

# Panel (a): scatter ΔE(65) vs ΔE(101)
ax = axes[0]
for data, label, color, marker in [(tio2, 'TiO₂', '#2166AC', 'o'), (asi, 'a-Si', '#B2182B', 's')]:
    x = [d['de_low'] for d in data]
    y = [d['de_high'] for d in data]
    ax.scatter(x, y, c=color, marker=marker, s=40, alpha=0.7, label=label, edgecolors='white', linewidth=0.3)

lim = max(max(d['de_high'] for d in asi), max(d['de_low'] for d in asi)) * 1.1
ax.plot([0, lim], [0, lim], 'k--', lw=0.8, alpha=0.5, label='1:1 line')
ax.axvline(2.3, color='gray', ls=':', lw=1.0, alpha=0.6)
ax.axhline(2.3, color='gray', ls=':', lw=1.0, alpha=0.6)
ax.text(2.5, lim*0.95, 'JND', fontsize=8, color='gray')
ax.set_xlabel(r'$\Delta E_{00}$ at nG = 65', fontsize=10)
ax.set_ylabel(r'$\Delta E_{00}$ at nG = 101', fontsize=10)
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.legend(fontsize=9, loc='lower right')
ax.set_title('(a) Per-structure convergence', fontsize=10)
ax.set_aspect('equal')

# Panel (b): histogram of spread
ax = axes[1]
tio2_spread = [d['spread'] for d in tio2]
asi_spread = [d['spread'] for d in asi]
bins = np.linspace(0, max(max(asi_spread), 5), 20)
ax.hist(tio2_spread, bins=bins, alpha=0.5, color='#2166AC', label=f'TiO₂ (mean={np.mean(tio2_spread):.2f})')
ax.hist(asi_spread, bins=bins, alpha=0.5, color='#B2182B', label=f'a-Si (mean={np.mean(asi_spread):.2f})')
ax.axvline(2.3, color='black', ls='--', lw=1.2, alpha=0.7)
ax.text(2.4, ax.get_ylim()[1]*0.9, '1 JND', fontsize=9, color='black')
ax.set_xlabel(r'$|\Delta E_{00}(\mathrm{nG\!=\!101}) - \Delta E_{00}(\mathrm{nG\!=\!65})|$', fontsize=10)
ax.set_ylabel('Count', fontsize=10)
ax.legend(fontsize=9)
ax.set_title('(b) Convergence spread distribution', fontsize=10)

# Stats annotation
tio2_pct = 100 * np.mean(np.array(tio2_spread) < 2.3)
asi_pct = 100 * np.mean(np.array(asi_spread) < 2.3)
ax.text(0.97, 0.75, f'TiO₂: {tio2_pct:.0f}% < 1 JND\na-Si: {asi_pct:.0f}% < 1 JND',
        transform=ax.transAxes, fontsize=9, va='top', ha='right',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', lw=0.5))

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "nG_convergence.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(FIG_DIR, "nG_convergence.png"), bbox_inches='tight', dpi=300)
print(f"\nSaved: figs/nG_convergence.pdf + .png")
print(f"  TiO2: {tio2_pct:.0f}% < 1 JND, mean spread = {np.mean(tio2_spread):.2f}")
print(f"  a-Si: {asi_pct:.0f}% < 1 JND, mean spread = {np.mean(asi_spread):.2f}")
plt.close()
