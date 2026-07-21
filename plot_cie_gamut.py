"""
Fig. X — CIE 1931 色度图: 目标色 vs naïve vs hybrid 达成色
Two panels: TiO₂ (left) and a-Si (right)
Shows target, naïve-achieved, hybrid-achieved as scatter on CIE xy diagram.
Output: figs/cie_gamut.pdf + .png
"""
import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(FIG_DIR, exist_ok=True)

# === CIE 1931 2-deg CMFs (380-780nm, 5nm) — simplified from color_utils ===
# Use color_utils if available, otherwise inline
try:
    from color_utils import spectrum_to_xyz
    HAS_COLOR_UTILS = True
except ImportError:
    HAS_COLOR_UTILS = False

def rgb_to_xy(rgb):
    """Convert sRGB [0-1] to CIE xy chromaticity."""
    # Linearize sRGB
    rgb_lin = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    # sRGB -> XYZ (D65)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = M @ rgb_lin
    s = xyz.sum()
    if s < 1e-10:
        return 0.3127, 0.3290  # D65 white
    return xyz[0] / s, xyz[1] / s

def load_hybrid_xy(fname):
    """Load hybrid pkl, extract target/naive/hybrid xy coordinates."""
    with open(os.path.join(DATA, fname), 'rb') as f:
        records = pickle.load(f)
    
    targets, naives, hybrids = [], [], []
    for r in records:
        if r.get('status') != 'ok':
            continue
        t_xy = rgb_to_xy(np.array(r['target_rgb']))
        targets.append(t_xy)
        
        # Naïve achieved
        if r.get('naive_ach_rgb') is not None:
            naives.append(rgb_to_xy(np.array(r['naive_ach_rgb'])))
        elif r.get('naive_ach_de2000') is not None and 'ach_rgb' in r:
            # fallback: use hybrid rgb if naive not stored separately
            naives.append(rgb_to_xy(np.array(r['ach_rgb'])))
        else:
            naives.append(None)
        
        # Hybrid achieved
        if r.get('ach_rgb') is not None:
            hybrids.append(rgb_to_xy(np.array(r['ach_rgb'])))
        else:
            hybrids.append(None)
    
    return targets, naives, hybrids

# CIE 1931 spectral locus (approximate, for background)
CIE_LOCUS_XY = [
    (0.1741, 0.0050), (0.1740, 0.0050), (0.1738, 0.0049), (0.1736, 0.0049),
    (0.1733, 0.0048), (0.1730, 0.0048), (0.1726, 0.0048), (0.1721, 0.0048),
    (0.1714, 0.0051), (0.1703, 0.0058), (0.1689, 0.0069), (0.1669, 0.0086),
    (0.1644, 0.0109), (0.1611, 0.0138), (0.1566, 0.0177), (0.1510, 0.0227),
    (0.1440, 0.0297), (0.1355, 0.0399), (0.1241, 0.0578), (0.1096, 0.0868),
    (0.0913, 0.1327), (0.0687, 0.2007), (0.0454, 0.2950), (0.0235, 0.4127),
    (0.0082, 0.5384), (0.0039, 0.6548), (0.0139, 0.7502), (0.0389, 0.8120),
    (0.0743, 0.8338), (0.1142, 0.8262), (0.1547, 0.8059), (0.1929, 0.7816),
    (0.2296, 0.7543), (0.2658, 0.7243), (0.3016, 0.6923), (0.3373, 0.6589),
    (0.3731, 0.6245), (0.4087, 0.5896), (0.4441, 0.5547), (0.4788, 0.5202),
    (0.5125, 0.4866), (0.5448, 0.4544), (0.5752, 0.4242), (0.6029, 0.3965),
    (0.6270, 0.3725), (0.6482, 0.3514), (0.6658, 0.3340), (0.6801, 0.3197),
    (0.6915, 0.3083), (0.7006, 0.2993), (0.7079, 0.2920), (0.7140, 0.2859),
    (0.7190, 0.2809), (0.7230, 0.2770), (0.7260, 0.2740), (0.7283, 0.2717),
    (0.7300, 0.2700), (0.7311, 0.2689), (0.7320, 0.2680), (0.7327, 0.2673),
    (0.7334, 0.2666), (0.7340, 0.2660), (0.7344, 0.2656), (0.7346, 0.2654),
    (0.7347, 0.2653),
]

# === Load data ===
print("Loading TiO2...")
t_targets, t_naives, t_hybrids = load_hybrid_xy("hybrid_TiO2_SiO2.pkl")
print(f"  {len(t_targets)} targets, {sum(1 for x in t_naives if x)} naïve, {sum(1 for x in t_hybrids if x)} hybrid")

print("Loading a-Si...")
a_targets, a_naives, a_hybrids = load_hybrid_xy("hybrid_aSi_SiO2.pkl")
print(f"  {len(a_targets)} targets, {sum(1 for x in a_naives if x)} naïve, {sum(1 for x in a_hybrids if x)} hybrid")

# === Plot ===
plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in',
})

fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.5), dpi=150)

def plot_cie_panel(ax, targets, naives, hybrids, title):
    # Spectral locus
    lx = [p[0] for p in CIE_LOCUS_XY]
    ly = [p[1] for p in CIE_LOCUS_XY]
    ax.plot(lx + [lx[0]], ly + [ly[0]], 'k-', lw=0.8, alpha=0.4)
    
    # D65 white point
    ax.plot(0.3127, 0.3290, 'k+', markersize=8, mew=1.5, alpha=0.5)
    
    # Targets
    tx = [t[0] for t in targets]
    ty = [t[1] for t in targets]
    ax.scatter(tx, ty, c='none', edgecolors='#333333', s=50, linewidth=1.2,
               marker='o', zorder=4, label='Target')
    
    # Naïve achieved
    nx = [n[0] for n in naives if n is not None]
    ny = [n[1] for n in naives if n is not None]
    ax.scatter(nx, ny, c='#D6604D', s=30, marker='^', alpha=0.7,
               zorder=3, label='Naïve achieved')
    
    # Hybrid achieved
    hx = [h[0] for h in hybrids if h is not None]
    hy = [h[1] for h in hybrids if h is not None]
    ax.scatter(hx, hy, c='#4393C3', s=30, marker='s', alpha=0.7,
               zorder=3, label='Hybrid achieved')
    
    # Arrows: target -> hybrid (showing error vector)
    for i, (t, h) in enumerate(zip(targets, hybrids)):
        if h is not None:
            dx = h[0] - t[0]
            dy = h[1] - t[1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 0.005:  # only draw visible arrows
                ax.annotate('', xy=h, xytext=t,
                           arrowprops=dict(arrowstyle='->', color='#2166AC',
                                          lw=0.6, alpha=0.4))
    
    ax.set_xlabel('CIE x', fontsize=10)
    ax.set_ylabel('CIE y', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.9)
    ax.set_aspect('equal')
    ax.legend(fontsize=8.5, loc='upper right', framealpha=0.9)

plot_cie_panel(axes[0], t_targets, t_naives, t_hybrids, r'TiO$_2$/SiO$_2$')
plot_cie_panel(axes[1], a_targets, a_naives, a_hybrids, r'a-Si/SiO$_2$')

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "cie_gamut.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(FIG_DIR, "cie_gamut.png"), bbox_inches='tight', dpi=300)
print(f"\nSaved: figs/cie_gamut.pdf + .png")
plt.close()
