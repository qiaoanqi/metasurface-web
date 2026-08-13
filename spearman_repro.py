# spearman_repro.py — Spearman 秩相关复现脚本 (审计要求 2026-08-08)
# 正文句子 (paper_oe.tex L211): rho=-0.43 p=0.34 @nG65; rho=+0.14 p=0.76 @nG101
# 输入向量 (统一 seed-mean forward, 与 Table 3 forward 列同口径):
#   forward seed-mean 来源:
#     a-Si 2.368  = mean(models/forward_mlp_rcwa_aSi_PS_s{42,123,456}_test_metrics.json 'mean')
#     Si3N4 13.75 = A2 独立冻结划分复现 (N=5400, 三衬底聚合)
#     TiO2  2.99  = 既有 metrics (frozen holdout)
#     GaN/Ta2O5/HfO2/Al2O3 = models/forward_mlp_rcwa_{MAT}_seedmean.json 'seed_mean'
#        (GaN 2.889, Ta2O5 3.037, HfO2 2.958, Al2O3 8.114; N=493/540/545/450)
#   RT 成功率 (closed_loop pkl, hybrid, JND 2.3):
#     nG65: a-Si 86, Si3N4 81, HfO2 82, Ta2O5 74, GaN 66, TiO2 62, Al2O3 43
#     nG101 (ng_verify 归档): a-Si 47, Si3N4 79, HfO2 80, Ta2O5 69, GaN 58, TiO2 53, Al2O3 43
# 输出全部变体, 与正文句对照.
import math
import numpy as np
from scipy.stats import t as tdist

def spearman(x, y):
    """Spearman rank correlation with t-approximation p-value (n<=30 standard)."""
    rx = np.argsort(np.argsort(np.asarray(x, dtype=float)))
    ry = np.argsort(np.argsort(np.asarray(y, dtype=float)))
    n = len(x)
    d = rx - ry
    rho = 1 - 6 * np.sum(d * d) / (n * (n * n - 1))
    if n > 2:
        t = rho * math.sqrt((n - 2) / max(1e-12, 1 - rho * rho))
        p = 2 * tdist.sf(abs(t), n - 2)
    else:
        p = 1.0
    return rho, p

# ---- 输入向量 (材料顺序固定: a-Si, Si3N4, HfO2, Ta2O5, GaN, TiO2, Al2O3) ----
MATS = ['a-Si', 'Si3N4', 'HfO2', 'Ta2O5', 'GaN', 'TiO2', 'Al2O3']
FWD_SEEDMEAN = {'a-Si': 2.368, 'Si3N4': 13.75, 'HfO2': 2.958, 'Ta2O5': 3.037,
                'GaN': 2.889, 'TiO2': 2.99, 'Al2O3': 8.114}
FWD_ENSEMBLE = {'a-Si': 2.15, 'Si3N4': 13.75, 'HfO2': 2.603, 'Ta2O5': 2.707,
                'GaN': 2.429, 'TiO2': 2.99, 'Al2O3': 8.114}  # a-Si/TiO2 无 ensemble 表值 -> 用同值占位(仅排序参考)
SUCC65 = {'a-Si': 86, 'Si3N4': 81, 'HfO2': 82, 'Ta2O5': 74, 'GaN': 66, 'TiO2': 62, 'Al2O3': 43}
SUCC101 = {'a-Si': 47, 'Si3N4': 79, 'HfO2': 80, 'Ta2O5': 69, 'GaN': 58, 'TiO2': 53, 'Al2O3': 43}

print('材料顺序:', MATS)
print('seed-mean forward:', [FWD_SEEDMEAN[m] for m in MATS])
print('nG65 成功率     :', [SUCC65[m] for m in MATS])
print('nG101 成功率    :', [SUCC101[m] for m in MATS])
print()

for fwd_name, fwd in [('seed-mean', FWD_SEEDMEAN), ('ensemble', FWD_ENSEMBLE)]:
    for s_name, s in [('nG65', SUCC65), ('nG101', SUCC101)]:
        for drop in [None, 'a-Si']:
            ms = [m for m in MATS if m != drop]
            x = [fwd[m] for m in ms]
            y = [s[m] for m in ms]
            rho, p = spearman(x, y)
            tag = 'all7' if drop is None else 'drop-aSi'
            star = '  <-- 正文句 @%s' % s_name if (fwd_name == 'seed-mean' and tag == 'all7') else ''
            print('%-9s %-6s %-8s rho=%+.2f p=%.2f (n=%d)%s' % (fwd_name, s_name, tag, rho, p, len(ms), star))
print()
print('正文句: rho=-0.43 p=0.34 @nG65; rho=+0.14 p=0.76 @nG101 (统一 seed-mean, 全七材料)')
