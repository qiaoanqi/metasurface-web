# _ng_probe_elliptical.py — 椭圆 × 双偏振 nG 收敛探针 (门 4, 2026-08-13)
# 自洽口径: 以最高阶 nG=201 为参考, 每偏振分别测 ΔE(color@nG, color@201).
# 预注册联合判据: max(ΔE_p, ΔE_s) < 2.3 (两偏振同时达标才算成功).
# 实测各 nG 档耗时 -> 池生成预算 (上报用户, 不用外推).
import os, pickle, sys, time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rcwa_batch import rcwa_spectrum, generate_params_elliptical
from color_utils import spectrum_to_srgb, rgb_to_lab, delta_e2000

WL = np.linspace(380, 780, 81)
NGS = [65, 101, 131, 201]
REF_NG = 201
SUB = 'SiO2'
N_STRUCT = 16

def spec(L, W, H, P, mat, pol, nG):
    for ng in [nG, nG + 50, nG + 100]:
        try:
            t0 = time.perf_counter()
            R, T = rcwa_spectrum(L, H, P, WL, nG_req=ng, material=mat, substrate=SUB,
                                 W_nm=W, pol=pol)
            return np.array(spectrum_to_srgb(WL, np.clip(R, 0, None))), time.perf_counter() - t0
        except Exception:
            continue
    return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--material', default='a-Si')
    ap.add_argument('--n-struct', type=int, default=16)
    ap.add_argument('--out', default='data/ng_probe_elliptical_aSi.pkl')
    args = ap.parse_args()
    MAT = args.material
    params = generate_params_elliptical(args.n_struct, seed=2026)
    print('探针 %d 个 %s 椭圆结构 x nG %s x pol p/s (参考 %d)' % (len(params), MAT, NGS, REF_NG))
    out = []
    timings = {ng: [] for ng in NGS}
    t_all = time.perf_counter()
    for i, (L, W, H, P) in enumerate(params):
        colors = {'p': {}, 's': {}}
        times = {'p': {}, 's': {}}
        for pol in ['p', 's']:
            for ng in NGS:
                rgb, dt = spec(L, W, H, P, MAT, pol, ng)
                colors[pol][ng] = rgb
                times[pol][ng] = dt
                if rgb is not None:
                    timings[ng].append(dt)
        rec = {'L': L, 'W': W, 'H': H, 'P': P, 'r': max(L, W) / min(L, W),
               'colors': {p: {ng: (None if c is None else list(c)) for ng, c in cc.items()} for p, cc in colors.items()},
               'times_s': {p: {ng: dt for ng, dt in tt.items()} for p, tt in times.items()}}
        # 自洽: 每偏振以 REF 为参考
        for pol in ['p', 's']:
            ref = colors[pol][REF_NG]
            rec['de_vs_ref_' + pol] = {}
            if ref is not None:
                lab_ref = rgb_to_lab(ref)
                for ng in NGS:
                    c = colors[pol][ng]
                    rec['de_vs_ref_' + pol][ng] = None if c is None else float(
                        delta_e2000(rgb_to_lab(c), lab_ref))
        rec['de_vs_ref_max'] = {}
        for ng in NGS:
            dp = rec['de_vs_ref_p'].get(ng)
            ds = rec['de_vs_ref_s'].get(ng)
            rec['de_vs_ref_max'][ng] = None if (dp is None or ds is None) else max(dp, ds)
        out.append(rec)
        print('  [%2d/%d] L=%.0f W=%.0f r=%.2f  de_max vs %d: %s' % (
            i + 1, len(params), L, W, rec['r'], REF_NG,
            {ng: ('%.2f' % v if v is not None else 'NA') for ng, v in rec['de_vs_ref_max'].items()}))

    pickle.dump(out, open(args.out, 'wb'))

    print('\n=== 汇总 (以 nG=%d 为参考, 联合判据 max(p,s)) ===' % REF_NG)
    for ng in NGS:
        if ng == REF_NG:
            continue
        vals = np.array([x['de_vs_ref_max'][ng] for x in out if x['de_vs_ref_max'].get(ng) is not None])
        if len(vals):
            print('  nG=%3d: mean=%.2f median=%.2f <1JND=%.0f%% <2.3JND=%.0f%% (n=%d)' % (
                ng, vals.mean(), np.median(vals), 100 * (vals < 1.0).mean(),
                100 * (vals < 2.3).mean(), len(vals)))
    print('\n=== 实测耗时 (s/结构/偏振) ===')
    for ng in NGS:
        ts = np.array(timings[ng])
        if len(ts):
            print('  nG=%3d: mean=%.1f median=%.1f (n=%d)' % (ng, ts.mean(), np.median(ts), len(ts)))
    print('\n池生成预算外推 (3000 结构 x 选定 nG x 2 偏振): 用上表实测 x 3000 x 2')
    print('归档: %s  总耗时 %.0f s' % (args.out, time.perf_counter() - t_all))

if __name__ == '__main__':
    main()
