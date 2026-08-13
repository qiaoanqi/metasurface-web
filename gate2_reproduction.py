# -*- coding: utf-8 -*-
"""gate2_reproduction.py — 门 2 L=W→圆柱复现门生成脚本 (血统纪律: 生成脚本永留档, 可重跑).

判据 (审计裁定 v2 收紧): 逐位 0.0 — R/T 全谱 max abs diff 与 ΔE2000 都是 0.0.
退化分支 (|W-D|<1e-12) 就是旧圆形公式本身, 路由正确则必然全零; 任何非零 = bug, 停下查因.
<0.1 仅在有书面归因时作后备容差, 不默认可放松.

对比源: TiO2 15 (closed_loop_TiO2_SiO2_roundtrip_N100.pkl, float eps 路径)
        + a-Si 15 (rcwa_aSi_PS_SiO2.pkl, complex eps 路径) — 两条代码路径都要过.
结构 ID (name 字段) 记录进 gate pkl.

输出: data/ellipse_gate2_result.pkl
用法: python gate2_reproduction.py [--n-tio2 15] [--n-asi 15]
"""
import argparse, os, pickle, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rcwa_batch import rcwa_spectrum, generate_params_elliptical
from color_utils import spectrum_to_srgb, rgb_to_lab, delta_e2000

WL = np.linspace(380, 780, 81)
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def spec(D, H, P, W=None, mat='TiO2', pol='p', nG=65):
    for ng in [nG, nG + 50, nG + 100]:
        try:
            R, T = rcwa_spectrum(D, H, P, WL, nG_req=ng, material=mat, substrate='SiO2',
                                 W_nm=W, pol=pol)
            return R, T
        except Exception:
            continue
    raise RuntimeError('all nG failed D=%s W=%s mat=%s' % (D, W, mat))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-tio2', type=int, default=15)
    ap.add_argument('--n-asi', type=int, default=15)
    ap.add_argument('--out', default=os.path.join(DATA, 'ellipse_gate2_result.pkl'))
    args = ap.parse_args()

    # 采样器约束 (门 1c/门 3, 同 gate 验证)
    params = generate_params_elliptical(500, seed=2026)
    ok_g = all(max(l, w) < p for l, w, h, p in params)
    ok_f = all(0.03 <= np.pi * (l / 2) * (w / 2) / p ** 2 <= 0.70 for l, w, h, p in params)
    ok_r = all(1.0 <= max(l, w) / min(l, w) <= 3.0 for l, w, h, p in params)
    assert ok_g and ok_f and ok_r, 'sampler constraints violated'
    print('sampler: 500 draws OK (geom/fill/ratio)')

    cl = [r for r in pickle.load(open(os.path.join(DATA, 'closed_loop_TiO2_SiO2_roundtrip_N100.pkl'), 'rb'))
          if r.get('status') == 'ok' and 'D' in r][:args.n_tio2]
    asi = [r for r in pickle.load(open(os.path.join(DATA, 'rcwa_aSi_PS_SiO2.pkl'), 'rb'))
           if r.get('success') and 'D' in r][:args.n_asi]
    cases = [('TiO2', r.get('name', 'cl_%d' % i), r) for i, r in enumerate(cl)] + \
            [('a-Si', 'pool_%d' % i, r) for i, r in enumerate(asi)]
    print('gate2: %d cases (TiO2 %d closed-loop + a-Si %d pool)' % (len(cases), len(cl), len(asi)))

    rows = []
    t0 = time.perf_counter()
    for mat, name, r in cases:
        D, H, P = r['D'], r['H'], r['P']
        R1, T1 = spec(D, H, P, None, mat)          # circular
        R2, T2 = spec(D, H, P, D, mat)             # elliptical W=D -> same code path
        dR = float(np.abs(R1 - R2).max())
        dT = float(np.abs(T1 - T2).max())
        de = float(delta_e2000(rgb_to_lab(np.array(spectrum_to_srgb(WL, np.clip(R1, 0, None)))),
                               rgb_to_lab(np.array(spectrum_to_srgb(WL, np.clip(R2, 0, None))))))
        assert dR <= 1e-12 and dT <= 1e-12 and de <= 1e-12, \
            'gate2 VIOLATION %s %s: dR=%.2e dT=%.2e dE=%.2e — routing bug, STOP' % (mat, name, dR, dT, de)
        rows.append({'name': name, 'material': mat, 'D': D, 'H': H, 'P': P,
                     'max_dR': dR, 'max_dT': dT, 'dE2000': de})
        print('  [%2d/%d] %s %s: dR=%.2e dT=%.2e dE=%.2e' % (len(rows), len(cases), mat, name, dR, dT, de))

    max_dr_all = max(r['max_dR'] for r in rows)
    max_de_all = max(r['dE2000'] for r in rows)
    with open(args.out, 'wb') as f:
        pickle.dump({'max_dR_all': max_dr_all, 'max_dE_all': max_de_all, 'rows': rows,
                     'n_cases': len(cases),
                     'script': os.path.basename(__file__),
                     'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}, f)
    print('persisted: %s (n=%d, max_dR=%.2e, max_dE=%.2e) [%.0f s]'
          % (args.out, len(rows), max_dr_all, max_de_all, time.perf_counter() - t0))


if __name__ == '__main__':
    main()
