# -*- coding: utf-8 -*-
"""gate1_invariants.py — 门 1 三条物理不变量 + 零回归生成脚本 (结果落 pkl, 血统纪律).

v2 方案 L41 写死 "结果入 gate pkl"。三条不变量 + 零回归, 全过才落盘.

1. 圆 + 正入射: spec(p) ~= spec(s) (旋转对称; float 噪声, <=1e-8)
2. 椭圆 L!=W: spec(p) != spec(s) (必须分裂, dE > 1.0)
3. 90° 旋转: spec(L,W,p) == spec(W,L,s) (<=1e-9)
零回归: pol='p' 圆形重算 vs 存档池 (同路径, <=1e-9)

输出: data/ellipse_gate1_invariants.pkl
用法: python gate1_invariants.py
"""
import os, pickle, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rcwa_batch import rcwa_spectrum
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


def cmp(R1, T1, R2, T2):
    dR = float(np.abs(R1 - R2).max())
    dT = float(np.abs(T1 - T2).max())
    de = float(delta_e2000(rgb_to_lab(np.array(spectrum_to_srgb(WL, np.clip(R1, 0, None)))),
                           rgb_to_lab(np.array(spectrum_to_srgb(WL, np.clip(R2, 0, None))))))
    return dR, dT, de


def main():
    cl = [r for r in pickle.load(open(os.path.join(DATA, 'closed_loop_TiO2_SiO2_roundtrip_N100.pkl'), 'rb'))
          if r.get('status') == 'ok' and 'D' in r]
    asi_pool = [r for r in pickle.load(open(os.path.join(DATA, 'rcwa_aSi_PS_SiO2.pkl'), 'rb'))
                if r.get('success') and 'D' in r]
    circ = [('TiO2', cl[0]), ('TiO2', cl[6]), ('a-Si', asi_pool[1]), ('a-Si', asi_pool[9])]
    ell = [('TiO2', cl[2], cl[2]['D'] / 2), ('a-Si', asi_pool[5], asi_pool[5]['D'] / 2)]
    rec = {'invariants': {}, 'zero_regression': []}

    # 1) circular p ~= s
    for mat, r in circ:
        D, H, P = r['D'], r['H'], r['P']
        R1, T1 = spec(D, H, P, None, mat, 'p')
        R2, T2 = spec(D, H, P, None, mat, 's')
        dR, dT, de = cmp(R1, T1, R2, T2)
        assert dR <= 1e-8 and dT <= 1e-8 and de <= 1e-8, 'inv1 VIOLATION %s D=%g' % (mat, D)
        rec['invariants']['inv1_circ_p_eq_s_%s_%d' % (mat, D)] = {'dR': dR, 'dT': dT, 'dE': de}

    # 2) elliptical split
    for mat, r, W in ell:
        D, H, P = r['D'], r['H'], r['P']
        R1, T1 = spec(D, H, P, W, mat, 'p')
        R2, T2 = spec(D, H, P, W, mat, 's')
        _, _, de = cmp(R1, T1, R2, T2)
        assert de > 1.0, 'inv2 VIOLATION %s: no split (dE=%.3f)' % (mat, de)
        rec['invariants']['inv2_ell_split_%s_%d' % (mat, D)] = {'dE': de}

    # 3) 90-deg rotation
    for mat, r, _w in ell:
        L, H, P, W = r['D'], r['H'], r['P'], r['D'] / 2
        R1, T1 = spec(L, H, P, W, mat, 'p')
        R2, T2 = spec(W, H, P, L, mat, 's')
        dR, dT, de = cmp(R1, T1, R2, T2)
        assert dR <= 1e-9 and dT <= 1e-9, 'inv3 VIOLATION %s L=%g' % (mat, L)
        rec['invariants']['inv3_90deg_%s_%d' % (mat, L)] = {'dR': dR, 'dT': dT, 'dE': de}

    # zero regression vs archived pool
    for r in asi_pool[:4]:
        D, H, P = r['D'], r['H'], r['P']
        R, T = spec(D, H, P, None, 'a-Si', 'p')
        dR = float(np.abs(R - np.array(r['R'])).max())
        de = float(delta_e2000(rgb_to_lab(np.array(spectrum_to_srgb(WL, np.clip(R, 0, None)))),
                               rgb_to_lab(np.array(r['rgb']))))
        assert dR <= 1e-9 and de <= 1e-9, 'zero-regression VIOLATION D=%.0f' % D
        rec['zero_regression'].append({'D': D, 'H': H, 'P': P, 'dR': dR, 'dE': de})

    rec['script'] = os.path.basename(__file__)
    rec['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    out = os.path.join(DATA, 'ellipse_gate1_invariants.pkl')
    with open(out, 'wb') as f:
        pickle.dump(rec, f)
    print('gate1 invariants ALL PASS -> %s' % out)
    for k, v in rec['invariants'].items():
        print('  %s: %s' % (k, {kk: ('%.2e' % vv) if isinstance(vv, float) else vv for kk, vv in v.items()}))
    print('  zero_regression n=%d, max dR=%.2e' % (len(rec['zero_regression']),
                                                   max(x['dR'] for x in rec['zero_regression'])))


if __name__ == '__main__':
    main()
