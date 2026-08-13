# -*- coding: utf-8 -*-
"""pool_generate_elliptical.py — 椭圆 x 双偏振训练池生成 (论文 2, 2026-08-14).

预注册规格 (审计 2026-08-14 批准方案 A, 5 条全部实现):
1. 重试纪律: nG=201 重试阶梯 [201, 251, 301]; 每记录 retry_nG; retry_nG != 201 的
   记录 isolated=True (不入训练池), 单独报数.
2. 记录字段: L/W/H/P/r/pol/nG_actual/material/substrate/R(81)/T(81)/xyz/rgb/
   R_plus_T_mean/time_s/retry_nG/quality_pass/isolated + 池元数据 (seed/采样器版本/日期/机器).
3. 断点续跑: 增量落盘 (每 --flush-every 条原子写), --resume 幂等跳过已完成 (L,W,H,P,pol).
4. 质量过滤预注册 (TiO2 无损): quality_pass = |R_plus_T_mean - 1.0| <= 0.05
   (无损期望 mean=1.0, 论文 1 无损规则; 口径审计复核), 通过率如实记录, 训练侧用 pass 子集.
5. 耗时口径: --n-jobs 16 (8P+8E 核, E 核拖后腿 -> 预期 15-19 h); 插电/高性能模式/关睡眠.

用法: python pool_generate_elliptical.py --samples 3000 --material TiO2 --pol both
      --nG 201 --n-jobs 16 --out data/rcwa_ellip_TiO2_3000.pkl [--resume]
"""
import argparse, os, pickle, platform, sys, time
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rcwa_batch import rcwa_spectrum, generate_params_elliptical
from color_utils import spectrum_to_xyz, xyz_to_srgb

WL = np.linspace(380, 780, 81)
SAMPLER_VERSION = 'generate_params_elliptical v1 (uniform rejection, r_max=3.0, seed-drawn)'
QUALITY_TOL = 0.05  # lossless: |R_plus_T_mean - 1.0| <= 0.05 (mean=1.0 lossless expectation)


def _process_ellip(task):
    """Pool worker (module-level for pickling). One (L,W,H,P,pol) -> one record or fail dict."""
    L, W, H, P, wls, nG, material, substrate, pol = task
    t0 = time.perf_counter()
    r = max(L, W) / min(L, W)
    for ng in [nG, nG + 50, nG + 100]:  # retry ladder 201 -> 251 -> 301
        try:
            R, T = rcwa_spectrum(L, H, P, wls, nG_req=ng, material=material,
                                 substrate=substrate, W_nm=W, pol=pol)
            xyz = spectrum_to_xyz(wls, R)
            rgb = xyz_to_srgb(xyz)
            rpt = float(np.mean(R + T))
            return {
                'L': L, 'W': W, 'H': H, 'P': P, 'r': r,
                'pol': pol, 'material': material, 'substrate': substrate,
                'nG_actual': ng, 'retry_nG': ng,
                'isolated': ng != nG,  # retried at higher order -> not for training pool
                'wl_nm': wls.copy(), 'R': R, 'T': T,
                'xyz': xyz, 'rgb': rgb, 'R_plus_T_mean': rpt,
                'quality_pass': abs(rpt - 1.0) <= QUALITY_TOL,
                'time_s': time.perf_counter() - t0,
                'success': True,
            }
        except Exception:
            continue
    return {'L': L, 'W': W, 'H': H, 'P': P, 'pol': pol, 'success': False}


def main():
    ap = argparse.ArgumentParser(description='elliptical x dual-pol training pool generator')
    ap.add_argument('--samples', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--nG', type=int, default=201)
    ap.add_argument('--material', default='TiO2')
    ap.add_argument('--substrate', default='SiO2')
    ap.add_argument('--pol', default='both', choices=['both', 'p', 's'])
    ap.add_argument('--n-jobs', type=int, default=16)
    ap.add_argument('--out', default='data/rcwa_ellip_TiO2_3000.pkl')
    ap.add_argument('--flush-every', type=int, default=50)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()

    pols = ['p', 's'] if args.pol == 'both' else [args.pol]
    params = generate_params_elliptical(args.samples, seed=args.seed)
    tasks = [(L, W, H, P, WL, args.nG, args.material, args.substrate, pol)
             for (L, W, H, P) in params for pol in pols]
    print('池: %d 结构 x %d 偏振 = %d 任务 | %s/%s | nG=%d (阶梯 %d,%d,%d) | jobs=%d | out=%s'
          % (len(params), len(pols), len(tasks), args.material, args.substrate, args.nG,
             args.nG, args.nG + 50, args.nG + 100, args.n_jobs, args.out))

    # resume: load existing, build done-set
    records = []
    done = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out, 'rb') as f:
            old = pickle.load(f)
        records = old.get('records', [])
        done = {(r['L'], r['W'], r['H'], r['P'], r['pol']) for r in records if r.get('success')}
        print('resume: %d existing records, %d done keys' % (len(records), len(done)))
    tasks = [t for t in tasks if (t[0], t[1], t[2], t[3], t[8]) not in done]
    print('remaining tasks: %d' % len(tasks))

    meta = {'seed': args.seed, 'sampler_version': SAMPLER_VERSION,
            'nG': args.nG, 'material': args.material, 'substrate': args.substrate,
            'pols': pols, 'n_samples': args.samples, 'n_jobs': args.n_jobs,
            'date': time.strftime('%Y-%m-%d %H:%M:%S'), 'machine': platform.node(),
            'quality_rule': 'lossless |R_plus_T_mean - 1.0| <= 0.05'}

    t_all = time.perf_counter()
    n_ok = n_fail = n_isolated = n_pass = 0
    try:
        with Pool(args.n_jobs) as pool:
            for i, rec in enumerate(pool.imap_unordered(_process_ellip, tasks, chunksize=4), 1):
                records.append(rec)
                if rec.get('success'):
                    n_ok += 1
                    if rec['isolated']:
                        n_isolated += 1
                    if rec['quality_pass']:
                        n_pass += 1
                else:
                    n_fail += 1
                if i % args.flush_every == 0:
                    tmp = args.out + '.tmp'
                    with open(tmp, 'wb') as f:
                        pickle.dump({'meta': meta, 'records': records}, f)
                    os.replace(tmp, args.out)
                    print('  [%d/%d] ok=%d fail=%d isolated=%d pass=%d | %.0f s'
                          % (i, len(tasks), n_ok, n_fail, n_isolated, n_pass,
                             time.perf_counter() - t_all))
    finally:
        tmp = args.out + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump({'meta': meta, 'records': records}, f)
        os.replace(tmp, args.out)

    ok_recs = [r for r in records if r.get('success')]
    print('\n完成: %d 任务, %d 成功, %d 失败 | isolated(retry!=%d)=%d | quality_pass=%d/%.0f%%'
          % (len(tasks), n_ok, n_fail, args.nG, n_isolated, n_pass,
             100 * n_pass / max(n_ok, 1)))
    print('总耗时 %.0f s (%.1f h) | 归档 %s' % (time.perf_counter() - t_all,
          (time.perf_counter() - t_all) / 3600, args.out))


if __name__ == '__main__':
    main()
