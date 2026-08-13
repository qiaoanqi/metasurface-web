# closed_loop_validate.py — ML 逆设计的闭环验证实验
#
# 协议 (paper-grade):
#   目标色 -> smart_grid_search (ML 逆设计) -> 最优 D/H/P
#          -> 独立 RCWA (grcwa, nG=65, Nxy=256, 与训练数据同参数) -> achieved 光谱
#          -> achieved RGB -> achieved ΔE2000 (目标 vs 独立验证)
#
# 报告:
#   - predicted ΔE  : ML 正向模型的"自称"误差 (目标 vs ML 预测色)
#   - achieved ΔE   : 独立 RCWA 重算的真实误差 (目标 vs RCWA 验证色)
#   - gap           : achieved - predicted (正向模型在逆设计相关区域的系统偏差)
#   - success rate  : achieved ΔE < 2.3 (JND 肉眼不可分辨阈值) 的比例
#   - speed         : ML 逆设计耗时 vs 等效 RCWA 暴力网格搜索耗时 (加速比)
#
# 用法:
#   python closed_loop_validate.py --material TiO2 --substrate SiO2 --limit 3   # 冒烟测试
#   python closed_loop_validate.py --material TiO2 --substrate SiO2             # 全量
#   python closed_loop_validate.py --material TiO2 --substrate SiO2,Si3N4,Al2O3 # 三衬底

# === BLAS 单线程 (必须在 import numpy 之前, 与 rcwa_batch.py 一致) ===
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import sys, time, argparse, colorsys, pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ml_module
from color_utils import spectrum_to_srgb, rgb_to_lab, delta_e2000
from rcwa_batch import rcwa_spectrum
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError

# ML 内部材料名 -> rcwa_batch 材料名
ML_TO_RCWA_MAT = {
    "TiO2 (anatase)": "TiO2",
    "a-Si (amorphous)": "a-Si",
    "Si3N4 (nitride)": "Si3N4",
    "Al2O3 (sapphire)": "Al2O3",
    "GaN (wurtzite)": "GaN",
    "HfO2 (hafnia)": "HfO2",
    "Ta2O5 (tantala)": "Ta2O5",
}
ML_TO_RCWA_SUB = {
    "SiO2 (fused silica)": "SiO2",
    "Si3N4 (nitride)": "Si3N4",
    "Al2O3 (sapphire)": "Al2O3",
}
# 短名 -> ML 内部名
SHORT_MAT = {"TiO2": "TiO2 (anatase)", "a-Si": "a-Si (amorphous)",
             "Si3N4": "Si3N4 (nitride)", "Al2O3": "Al2O3 (sapphire)",
             "GaN": "GaN (wurtzite)", "HfO2": "HfO2 (hafnia)",
             "Ta2O5": "Ta2O5 (tantala)"}
SHORT_SUB = {"SiO2": "SiO2 (fused silica)", "Si3N4": "Si3N4 (nitride)",
             "Al2O3": "Al2O3 (sapphire)"}

WL = np.linspace(380, 780, 81)
JND = 2.3  # just-noticeable-difference threshold


def build_target_colors(n=100):
    """生成覆盖色域内部 + 边界 + 中性轴的目标色集合.

    - 高饱和色相 (色域边界探测, S=0.95, V=0.95)
    - 中饱和色相 (色域内部, S=0.50, V=0.85)
    - 低饱和色相 (近中性轴, S=0.20, V=0.70)
    - 中性灰阶 (消色差能力)
    n: 目标总数, 默认 100. 在各层间均匀分配.
    返回: list of (name, rgb[0..1])
    """
    targets = []
    n_high = max(4, int(n * 0.35))
    n_mid = max(4, int(n * 0.30))
    n_low = max(2, int(n * 0.15))
    n_neutral = n - n_high - n_mid - n_low

    # 高饱和色相环 (S=0.95, V=0.95)
    for i in range(n_high):
        h = i / n_high
        r, g, b = colorsys.hsv_to_rgb(h, 0.95, 0.95)
        targets.append((f"sat_hue{int(h*360):03d}", np.array([r, g, b])))

    # 中饱和色相环 (S=0.50, V=0.85)
    for i in range(n_mid):
        h = i / n_mid
        r, g, b = colorsys.hsv_to_rgb(h, 0.50, 0.85)
        targets.append((f"mid_hue{int(h*360):03d}", np.array([r, g, b])))

    # 低饱和色相环 (S=0.20, V=0.70)
    for i in range(n_low):
        h = i / n_low
        r, g, b = colorsys.hsv_to_rgb(h, 0.20, 0.70)
        targets.append((f"low_hue{int(h*360):03d}", np.array([r, g, b])))

    # 中性灰阶
    for i in range(n_neutral):
        v = 0.95 - i * 0.90 / max(1, n_neutral - 1)
        targets.append((f"neutral_{i:02d}", np.array([v, v, v])))

    return targets


def build_roundtrip_targets(material_short, substrate_short, n=30, seed=2024, nG=65):
    """往返(round-trip)目标: 随机生成有效结构 -> RCWA 算真实颜色 -> 作为可达目标.

    目标色按构造一定可达 (它本身就是某个真实结构的颜色), 因此纯粹测试
    逆设计能否找回产生同色的结构, 而不被色域外目标污染成功率.
    参数分布与训练数据一致 (D 80-350, H 100-600, P 200-600, D<P, 填充率 0.03-0.70).
    返回: list of (name, rgb[0..1])
    """
    rng = np.random.RandomState(seed)
    targets = []
    attempts = 0
    while len(targets) < n and attempts < n * 20:
        attempts += 1
        D = rng.uniform(80, 350)
        H = rng.uniform(100, 600)
        P = rng.uniform(200, 600)
        if D >= P:
            continue
        fr = np.pi * (D / 2) ** 2 / (P ** 2)
        if fr < 0.03 or fr > 0.70:
            continue
        ver = rcwa_verify(D, H, P, material_short, substrate_short, nG=nG)
        if ver is None:
            continue
        rgb, R_spec, T_spec = ver
        # 只保留能量守恒良好的样本 (与训练质量过滤一致);
        # lossy (a-Si): 物理判据 0 < R+T <= 1.05 — 强吸收下 R+T 远小于 1 是正常物理
        # (旧判据 |R+T-1|<0.05 会把全部 a-Si 目标滤掉, A1 审计 2026-08-07)
        rt = float(np.mean(R_spec + T_spec))
        if material_short in ('a-Si', 'a-Si (amorphous)'):
            if rt <= 0.0 or rt > 1.05:
                continue
        else:
            if abs(rt - 1.0) > 0.05:
                continue
        targets.append((f"rt_{len(targets):02d}_D{D:.0f}H{H:.0f}P{P:.0f}", np.array(rgb)))
        print(f"  [roundtrip target {len(targets):2d}/{n}] D={D:.0f} H={H:.0f} P={P:.0f} "
              f"-> RGB=({rgb[0]:.2f},{rgb[1]:.2f},{rgb[2]:.2f})")
    return targets


def _rcwa_compute(D, H, P, material_short, substrate_short, nG, Nxy):
    """Internal: single RCWA computation without timeout protection."""
    R, T = rcwa_spectrum(D, H, P, WL, nG_req=nG, Nxy=Nxy,
                         material=material_short, substrate=substrate_short, angle_deg=0.0)
    rgb = spectrum_to_srgb(WL, np.clip(R, 0, None))
    return np.array(rgb), R, T


def rcwa_verify(D, H, P, material_short, substrate_short, nG=65, Nxy=256, timeout_s=120):
    """独立 RCWA 验证: 对给定结构从头算反射光谱 -> RGB.

    包含 per-sample 超时保护 (默认 120 s), 超时返回 None.
    返回 (rgb, R_spec, T_spec) 或 None.
    """
    try:
        with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context('spawn')) as executor:
            future = executor.submit(_rcwa_compute, D, H, P, material_short,
                                     substrate_short, nG, Nxy)
            return future.result(timeout=timeout_s)
    except FuturesTimeoutError:
        print(f"    [RCWA timeout {timeout_s}s] D={D:.0f} H={H:.0f} P={P:.0f}")
        return None
    except Exception as e:
        print(f"    [RCWA fail] D={D:.0f} H={H:.0f} P={P:.0f}: {e}")
        return None


def run_one_target(name, target_rgb, ml_mat, ml_sub, rcwa_mat, rcwa_sub, nG):
    """对单个目标色跑完整闭环. 返回结果 dict."""
    rec = {'name': name, 'target_rgb': list(target_rgb),
           'material': rcwa_mat, 'substrate': rcwa_sub}

    # --- ML 逆设计 ---
    t0 = time.perf_counter()
    res = ml_module.smart_grid_search(list(target_rgb), material=ml_mat, substrate=ml_sub)
    t_ml = time.perf_counter() - t0
    rec['ml_time_s'] = t_ml

    if not res:
        rec['status'] = 'ml_no_result'
        return rec

    _, bp, pred_rgb, de76, pred_de = res[0]
    D, H, P = bp.diameter_nm, bp.height_nm, bp.period_nm
    rec.update({'D': D, 'H': H, 'P': P,
                'pred_rgb': list(pred_rgb), 'pred_de2000': float(pred_de)})

    # --- 独立 RCWA 验证 ---
    t0 = time.perf_counter()
    ver = rcwa_verify(D, H, P, rcwa_mat, rcwa_sub, nG=nG)
    t_rcwa = time.perf_counter() - t0
    rec['rcwa_time_s'] = t_rcwa

    if ver is None:
        rec['status'] = 'rcwa_fail'
        return rec

    ach_rgb, R_spec, T_spec = ver
    ach_de = delta_e2000(rgb_to_lab(ach_rgb), rgb_to_lab(np.array(target_rgb)))
    rec.update({
        'ach_rgb': list(ach_rgb),
        'ach_de2000': float(ach_de),
        'gap': float(ach_de - pred_de),       # 正向模型偏差
        'R_plus_T': float(np.mean(R_spec + T_spec)),
        'status': 'ok',
    })
    return rec


def run_one_target_hybrid(name, target_rgb, ml_mat, ml_sub, rcwa_mat, rcwa_sub, nG, rerank_k=20):
    """混合逆设计: ML 取 top-K 候选 -> 逐个 RCWA 验证 -> 选实测最优.

     against 优化器诅咒: 最终选择基于真实 RCWA, 而非 ML 代理的乐观预测.
    同时记录 naive (ML 首选) 结果用于量化重排收益.
    """
    target_lab = rgb_to_lab(np.array(target_rgb))
    rec = {'name': name, 'target_rgb': list(target_rgb),
           'material': rcwa_mat, 'substrate': rcwa_sub}

    # --- ML 取 top-K 候选 ---
    t0 = time.perf_counter()
    res = ml_module.smart_grid_search(list(target_rgb), material=ml_mat, substrate=ml_sub,
                                      max_results=rerank_k)
    t_ml = time.perf_counter() - t0
    rec['ml_time_s'] = t_ml
    if not res:
        rec['status'] = 'ml_no_result'
        return rec

    # --- 逐个 RCWA 验证, 选实测最优 ---
    t_rcwa_total = 0.0
    best = None  # (ach_de, D, H, P, ach_rgb, pred_de, R_plus_T)
    naive = None  # candidate[0] 的实测
    n_verified = 0
    all_verified = []  # stores (ach_de, pred_de) for each verified candidate in ML rank order
    for idx, (_, bp, pred_rgb, de76, pred_de) in enumerate(res):
        D, H, P = bp.diameter_nm, bp.height_nm, bp.period_nm
        t0 = time.perf_counter()
        ver = rcwa_verify(D, H, P, rcwa_mat, rcwa_sub, nG=nG)
        t_rcwa_total += time.perf_counter() - t0
        if ver is None:
            continue
        ach_rgb, R_spec, T_spec = ver
        ach_de = delta_e2000(rgb_to_lab(ach_rgb), target_lab)
        n_verified += 1
        all_verified.append((float(ach_de), float(pred_de)))
        rt = float(np.mean(R_spec + T_spec))
        if idx == 0:
            naive = (float(ach_de), D, H, P, list(ach_rgb), float(pred_de))
        if best is None or ach_de < best[0]:
            best = (float(ach_de), D, H, P, list(ach_rgb), float(pred_de), rt)

    rec['rcwa_time_s'] = t_rcwa_total
    rec['n_verified'] = n_verified
    rec['all_verified'] = all_verified
    if best is None:
        rec['status'] = 'rcwa_fail'
        return rec

    ach_de, D, H, P, ach_rgb, pred_de, rt = best
    rec.update({
        'D': D, 'H': H, 'P': P,
        'pred_rgb': list(res[0][2]),
        'pred_de2000': pred_de,            # hybrid-best 候选的 ML 自称
        'ach_rgb': ach_rgb,
        'ach_de2000': ach_de,              # hybrid 实测 (headline)
        'gap': float(ach_de - pred_de),
        'R_plus_T': rt,
        'naive_ach_de2000': naive[0] if naive else None,   # ML 首选的实测
        'naive_pred_de2000': naive[5] if naive else None,
        'status': 'ok',
    })
    return rec


def summarize(records, rcwa_mat):
    """打印汇总表."""
    ok = [r for r in records if r.get('status') == 'ok']
    if not ok:
        print("  无有效结果"); return
    ach = np.array([r['ach_de2000'] for r in ok])
    pred = np.array([r['pred_de2000'] for r in ok])
    gap = np.array([r['gap'] for r in ok])
    ml_t = np.array([r['ml_time_s'] for r in ok])
    rcwa_t = np.array([r['rcwa_time_s'] for r in ok])

    print(f"\n{'='*64}")
    print(f"  {rcwa_mat} 闭环验证结果  (n={len(ok)}/{len(records)} 有效)")
    print(f"{'='*64}")
    print(f"  achieved ΔE2000 : mean={ach.mean():.2f}  median={np.median(ach):.2f}  "
          f"P95={np.percentile(ach,95):.2f}")
    print(f"  predicted ΔE    : mean={pred.mean():.2f}  median={np.median(pred):.2f}")
    print(f"  gap (ach-pred)  : mean={gap.mean():+.2f}  (正向模型系统偏差)")
    print(f"  成功率 ΔE<2.3   : {100*(ach<JND).mean():.0f}%  ({(ach<JND).sum()}/{len(ok)})")
    print(f"  ML 逆设计耗时   : {ml_t.mean():.1f}s/目标")
    print(f"  RCWA 验证耗时   : {rcwa_t.mean():.1f}s/结构")
    # 速度基准: 等效 RCWA 暴力网格 (coarse 12^3, 过滤后约 1000 有效点)
    n_grid_eff = 1000
    brute_est = n_grid_eff * rcwa_t.mean()
    print(f"  等效暴力网格    : ~{n_grid_eff} 点 × {rcwa_t.mean():.1f}s ≈ {brute_est:.0f}s "
          f"({brute_est/60:.0f}min)  →  加速比 ~{brute_est/ml_t.mean():.0f}×")
    print(f"{'='*64}\n")


def summarize_hybrid(records, rcwa_mat):
    """混合逆设计汇总: 对比 naive (ML 首选) 与 hybrid (RCWA 重排) 的实测 ΔE.

    核心论点: 优化器诅咒使 ML 首选在实测下偏乐观; 用 RCWA 对 top-K 重排后,
    最终选择基于真实物理, 实测 ΔE 应显著下降, gap 应收窄.
    """
    ok = [r for r in records if r.get('status') == 'ok' and r.get('naive_ach_de2000') is not None]
    if not ok:
        print("  无有效 hybrid 结果"); return
    naive = np.array([r['naive_ach_de2000'] for r in ok])
    hybrid = np.array([r['ach_de2000'] for r in ok])
    naive_pred = np.array([r['naive_pred_de2000'] for r in ok])
    n_ver = np.array([r['n_verified'] for r in ok])
    ml_t = np.array([r['ml_time_s'] for r in ok])
    rcwa_t = np.array([r['rcwa_time_s'] for r in ok])

    print(f"\n{'='*68}")
    print(f"  {rcwa_mat} 混合逆设计 (ML top-K 筛选 + RCWA 重排)  n={len(ok)}/{len(records)}")
    print(f"{'='*68}")
    print(f"  naive  (ML 首选)   实测 ΔE : mean={naive.mean():.2f}  median={np.median(naive):.2f}  "
          f"成功率={100*(naive<JND).mean():.0f}%")
    print(f"  hybrid (RCWA 重排) 实测 ΔE : mean={hybrid.mean():.2f}  median={np.median(hybrid):.2f}  "
          f"成功率={100*(hybrid<JND).mean():.0f}%")
    print(f"  ΔE 改善 (naive-hybrid)     : mean={ (naive-hybrid).mean():+.2f}  "
          f"max={ (naive-hybrid).max():+.2f}")
    print(f"  ML 自称 ΔE (naive)         : mean={naive_pred.mean():.2f}  "
          f"(对比 naive 实测 {naive.mean():.2f} → 优化器诅咒 {naive.mean()-naive_pred.mean():+.2f})")
    print(f"  平均 RCWA 验证次数/目标     : {n_ver.mean():.1f}")
    print(f"  耗时: ML 筛选 {ml_t.mean():.1f}s + RCWA 重排 {rcwa_t.mean():.1f}s "
          f"= {(ml_t+rcwa_t).mean():.1f}s/目标")
    print(f"{'='*68}\n")


def convergence_verify(input_pkl, nG_list, Nxy=256, out_path=None):
    """对闭环结果中选出的结构做多 nG 收敛性复验.

    目的: 证明 achieved 颜色随傅里叶阶数 nG 收敛, 而非 nG=65 的求解器噪声.
    判据: 若每个结构的 achieved ΔE 跨 nG 极差 < 1 JND (2.3), 且 R+T 随 nG
          趋近 1.0, 则 achieved 颜色可信, 可堵审稿人"求解器噪声"的质疑.
    out_path: 若给出, 把 per-structure 数据 (D,H,P,mat,sub,de_by_nG,rt_by_nG,spread)
              存成 pkl 归档 (fig5 重生成 / Table 3 数字溯源用).
    """
    with open(input_pkl, 'rb') as f:
        recs = pickle.load(f)
    ok = [r for r in recs if r.get('status') == 'ok' and 'D' in r]
    if not ok:
        print("无有效结构可复验"); return

    print(f"\n{'='*72}")
    print(f"  nG 收敛性复验  (输入: {input_pkl}, n={len(ok)} 结构, nG={nG_list})")
    print(f"{'='*72}")

    de_by_nG = {g: [] for g in nG_list}
    rt_by_nG = {g: [] for g in nG_list}
    spread = []  # 每结构 ΔE 跨 nG 的极差
    records = []  # per-structure archive (out_path)
    for r in ok:
        target_lab = rgb_to_lab(np.array(r['target_rgb']))
        D, H, P = r['D'], r['H'], r['P']
        mat, sub = r['material'], r['substrate']
        de_this = []
        rec = {'D': D, 'H': H, 'P': P, 'material': mat, 'substrate': sub,
               'de': {}, 'rt': {}}
        for g in nG_list:
            ver = rcwa_verify(D, H, P, mat, sub, nG=g, Nxy=Nxy)
            if ver is None:
                continue
            rgb, R_spec, T_spec = ver
            de = delta_e2000(rgb_to_lab(rgb), target_lab)
            de_by_nG[g].append(de)
            rt_by_nG[g].append(float(np.mean(R_spec + T_spec)))
            de_this.append(de)
            rec['de'][g] = float(de)
            rec['rt'][g] = float(np.mean(R_spec + T_spec))
        if len(de_this) >= 2:
            spread.append(max(de_this) - min(de_this))
            rec['spread'] = float(max(de_this) - min(de_this))
        records.append(rec)

    print(f"  {'nG':>5} | {'mean ΔE':>8} | {'median':>7} | {'mean R+T':>9} | {'n':>3}")
    print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*7}-+-{'-'*9}-+-{'-'*3}")
    for g in nG_list:
        if de_by_nG[g]:
            a = np.array(de_by_nG[g]); rt = np.array(rt_by_nG[g])
            print(f"  {g:>5} | {a.mean():>8.2f} | {np.median(a):>7.2f} | "
                  f"{rt.mean():>9.4f} | {len(a):>3}")

    if spread:
        sp = np.array(spread)
        print(f"\n  每结构 ΔE 跨 nG 极差: mean={sp.mean():.2f}  max={sp.max():.2f}  "
              f"<1 JND(2.3) 比例={100*(sp<2.3).mean():.0f}%")
        if sp.max() < 2.3:
            print(f"  [OK] 所有结构 achieved ΔE 跨 nG 变化 < 1 JND → 颜色收敛, 非求解器噪声")
        else:
            print(f"  [!] 有结构跨 nG 变化 >= 1 JND, 需在论文中如实报告并讨论")
    print(f"{'='*72}\n")

    if out_path:
        with open(out_path, 'wb') as f:
            pickle.dump(records, f)
        print(f"  [归档] {len(records)} 条 per-structure 记录 -> {out_path}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--material', default='TiO2')
    ap.add_argument('--substrate', default='SiO2', help='逗号分隔多衬底')
    ap.add_argument('--nG', type=int, default=65)
    ap.add_argument('--limit', type=int, default=0, help='只跑前 N 个目标 (0=全部)')
    ap.add_argument('--mode', default='fixed', choices=['fixed', 'roundtrip'],
                    help='fixed=预设目标色(含色域外); roundtrip=RCWA可达色目标(纯逆设计能力)')
    ap.add_argument('--n-roundtrip', type=int, default=30, help='roundtrip 模式的目标数量')
    ap.add_argument('--rerank', type=int, default=0,
                    help='0=naive(ML首选); >0=混合重排(取该数量ML候选, RCWA逐个验证选实测最优)')
    ap.add_argument('--resume', action='store_true',
                    help='若输出文件已存在, 跳过已完成的目标(按名字匹配), 增量续跑')
    ap.add_argument('--verify-nG', default='',
                    help='收敛复验模式: 逗号分隔的 nG 列表(如 "65,101,131"). '
                         '指定时不跑闭环, 而是读 --input 结果对每个结构多 nG 复验')
    ap.add_argument('--input', default='', help='收敛复验模式的输入结果 pkl')
    ap.add_argument('--output', default='')
    args = ap.parse_args()

    # --- 收敛复验模式: 不跑闭环, 读已有结果做多 nG 复验 (只需 RCWA, 不加载 ML) ---
    if args.verify_nG:
        if not args.input:
            ap.error("--verify-nG 需要 --input 指定要复验的结果 pkl")
        nG_list = [int(x) for x in args.verify_nG.split(',') if x.strip()]
        convergence_verify(args.input, nG_list, Nxy=256, out_path=args.output or None)
        return

    ml_module.init_ml()
    ml_module.init_rcwa_ml()

    ml_mat = SHORT_MAT[args.material]
    rcwa_mat = ML_TO_RCWA_MAT[ml_mat]
    substrates = [s.strip() for s in args.substrate.split(',')]

    # fixed 模式目标与衬底无关, 循环外构建一次
    fixed_targets = None
    if args.mode == 'fixed':
        fixed_targets = build_target_colors(n=100)
        if args.limit > 0:
            fixed_targets = fixed_targets[:args.limit]
    out = args.output or f"data/closed_loop_{rcwa_mat}_{'-'.join(substrates)}.pkl"
    print(f"模式: {args.mode} | 材料: {args.material} | 衬底: {substrates} | nG={args.nG} | "
          f"rerank={'naive' if args.rerank <= 0 else f'hybrid top-{args.rerank}'} | out={out}")

    # --- resume: 载入已完成记录, 按 (name, substrate) 跳过 ---
    all_records = []
    done_keys = set()
    if args.resume and os.path.exists(out):
        try:
            with open(out, 'rb') as f:
                all_records = pickle.load(f)
            done_keys = {(r.get('name'), r.get('substrate')) for r in all_records}
            print(f"[resume] 已载入 {len(all_records)} 条完成记录, 跳过同名目标")
        except Exception as e:
            print(f"[resume] 载入失败 ({e}), 从头开始")
            all_records = []

    for sub_short in substrates:
        ml_sub = SHORT_SUB[sub_short]
        rcwa_sub = ML_TO_RCWA_SUB[ml_sub]
        print(f"\n>>> 衬底 {sub_short}")
        # roundtrip 模式: 每个衬底单独生成可达目标
        if args.mode == 'roundtrip':
            print(f"  生成 {args.n_roundtrip} 个 roundtrip 可达目标 (RCWA 真实颜色)...")
            targets = build_roundtrip_targets(rcwa_mat, rcwa_sub, n=args.n_roundtrip, nG=args.nG)
        else:
            targets = fixed_targets
        sub_records = []
        for i, (name, rgb) in enumerate(targets):
            if (name, rcwa_sub) in done_keys:
                print(f"  [{i+1:2d}/{len(targets)}] {name:14s} -> [skip: 已完成]")
                continue
            if args.rerank > 0:
                rec = run_one_target_hybrid(name, rgb, ml_mat, ml_sub, rcwa_mat, rcwa_sub,
                                            args.nG, rerank_k=args.rerank)
            else:
                rec = run_one_target(name, rgb, ml_mat, ml_sub, rcwa_mat, rcwa_sub, args.nG)
            sub_records.append(rec)
            all_records.append(rec)
            done_keys.add((name, rcwa_sub))
            # 增量保存: 每完成一个目标即落盘, 中断不丢进度
            with open(out, 'wb') as f:
                pickle.dump(all_records, f)
            if rec.get('status') == 'ok':
                if args.rerank > 0 and rec.get('naive_ach_de2000') is not None:
                    print(f"  [{i+1:2d}/{len(targets)}] {name:14s} D={rec['D']:.0f} H={rec['H']:.0f} "
                          f"P={rec['P']:.0f} | naive={rec['naive_ach_de2000']:.1f} "
                          f"hybrid={rec['ach_de2000']:.1f} (n_ver={rec['n_verified']})")
                else:
                    print(f"  [{i+1:2d}/{len(targets)}] {name:14s} D={rec['D']:.0f} H={rec['H']:.0f} "
                          f"P={rec['P']:.0f} | pred={rec['pred_de2000']:.1f} ach={rec['ach_de2000']:.1f} "
                          f"gap={rec['gap']:+.1f}")
            else:
                print(f"  [{i+1:2d}/{len(targets)}] {name:14s} -> {rec.get('status')}")
        if args.rerank > 0:
            summarize_hybrid(sub_records, f"{rcwa_mat}/{rcwa_sub}")
        else:
            summarize(sub_records, f"{rcwa_mat}/{rcwa_sub}")

    print(f"结果已保存: {out}  (共 {len(all_records)} 条)")


if __name__ == '__main__':
    main()
