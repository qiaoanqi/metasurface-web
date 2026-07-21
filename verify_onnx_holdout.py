# verify_onnx_holdout.py — 部署一致性校验 (Si3N4 教训固化为流程)
#
# 用途: 每个材料训练 + 导出 ONNX 之后, 跑 closed-loop 之前必须先过这关.
#       用 ml_module 的【真实部署路由】(含衬底专用模型 override) 在冻结 holdout 上算 ΔE.
#       - 复现训练报告的 holdout ΔE  → ONNX 部署正确, 可跑 closed-loop
#       - 显著偏高                  → 导出坏 / 路由到旧模型 / 平谱塌缩
#       按衬底分别报告, 故能抓出"某衬底被偷偷路由到旧衬底专用模型"的隐患.
#
# 用法: python verify_onnx_holdout.py --test-file data/TiO2_holdout_test.pkl
#       python verify_onnx_holdout.py --test-file data/aSi_holdout_test.pkl --limit 500
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import sys, argparse, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml_module
from color_utils import spectrum_to_srgb, rgb_to_lab, delta_e2000

WL = np.linspace(380, 780, 81)
INV_MAT = {v: k for k, v in ml_module.MATERIAL_CODES.items()}
INV_SUB = {v: k for k, v in ml_module.SUBSTRATE_CODES.items()}


def denorm(x):
    """把 7 维归一化输入反解回物理量 (与 _build_rcwa_input 互逆)."""
    D = x[0] * 300 + 50
    H = x[1] * 520 + 80
    P = x[2] * 400 + 200
    angle = x[3] * 80
    pol = "TE" if x[4] < 0.5 else "TM"
    mat = INV_MAT.get(int(round(x[5])))
    sub = INV_SUB.get(int(round(x[6])))
    return D, H, P, angle, pol, mat, sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-file', required=True, help='冻结 holdout pkl, 含 {X, Y, indices}')
    ap.add_argument('--limit', type=int, default=0, help='只验前 N 条 (0=全部)')
    args = ap.parse_args()

    ml_module.init_rcwa_ml()
    import torch
    obj = torch.load(args.test_file, weights_only=False)
    X = np.asarray(obj['X'], dtype=np.float64)
    Y = np.asarray(obj['Y'], dtype=np.float64)
    if args.limit > 0:
        X, Y = X[:args.limit], Y[:args.limit]

    des, by_sub, by_mat = [], {}, {}
    nfail = 0
    for i in range(len(X)):
        D, H, P, angle, pol, mat, sub = denorm(X[i])
        if mat is None or sub is None:
            nfail += 1; continue
        spec = ml_module.predict_spectrum(D, H, P, angle_deg=angle, polarization=pol,
                                          material=mat, substrate=sub)
        if spec is None:
            nfail += 1; continue
        spec = np.asarray(spec)
        rgb_pred = spectrum_to_srgb(WL, np.clip(spec, 0, None))
        rgb_true = spectrum_to_srgb(WL, np.clip(Y[i], 0, None))
        de = delta_e2000(rgb_to_lab(np.array(rgb_pred)), rgb_to_lab(np.array(rgb_true)))
        des.append(de)
        by_sub.setdefault(sub, []).append(de)
        by_mat.setdefault(mat, []).append(de)

    des = np.array(des)
    print(f"\n{'='*66}")
    print(f"  ONNX 部署一致性校验: {os.path.basename(args.test_file)}")
    print(f"{'='*66}")
    print(f"  有效样本: {len(des)}  ({nfail} 失败/跳过)")
    print(f"  整体 ΔE : mean={des.mean():.2f}  median={np.median(des):.2f}  "
          f"<2.3={100*(des<2.3).mean():.0f}%")
    for mat, v in by_mat.items():
        v = np.array(v)
        print(f"  材料 {mat:20s} mean={v.mean():.2f}  (n={len(v)})")
    for sub, v in by_sub.items():
        v = np.array(v)
        print(f"    衬底 {sub:20s} mean={v.mean():.2f}  (n={len(v)})")
    print(f"\n  判读: 与训练报告的 holdout ΔE 一致 → 部署正确, 可跑 closed-loop;")
    print(f"        显著偏高 → 导出坏/平谱塌缩/路由到旧模型 (查对应衬底那行).")
    print(f"{'='*66}\n")


if __name__ == '__main__':
    main()
