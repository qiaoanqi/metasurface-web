# 椭圆 × 双偏振管道：门 1 + 门 2 闭合记录（2026-08-13）

> 按 second_paper_elliptical_plan.md v2 执行。Qwen 审计限额期间由 Hermes 自主推进，
> 全程遵守执行纪律（门 2 通过前不 commit；论文文件与 data/*. 一字节不碰）。

## 门 1：椭圆 × 双偏振管道（已实现 + 验证）

**代码变更（rcwa_batch.py）**：
1. 椭圆几何：`rcwa_spectrum(..., W_nm=None)` —— None 走原圆形公式（圆柱零回归）；
   W_nm 给定走椭圆 mask `(X/(D/2))²+(Y/(W/2))² ≤ 1`；`|W−D|<1e-12` 路由回圆形公式。
2. 偏振：`pol ∈ {'p','s'}` —— 默认 'p'（legacy (1,0,0,0)）；'s' = (0,0,1,0)。
   grcwa `MakeExcitationPlanewave(p_amp,p_phase,s_amp,s_phase)`，phi=0 时 p 激发后半块、
   s 激发前半块（源码核实）。TE/TM 物理标签由三条不变量实测确立，非假定。
3. 采样器：`generate_params_elliptical(n, seed, r_max=3.0)` —— 均匀随机 + 拒绝法
   （与论文 1 `generate_params` 同协议族，不用 LHS）；约束硬断言。

**三条物理不变量（13/13 PASS，91 s；落盘 data/ellipse_gate1_invariants.pkl，脚本 gate1_invariants.py 可重跑）**：
| 不变量 | 结果 |
|---|---|
| 1. 圆 + 正入射：p ≡ s（旋转对称） | max\|dR\| ~1e-13（浮点噪声），全过 |
| 2. 椭圆 L≠W：p ≠ s（必须分裂） | TiO₂ dE=11.8、a-Si dE=28.9 —— 显著分裂 |
| 3. 90° 旋转：spec(L,W,p) ≡ spec(W,L,s) | max\|dR\| ~1e-12，全过 |
| 零回归：pol='p' 圆形 vs 存档 | max\|dR\| ~1e-14（逐位一致） |

## 门 2：L=W 复现门（30/30 PASS，397 s，逐位 0.0）

- **实测性质（口径写清）**：椭圆 `W_nm=D` 与圆形 `W_nm=None` 走**同一条代码路径**
  （`abs(W-D)<1e-12` 路由回原圆形公式）——验证的是"退化路由正确性"，非"与存档对比"。
- 对比源：TiO₂ 15（closed_loop_TiO2_SiO2_roundtrip_N100.pkl，float eps 路径）
  + a-Si 15（rcwa_aSi_PS_SiO2.pkl，complex eps 路径）——两条代码路径都要过。
- 判据：R/T 全谱 max abs diff 与 ΔE2000 **逐位 0.0**（同代码路径，路由正确则必然全零；
  任何非零 = 路由/因子化 bug，停下查因；<0.1 仅作有书面归因的后备容差）。
- 结果：**max_dR = 0.00e+00，max_dE = 0.00e+00**（30/30 完全一致）
- 结构 ID + 逐结构结果落盘：`data/ellipse_gate2_result.pkl`（含 script 字段）
- **存档回归证据在门 1 零回归行**（非门 2）：pol='p' 圆形重算 vs 存档池
  max dR=4.07e-14（gate1_invariants.pkl）；审计独立复算 (165.1, 546.7, 200.0)
  颜色差 2.2e-16、R+T=0.303 与存档一致——两条证据线勿混。
- 生成脚本：`gate2_reproduction.py`（随 pkl 入库，血统纪律，可重跑）

## 门 3：约束口径（内嵌于门 1c 采样器，已验证）

`max(L,W) < P`；`f = π(L/2)(W/2)/P² ∈ [0.03, 0.70]`；`r = L/W ∈ [1, 3]` ——
500 采样全过（r 1.00-3.00、f 0.030-0.700）。

## 门 4：nG 收敛探针（进行中，后台）

- 16 个 a-Si 椭圆结构 × nG {65, 101, 131, 201} × pol {p, s}（最坏情况材料）
- 自洽口径（以 nG=201 为参考）+ 预注册联合判据 max(ΔE_p, ΔE_s)
- 实测各档耗时 → 池生成预算（上报用户）
