# a-Si 新数字包 v3（Pierce & Spicer 1972 重跑，2026-08-07 定稿，2026-08-08 审查更新）

> 数据源：`data/rcwa_aSi_PS_SiO2.pkl`（2725 结构）→ 3 籽重训（42/123/456，统一协议）
> → `data/closed_loop_aSi_PS_SiO2_roundtrip_N100.pkl` + `data/closed_loop_aSi_PS_SiO2_probe29.pkl`
> 全部数字可从上述 pkl 复算核验。**curse gap 一律 median 口径**（LESSONS.md 裁定）。

## 1. Forward 精度（frozen test N=1635）

| 口径 | 值 | 说明 |
|---|---|---|
| 三籽 seed-mean（延续旧论文口径） | **2.37**（2.448/2.341/2.314） | metrics json（`models/forward_mlp_rcwa_aSi_PS_s*_test_metrics.json`） |
| 3 籽 ensemble 部署（ml_module 路由实测） | **2.15**（median 1.57，<2.3 = 66.6%） | `data/forward_aSi_PS_test_per_target_ensemble.pkl`（逐目标落盘，1635 条，ensemble 口径） |
| 旧值（杂合常数） | 2.68 | 已被取代 |

⚠️ **口径必须明确**：与旧 a-Si 的 2.68（seed-mean）/2.38（ensemble）之争同构。**建议正文用 seed-mean 2.37**（与 TiO₂ 2.99、Si₃N₄ 13.75 同口径），ensemble 2.15 作脚注披露。

## 2. 闭环 RT（N=100，roundtrip 目标）

| 指标 | 新值 | 旧值 |
|---|---|---|
| hybrid 成功率（nG=65 验证） | **86/100 = 86%** | 18% |
| **hybrid 成功率（nG=101 复验）** | **47/100 = 47%** ⚠️ 见 nG_convergence_audit.md | — |
| hybrid mean/median ΔE（nG65） | 1.37 / 1.15 | 4.68 / 3.95 |
| naive 成功率 | **53%**（51/97，3 个 naive None） | 19% |
| curse gap median | **+1.46** | +3.68 |
| curse gap mean（仅披露辅助） | +2.05 | — |
| hybrid ≤ naive | 100/100 | 98/98 |

> ⚠️ **求解器阶数依赖（2026-08-07 nG 审计）**：a-Si 强吸收高对比导致 nG=65 验证伪影——
> nG=101 下成功率 47%，nG=131 仍未收敛（ΔE 1.86→3.09→4.23 持续上涨）。
> **"86% 全场最高"表述不成立**（nG101 口径 47%，全配置倒数第二）；probe 0% 稳健。
> 无损材料全部稳健（掉幅 ≤9 点）。论文须报双口径 + 收敛率，详见 nG_convergence_audit.md。

## 3. 色域 Probe（N=29，sRGB 边界目标；**29 条全 ok，无丢目标**）

| 指标 | 新值 | 旧值 |
|---|---|---|
| hybrid 成功率 | **0/29 = 0%** | 0% |
| hybrid mean/median ΔE | 22.36 / 22.74 | 18.04 / 14.18 |
| naive 成功率 | 0%（0/28，1 个 naive None） | — |
| curse gap median | +0.11 | +1.87 |
| curse gap mean（披露辅助） | +2.26 | +4.65 |

> 澄清：汇总输出 "n=28/29" 是 naive-有效口径（29 − 1 个 naive None），**hybrid N=29 成立**。
> 目标清单归档：`data/externals/probe_aSi_PS_29_targets.json`（29 个 RGB + name）。

## 4. 色域与材料参数

| 指标 | 新值 | 旧值 |
|---|---|---|
| sRGB std（cross-sample） | **0.112 / 0.081 / 0.084** | 0.21 / 0.10 / 0.09 |
| R_range（per-sample median） | **0.209**（mean 0.244，max 1.000） | 0.635 |
| n(550) / k(550) | **4.39 / 0.77**（Pierce & Spicer 1972） | 4.06 / 0.125（杂合） |
| **Δn(550)** | **≈ 2.93**（4.39 − 1.458） | 2.61（旧常数残留，勿带入） |

## 5. A4 有界对照（holdout 目标 + random-20，预注册 ΔE<2.3 / 种子 2026）

| 臂 | naive | hybrid (ML top-20) | random-20 | hybrid−random20 |
|---|---|---|---|---|
| TiO₂（N=100） | 24%（mean 6.11） | **63%**（mean 2.28） | **9%**（mean 8.10） | −5.82 |
| a-Si（N=100） | 53%（mean 3.41，n=96） | **87%**（mean 1.51） | **4%**（mean 7.16） | −5.64 |

- 归档：`data/controlled_TiO2_SiO2_holdout_N100.pkl`、`data/controlled_aSi_PS_SiO2_holdout_N100.pkl`（逐目标明细 + 来源 DHP/index）
- **M6 终结**：holdout（surrogate 未见结构）目标下 hybrid ≈ 训练集 roundtrip（TiO₂ 63%≈62%、a-Si 87%≈86%）——无记忆泄漏伪影
- **M7 终结**：ML top-20 远胜 random-20（63%/87% vs 9%/4%）——审稿人"random≈64%"估算（5% 候选密度假设）被实测推翻，**ML 筛选价值首次被直接对照证明**
- ⚠️ a-Si 臂成功率同样基于 nG=65 验证（nG 依赖见 nG_convergence_audit.md）；hybrid vs random 对照不受影响（同为 nG65，公平）

## 6. 叙事定位（审计拍板 2026-08-07 + nG 审计修正 2026-08-07）

- **头条反例：Si₃N₄/TiO₂ 双向对**（前向 13.75 vs 2.99 → 逆向 81% vs 62%；nG101 口径 79% vs 53% 差距更大）；统计用"7 材料秩解耦（Spearman ρ+p）+ 极端对示意"，不用成对显著性
- **第二支柱：a-Si 窄色域高可达**——probe 0%（钉死色域边界，稳健）+ RT 双口径披露（86%@nG65 / 47%@nG101，nG131 未收敛）；"高对比强吸收材料的 RCWA 阶数依赖"写入讨论作方法学发现；"旧 18% 是无出处常数的方法伪影"写入审计档案 + REPRODUCIBILITY
- **A4 新增支柱：ML 筛选价值**——holdout 目标下 hybrid 63%/87% vs random-20 9%/4%（M6/M7 双终结），可作 §3.1 核心证据
- §3.1.2 必须补 **Si₃N₄ 前向 13.75 的机制句**（摘要已扛此数字）
- A4 有界对照加 a-Si 臂（holdout + random-20，预注册 ΔE<2.3 / 种子 2026）

## 6. 归档索引（审计要求 2026-08-07 已执行）

- `data/forward_aSi_PS_test_per_target_ensemble.pkl` — forward 逐目标预测（1635 条，ensemble 口径，含 DHP/index/pred_rgb/target_rgb/de2000）
- `data/externals/probe_aSi_PS_29_targets.json` — probe 29 目标 RGB 清单
- `data/forward_mlp_rcwa_aSi_PS_s{42,123,456}_test.pkl` — frozen test（3 份 indices 一致，213→1635 为全量重训版）
- `data/ng_verify_aSi_PS_*_65_101.pkl` — nG 收敛归档（已完成：RT 72%、probe 34%、k0 12%）

## 7. 统计量（2026-08-07 已完成，双证据线通过）

- Spearman（统一 seed-mean 口径）：nG65 ρ=−0.43 p=0.34、nG101 ρ=+0.14 p=0.76（审计混口径版 −0.50/0.25 数值不同，结论一致——正文用统一口径值）
- Wilson CI：a-Si 86% → [77.9, 91.5]；probe 0/29 单侧上界 9.8%；A4 TiO₂ 63% [53.2, 71.8]、a-Si 87% [79.0, 92.2]
- McNemar（a-Si RT）：b=32 c=0 p=4.66e-10；A4：TiO₂ b=54、a-Si b=83（均 c=0）
- bootstrap（curse gap median +1.46）：CI [+1.15, +1.93]（20000 次，seed 2026）
