# a-Si 光学常数取证档案（2026-08-06）

> 对抗性预审（审稿人 M1 + 数字红队）触发。目的：确认现有 a-Si 光学常数来源，为"重跑 or 改标"决策留档。

## 一、现状（代码/文档实测）

**k 表（rcwa_batch.py `_A_SI_K_TABLE`，L40-50）**，注释："Green & Keevers 1995, Prog. Photovoltaics"：

| λ(nm) | 380 | 400 | 450 | 500 | 550 | 600 | 650 | 700 | 780 |
|---|---|---|---|---|---|---|---|---|---|
| k | 0.520 | 0.445 | 0.325 | 0.225 | 0.125 | 0.050 | 0.016 | 0.005 | 0.001 |

**n 侧**：`CAUCHY["a-Si"] = (3.80, 0.08, 0.0)`（rcwa_batch.py，无来源注释）→ n(550) = 3.8 + 0.08/0.55² ≈ **4.064**

## 二、矛盾点

1. **注释 vs 表值**：G&K 1995 是 *intrinsic silicon*（**c-Si**）数据；c-Si 在可见光区 k < 0.1（550 nm 约 0.02 量级）。表中 k(550)=0.125 高约 6 倍，k(380)=0.52 高约 10 倍——**形态是 a-Si 的**（短波强吸收、指数衰减）。
2. **n 侧无出处且自相矛盾**：旧稿 paper_full.md 材料表记 a-Si n=**3.80**（常量），rcwa_batch 的 Cauchy 算 n(550)=**4.064**——两个口径对不上；标准 a-Si 的 n(550)≈4.2–4.4，两者都偏低；c-Si n(550)≈4.0。
3. **遗留线索**（_archive/old_drafts/paper_draft_results_discussion.md L159）："a-Si 光学常数引用：Palik Handbook / refractiveindex.info (Green & Keevers 1995)"——**k 表疑似实际抄自 refractiveindex.info 的 a-Si 条目，注释误挂 G&K**。

## 三、定性结论

现有 "a-Si" 是**无可靠出处的杂合模型**：k 形态≈a-Si、出处错标；n 侧来源不明（疑似 c-Si 量级）。既不能声称"忠实于 G&K 的 c-Si"，也不能声称"标准 a-Si"。**改标方案（a-Si→c-Si）前提不成立**——k 表不是 c-Si 数据，改标后反而穿帮。

## 四、重跑方案（推荐，待导师拍板）

**目标**：a-Si 层改用可引用的标准数据集，其余 6 材料冻结不动。

1. **数据源**：Palik《Handbook of Optical Constants of Solids》卷 III "Amorphous silicon" 条目（项目已引 ref30 Palik，风格统一；备选 Jellison 1999 椭偏数据）。⚠️ 需要拿到 380–780 nm 的 n,k 数值表（当前网络受限，refractiveindex.info 等不可达，需人工提供或确认获取渠道）。
2. **脚本改动点**（全部现成，改动面小）：
   - `rcwa_batch.py`：替换 `_A_SI_K_TABLE` + `CAUCHY["a-Si"]` → Palik 表（或插值函数）
   - 重跑：a-Si/SiO₂ RCWA 池（~3000 结构）→ 训练 3 籽 → N=100 roundtrip → 29-probe → 数据核验
3. **波及面**（重跑后全链条更新）：
   - Table 3 a-Si 两列（forward ΔE、成功率 0–18%、gap +3.68 等）
   - Fig.2b（a-Si 点位置）、Fig.3（a-Si 色域面板）、Fig.4（curse gap）
   - Q&A backup（reviewer_qa.md）、cover letter、highlights
   - 幻灯片第 3/9 页 + 讲稿（Qwen 侧同步）
   - REPRODUCIBILITY.md 数据源描述修正
4. **招牌例切换预案**：若真 a-Si 前向误差变大、反转消失 → 摘要招牌例改用 **Si₃N₄ 对**（forward 差但成功率高）。⚠️ 与 A2 耦合：Si₃N₄ 的 3.5 数字需同步定稿（部署实测 13.75），否则新招牌例带病。
5. **预期**：真 a-Si k≈0.5 起步（可见光），吸收叙事只会更强，0–18% 大概率变更干净的 0%。

## 五、待办

- [ ] 拿到 Palik（或 Jellison）a-Si 的 380–780nm n,k 数值表（人工提供或确认渠道）
- [ ] 导师拍板：重跑 + A4（holdout 目标 + random-20 有界对照，一并批）
- [ ] 拍板后执行：RCWA 池 → 训练 → 闭环 → 全链条更新 → 重锁 MD5

## 六、指标口径注记（2026-08-06 交叉审计对齐）

- "max 相邻跳变 0.070"：指 **5 nm 网格**（380–780 步长 5）上 k 的相邻采样值差分绝对值最大值，位于 ~430 nm 陡峭段，是平滑导数的采样表现，非台阶（线性插值已保证连续）。
- 审计方独立复算（10 nm 网格）实测原始相邻差分最大 0.1395 ≈ 2 × 0.070——同区段、同来源，差异仅因采样步长不同，结论一致。
- 决定性判据：rcwa_batch 的 `_aSi_ps_nk` 与原始 CSV 的 np.interp 在 380–780 逐 10 nm 点 1e-9 级一致（已复现）。
