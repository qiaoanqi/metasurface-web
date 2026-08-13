# 级联重写方案（paper_oe.tex 等，2026-08-07 设计稿，待统计量 v2 后执行）

> 数据依据：aSi_PS_new_numbers.md（v3）+ nG_convergence_audit.md + stats 第一批核验。
> 执行顺序：k=0 对照重跑 → tex 批量 patch → figures 重生成 → 外部文档 → 双遍编译 → MD5 重锁。

## A. 前置计算任务（级联前）
- [ ] **a-Si(k=0) 对照重跑**：新常数下 k=0 的闭环 + nG 收敛（fig5 第三条线数据源，旧 40% 无源）
- [ ] fig2b/fig3/fig4 数据更新：a-Si 点（R_range 0.209、sRGB std 0.112/0.081/0.084、RT 86/47、gap +1.46）→ 重生成 3 图
- [ ] fig5 重生成：_make_fig5_from_pkl.py（TiO₂ 93%、a-Si RT 72%、a-Si k=0 新值）

## B. tex 修改点（paper_oe.tex）

| 位置 | 修改 |
|---|---|
| L21 摘要 | a-Si 反转例 → Si₃N₄/TiO₂ 双向对（13.75 vs 2.99 → 81% vs 62%）；Δn 上限 2.61→2.93；+3-4 收窄按统计量 v2 |
| L31 引言 | 反转例换 Si₃N₄/TiO₂；a-Si 2.68/0-18% 删除 |
| L39 | material-independent +3-4 → 收窄表述（按统计量 v2） |
| L43 | Δn 2.61 → 2.93 |
| L49 | CIE 1931 → 补 D65/2°（minor 1） |
| L69 | a-Si 常数描述 → Pierce & Spicer 1972（n/k 同源、60nm 膜、CSV 加载、线性插值） |
| L133 | **"drawn from the training set" → 随机结构（分布一致）**——红队溯源 0/100 根因，表述修正 |
| L135 | McNemar 句保留 + A4 结果句（holdout 63% vs random 9%） |
| L139-145 | a-Si 段重写：forward 2.37（seed-mean，脚注 ensemble 2.15 口径）、probe 0%（mean 22.4）、RT 86%@nG65/47%@nG101 双口径、sRGB std 0.112/0.081/0.084、k(400)=2.21、R+T≈0.51 |
| L168 | Table 1：a-Si forward 2.37、Si₃N₄ 13.75‡；**Table 扩列**（加 GaN/Ta₂O₅/HfO₂ 三列 forward + 成功率——审计要求 7 材料全网格） |
| L170-176 | a-Si 两列全数字更新（naive 53%、gap +1.46、hybrid 1.37/1.15、86%、100/100） |
| L177 | nG 行：TiO₂ 93%、a-Si RT 72%、probe 34%；可加收敛率列 |
| L182 | 脚注 ‡ 改写（A2：独立冻结划分复现 N=5400 三衬底聚合；训练期自报弃用） |
| L192 | +3-4 收窄 + 反转句改写（Si₃N₄/TiO₂） |
| L211 | k≈0.52@400 → k(400)=2.21（P&S）；R+T 0.731 → 0.51；sRGB std 更新；impedance+absorption 叙事保留 |
| L219 | a-Si ML error ~2.7 → ~2.4 |
| L234+ §4.2 | nG 数字：90%/72% → 93%/72%（TiO₂/a-Si）+ "高对比强吸收材料 RCWA 阶数依赖"方法学发现段（nG 审计结论） |
| L258 | testable prediction 保留（无损材料数据不变） |
| §3.1.2 新增 | **Si₃N₄ 前向 13.75 机制句**（三衬底聚合口径 + 模型容量讨论，摘要已扛此数字） |

## C. 外部文档
- cover letter：a-Si 数字（2.68→2.37、0-18%→新口径）、头条例换 Si₃N₄/TiO₂
- highlights：5 条重写（Si₃N₄/TiO₂ 头条、a-Si 窄色域高可达、ML 筛选价值 A4、Δn 判据、方法学发现）
- reviewer_qa.md backup：a-Si 行更新（2.37/86/47/+1.46/probe 0）、Si₃N₄ 13.75、nG 双口径、A4 结果
- REPRODUCIBILITY.md：a-Si P&S（已改）+ k=0 对照 + A4 实验记录
- 幻灯片第 3/8/9/13 页 + 讲稿（Qwen 侧同步）

## D. minor 14 条批量（随级联）
1 照明体/2 几何定义/3 Cauchy 引用补全/5 双过滤说明/6 计时统一/7 阈值 CDF+Fig1 措辞/10 θ/pol 预留说明/12 Lab 体积叙述/14 多重比较说明（4/8/9/11/13 视情况）

## E. 收尾
双遍 pdflatex → 0 overfull/undefined → Copy-Item 论文.pdf → MD5 重锁 → 数字包 v4 归档
