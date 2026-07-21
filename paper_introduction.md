# Introduction

机器学习代理模型在超表面逆设计中被广泛使用，但其有效性从未被严格验证。我们发现一个反直觉现象：正向预测精度更高的材料体系（a-Si，holdout ΔE₀₀ = 2.38），逆设计成功率为零；而精度较低的材料（TiO₂，ΔE₀₀ = 2.99），成功率却达 63%。精度指标本身无法预测逆设计的成败——这一发现迫使重新审视 ML 辅助光子学设计的基本假设。

结构色源于亚波长尺度光与微纳结构的相互作用，介电超表面（周期性高折射率纳米柱阵列）因支持导模共振（GMR）和米氏共振，能在极薄层内实现高饱和度色彩，在显示、防伪、传感等领域前景广阔[1-5]。逆设计（给定目标颜色求解几何参数）面临高度非线性的多对一映射和高昂仿真代价（单次 RCWA 约 4 s），传统穷举搜索不可行[6-8]。ML 代理模型将单次推理压缩至毫秒级（本研究 5.13 ms，加速比 811×），但现有文献存在三个被普遍忽视的关键问题[9-12]：

**第一，正向精度与逆设计成功之间的脱节未被认识。** 现有工作普遍以正向 holdout 精度（代理模型在测试集上的误差）作为模型质量的最终指标，隐含假设"预测得准就能设计得出"。然而，逆设计是在候选池中寻找最优解的优化问题，其成功取决于材料的可实现色域是否覆盖目标色，而非代理模型在测试集上的平均误差。这一根本区别从未被系统验证。

**第二，缺乏严格的闭环验证，优化器诅咒效应未被量化。** 绝大多数 ML 辅助超表面设计工作仅报告正向精度，而未验证逆设计结果在全波仿真中的实际表现[9-11]。当代理模型被用作优化目标时，其预测误差会被选择过程系统性放大（"优化器诅咒"，optimizer's curse）[13,14]，使得"代理最优"结构在真实物理验证中表现远逊于预期。这一效应在超表面领域从未被量化。

**第三，缺乏材料筛选的定量判据。** 并非所有介电材料都适合产生结构色。低折射率对比度（Δn）体系中导模共振被抑制，光谱几乎无变化，ML 模型无论多精确都无法逆设计出有意义的颜色。然而，现有文献中材料选择多基于经验或直觉，缺乏基于物理的定量筛选准则[15]。

本文的核心论点是：**ML 代理模型的逆设计天花板不由模型精度决定，而由材料的物理可实现色域决定。** 围绕这一论点，我们做出以下贡献：

（1）**正向精度≠逆设计成功的实证**：通过 TiO₂（成功，63%）与 a-Si（失败，0%）的严格对比，首次证明正向 holdout ΔE 是逆设计成功的必要非充分条件，色域覆盖度才是充要条件。

（2）**优化器诅咒量化与混合重排修复**：首次量化超表面逆设计中的优化器诅咒（~+5–6 ΔE₀₀，材料无关），提出混合重排策略（ML top-K → RCWA 逐一验证），数学保证 hybrid ≤ naïve，TiO₂ 成功率 23%→63%。

（3）**折射率对比度二阶筛选判据**：基于 11 组材料/衬底数据，建立"Δn > ~0.5 共振截止 + k ≈ 0 色域最大化"的二阶判据，Δn 变化 0.013 导致 7× 共振崩塌，成功解释四材料体系的实验表现。

本文结构如下：§2 描述 RCWA 数据生成、ResMLP 代理模型架构和混合逆设计流程；§3 报告正向预测精度、闭环逆设计验证和材料筛选判据的实验结果；§4 讨论代理保真度的材料依赖性、求解器收敛性和方法局限；§5 总结全文。

---

## 引用占位

[1] Kinoshita et al., Rep. Prog. Phys. 2008 — structural color in nature
[2] Tan et al., Adv. Mater. 2014 — structural color review
[3] Yu et al., Nat. Rev. Mater. 2023 — metasurface structural color
[4] Kuznetsov et al., Science 2016 — dielectric metasurfaces
[5] Tseng et al., ACS Nano 2017 — TiO2 metasurface color
[6] Moharam & Gaylord, JOSA 1981 — RCWA
[7] Taflove & Hagness, Computational Electrodynamics 2005 — FDTD
[8] Molesky et al., Nat. Photonics 2018 — inverse design in photonics
[9] Tahersima et al., Sci. Rep. 2019 — NN for metasurface design
[10] Liu et al., ACS Photonics 2021 — deep learning metasurface
[11] So et al., Adv. Photonics 2022 — ML-assisted nanophotonics
[12] 本领域最新综述 — 2024/2025
[13] Jones et al., J. Global Optim. 1998 — Efficient Global Optimization
[14] Gonzalez et al., AISTATS 2016 — Batch Bayesian Optimization
[15] Magnusson & Wang, Appl. Phys. Lett. 1992 — GMR principle
