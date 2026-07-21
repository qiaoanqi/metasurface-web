# Introduction

结构色（structural color）源于亚波长尺度光与微纳结构的相互作用，通过调控几何参数而非化学染料实现色彩生成，具有不褪色、环保、分辨率高等优势，在显示技术、防伪、传感和装饰涂层等领域展现出广泛应用前景[1-3]。介电超表面（dielectric metasurface）——由周期性排列的高折射率纳米柱阵列构成的二维光子结构——因其支持导模共振（guided-mode resonance, GMR）和米氏共振（Mie resonance），能在极薄层内实现高饱和度、宽色域的结构色，已成为该领域的研究前沿[4,5]。

超表面结构色的正向设计（给定几何参数预测颜色）可通过严格耦合波分析（RCWA）[6]或有限时域差分（FDTD）[7]等全波电磁仿真精确求解。然而，逆设计（给定目标颜色求解几何参数）面临根本性困难：电磁响应与几何参数之间存在高度非线性的多对一映射，且仿真计算代价高昂（单次 RCWA 求解约 4 s，FDTD 更甚），使得传统穷举搜索或梯度优化在工程上不可行[8]。

近年来，机器学习（ML）代理模型（surrogate model）被引入超表面逆设计以加速搜索过程[9-12]。典型策略是：先训练一个神经网络近似正向映射（几何→光谱/颜色），再利用该代理模型在候选空间中快速筛选最优设计。这一范式将单次推理时间从秒级压缩至毫秒级（本研究中 ML 推理 5.13 ms vs RCWA 4.16 s，加速比 811×），展现了巨大的加速潜力。然而，现有文献中存在三个被普遍忽视的关键问题：

**第一，缺乏严格的闭环验证。** 绝大多数 ML 辅助超表面设计工作仅报告正向预测精度（代理模型在测试集上的误差），而未验证逆设计结果在全波仿真或实验中的实际表现[9-11]。正向精度低并不保证逆设计成功——二者之间存在由"优化器诅咒"（optimizer's curse）[13,14]导致的系统性鸿沟：当代理模型被用作优化目标时，其预测误差会被选择过程系统性放大，使得"代理最优"结构在真实物理验证中表现远逊于预期。

**第二，缺乏材料筛选的定量判据。** 并非所有介电材料都适合产生结构色。低折射率对比度（Δn）体系中导模共振被抑制，光谱几乎无变化，ML 模型无论多精确都无法逆设计出有意义的颜色。然而，现有文献中材料选择多基于经验或直觉，缺乏基于物理的定量筛选准则[15]。

**第三，代理模型保真度对逆设计成功率的决定性作用未被充分认识。** 即使采用相同的模型架构和训练协议，不同材料体系的代理精度可能存在数倍差异，这直接决定了逆设计的成败。理解这一材料依赖性对于指导实验设计和计算资源分配至关重要。

本文针对上述三个问题，提出一套完整的 ML 辅助介电超表面结构色智能设计系统，并做出以下贡献：

（1）**闭环验证框架与优化器诅咒量化**：我们建立了从目标色到 ML 逆设计再到独立 RCWA 全波验证的严格闭环协议。通过四材料体系（TiO₂, a-Si, Si₃N₄, Al₂O₃）的系统实验，首次量化了超表面逆设计中的优化器诅咒效应（~+5–6 ΔE₀₀ 的过度乐观偏差），并证明该偏差近似材料无关。进而提出混合重排策略（hybrid re-ranking）：ML 筛选 top-K 候选后由 RCWA 逐一验证取最优，在数学上保证 hybrid ≤ naïve，实验上将 TiO₂ 逆设计成功率从 23% 提升至 63%。

（2）**折射率对比度二阶筛选判据**：通过 11 组材料/衬底组合的系统 RCWA 数据，建立了"Δn > ~0.5 共振截止阈值 + 低消光系数 k ≈ 0 充分条件"的二阶材料筛选准则。该判据成功解释了 TiO₂（正例）、Si₃N₄/Al₂O₃（负对照）和 a-Si（高 Δn 但受损耗限制）的实验表现，为超表面结构色的材料选择提供了物理基础。

（3）**代理保真度决定逆设计天花板的实证**：通过 TiO₂（成功）与 a-Si（失败）的对比实验，证明混合重排消除统计偏差（优化器诅咒）但不消除系统误差（代理基线精度），逆设计的最终精度天花板由材料本征属性决定。这一发现为 ML 辅助光子学设计的适用范围划定了明确边界。

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
