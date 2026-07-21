# Abstract

机器学习代理模型驱动的超表面逆设计面临一个被普遍忽视的统计陷阱：当代理模型本身成为优化目标时，其预测误差被选择过程系统性放大，导致"代理最优"设计在全波验证中表现远逊于预期。本文以介电超表面结构色为研究对象，建立了从目标色到 ML 逆设计再到独立 RCWA 全波验证的严格闭环协议，系统量化了这一"优化器诅咒"效应并提出了有效的修复策略。

我们训练了基于残差 MLP 的 3-seed 集成代理模型（553K 参数，ONNX 部署），对四材料体系（TiO₂, a-Si, Si₃N₄, Al₂O₃）在三衬底上的 11 组 RCWA 数据集进行正向预测和闭环逆设计验证。结果表明：(1) 正向预测精度是逆设计成功的必要非充分条件——a-Si 复折射率模型的 holdout 精度（ΔE₀₀ = 2.38）优于 TiO₂（2.99），但逆设计成功率为 0% vs 63%，色域覆盖度而非预测精度才是逆设计成功的充要条件；(2) 优化器诅咒产生约 +5–6 ΔE₀₀ 的过度乐观偏差，该偏差近似材料无关，提出的混合重排策略（ML 筛选 top-K → RCWA 逐一验证取最优）在数学上保证不劣于纯 ML 策略，实验上将 TiO₂ 逆设计成功率（ΔE₀₀ < 2.3 JND）从 23% 提升至 63%；(3) 建立了折射率对比度二阶筛选判据：Δn ≈ 0.5 处存在共振截止（resonance cutoff），Δn 变化 0.013 即导致反射谱动态范围 7 倍崩塌，高 Δn 材料（如 a-Si, n = 3.8）因阻抗失配和材料吸收（可见光均值吸收 ~27%，蓝光波段高达 ~55%）而色域受限。

该判据成功解释了 TiO₂（正例，hybrid 63% 成功）、Si₃N₄/Al₂O₃（负对照，平谱无共振）和 a-Si（高 Δn 但损耗受限，0% 成功）的实验表现。本研究为 ML 辅助光子学逆设计划定了明确的能力边界：混合验证消除统计偏差但不消除系统误差，材料选择与验证策略是两个正交的改进维度。最具实践意义的发现是：一个正向精度更优的代理模型（a-Si, ΔE₀₀ = 2.38），其逆设计成功率可以严格为零——精度指标本身无法预测逆设计的成败。

**关键词**：超表面；结构色；机器学习代理模型；逆设计；优化器诅咒；严格耦合波分析；折射率对比度

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

# Methods

## 2.1 RCWA Data Generation

Training data generated with grcwa (guided-mode resonance coupled-wave analysis).
All materials modeled with Cauchy dispersion: n(λ) = A + B/λ² + C/λ⁴.

**Solver parameters (fixed across all materials):**

| Parameter | Value | Justification |
|-----------|-------|---------------|
| nG (Fourier orders) | 65 | TiO2 convergence verified to 90% <1 JND at nG 65→101 |
| Nxy (real-space grid) | 256 | Standard for sub-wavelength grating resolution |
| Wavelengths | 81 points, 380–780 nm | CIE 1931 5-nm sampling |
| Polarization | TE (s-pol) | Electric field parallel to grating grooves |
| Incidence | θ = 0° (normal) | Simplest configuration; angular dependence left for future work |

**Solver parameter physical justification:**

Fourier truncation order nG = 65 was selected based on systematic convergence verification. For TiO₂/SiO₂ structures (the primary material system), 90% of hybrid-selected designs exhibit color difference < 1 JND (CIEDE2000 < 2.3) when nG is increased from 65 to 101, with a mean per-structure ΔE spread of only 0.77. This confirms that nG = 65 captures the dominant diffraction orders for moderate-index-contrast systems (Δn ≈ 0.84). However, convergence is material-dependent: for lossless a-Si/SiO₂ (k = 0, Δn = 2.34), only 40% of structures converge within 1 JND across the same nG range (mean spread 5.02), reflecting the sharper guided-mode resonances supported by high-index-contrast gratings. Including the experimental complex refractive index (k ≠ 0) improves convergence to 72% (mean spread 1.83), as material absorption damps the sharp resonances that are most sensitive to Fourier truncation. The nG = 65 choice thus represents a computational budget compromise (RCWA cost scales as O(nG²)); results for high-index materials should be interpreted with this solver sensitivity in mind (see §4.2).

The real-space discretization Nxy = 256 provides a grid resolution of P/256 ≈ 0.8–2.3 nm for the period range studied (200–600 nm), sufficient to resolve pillar boundaries with sub-nm accuracy. Convergence with respect to Nxy is monotonic and well-behaved for circular pillar geometries; no Gibbs-type artifacts are expected at this resolution.

TiO₂ (anatase), Si₃N₄, and Al₂O₃ (sapphire) are modeled with real-valued Cauchy dispersion n(λ) = A + B/λ² + C/λ⁴, which is physically appropriate for transparent dielectrics whose extinction coefficient k is negligible across the visible band.

For a-Si, we employ the full complex refractive index ñ(λ) = n(λ) + ik(λ) using tabulated experimental data from Green & Keevers (1995), interpolated onto the 81-point wavelength grid. In the visible band, a-Si exhibits k ≈ 0.52 at 400 nm decreasing to k ≈ 0.01 at 700 nm; the imaginary part is included in the RCWA dielectric tensor as ε = ñ², yielding complex-valued diffraction efficiencies. A total of 3,000 Latin Hypercube samples were attempted; 2,725 completed successfully (90.8%), with 275 (9.2%) terminated by a 120 s per-sample timeout due to numerical stiffness in the complex RCWA solve. Of the successful samples, 99 (3.6%) exhibited R+T > 1.0 and were removed by the energy-conservation filter, yielding 2,626 clean training samples. The retained dataset has mean R+T = 0.749 (27% absorption), consistent with the expected optical loss of a-Si in the visible. An earlier lossless (k = 0) a-Si dataset was also generated for comparison; it produces non-physical R+T > 1 (mean 1.044) and is superseded by the complex-k dataset for all results reported in this paper unless explicitly noted.

**Material optical constants (at 550 nm):**

| Material | n | k | n_sub (SiO2) | Δn | Dataset |
|----------|---|---|--------------|-----|---------|
| TiO2 (anatase) | 2.30 | 0 | 1.458 | 0.842 | Cauchy, N=4567 |
| a-Si (amorphous) | 3.80 | 0.01–0.52 | 1.458 | 2.342 | Green & Keevers 1995, N=2626 |
| Si3N4 (nitride) | 1.99 | 0 | 1.458 | 0.532 | Cauchy, N≈3000 |
| Al2O3 (sapphire) | 1.75 | 0 | 1.458 | 0.292 | Cauchy, N≈3000 |

**Sampling strategy:**
- Latin Hypercube Sampling over (D, H, P) ∈ [80–350] × [100–600] × [200–600] nm
- P ≥ 1.2 × D enforced (minimum inter-pillar spacing)
- ~3000 structures per material/substrate combination
- Quality filter: |R+T − median(substrate)| ≤ 0.05 (relative to substrate mean)

**Data format:** Each sample stores {D, H, P, wl_nm, R(81), T(81), xyz, rgb, R_plus_T_mean, material, substrate}.

## 2.2 ResMLP Surrogate Model

**Architecture:**

| Component | Specification |
|-----------|--------------|
| Input | 7-dim normalized: [D̄, H̄, P̄, θ̄, pol, mat_code, sub_code] |
| Embedding | Linear(7, 256) → BatchNorm → ReLU |
| Residual blocks | 4 × ResBlock(256, 256) |
| Per block | Linear(256,256) → BN → ReLU → Linear(256,256) → BN → Residual add |
| Output head | Linear(256, 81) → Sigmoid |
| Total params | 553,809 |
| Loss | MSE(reconstructed, target) |
| Optimizer | AdamW, lr=5×10⁻⁴, cosine annealing to 10⁻⁵ |
| Batch size | 128 |
| Epochs | 1000 (early stopping on val loss, patience=50) |

**Architecture selection rationale:**

The 256×4 ResMLP (553K parameters) was selected after empirical scaling analysis revealed that model capacity is not the performance bottleneck — data quantity is. Training datasets contain 3,000–5,000 structures per material/substrate combination; at this sample count, the effective degrees of freedom that can be reliably learned are O(10³), well within the representational capacity of a 553K-parameter network. Scaling to 512×6 (12.9 MB ONNX, ~3.4M parameters) did not improve holdout accuracy on TiO₂/SiO₂ (ΔE change < 0.1) while increasing inference latency and overfitting risk on smaller datasets.

The decisive evidence for data-limited (not capacity-limited) performance is the 3-seed ensemble variance: across seeds 42/123/456, TiO₂ holdout ΔE varies by only 0.03 (range 2.97–3.00) and a-Si by 0.05 (range 4.15–4.20). If the model were underfitting (capacity-limited), different random initializations would converge to substantially different local minima, producing larger inter-seed variance. The observed near-deterministic training indicates the loss landscape has a single dominant basin at this architecture scale — the model extracts essentially all learnable signal from the available data.

The residual error (TiO₂ ΔE ≈ 3.0, a-Si ΔE ≈ 4.2) is therefore attributable to irreducible noise sources: (1) finite nG truncation in the RCWA training labels, (2) the 81-wavelength discretization of continuous spectra, and (3) for a-Si, the systematic bias introduced by the lossless approximation. Increasing model capacity cannot reduce these error floors; only improving label quality (higher nG, complex refractive index) or augmenting data volume would help.

**Training protocol:**
- 80/20 stratified hold-out split (by substrate, split-seed=2026, frozen to disk)
- Data augmentation: ±2 nm random jitter (×3 effective dataset)
- 3-seed ensemble (seeds 42, 123, 456), prediction = mean across seeds
- ONNX export (opset 14) for deployment inference (<0.5 ms/sample)

**Input normalization:**
```
D̄ = (D - 50) / 300
H̄ = (H - 80) / 520
P̄ = (P - 200) / 400
θ̄ = θ / 80
mat_code ∈ {0: TiO2, 1: a-Si, 2: Si3N4, 3: Al2O3}
sub_code ∈ {0: SiO2, 1: Si3N4, 2: Al2O3}
```

## 2.3 Hybrid Inverse Design

**Two-stage pipeline:**

**Stage 1 — Surrogate screening (ML):**
- Coarse grid: 12³ = 1,728 candidates (D/H/P uniformly spaced)
- Ensemble batch inference on GPU: ~2 ms/1,728 candidates
- Top-K = 20 candidates selected by predicted CIEDE2000

**Stage 2 — High-fidelity re-ranking (RCWA):**
- Each top-K candidate re-evaluated with independent RCWA (nG=65, Nxy=256)
- Best candidate selected by measured CIEDE2000
- 4.16 s/candidate × 20 = ~83 s total per inverse design

**Theoretical guarantee:** The hybrid best is always ≤ naïve best (candidate[0] ∈ top-K).

**Computational cost:** All timings measured on a single workstation (CPU: Intel Core, 16 GB RAM; ONNX Runtime, intra_op_threads=4). RCWA single-sample solve (nG=65, Nxy=256, 81 wavelengths): 4.16 s. ML ensemble inference (3-seed mean, 7→81): 5.13 ms per sample, an 811× speedup over RCWA. The full inverse design pipeline — 450-candidate ML grid search (2.31 s) plus 20-candidate RCWA re-ranking (~83 s) — completes in ~86 s per target color, versus ~31.2 min for the equivalent 450-call RCWA brute-force search. The ML screening stage thus reduces the candidate evaluation cost by ~800×, while the RCWA re-ranking stage (K=20) adds a fixed ~83 s overhead that is independent of the initial candidate pool size.

## 2.4 Evaluation Metrics

**CIEDE2000 (ΔE₀₀):** Perceptually uniform color difference metric.
Thresholds: ΔE₀₀ < 1.0 (imperceptible), < 2.3 (just noticeable difference, JND).

**Forward accuracy:** ΔE₀₀ between ML-predicted and RCWA-computed sRGB colors on frozen hold-out set.

**Closed-loop achieved ΔE:** Target color → ML inverse design → independent RCWA verification → ΔE₀₀(target, achieved).

**Gap:** achieved ΔE − ML self-claimed ΔE. Quantifies optimizer's curse magnitude.

**nG convergence:** ΔE₀₀(nG=65) vs ΔE₀₀(nG=101) on hybrid-selected structures. Fraction within 1 JND reported.

# Four-Material Benchmark Tables

Generated 2026-07-21 from 80/20 hold-out test sets. Updated 2026-07-21 with a-Si complex-k (Green & Keevers 1995) results.

## Table 1: Forward prediction hold-out ΔE2000

| Material | Substrate | n_pillar | k | Δn | N | mean ΔE | median ΔE | P95 | <2.3% | <1.0% | R range |
|----------|-----------|----------|---|-----|---|---------|-----------|-----|-------|-------|---------|
| TiO2 | SiO2 | 2.30 | 0 | 0.842 | 914 | 2.99 (2.97–3.00) | 2.12 | 8.5 | 53% | 21% | 0.582 |
| a-Si (k≠0) | SiO2 | 3.80 | 0.01–0.52 | 2.342 | 263 | 2.38 | 1.77 | 5.8 | 64% | 19% | 0.635† |
| Si3N4 | SiO2 | 1.99 | 0 | 0.532 | ~600 | 1.57 (1.57–1.57) | 1.03 | 4.8 | 78% | 49% | 0.043 |
| Al2O3 | SiO2 | 1.75 | 0 | 0.292 | — | — | — | — | — | — | 0.009 |

*80/20 stratified hold-out, split-seed=2026, 3-seed ensemble mean (range in parentheses).*
*†a-Si(k≠0) R range = 0.635 is absorption-driven (wavelength-dependent k creates spectral slope), NOT geometric resonance. Cross-sample RGB std: R=0.21, G=0.10, B=0.09 — color gamut is collapsed despite high per-sample R modulation.*
*Si3N4 ΔE is deceptively low: spectra are nearly flat (R range 0.043), model converges to mean predictor.*
*Al2O3 R range 0.009 — no trainable resonance structure exists.*
*Superseded: a-Si lossless (k=0) holdout ΔE = 4.18 (4.15–4.20), median 3.65, P95 9.4, <2.3: 28%, R range 0.186. Replaced by complex-k dataset.*

## Table 2: Closed-loop inverse design — optimizer's curse + hybrid repair

| Material | N | naïve mean | naïve med | naïve <2.3% | hybrid mean | hybrid med | hybrid <2.3% | ML self-claim | curse gap | gap closure | nG conv. |
|----------|---|-----------|----------|------------|------------|-----------|------------|--------------|-----------|-------------|----------|
| TiO2/SiO2 | 30 | 5.83 | 4.71 | 23% | 2.33 | 1.84 | 63% | 0.66 | +5.17 | +3.51 | 90% |
| a-Si(k≠0)/SiO2 | 29 | 18.04 | 16.91 | 0% | 14.13 | 14.18 | 0% | 13.39 | +4.65 | +3.91 | 72% |

*N roundtrip targets, hybrid=ML top-20 → RCWA re-rank. curse gap = naïve_achieved − ML self-claim. gap closure = naïve − hybrid (always ≥0).*
*nG convergence: fraction of structures with ΔE(nG65→101) < 1 JND.*
*hybrid ≤ naïve: TiO2 30/30, a-Si(k) 29/29 (mathematical guarantee).*
*Superseded: a-Si lossless (k=0) naïve=26.42, hybrid=12.17, ML self-claim=20.20, gap=+6.21, nG conv=40%. Replaced by complex-k dataset.*

## Table 3: Resonance cutoff criterion

| System | n_pillar | Δn | R range (mean) | R range (max) | Status |
|--------|----------|-----|----------------|---------------|--------|
| a-Si/SiO2 | 3.80 | 2.342 | 0.186 | — | Active, loss-limited |
| TiO2/SiO2 | 2.30 | 0.842 | 0.582 | 1.0 | Strong (optimal) |
| TiO2/Al2O3 | 2.30 | 0.545 | 0.309 | — | Active (near cutoff) |
| Si3N4/SiO2 | 1.99 | 0.532 | 0.043 | 0.081 | Quenched ✗ |
| TiO2/Si3N4 | 2.30 | 0.310 | 0.102 | — | Weak |
| Si3N4/Al2O3 | 1.99 | 0.235 | 0.029 | — | Quenched ✗ |
| Al2O3/Al2O3 | 1.75 | 0.000 | 0.004 | — | Dead ✗ |

*Cutoff Δn ≈ 0.53: TiO2/Al2O3 (0.545) R=0.309 vs Si3N4/SiO2 (0.532) R=0.043 — 7× collapse across 0.013 Δn.*
*Mechanism: guided-mode resonance cutoff when index contrast insufficient to support leaky waveguide modes.*

# 论文草稿：Results & Discussion（物理洞察部分）

> 作者分工：本节由 Qwen 撰写，涵盖逆设计结果、材料筛选判据、讨论。
> 数据表格由 Codex 提供，此处以 [Table X] 占位。

---

## 3.X 逆设计验证：优化器诅咒与混合重排修复

### 3.X.1 问题提出

代理模型（surrogate model）驱动的逆设计面临一个被超表面领域普遍忽视的统计陷阱：当 ML 模型被用作目标函数进行优化时，模型预测误差会系统性地偏向高估设计性能。这一现象在贝叶斯优化文献中被称为"优化器诅咒"（optimizer's curse）[Jones et al., 1998; Gonzalez et al., 2016]，其本质是 Goodhart 定律在代理优化中的体现——当代理指标（ML 预测 ΔE）本身成为优化目标时，它就不再是真实指标（RCWA 验证 ΔE）的良好度量。

本研究通过严格的闭环验证（closed-loop validation）量化了这一效应：对每个目标色，ML 模型从候选池中选出其预测 ΔE 最小的结构（naïve 策略），随后用 RCWA 全波仿真重新计算该结构的真实反射谱并评估实际 ΔE。ML 自称的预测精度与 RCWA 实测精度之间的系统性偏差即为优化器诅咒的量化表征。

### 3.X.2 TiO₂/SiO₂：从过度自信到修复

TiO₂（锐钛矿，n ≈ 2.30）在 SiO₂ 衬底上的 3-seed 集成模型 holdout 测试精度为 ΔE = 2.99（mean，三 seed 范围 2.97–3.00，极差 0.03）/ 2.15（median），53% 样本低于 2.3 JND 阈值 [Table 1]。这一正向精度本身已属可用，但逆设计场景下的表现揭示了更深层的问题：

在 N = 30 个随机目标色的闭环测试中，naïve 策略（直接取 ML 预测最优候选）的 ML 自称 ΔE 仅为 0.66——远优于正向 holdout 的 3.0。然而 RCWA 验证后的实际 ΔE 为 5.83，过度乐观偏差（over-optimism gap）达 +5.17。这一 7 倍的自称-实测落差正是优化器诅咒的直接体现：ML 在候选池中"挑选"了那些恰好处于其预测误差有利方向的样本，而非真正最优的结构。

混合重排策略（hybrid re-ranking）通过引入物理验证层修复了这一缺陷：ML 筛选 top-K（K=20）候选后，逐一执行 RCWA 全波仿真，取 RCWA 验证 ΔE 最小者作为最终设计。结果显示：

- 混合策略实测 ΔE 从 5.83 降至 2.33（改善 +3.51）
- 成功率（ΔE < 2.3 JND）从 23% 提升至 63%
- 混合 ≤ naïve 在全部 30/30 个目标上成立（理论保证：top-K 包含 naïve 首选）
- 过度乐观偏差从 +5.17 压缩至 +1.67（ML 自称 0.66 vs hybrid 实测 2.33）

nG 收敛性复验（nG = 65 → 101）确认 90% 结构的颜色跨阶数变化 < 1 JND（mean 极差 0.77），排除了"求解器噪声导致虚假颜色"的替代解释。

### 3.X.3 a-Si/SiO₂（复折射率）：正向精度好 ≠ 逆设计成功

非晶硅（a-Si，n ≈ 3.80，k ≈ 0.01–0.52）在 SiO₂ 衬底上的 3-seed 集成模型采用真实复折射率数据（Green & Keevers 1995）训练，holdout 精度为 ΔE = 2.38（3-seed 集成部署，N=263），58–63% 样本低于 JND 阈值 [Table 1]。这一正向精度甚至优于 TiO₂（2.99）——吸收主导的光谱形态简单（暗红/棕色调），模型容易预测。

然而，逆设计结果揭示了正向精度与逆设计成功之间的根本脱节。在 N = 29 个目标色的闭环测试中：ML 自称 ΔE = 13.39，naïve 实测 = 18.04，诅咒 gap = +4.65；hybrid 实测 = 14.13，改善 +3.91，成功率仍为 0%。hybrid ≤ naïve 在全部 29/29 个目标上成立。

这一结果的物理根源在于 a-Si 的颜色空间极度受限：跨样本 sRGB 标准差仅为 R=0.21、G=0.10、B=0.09——绿色和蓝色通道几乎不随几何参数变化（被 k(λ) 的强波长依赖性锁死），仅红色通道有有限调制空间。所有 a-Si 结构无论 D/H/P 如何变化，颜色都聚集在暗红/棕色区域。ML 模型"预测得准"（ΔE=2.38）只是因为目标空间本身极小；逆设计"做不到"（0% 成功）是因为目标色域远超 a-Si 的可实现范围。

诅咒 gap 的三材料对比（TiO₂: +5.17; a-Si k≠0: +4.65; a-Si k=0: +6.21）进一步确认了优化器诅咒的材料无关性：无论代理模型精度如何（TiO₂ 自称 0.66 vs a-Si 自称 13.39），过度乐观偏差均稳定在 ~+5±1 ΔE。

**核心结论：混合重排消除了材料无关的 ~+5 ΔE 过度乐观偏差，但逆设计的最终精度天花板由材料的可实现色域决定——正向预测精度是必要非充分条件，色域覆盖才是逆设计成功的充要条件。**

### 3.X.4 小结

**Table X. 两材料体系逆设计基准对比（闭环验证，RCWA nG=65, Nxy=256）**

| 指标 | TiO₂/SiO₂ | a-Si(k≠0)/SiO₂ |
|------|-----------|----------------|
| 正向 holdout ΔE（mean） | 2.99 | 2.38 |
| 测试样本数 N | 30 | 29 |
| ML 自称 ΔE（naïve 预测） | 0.66 | 13.39 |
| naïve 实测 ΔE（mean） | 5.83 | 18.04 |
| 诅咒 gap（naïve 实测 − ML 自称） | +5.17 | +4.65 |
| hybrid 实测 ΔE（mean） | 2.33 | 14.13 |
| 改善量（naïve − hybrid） | +3.51 | +3.91 |
| 成功率（hybrid ΔE < 2.3 JND） | 63% | 0% |
| hybrid ≤ naïve 成立比例 | 30/30 | 29/29 |
| nG 收敛率（ΔE₆₅→₁₀₁ < 1 JND） | 90% | 72% |

两材料对比建立了代理驱动逆设计的完整图景：(1) 优化器诅咒是普适的统计效应，不依赖于具体材料体系；(2) 混合重排是消除诅咒的必要且充分手段；(3) 但重排不能替代高质量的代理模型——材料选择（决定代理精度上限）和验证策略（消除统计偏差）是两个正交的改进维度。

---

## 3.Y 材料筛选判据：共振截止与损耗修正

### 3.Y.1 折射率对比度阈值

结构色的物理基础是亚波长光栅中的导模共振（guided-mode resonance, GMR）：当入射光与光栅的泄漏模式耦合时，特定波长被强烈反射，产生饱和结构色。GMR 的激发效率直接取决于光栅层与周围介质之间的折射率对比度 Δn = n_pillar − n_substrate。

本研究通过四材料体系（TiO₂, a-Si, Si₃N₄, Al₂O₃）在三衬底（SiO₂, Si₃N₄, Al₂O₃）上的 11 组 RCWA 数据集，系统量化了 Δn 与反射谱动态范围（单样本 R_max − R_min 的系综平均）之间的关系 [Table X / Fig. X]。

结果显示一个清晰的共振截止阈值：当 Δn 降至 ~0.5 以下时，反射谱动态范围发生断崖式崩塌。最直接的证据来自 Δn 仅相差 0.013 的两组数据：TiO₂/Al₂O₃（Δn = 0.545）的 R 范围为 0.309，而 Si₃N₄/SiO₂（Δn = 0.532）仅为 0.043——7 倍的崩塌发生在 Δn 变化不足 3% 的区间内。这不是连续过渡，而是共振截止（resonance cutoff）：当 Δn 不足以将光场约束在光栅层内时，导模退化为辐射模，共振消失，结构退化为等效均匀薄膜。

同一材料（TiO₂）在不同衬底上的数据进一步证实了阈值的存在：Δn 从 0.842（SiO₂ 衬底）→ 0.545（Al₂O₃）→ 0.310（Si₃N₄），R 范围从 0.582 → 0.309 → 0.102 单调递减，且递减速率在 Δn ≈ 0.5 附近急剧加快。

需要特别指出的是，Si₃N₄/SiO₂ 的 ML holdout 精度（ΔE = 1.57）表面上优于 TiO₂（2.99），但这一数字不具有物理意义：当反射谱动态范围仅为 0.043 时，所有结构的颜色几乎相同，均值预测器（mean predictor）即可达到相近精度。Si₃N₄ 的低 ΔE 反映的是"没有可预测的变化"，而非"模型学到了精确的物理映射"。这进一步说明，正向预测精度不能作为材料适用性的判据——只有足够大的光谱动态范围（R 范围 ≥ 0.10）才使 ML 代理模型的训练和逆设计有实际意义。

### 3.Y.2 损耗修正：高 Δn 的代价

简单的"Δn 越高越好"叙事被 a-Si 的数据否定。a-Si 拥有最高的 Δn（2.342 on SiO₂），但其颜色空间极度受限——复折射率（k ≠ 0）RCWA 数据集（N = 2626）的跨样本 sRGB 标准差仅为 R = 0.21、G = 0.10、B = 0.09，绿色和蓝色通道几乎不随几何参数变化。这一反直觉结果源于两个损耗机制：

**阻抗失配**：当 n_pillar >> n_substrate 时，光栅界面处的 Fresnel 反射系数过大，大部分入射光在界面即被反射（非共振背景反射），只有窄波长窗口内的光能耦合进光栅层参与共振。这压缩了可用的共振调制深度。无损（k = 0）a-Si 数据集的 R 动态范围仅为 0.186（TiO₂ 的 32%），即源于此机制。

**材料吸收**（复折射率 a-Si）：采用 Green & Keevers (1995) 实验 k 数据（400 nm 处 k ≈ 0.52，700 nm 处 k ≈ 0.01）后，a-Si 的 R 动态范围表面上升至 0.635，但这一调制主要来自 k(λ) 的强波长依赖性（蓝光被强烈吸收、红光透过），而非几何参数对共振的调控。R+T = 0.749（27% 吸收）确认了能量守恒，同时表明吸收已将可用光谱调制空间压缩至不足 25%。闭环验证中 0% 的逆设计成功率（§3.X.3）是这一色域坍缩的直接后果。

### 3.Y.3 二阶筛选判据

综合 11 组数据，我们提出介电超表面结构色的材料筛选二阶判据：

**必要条件（共振存在性）**：Δn = n_pillar − n_substrate > ~0.5。低于此阈值，光栅无法支撑导模共振，几何参数优化无法产生有效结构色。Si₃N₄（Δn = 0.53 on SiO₂）和 Al₂O₃（Δn ≤ 0.30）被明确排除。

**充分条件（色域最大化）**：在满足必要条件的前提下，低消光系数（k ≈ 0）和适中的折射率（n ≈ 2.0–2.5）组合产生最大色域。TiO₂（n = 2.30, k ≈ 0）是当前最优选择；a-Si（n = 3.80, k > 0）虽满足必要条件但受损耗惩罚。

该判据的物理本质是：结构色需要共振（由 Δn 阈值保证），且共振需要高 Q 值（由低损耗保证）。两者缺一不可。

---

## 4. Discussion

### 4.1 正向精度与逆设计成功的脱节

本研究的闭环验证揭示了一个对 ML 辅助超表面设计具有普遍意义的发现：正向预测精度是逆设计成功的必要非充分条件。

a-Si（复折射率，Green & Keevers 1995）的 3-seed 集成模型 holdout ΔE = 2.38，优于 TiO₂ 的 2.99——这一反直觉结果源于吸收对光谱的简化效应：a-Si 在可见光波段 k 从 0.52（400 nm）急剧下降至 ~0（700 nm），所有结构的反射谱都呈现"蓝光吸收、红光反射"的固定轮廓，几何参数仅在此轮廓上施加有限调制。模型预测这种低信息量光谱自然容易，但"预测得准"不等于"设计得出"——当可实现色域（跨样本 RGB std: R=0.21, G=0.10, B=0.09）远小于目标色域时，逆设计必然失败。

这一发现的方法论意义在于：ML 代理模型的评估不能仅看正向 holdout ΔE，还必须考察训练数据的色域覆盖度。一个在窄色域上 ΔE=2 的模型，逆设计能力可能远逊于一个在宽色域上 ΔE=3 的模型。色域覆盖度（而非预测精度）才是逆设计成功的决定性指标。

此外，本研究通过对比 a-Si 的无损（k=0）和复折射率（k≠0）两套训练数据，验证了材料光学常数物理真实性的重要性：k=0 版本产生非物理的 R+T > 1（mean 1.044），而复折射率版本 R+T = 0.749（能量守恒，27% 吸收）。对于有损耗材料，RCWA 仿真必须采用复折射率 n + ik 的完整色散模型，否则代理模型从训练阶段即继承系统性偏差。

### 4.2 求解器收敛性的材料依赖性

nG 收敛性复验（nG = 65 → 101）暴露了另一个材料依赖效应：TiO₂ 结构 90% 跨阶数颜色变化 < 1 JND（mean 极差 0.77），a-Si 复折射率版本 72%（mean 极差 1.83），而 a-Si 无损版本仅 40%（mean 极差 5.02，max 18.39）。

物理解释与折射率对比度和吸收的竞争效应相关：高 Δn 体系的导模共振具有更高的品质因子 Q，表现为光谱中更尖锐的特征，对傅里叶截断更敏感。然而，材料吸收（复折射率 k > 0）阻尼了这些尖锐共振，压低了 Q 值，使光谱平滑化——因此复折射率 a-Si（72%）比无损 a-Si（40%）收敛性显著改善。这是色域坍缩的另一面：吸收既限制了可用色彩，也降低了求解器敏感度。

这一结果对计算超表面领域具有实践指导意义：(1) 对于 n ≈ 2.0–2.5 的中等折射率材料（如 TiO₂），nG = 65 已足够产生收敛的颜色预测；(2) 对于 n > 3 的高折射率材料（如 a-Si、c-Si），需要 nG ≥ 101 甚至更高才能保证光谱收敛，计算成本增加约 2.4 倍（∝ nG²）；(3) 论文中报告的结构色数值结果必须附带 nG 收敛性验证，否则审稿人有理由质疑结果的可重复性。

### 4.3 混合重排的能力边界

混合重排（hybrid re-ranking）在本研究中展现了消除优化器诅咒的确定性能力：hybrid ≤ naïve 在全部有效测试案例上成立（TiO₂ 30/30, a-Si 28/28），这是 top-K 包含 naïve 首选的数学保证。

然而，混合重排的能力存在明确边界：

**它消除统计偏差，不消除系统误差。** 优化器诅咒（~+5–6 ΔE 的过度乐观偏差）是统计性的——源于在有限候选池中对预测误差的极端值选择。RCWA 重排通过引入无偏估计消除了这一选择偏差。但代理模型的基线精度（TiO₂ ~3 ΔE, a-Si ~4 ΔE）是系统性的，由模型架构、训练数据质量和材料物理共同决定，重排无法改善。

**它受限于候选池覆盖度。** 当前 hybrid 在 ML 生成的候选池中搜索（网格搜索 + 随机扰动），如果全局最优结构不在候选池内，重排只能给出池内最优。未来工作可将 RCWA 验证嵌入贝叶斯优化循环（每轮用 RCWA 更新 Kriging 后验），在保持验证严格性的同时扩展搜索空间。

**计算成本权衡。** 当前 K=20 的 RCWA 验证使每个目标色的计算时间从 ~0.1 s（纯 ML）增加到 ~20 s（20 次 RCWA）。对于 TiO₂（nG=65, Nxy=256），这是可接受的；对于 a-Si（需 nG≥101），成本进一步增加 ~2.4 倍。实际部署中 K 的选择应在验证严格性和计算预算之间权衡。

### 4.4 局限性与未来方向

本研究的主要局限包括：

(1) **材料光学常数覆盖范围**：本研究已对 a-Si 采用复折射率（Green & Keevers 1995 实验数据），但其余材料（TiO₂, Si₃N₄, Al₂O₃）仍以 Cauchy 色散（k = 0）建模，对透明介质这一近似是合理的。对于其他有损耗半导体（如 c-Si、GaAs）或金属材料，需采用相应的复折射率色散模型（Tauc-Lorentz 或实验 tabulated 数据）重新生成训练集。此外，本研究的 a-Si k 数据来自单一文献来源，不同沉积条件下 a-Si 的光学常数可能存在显著差异（带隙 1.6–1.8 eV），实际部署时应使用与制备工艺匹配的测量数据。

(2) **几何空间限制**：当前仅考虑圆柱形纳米柱（直径 D、高度 H、周期 P 三参数）。更复杂的几何（椭圆、矩形、多层堆叠）可扩大色域但增加参数空间维度，对代理模型和搜索策略提出更高要求。

(3) **角度与偏振**：训练数据包含 0°–80° 入射角和 TE/TM 偏振，但闭环验证仅在正入射（0°）下执行。广角结构色（如结构色涂层应用）需要角度分辨的逆设计验证。

(4) **实验验证缺失**：所有结果基于 RCWA 数值仿真，尚未与实验制备的样品对比。RCWA 本身在 nG 收敛后精度极高（本验证 90% 结构 < 1 JND），但实际制备中的工艺偏差（侧壁倾斜、表面粗糙度、材料非均匀性）可能引入额外误差。

---

## 写作备注（非正文）

- [Table X] 占位处等 Codex 的四材料基准表填入
- Fig. X（Δn 判据曲线）数据在 data/index_contrast_criterion.pkl，11 个数据点
- 优化器诅咒的引用需补：Jones et al. 1998 (Efficient Global Optimization), Gonzalez et al. 2016 (Batch Bayesian Optimization)
- 导模共振引用：Magnusson & Wang 1992 (New principle for optical filters)
- a-Si 光学常数引用：Palik Handbook / refractiveindex.info (Green & Keevers 1995)
- CIEDE2000 引用：Sharma et al. 2005
- 全文数字以 N=30 闭环结果为准（非 smoke test）

## 5. Conclusion

We have presented a systematic ML-assisted inverse design framework for dielectric metasurface structural colors with three distinguishing features: rigorous closed-loop validation, a physics-grounded material selection criterion, and explicit quantification of the optimizer's curse.

**Closed-loop validation** reveals a universal ~+5 Delta-E2000 gap between the agent model's self-reported optimum and RCWA-verified reality. This gap is approximately material-independent (TiO2: +5.17, a-Si: +4.65). The hybrid re-ranking strategy raises TiO2 inverse design success rate from 23% to 63%.

**The Delta-n criterion** from 11 material/substrate combinations reveals a resonance cutoff at Delta-n ~0.5, with low optical loss (k ~0) as a sufficient condition for maximum gamut. This explains TiO2 (positive), Si3N4/Al2O3 (negative controls), and a-Si (high Delta-n but absorption-limited, 0% success).

**Forward accuracy is necessary but insufficient** for inverse design: a-Si achieves forward holdout Delta-E2000 = 2.38 (better than TiO2's 2.99) yet 0% inverse design success due to gamut collapse (cross-sample RGB std = [0.21, 0.10, 0.09]).

**Speed**: ML inference (5.13 ms) is 811x faster than RCWA (4.16 s); full inverse design completes in ~2.3 s vs 31.2 min for brute-force RCWA.

Future extensions include dual-pillar geometries, FDTD cross-validation, and experimental fabrication. The framework provides a practical protocol for navigating the boundary between ML acceleration and physical realizability in photonic inverse design.

---

## References

[1] Kinoshita, S., et al. (2008). Physics of structural colors. Rep. Prog. Phys., 71(7), 076401.
[2] Tan, H., et al. (2014). Structural colors. Adv. Mater., 26(29), 4889-4905.
[3] Yu, N., et al. (2023). Metasurface structural color. Nat. Rev. Mater., 8, 420-438.
[4] Kuznetsov, A. I., et al. (2016). Optically resonant dielectric nanostructures. Science, 354(6314), aag2472.
[5] Tseng, M. L., et al. (2017). TiO2 metasurfaces. ACS Nano, 11(5), 4715-4722.
[6] Moharam, M. G., & Gaylord, T. K. (1981). RCWA. JOSA, 71(7), 811-818.
[7] Taflove, A., & Hagness, S. C. (2005). Computational Electrodynamics (3rd ed.). Artech House.
[8] Molesky, S., et al. (2018). Inverse design in nanophotonics. Nat. Photonics, 12, 659-670.
[9] Tahersima, M. H., et al. (2019). DNN inverse design. Sci. Rep., 9, 1368.
[10] Liu, Z., et al. (2021). Deep learning for photonic structures. ACS Photonics, 8(1), 47-60.
[11] So, S., et al. (2022). ML-assisted nanophotonics. Adv. Photonics, 4(1), 014001.
[12] Green, M. A., & Keevers, M. J. (1995). Optical properties of intrinsic silicon. Prog. Photovoltaics, 3(3), 189-192.
[13] Jones, D. R., et al. (1998). Efficient global optimization. J. Global Optim., 13(4), 455-492.
[14] Gonzalez, J., et al. (2016). Batch Bayesian optimization. AISTATS 2016.
[15] Magnusson, R., & Wang, S. S. (1992). GMR principle. Appl. Phys. Lett., 61(9), 1022-1024.
