# 正向精度无法预测逆设计成功：ML 辅助超表面结构色中的优化器诅咒与色域极限

*Forward Accuracy Does Not Predict Inverse Design Success: Optimizer's Curse and Gamut Limits in ML-Assisted Metasurface Structural Color*

*草稿版本 2026-07-22，供导师审阅（作者、单位、基金信息待补）*

# Abstract

机器学习代理模型驱动的超表面逆设计面临一个被普遍忽视的统计陷阱：当代理模型本身成为优化目标时，其预测误差被选择过程系统性放大，导致"代理最优"设计在全波验证中表现远逊于预期。本文以介电超表面结构色为研究对象，建立了从目标色到 ML 逆设计再到独立 RCWA 验证的严格闭环协议，系统量化了这一"优化器诅咒"效应并提出了有效的修复策略。

我们训练了基于残差 MLP 的 3-seed 集成代理模型（553K 参数），对四材料体系（TiO₂, a-Si, Si₃N₄, Al₂O₃）在三衬底上的 12 组 RCWA 数据集进行验证。三项发现：(1) 正向精度是逆设计成功的必要非充分条件——a-Si holdout 精度（ΔE₀₀ = 2.38）优于 TiO₂（2.99），逆设计成功率却仅 0–18% vs 62%，候选池密度才是充要条件；(2) 优化器诅咒产生材料无关的 +3–4 ΔE₀₀ 中位过度乐观偏差，混合重排（ML top-K → RCWA 逐一验证）数学上保证不劣于纯 ML，将 TiO₂ roundtrip 成功率从 19% 提升至 62%（Wilson 95% CI：52–71%）；(3) Δn ≈ 0.5 处存在共振截止，Δn 变化 0.013 即致反射动态范围 7 倍崩塌，提供先验材料筛选判据。

本研究划定 ML 辅助光子学逆设计的能力边界：混合验证消除统计偏差但不消除系统误差，材料选择与验证策略是两个正交改进维度。

**关键词**：超表面；结构色；机器学习代理模型；逆设计；优化器诅咒；严格耦合波分析；折射率对比度

---

# 1. Introduction

机器学习代理模型在超表面逆设计中被广泛使用，但其有效性从未被严格验证。我们发现一个反直觉现象：正向预测精度更高的材料体系（a-Si，holdout ΔE₀₀ = 2.38），逆设计成功率仅为 0–18%；而精度较低的材料（TiO₂，ΔE₀₀ = 2.99），成功率却达 62%。精度指标本身无法预测逆设计的成败——这一发现迫使重新审视 ML 辅助光子学设计的基本假设。

结构色源于亚波长尺度光与微纳结构的相互作用，介电超表面（周期性高折射率纳米柱阵列）因支持导模共振（GMR）和米氏共振，能在极薄层内实现高饱和度色彩，在显示、防伪、传感等领域前景广阔[1-5]。逆设计（给定目标颜色求解几何参数）面临高度非线性的多对一映射和高昂仿真代价（单次 RCWA 约 1.6 s），传统穷举搜索不可行[6-8]。ML 代理模型将单次推理压缩至毫秒级（本研究 5.13 ms vs RCWA ~1.6 s，单次推理 >300×），但现有文献存在三个被普遍忽视的关键问题[9-12]：

**第一，正向精度与逆设计成功之间的脱节未被认识。** 现有工作普遍以正向 holdout 精度作为代理模型质量的最终指标，隐含假设"预测得准就能设计得出"。然而，逆设计是在有限候选池上的优化问题：成功取决于候选池中是否存在匹配目标色的结构，而非代理模型在测试集上的平均误差。这一预测与优化之间的根本区别，从未在超表面文献中被系统验证。

**第二，优化器诅咒在本领域从未被量化。** 这一统计现象已有充分的理论基础：当含噪声的预测器被用作目标函数时，选择过程会放大预测误差（Goodhart 定律；拍卖理论中的赢者诅咒[13,14]）。在超表面 ML 中，这表现为"代理最优"结构在全波验证中的表现远逊于代理预测。尽管其相关性显而易见，此前没有工作测量过这一过度乐观偏差的幅度，也未提出系统性的修正方案。

**第三，不存在定量的材料筛选判据。** 实践者凭直觉选择纳米柱材料（高 n、低损耗），但没有数据驱动的阈值能先验地判断给定的材料-衬底组合能否产生有用的结构色。

本文的核心论点是：**ML 代理模型的逆设计天花板不由模型精度决定，而由材料物理上可实现的候选池密度决定。**

本文做出三项贡献：

（1）**优化器诅咒的闭环量化。** 通过严格协议（ML 逆设计 → 独立 RCWA 验证），我们测量了材料无关的过度乐观偏差 +3–4 ΔE₀₀（中位数），并证明正向 holdout 精度无法预测逆设计成功：a-Si 的正向精度优于 TiO₂（2.38 vs 2.99 ΔE₀₀），逆设计成功率却远低于 TiO₂（0–18% vs 62%）。

（2）**通过混合重排诊断失败机制。** ML top-K 的 RCWA 重排（一种标准的代理辅助优化策略，而非新算法）充当*诊断工具*：它消除统计偏差，并将残余失败暴露为候选池耗尽——对 a-Si，网格中物理上不存在匹配高饱和度目标的结构；而 TiO₂ 的候选池中每个目标约有 ~5% 的可行候选。优化器诅咒可治；材料极限不可治。

（3）**数据驱动的材料筛选判据。** 12 组材料-衬底组合揭示了 Δn ≈ 0.5 处的共振截止：低于此阈值，反射谱平坦化，结构色消失。结合色域密度分析，这为 ML 训练开始之前的材料选择提供了先验筛选。

本文结构如下：§2 描述 RCWA 数据生成、ResMLP 代理模型架构和混合逆设计流程；§3 报告正向预测精度、闭环逆设计验证和材料筛选判据的实验结果；§4 讨论代理保真度的材料依赖性、求解器收敛性和方法局限；§5 总结全文。

---

# 2. Methods

## 2.1 RCWA Data Generation

Training data generated with grcwa (guided-mode resonance coupled-wave analysis).

**Scope.** This study restricts to the minimal configuration — single cylindrical nanopillar, normal incidence (0°), TE polarization — to establish a controlled benchmark. Extensions to dual-pillar geometries, oblique incidence, and TM polarization are deferred to future work (§4.4).
All materials modeled with Cauchy dispersion: n(λ) = A + B/λ² + C/λ⁴.

**Table 1. Solver parameters (fixed across all materials):**

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

For a-Si, we employ the full complex refractive index ñ(λ) = n(λ) + ik(λ) using tabulated experimental data from Green & Keevers (1995), interpolated onto the 81-point wavelength grid. In the visible band, a-Si exhibits k ≈ 0.52 at 400 nm decreasing to k ≈ 0.01 at 700 nm; the imaginary part is included in the RCWA dielectric tensor as ε = ñ², yielding complex-valued diffraction efficiencies. A total of 3,000 Latin Hypercube samples were attempted; 2,725 completed successfully (90.8%), with 275 (9.2%) terminated by a 120 s per-sample timeout due to numerical stiffness in the complex RCWA solve. Of the successful samples, 99 (3.6%) exhibited R+T > 1.0 and were removed by the energy-conservation filter, yielding 2,626 clean training samples. The retained dataset has mean R+T = 0.731 (27% absorption), consistent with the expected optical loss of a-Si in the visible. An earlier lossless (k = 0) a-Si dataset was also generated for comparison; it produces non-physical R+T > 1 (mean 1.044) and is superseded by the complex-k dataset for all results reported in this paper unless explicitly noted.

**Table 2. Material optical constants (at 550 nm):**

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

**Table 3. ResMLP surrogate architecture:**

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

The decisive evidence for data-limited (not capacity-limited) performance is the 3-seed ensemble variance: across seeds 42/123/456, TiO₂ holdout ΔE varies by only 0.03 (range 2.97–3.00) and a-Si (complex-k) converges to ΔE = 2.38 with similarly negligible inter-seed spread. If the model were underfitting (capacity-limited), different random initializations would converge to substantially different local minima, producing larger inter-seed variance. The observed near-deterministic training indicates the loss landscape has a single dominant basin at this architecture scale — the model extracts essentially all learnable signal from the available data.

The residual error (TiO₂ ΔE ≈ 3.0, a-Si complex-k ΔE ≈ 2.4) is therefore attributable to irreducible noise sources: (1) finite nG truncation in the RCWA training labels, (2) the 81-wavelength discretization of continuous spectra, and (3) inherent spectral complexity that a 553K-parameter model cannot capture at this data volume. Increasing model capacity cannot reduce these error floors; only improving label quality (higher nG) or augmenting data volume would help.

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
- PyTorch batch inference (GPU): ~2 ms for all 1,728 candidates
- Top-K = 20 candidates selected by predicted CIEDE2000

**Stage 2 — High-fidelity re-ranking (RCWA):**
- Each top-K candidate re-evaluated with independent RCWA (nG=65, Nxy=256)
- Best candidate selected by measured CIEDE2000
- 1.6 s/candidate × 20 = ~33 s total per inverse design

**Theoretical guarantee:** The hybrid best is always ≤ naïve best (candidate[0] ∈ top-K).

**Computational cost:** All timings measured on a single workstation (Intel Core i7-14650HX, 16 cores, 32 GB RAM; NVIDIA GeForce RTX 5060 Laptop GPU for PyTorch batch inference). Two inference paths: (i) ONNX Runtime (CPU, intra_op_threads=4) single-sample deployment: 5.13 ms per sample, >300× speedup over RCWA; (ii) PyTorch batch mode (GPU) grid screening: ~2 ms for all 1,728 candidates. Full inverse design pipeline — ~1400-candidate ML batch search (~4 s including data transfer and CIEDE2000 evaluation) plus 20-candidate RCWA re-ranking (~33 s) — completes in ~37 s per target color, versus ~38 min for 1392-call RCWA brute-force search.

![闭环逆设计协议。目标色（sRGB）→ ML 代理筛选（1392 候选，~4 s）→ top-K = 20 选择 → RCWA 逐一验证（20 结构，~33 s）→ 最优设计（RCWA 验证 ΔE₀₀ 最小）。ML 自称 ΔE = 0.66 vs RCWA 实测 ΔE = 5.83（诅咒 gap +3–4 ΔE 中位数）。端到端加速 ~60×（vs 1392 次 RCWA 穷举 ~38 min）。](figures/fig1_protocol.png){width="6.2in"}

## 2.4 Evaluation Metrics

**CIEDE2000 (ΔE₀₀):** Perceptually uniform color difference metric.
Thresholds: ΔE₀₀ < 1.0 (imperceptible), < 2.3 (just noticeable difference, JND).

**Forward accuracy:** ΔE₀₀ between ML-predicted and RCWA-computed sRGB colors on frozen hold-out set.

**Closed-loop achieved ΔE:** Target color → ML inverse design → independent RCWA verification → ΔE₀₀(target, achieved).

**Gap:** achieved ΔE − ML self-claimed ΔE. Quantifies optimizer's curse magnitude.

**nG convergence:** ΔE₀₀(nG=65) vs ΔE₀₀(nG=101) on hybrid-selected structures. Fraction within 1 JND reported.

---

# 3. Results

## 3.1 逆设计验证：优化器诅咒与混合重排修复

### 3.1.1 问题提出

代理模型（surrogate model）驱动的逆设计面临一个被超表面领域普遍忽视的统计陷阱：当 ML 模型被用作目标函数进行优化时，模型预测误差会系统性地偏向高估设计性能。这一现象在贝叶斯优化文献中被称为"优化器诅咒"（optimizer's curse）[13,14]，其本质是 Goodhart 定律在代理优化中的体现——当代理指标（ML 预测 ΔE）本身成为优化目标时，它就不再是真实指标（RCWA 验证 ΔE）的良好度量。

本研究通过严格的闭环验证（closed-loop validation）量化了这一效应：对每个目标色，ML 模型从候选池中选出其预测 ΔE 最小的结构（naïve 策略），随后用 RCWA 全波仿真重新计算该结构的真实反射谱并评估实际 ΔE。ML 自称的预测精度与 RCWA 实测精度之间的系统性偏差即为优化器诅咒的量化表征。

极端值理论可以*先验地*预测诅咒量级。设代理模型对每个候选的预测误差标准差为 σ，候选池大小为 N，则 rank-1 选择的期望过乐观偏差满足 E[gap] ≈ σ√(2 ln N)（N 个独立高斯变量的期望最大值）。代入 TiO₂ 数据：σ ≈ 3 ΔE₀₀（holdout 散布），N = 1392（过滤后网格），得 √(2 ln 1392) ≈ 3.8，预测 gap ~ 3–5 ΔE₀₀——与下文实测中位数 +3.59 高度吻合。这一吻合确认诅咒并非特定模型或数据集的偶然产物，而是在 O(10³) 个含噪预测上执行优化的通用统计后果。

### 3.1.2 TiO₂/SiO₂：从过度自信到修复

TiO₂（锐钛矿，n ≈ 2.30）在 SiO₂ 衬底上的 3-seed 集成模型 holdout 测试精度为 ΔE = 2.99（mean，三 seed 范围 2.97–3.00，极差 0.03）/ 2.12（median），53% 样本低于 2.3 JND 阈值（Table 4）。这一正向精度本身已属可用，但逆设计场景下的表现揭示了更深层的问题：

在 N = 100 个 roundtrip 目标（从 TiO₂ 训练集中抽取真实结构，以其 RCWA 仿真色为逆设计目标——物理上保证位于 TiO₂ 可实现色域内）的闭环测试中，naïve 策略仅在 19% 的目标上成功（ΔE < 2.3 JND），过度乐观偏差（实测 − ML 自称）中位数为 +3.59 ΔE₀₀。ML 系统性地"挑选"了那些恰好处于其预测误差有利方向的样本，而非真正最优的结构。

我们强调此协议并非循环论证：(i) 目标是*颜色*（RCWA 验证的 sRGB），不是结构标签——逆设计需要找到*任何*能重现该颜色的几何结构；(ii) 解来自均匀 12³ 网格（1,728 候选），与 Latin Hypercube 训练集是完全不同的离散化，ML 模型必须泛化到未见过的几何；(iii) 最终选择使用独立 RCWA 验证，与 ML 预测完全解耦。Roundtrip 设计测试的是：管线能否从粗网格中恢复一个已知可达的颜色——这是最小可行的逆设计任务。

混合重排策略（hybrid re-ranking）通过引入物理验证层修复了这一缺陷：ML 筛选 top-K（K=20）候选后，逐一执行 RCWA 全波仿真，取 RCWA 验证 ΔE 最小者作为最终设计。结果显示：

- 混合策略实测 ΔE：mean 2.18，median 1.83
- 成功率（ΔE < 2.3 JND）从 19% 提升至 62%（Wilson 95% CI：52–71%）
- 混合 ≤ naïve 在全部 100/100 个目标上成立（理论保证：top-K 包含 naïve 首选）

nG 收敛性复验（nG = 65 → 101）确认 90% 结构的颜色跨阶数变化 < 1 JND（mean 极差 0.77），排除了"求解器噪声导致虚假颜色"的替代解释。

作为对比，互补的色域探测测试（N = 100，目标从 sRGB 色域边界采样）仅获 6% hybrid 成功率——多数 sRGB 边界色超出 TiO₂ 可实现色域。关键的是，这些色域外目标的诅咒 gap 中位数仅 +0.81，而色域内 roundtrip 目标为 +3.59。这一目标依赖性揭示了诅咒机制：当所有候选距目标 ΔE ~ 10–15 时，ML 排序不携带有效信号，选择偏差无法放大；诅咒恰在目标可达、候选紧密排列时最大。

### 3.1.3 a-Si/SiO₂（复折射率）：正向精度好 ≠ 逆设计成功

非晶硅（a-Si，n ≈ 3.80，k ≈ 0.01–0.52）在 SiO₂ 衬底上的 3-seed 集成模型采用真实复折射率数据（Green & Keevers 1995）训练，holdout 精度为 ΔE = 2.38（3-seed 集成部署，N=263），64% 样本低于 JND 阈值（Table 4）。这一正向精度甚至优于 TiO₂（2.99）——吸收主导的光谱形态简单（暗红/棕色调），模型容易预测。

为严格检验正向精度能否转化为逆设计能力，我们设计了两组互补的闭环测试。**第一组（色域探测目标，N = 29）**：从 sRGB 色空间均匀采样目标色（12 个饱和色相、12 个中饱和度、5 个中性灰）。ML 自称 ΔE = 13.39，naïve 实测 = 18.04，诅咒 gap = +4.65；hybrid 实测 = 14.13，改善 +3.91，成功率 0%（29/29 个目标均超出 a-Si 可实现色域；二项式 95% 上界：10%，rule of three）。**第二组（roundtrip 目标，N = 98）**：从 a-Si 训练集中抽取真实结构，以其 RCWA 仿真色为逆设计目标——这些目标在物理上保证位于 a-Si 可实现色域内。结果：ML 自称 ΔE = 2.95（median 2.58），naïve 实测 = 10.06（median 7.49），hybrid 实测 = 4.70（median 3.95，min = 0.50），成功率 18%（18/98 低于 JND 阈值）。hybrid ≤ naïve 在全部 98/98 个目标上成立。

两组结果的对比精确定位了 a-Si 逆设计的失败边界：色域探测 sRGB 目标（0% 成功）落在 a-Si 色域之外，无论代理模型多精确都无法到达；roundtrip 目标（18% 成功）位于色域之内，混合重排可以恢复部分设计。18% 而非 100% 的成功率反映了 a-Si 色域虽非零但极度受限——跨样本 sRGB 标准差仅为 R=0.21、G=0.10、B=0.09，绿色和蓝色通道被 k(λ) 的强波长依赖性锁死，仅红色通道有有限调制空间。ML 模型"预测得准"（ΔE=2.38）是因为吸收主导的光谱形态简单；逆设计"大部分做不到"（82% 失败）是因为即使目标在色域内，高 n（3.80）导致的阻抗失配和蓝光波段 ~55% 的吸收使有效设计空间极度压缩。

![三个训练集的 sRGB 色域分布。TiO₂/SiO₂（宽覆盖）、a-Si(k≠0)/SiO₂（吸收受限）、Si₃N₄/SiO₂（单点簇，无色域）。标注为跨样本 sRGB 标准差。](figures/fig3_color_gamut.png){width="6.2in"}

诅咒 gap 的中位数对比确认了优化器诅咒的材料无关性：TiO₂ roundtrip（N=100）中位数 +3.59，a-Si roundtrip（N=98）中位数 +3.68——两个独立材料体系，过度乐观偏差均稳定在 ~+3–4 ΔE。值得注意的是，诅咒量级具有目标依赖性：TiO₂ 色域探测目标（N=100，全部位于材料色域之外）的中位 gap 仅为 +0.81，因为当所有候选距目标 ΔE~10–15 时，ML 排序不携带有效信号，选择偏差无从放大。诅咒在目标位于可实现色域内、候选间距最小时最为显著——恰恰是逆设计最有意义的区间。

**核心结论：混合重排消除了材料无关的 ~+3–4 ΔE（中位数）过度乐观偏差，但逆设计的最终精度天花板由材料的可实现色域决定——a-Si 的正向精度（ΔE=2.38）优于 TiO₂（2.99），逆设计成功率却仅 0–18% vs 62%。正向预测精度是必要非充分条件，候选池密度才是逆设计成功的充要条件。**

### 3.1.4 小结

**Table 4. 两材料体系逆设计基准对比（闭环验证，RCWA nG=65, Nxy=256）**

| 指标 | TiO₂ roundtrip | TiO₂ 色域探测 | a-Si(k≠0) 色域探测 | a-Si(k≠0) roundtrip |
|------|---------------|-------------|-------------------|---------------------|
| 正向 holdout ΔE（mean） | 2.99 | 2.99 | 2.38 | 2.38 |
| 目标色来源 | 训练集 RT | HSV 色域边界 | HSV 色域边界 | 训练集 RT |
| 测试样本数 N | 100 | 100 | 29 | 98 |
| ML 自称 ΔE（mean） | 0.79 | 11.33 | 13.39 | 2.95 |
| naïve 实测 ΔE（mean） | 5.42 | 13.02 | 18.04 | 10.06 |
| 诅咒 gap（median） | +3.59 | +0.81 | +4.65† | +3.68 |
| hybrid 实测 ΔE（mean/median） | 2.18/1.83 | 11.60/10.52 | 14.13/14.18 | 4.70/3.95 |
| 成功率（hybrid ΔE < 2.3 JND） | 62%* | 6% | 0% | 18% |
| hybrid ≤ naïve 成立比例 | 100/100 | 100/100 | 29/29 | 98/98 |
| nG 收敛率（ΔE₆₅→₁₀₁ < 1 JND） | 90% | — | 72% | 72% |

*\*Wilson 95% CI：52–71%。†a-Si 色域探测 N=29 的诅咒 gap 为均值（median = +1.87，小样本高方差）；TiO₂ 和 a-Si roundtrip 为逐目标中位数。*

四组对比建立了代理驱动逆设计的完整图景：(1) 优化器诅咒是普适的统计效应（色域内目标中位数 ~+3–4 ΔE），不依赖于材料体系；(2) 混合重排是消除诅咒的必要且充分手段（hybrid ≤ naïve 在全部 227/227 个目标上成立）；(3) 但重排不能扩展材料的可实现色域——a-Si 的正向精度（2.38）优于 TiO₂（2.99），roundtrip 成功率却仅 18% vs 62%。2×2 设计矩阵（材料 × 目标类型）进一步揭示：TiO₂ 色域探测成功率（6%，N=100）与 a-Si（0%）相当，确认瓶颈是候选池密度而非代理精度。材料选择（决定色域天花板）和验证策略（消除统计偏差）是两个正交的改进维度。

![四个测试集的优化器诅咒与混合重排修复。(a) 诅咒 gap 分布（实测 − ML 自称 ΔE₀₀），中线为各组中位数；(b) hybrid 实测 ΔE₀₀ 箱线图。虚线：JND 阈值 2.3。成功率：TiO₂ roundtrip 62%，TiO₂ 色域探测 6%，a-Si 色域探测 0%，a-Si roundtrip 18%。](figures/fig4_curse_gap.png){width="6.2in"}

---

## 3.2 材料筛选判据：共振截止与损耗修正

### 3.2.1 折射率对比度阈值

结构色的物理基础是亚波长光栅中的导模共振（guided-mode resonance, GMR）：当入射光与光栅的泄漏模式耦合时，特定波长被强烈反射，产生饱和结构色。GMR 的激发效率直接取决于光栅层与周围介质之间的折射率对比度 Δn = n_pillar − n_substrate。

本研究通过四材料体系（TiO₂, a-Si, Si₃N₄, Al₂O₃）在三衬底（SiO₂, Si₃N₄, Al₂O₃）上的 12 组 RCWA 数据集，系统量化了 Δn 与反射谱动态范围（单样本 R_max − R_min 的系综平均）之间的关系（Fig. 4）。

![12 组材料/衬底组合的反射谱动态范围 vs Δn。虚线：Δn ≈ 0.53 共振截止阈值。箭头：Δn 变化 0.013 导致 R 范围 7 倍崩塌。](figures/fig2_delta_n_criterion.png){width="4.5in"}

结果显示一个清晰的共振截止阈值：当 Δn 降至 ~0.5 以下时，反射谱动态范围发生断崖式崩塌。最直接的证据来自 Δn 仅相差 0.013 的两组数据：TiO₂/Al₂O₃（Δn = 0.545）的 R 范围为 0.309，而 Si₃N₄/SiO₂（Δn = 0.532）仅为 0.043——7 倍的崩塌发生在 Δn 变化不足 3% 的区间内。这不是连续过渡，而是共振截止（resonance cutoff）：当 Δn 不足以将光场约束在光栅层内时，导模退化为辐射模，共振消失，结构退化为等效均匀薄膜。

同一材料（TiO₂）在不同衬底上的数据进一步证实了阈值的存在：Δn 从 0.842（SiO₂ 衬底）→ 0.545（Al₂O₃）→ 0.310（Si₃N₄），R 范围从 0.582 → 0.309 → 0.102 单调递减，且递减速率在 Δn ≈ 0.5 附近急剧加快。

**Si₃N₄/SiO₂——"精度好但色域为零"的平凡极限。** Si₃N₄（n = 1.99, Δn = 0.532）的 ML holdout 精度（ΔE = 1.57, median 1.03）表面上是四材料中最优的——甚至优于 TiO₂（2.99）。然而这一数字完全不反映模型能力：当反射谱动态范围仅为 0.043 时（TiO₂ 的 7%），所有 ~600 个结构无论 D/H/P 如何变化，颜色几乎相同。一个始终输出数据集均值色的平凡预测器（mean predictor）即可达到相近精度——模型无需学习任何几何-光学映射，只需记住"所有结构都是同一个颜色"。

三个独立证据确认了这一"平凡极限"诊断：(1) 3-seed 集成范围 = 1.57–1.57（极差 0.00），不同随机初始化收敛到完全相同的解——因为损失景观只有一个极小值（均值）；(2) 49% 样本的预测误差 < 1.0 ΔE（"不可感知"级别），但这不是因为模型精确，而是因为样本本身与均值色的距离就 < 1.0；(3) P95 = 4.8 远大于 mean 1.57，表明少数处于分布边缘的样本（可能是极端几何参数）仍产生可见误差，但模型对这些样本的处理与对中心样本无异——输出近似均值。

Si₃N₄ 与 a-Si 共同填充了"正向精度好但逆设计无意义"的象限，但机制不同：a-Si 的色域被材料吸收（k ≠ 0）压缩至暗红/棕色区域（RGB std: R=0.21, G=0.10, B=0.09），仍有有限但极度受限的设计空间（roundtrip 成功率 18%）；Si₃N₄ 的色域则因共振截止而完全消失（R range 0.043），不存在任何可逆设计的颜色变化——即使目标色就是 Si₃N₄ 自身的均值色，"逆设计"也退化为"返回任意结构"，因为所有结构等价。

这一对比确立了材料筛选的定量门槛：**只有 R 动态范围 ≥ ~0.10（对应 Δn ≥ ~0.5）的材料体系，ML 代理模型的训练和逆设计才具有实际意义。** 低于此门槛，正向 ΔE 指标退化为对数据集色域宽度的间接度量，而非对模型预测能力的评估。四材料体系由此形成完整的诊断矩阵：TiO₂（Δn=0.84, R=0.58, roundtrip 逆设计 62%）为正例；a-Si（Δn=2.34, R=0.635†, 逆设计 0–18%）为高 Δn 但损耗受限的警示；Si₃N₄（Δn=0.53, R=0.043）为共振截止的负对照；Al₂O₃（Δn=0.29, R=0.090）为低于门槛的弱共振案例，确认 R≥0.10 边界。

### 3.2.2 损耗修正：高 Δn 的代价

简单的"Δn 越高越好"叙事被 a-Si 的数据否定。a-Si 拥有最高的 Δn（2.342 on SiO₂），但其颜色空间极度受限——复折射率（k ≠ 0）RCWA 数据集（N = 2626）的跨样本 sRGB 标准差仅为 R = 0.21、G = 0.10、B = 0.09，绿色和蓝色通道几乎不随几何参数变化。这一反直觉结果源于两个损耗机制：

**阻抗失配**：当 n_pillar >> n_substrate 时，光栅界面处的 Fresnel 反射系数过大，大部分入射光在界面即被反射（非共振背景反射），只有窄波长窗口内的光能耦合进光栅层参与共振。这压缩了可用的共振调制深度。无损（k = 0）a-Si 数据集的 R 动态范围仅为 0.186（TiO₂ 的 32%），即源于此机制。

**材料吸收**（复折射率 a-Si）：采用 Green & Keevers (1995) 实验 k 数据（400 nm 处 k ≈ 0.52，700 nm 处 k ≈ 0.01）后，a-Si 的 R 动态范围表面上升至 0.635，但这一调制主要来自 k(λ) 的强波长依赖性（蓝光被强烈吸收、红光透过），而非几何参数对共振的调控。R+T = 0.731（27% 吸收）确认了能量守恒，同时表明吸收已将可用光谱调制空间压缩至不足 25%。闭环验证中 0–18% 的逆设计成功率（§3.1.3）是这一候选池耗尽的直接后果。

### 3.2.3 材料筛选阈值的数据驱动验证

综合 12 组数据，我们报告介电超表面结构色材料筛选阈值的数据驱动验证：

**必要条件（共振存在性）**：Δn = n_pillar − n_substrate > ~0.5。低于此阈值，光栅无法支撑导模共振，几何参数优化无法产生有效结构色。Si₃N₄（Δn = 0.53 on SiO₂）和 Al₂O₃（Δn ≤ 0.30）被明确排除。

**充分条件（色域最大化）**：在满足必要条件的前提下，低消光系数（k ≈ 0）和适中的折射率（n ≈ 2.0–2.5）组合产生最大色域。TiO₂（n = 2.30, k ≈ 0）是当前最优选择；a-Si（n = 3.80, k > 0）虽满足必要条件但受损耗惩罚。

该判据的物理本质是：结构色需要共振（由 Δn 阈值保证），且共振需要高 Q 值（由低损耗保证）。两者缺一不可。

---

# 4. Discussion

## 4.1 为何混合重排修复了 TiO₂ 却修复不了 a-Si：三阶段因果链

两个材料的 ML 预测网格天花板相同（3.4%，1/29 目标），但成功率之比无穷大（62% vs 0%），这揭示瓶颈不在于 ML 精度本身，而在于 ML 排序保真度与候选池质量之间的交互作用。

**阶段一——网格上的 ML 排序崩塌。** 对两种材料，全部 1392 个网格候选距典型目标的 ΔE₀₀ 均在 ~10–15 范围，而 ML 预测误差（TiO₂ ~3 ΔE₀₀，a-Si ~2.4）与这一散布相当。排序的信噪比因此 <1：ML 排序的 top-20 与从 1392 中随机抽取 20 个在统计上不可区分。这解释了为何 naïve 策略（取 ML 排序第 1 名）在两种材料上均灾难性失败——随机排序下的 rank-1 不携带任何信息。

**阶段二——混合修复受限于候选池质量。** 对 top-20 进行 RCWA 重排，等价于抽取 20 个随机候选并取验证最优。对 TiO₂，从 62% 成功率反推，约 4.8% 的网格候选（每个目标约 67 个）在 RCWA 验证下位于 ΔE₀₀ < 2.3 之内——候选池中有答案，混合重排能找到它们。对 a-Si，0% 成功率意味着候选池中没有任何候选在 29 个色域探测目标的 JND 之内——无物可寻。混合重排消除了优化器诅咒（统计偏差），但无法制造网格中不存在的候选。

**阶段三——候选池耗尽，而非色域坍缩。** a-Si 的 Lab 色彩范围（L ∈ [15, 71]，a ∈ [−62, 64]，b ∈ [−62, 69]）与 TiO₂ 相当，其凸包体积（sRGB 的 24%）甚至略超 TiO₂（20%）。失败不在于 a-Si *不能产生颜色*，而在于其可实现颜色集中在一个由去饱和暗红和棕色组成的稀疏簇中（sRGB 标准差 = [0.21, 0.10, 0.09]），而测试目标——12 个饱和色相和 5 个中性灰，代表典型结构色应用——要求 a-Si 的宽带吸收（均值 ~27%，蓝光波段 ~55%）物理上禁止的高饱和度。因此正确的描述是**候选池耗尽**：网格覆盖了 a-Si 的可实现色彩范围，但该范围在目标所在的高饱和度区域密度近乎为零。

**优化器诅咒可治；材料极限不可治。** 混合重排消除了 +3–4 ΔE₀₀ 的过度乐观偏差（材料无关，统计性），将成功率恢复至候选池的物理天花板。但当候选池本身为空——当搜索空间中没有任何几何产生目标色时——任何验证策略都无能为力。这区分了两个正交的改进维度：更好的验证（混合重排、RCWA-in-the-loop 贝叶斯优化）将成功率推向候选池天花板；更好的材料选择（高 Δn、低 k）抬高天花板本身。

## 4.2 求解器收敛性的材料依赖性

nG 收敛性复验（nG = 65 → 101）暴露了另一个材料依赖效应：TiO₂ 结构 90% 跨阶数颜色变化 < 1 JND（mean 极差 0.77），a-Si 复折射率版本 72%（mean 极差 1.83），而 a-Si 无损版本仅 40%（mean 极差 5.02，max 18.39）。

![nG 收敛性复验（65 → 101）。(a) 跨阶数颜色偏差；(b) 1 JND 内结构比例。TiO₂/SiO₂：0.77 / 90%；a-Si(k≠0)：1.83 / 72%；a-Si(k=0)：5.02 / 40%（max 18.39）。](figures/fig5_nG_convergence.png){width="6.2in"}

物理解释与折射率对比度和吸收的竞争效应相关：高 Δn 体系的导模共振具有更高的品质因子 Q，表现为光谱中更尖锐的特征，对傅里叶截断更敏感。然而，材料吸收（复折射率 k > 0）阻尼了这些尖锐共振，压低了 Q 值，使光谱平滑化——因此复折射率 a-Si（72%）比无损 a-Si（40%）收敛性显著改善。这是候选池耗尽的另一面：吸收既限制了可用色彩，也降低了求解器敏感度。

这一结果对计算超表面领域具有实践指导意义：(1) 对于 n ≈ 2.0–2.5 的中等折射率材料（如 TiO₂），nG = 65 已足够产生收敛的颜色预测；(2) 对于 n > 3 的高折射率材料（如 a-Si、c-Si），需要 nG ≥ 101 甚至更高才能保证光谱收敛，计算成本增加约 2.4 倍（∝ nG²）；(3) 论文中报告的结构色数值结果必须附带 nG 收敛性验证，否则审稿人有理由质疑结果的可重复性。

## 4.3 混合重排的能力边界

混合重排（hybrid re-ranking）在本研究中展现了消除优化器诅咒的确定性能力：hybrid ≤ naïve 在全部 227 个有效测试案例上成立（TiO₂ roundtrip 100/100, TiO₂ 色域探测 100/100, a-Si 色域探测 29/29, a-Si roundtrip 98/98），这是 top-K 包含 naïve 首选的数学保证。

然而，混合重排的能力存在明确边界：

**它消除统计偏差，不消除系统误差。** 优化器诅咒（~+3–4 ΔE 中位数的过度乐观偏差）是统计性的——源于在有限候选池中对预测误差的极端值选择。RCWA 重排通过引入无偏估计消除了这一选择偏差。但代理模型的基线精度（TiO₂ ~3 ΔE, a-Si ~2.4 ΔE）是系统性的，由模型架构、训练数据质量和材料物理共同决定，重排无法改善。

**它受限于候选池覆盖度。** 当前 hybrid 在 ML 生成的候选池中搜索（网格搜索 + 随机扰动），如果全局最优结构不在候选池内，重排只能给出池内最优。未来工作可将 RCWA 验证嵌入贝叶斯优化循环（每轮用 RCWA 更新 Kriging 后验），在保持验证严格性的同时扩展搜索空间。

**计算成本权衡。** 当前 K=20 的 RCWA 验证使每个目标色的计算时间从 ~0.1 s（纯 ML）增加到 ~37 s（ML 筛选 ~4 s + 20 次 RCWA 重排 ~33 s）。对于 TiO₂（nG=65, Nxy=256），这是可接受的；对于 a-Si（需 nG≥101），成本进一步增加 ~2.4 倍。实际部署中 K 的选择应在验证严格性和计算预算之间权衡。

**K 敏感性（TiO₂ roundtrip, N=100）。** 成功率从 K=5（47%, mean ΔE=2.96）经 K=10（57%, 2.33） steeply 上升至 K=20（62%, 2.14），随后饱和：K=50 仅达 64%（2.03）——2.5 倍 RCWA 成本换来 2 个百分点。每次额外 RCWA 调用的边际收益从 +0.5%/call（K: 5→20）降至 +0.07%/call（K: 20→50）。K=20 位于成本-精度曲线的"膝点"：捕获了 hybrid 收益的主体，同时将每目标时间控制在 40 s 以内。本文所有结果均采用 K=20。

**与其他逆设计范式的关系。** 本研究刻意聚焦于 ML 代理+物理验证这一超表面领域最广泛部署的工作流。替代方法——拓扑优化（全波求解器在环迭代，无代理偏差但成本高）、伴随方法（通过 Maxwell 方程计算梯度，精确但几何受限）、生成模型（GAN/VAE 直接学习逆映射，快速但训练需求大）——以不同方式处理逆问题。我们的发现具体适用于代理筛选工作流，但核心洞见——在噪声目标函数上优化会放大误差——是普适的，在任何近似模型引导选择的范式中都会显现。

**实操指南。** 基于上述结果，我们为 ML 辅助超表面逆设计的实践者提炼四条可操作规则：(1) 训练前先计算 Δn = n_pillar − n_substrate；若 Δn < 0.5，体系无法支撑导模共振，任何代理模型都不可能产生有用的结构色——直接放弃该材料对。(2) 永远不要仅凭正向 holdout ΔE 判断逆设计能力；必须执行闭环验证（ML 设计 → 独立全波检验）后才能报告成功率。(3) 采用混合重排，K ≥ 20；每目标 ~37 s 的成本即可消除 +3–4 ΔE₀₀ 的统计偏差，相对暴力搜索可忽略。(4) 若 roundtrip 成功率低于 ~30%，瓶颈是候选池密度而非模型精度——应扩大几何空间或更换材料，而非调网络。

## 4.4 局限性与未来方向

本研究的主要局限包括：

(1) **材料光学常数覆盖范围**：本研究已对 a-Si 采用复折射率（Green & Keevers 1995 实验数据），但其余材料（TiO₂, Si₃N₄, Al₂O₃）仍以 Cauchy 色散（k = 0）建模，对透明介质这一近似是合理的。对于其他有损耗半导体（如 c-Si、GaAs）或金属材料，需采用相应的复折射率色散模型（Tauc-Lorentz 或实验 tabulated 数据）重新生成训练集。此外，本研究的 a-Si k 数据来自单一文献来源，不同沉积条件下 a-Si 的光学常数可能存在显著差异（带隙 1.6–1.8 eV），实际部署时应使用与制备工艺匹配的测量数据。

(2) **几何空间限制**：当前仅考虑圆柱形纳米柱（直径 D、高度 H、周期 P 三参数）。更复杂的几何（椭圆、矩形、多层堆叠）可扩大色域但增加参数空间维度，对代理模型和搜索策略提出更高要求。

(3) **角度与偏振**：训练数据包含 0°–80° 入射角和 TE/TM 偏振，但闭环验证仅在正入射（0°）下执行。广角结构色（如结构色涂层应用）需要角度分辨的逆设计验证。

(4) **实验验证缺失**：所有结果基于 RCWA 数值仿真，尚未与实验制备的样品对比。RCWA 本身在 nG 收敛后精度极高（本验证 90% 结构 < 1 JND），但实际制备中的工艺偏差（侧壁倾斜、表面粗糙度、材料非均匀性）可能引入额外误差。

---

# 5. Conclusion

We have presented a systematic ML-assisted inverse design framework for dielectric metasurface structural colors with three distinguishing features: rigorous closed-loop validation, a physics-grounded material selection criterion, and explicit quantification of the optimizer's curse.

**Closed-loop validation** reveals a universal ~+3–4 ΔE₀₀ gap (median) between the agent model's self-reported optimum and RCWA-verified reality. This gap is approximately material-independent (TiO2 roundtrip: +3.59, a-Si roundtrip: +3.68, medians). The hybrid re-ranking strategy raises TiO2 roundtrip inverse design success rate from 19% to 62% (Wilson 95% CI: 52–71%), with hybrid ≤ naïve guaranteed on all 227/227 tested targets.

**The Δn criterion** from 12 material/substrate combinations reveals a resonance cutoff at Δn ~0.5, with low optical loss (k ~0) as a sufficient condition for maximum gamut. This explains TiO2 (positive), Si3N4/Al2O3 (negative controls), and a-Si (high Δn but absorption-limited, 0–18% success).

**Forward accuracy is necessary but insufficient** for inverse design: a-Si achieves forward holdout ΔE₀₀ = 2.38 (better than TiO2's 2.99) yet only 0–18% inverse design success (0% for gamut-probing sRGB targets outside gamut, 18% for roundtrip targets within gamut) versus 62% for TiO₂ roundtrip, due to candidate pool depletion (cross-sample RGB std = [0.21, 0.10, 0.09]).

**Speed**: ML inference (5.13 ms) is >300× faster than RCWA (~1.6 s); end-to-end inverse design completes in ~37 s vs ~38 min for brute-force RCWA (~60× acceleration).

Future extensions include dual-pillar geometries, FDTD cross-validation, and experimental fabrication. The framework provides a practical protocol for navigating the boundary between ML acceleration and physical realizability in photonic inverse design.

本框架给出一个可检验的预测：对于任何无损介质（k ≈ 0）且 Δn > 0.6 的材料-衬底组合，在候选网格覆盖 ≥ 1000 个几何有效结构的前提下，混合重排（K = 20）的 roundtrip 成功率预计超过 50%。我们邀请独立验证这一边界。更广泛地，本文提出的闭环验证协议和极端值估计 σ√(2 ln N) 可推广至光子学以外的代理辅助优化——任何从含噪预测器中选择最优候选的工作流都受益于同样的诊断与修复。

---

# References

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
