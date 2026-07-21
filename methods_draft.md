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
