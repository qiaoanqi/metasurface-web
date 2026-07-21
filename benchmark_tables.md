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