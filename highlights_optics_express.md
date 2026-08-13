# Highlights

For submission system "highlights" / "at a glance" fields. Each item is a single self-contained sentence.

- We establish a closed-loop protocol (target color -> ML inverse design -> independent RCWA verification) and show that forward holdout accuracy does not predict inverse design success: the least accurate surrogate (Si3N4, CIEDE2000 13.75) attains 79% success at verified solver convergence while a far more accurate surrogate (TiO2, 2.99) reaches only 53%, with no significant rank association across seven dielectrics.

- We quantify the "optimizer's curse" in metasurface ML: selecting the surrogate's rank-1 candidate amplifies prediction error by a +2 to +4 CIEDE2000 median over-optimism for in-gamut targets, a magnitude predicted a priori by the extreme-value estimate sigma*sqrt(2 ln N).

- Hybrid re-ranking (ML top-K screening followed by RCWA verification) repairs the curse deterministically, raising TiO2 roundtrip success from 19% to 62%; a controlled comparison on unseen structures shows ML top-20 candidates succeed on 63% of targets versus 9% for random-20, establishing the screening value directly.

- We identify a resonance cutoff at an index contrast of approximately 0.5, validated across seven dielectrics, as an a priori material-screening criterion: all lossless materials above it are active (62-82% success) while Al2O3 below it is inactive.

- We characterize a solver-convergence caveat for strongly absorbing high-index materials (a-Si): its RCWA Fourier orders converge slowly (72% of structures within 1 JND across nG 65->101, versus >=93% for lossless dielectrics), the first systematic report of this effect in metasurface color.
