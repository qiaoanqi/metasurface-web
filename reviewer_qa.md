# Anticipated Reviewer Q&A — Response Talking Points

Pre-drafted answers to the points reviewers are most likely to raise. All grounded in the
manuscript; cite section/line when responding. Technical, advisor-independent.

## Q1. Why define the index contrast as Δn = n(550 nm) − n_sub rather than the Cauchy asymptote A?

The cutoff is convention-invariant; n(550 nm) is a convenience, not a necessity (see §4.3, the
robustness paragraph). All six dielectrics have normal dispersion (B > 0), so n(550 nm) exceeds
the Cauchy asymptote A by only 0.02–0.13; the two metrics order the materials identically and
differ by a small monotonic offset. Critically, the classification boundary is unchanged: the dead
material Al₂O₃ and the weakest live material HfO₂ are separated by 0.30 → 0.51 under the
Cauchy-A convention and 0.31 → 0.54 under n(550 nm). In both conventions the Δn ≈ 0.5 cutoff falls
inside the dead/live gap, so the two-stage screening rule does not depend on the reference
wavelength. We use n(550 nm) because it references the wavelength at which the inverse-design color
error is evaluated.

## Q2. Why does HfO₂ (Δn = 0.545) reach 82% success among lossless materials — isn't that anomalous for a low-contrast material?

Success is governed by in-gamut reachability and surrogate ranking accuracy, not by gamut size
(§4.1; Fig. 2b). Moderate-Δn materials (HfO₂, Si₃N₄) support weaker, smoother resonances whose
spectra the surrogate predicts more accurately, so forward accuracy translates more directly into
design success; high-Δn systems (TiO₂ at ~0.96, GaN at ~0.94) support sharper, higher-Q resonances that
are more prediction-error-prone, partially offsetting their larger gamut. The non-monotonic
success-vs-Δn curve is confirmed by the substrate controls: holding the pillar fixed at TiO₂ and
varying only the substrate gives success 62% / 92% / 96% as Δn falls 0.958 / 0.645 / 0.386 —
success rises as contrast falls. Roundtrip success therefore tracks reachability, not gamut volume.

## Q3. a-Si has the highest contrast (Δn = 2.93) — what limits its gamut, and why does it still design well in-gamut?

Two loss mechanisms restrict its usable gamut despite high Δn (see §3.2 / Discussion §4.1):
(i) impedance mismatch — excessive Fresnel reflection at the grating interface compresses the
resonant modulation; (ii) material absorption — k ≈ 2.2 at 400 nm removes blue-band photons
(mean R+T = 0.485, ~52% absorption). The achievable colors concentrate in a sparse cluster of
desaturated dark tones (sRGB std R = 0.11, G = 0.08, B = 0.08; only dark reds and browns). Note this
is a genuine gamut restriction driven by absorption (k ≈ 2.2 at 400 nm; mean R+T = 0.485), and it does
not prevent in-gamut design: a-Si attains 86% roundtrip success (nG 65; 47% at nG 101, see the
convergence caveat) and 87% on unseen-structure holdout targets. The median curse gap (+1.46) is smaller
than TiO₂'s (+3.59) because the absorbing material's simple spectra are easier to predict.

## Q4. Why report "seven dielectrics" rather than a fixed dataset count (e.g. twelve)?

The scientific claim rests on the seven dielectrics and the Δn span (0.31 for Al₂O₃ to 2.93 for
a-Si), not on a material×substrate product count. The primary comparison is on SiO₂; other substrates
enter only as substrate-variation controls. Stating "seven dielectrics" is defensible and avoids a
brittle combinatorial count. The screening claim itself rests on resonance amplitude (R range), not
on the dataset count.

## Q5. Al₂O₃ achieves 43% roundtrip success — how can you call it "dead"?

Roundtrip success and gamut usefulness are different quantities (Fig. 2, §3.2). Roundtrip targets are
sampled from the material's own parameter distribution: a degenerate gamut is trivially self-consistent, because most
geometries map to the same few dark, desaturated tones, so a roundtrip target is reachable by many
structures even when the surrogate is comparatively inaccurate (mean claimed ΔE ≈ 7). The screen uses
the reflectance dynamic range instead: Al₂O₃'s R range collapses to 0.090, ~3× below the weakest live
material (HfO₂, 0.264), so no useful saturated structural color exists. Roundtrip measures in-gamut
reachability; R range measures gamut volume. The two-stage rule is: (i) Δn ≳ 0.5 necessary for
resonance existence; (ii) low k with moderate n for a useful gamut.

## Q6. Is the R-range threshold chosen post hoc to fit the data?

No — the classification is insensitive to the exact threshold. The R range values fall into two
clearly separated groups: Al₂O₃ at 0.090 versus all live materials at ≥ 0.26 (HfO₂ 0.264, Si₃N₄ 0.290,
TiO₂ 0.582). The gap is a factor of ~3 with no material in between, so any threshold in (0.090, 0.264)
gives the same dead/live split. The Δn ≈ 0.5 cutoff inherits this insensitivity: it is also
convention-invariant (Q1) and falls inside the 0.31–0.54 gap with no material near the boundary.

## Q7. Is 43% vs 96% statistically distinguishable, or is it sampling noise?

Clearly distinguishable. Wilson 95% CIs (n = 100 each): Al₂O₃ 43% → [33.7%, 52.8%]; TiO₂/Si₃N₄ 96% →
[90.2%, 98.4%]. The intervals do not overlap (upper bound of Al₂O₃ 52.8% < lower bound of the
96% case 90.2%), so the anti-monotonic ordering is significant at the 5% level. A two-sided Fisher
exact test on the raw counts (43/57 vs 96/4 successes) gives p = 1.9×10⁻¹⁷, decisively below the 5%
level. All rates in Fig. 2b
carry Wilson CIs; Table 3 reports the raw counts.

## Q8. Why does Fig. 2 show two panels with different y-axes?

The two panels answer two different questions. Panel (a) plots reflectance dynamic range vs Δn — the
physical quantity that defines whether a useful gamut exists (the screening criterion). Panel (b)
plots roundtrip success vs Δn — included to expose the anti-monotonic behavior (62/92/96 as Δn falls)
and to demonstrate explicitly that roundtrip success is not a valid material screen. Showing both
prevents a reader from re-interpreting the cutoff as a statement about inverse-design success.
## Q9. What exactly is "nG = 65" in grcwa — is it "65 Fourier orders"?

No. grcwa's nG is a reciprocal-lattice truncation parameter whose numerical meaning is the
*square of the grid side length*, not the harmonic count. In the parallelogramic scheme
(Gmethod=1, `Gsel_parallelogramic`): `NGroot = int(sqrt(nG))`, decremented by 1 if even.
Measured (grcwa.kbloch.Lattice_getG, method=1):

| nG parameter | actual plane waves | grid |
|---|---|---|
| 65 | 49 | 7×7 |
| 101 | 81 | 9×9 |
| 131 / 151 | 121 | 11×11 |
| 201 | 169 | 13×13 |

A self-consistent probe (reference = highest order, 20 roundtrip structures,
ng_probe_selfconsistent_result.md) shows a-Si converges at the 11×11 grid (nG ≈ 131;
mean ΔE 0.85, 80% < 1 JND), while 7×7 (nG = 65, ΔE 4.41) and 9×9 (nG = 101, ΔE 2.23) are
insufficient. The manuscript's "nG = 131 probing still drifts" (old reference convention)
is defensible only as a residual-vs-13×13 statement (0.85, 20% > 1 JND); the solver
semantics are documented here and in §convergence (L49/L60 fixed in relay #6).

## Backup numbers (cross-verified against pkl ground truth)
- TiO₂ roundtrip: naive 19% → hybrid 62% (Wilson 95% CI 52–71%); curse gap median +3.59 (bootstrap 95% CI [3.06, 4.49]; n = 100, 20,000 resamples, seed 2026); nG101 re-verification 53%.
- a-Si roundtrip (Pierce & Spicer 1972 constants, 2026-08-07 rerun): 86% at nG65 (47% at nG101; nG131 still drifting — solver-convergence caveat, see nG_convergence_audit.md); gamut-probe: 0% (N=29, binomial upper bound 10%); curse gap median +1.46 (bootstrap CI [+1.15, +1.93]); forward seed-mean 2.37 (ensemble 2.15).
- Si₃N₄ roundtrip: 81% (retrained model; nG101 79%); forward holdout 13.75 (independent frozen-split reproduction, N=5400, three substrates); curse gap median +2.25.
- HfO₂: 82% (nG101 80%); GaN: 66% (58%); Ta₂O₅: 74% (69%).
- Al₂O₃ roundtrip: 43% (naive 4%, gap +1.36) — gamut-dead by R range (0.090), not by roundtrip.
- A4 holdout control (unseen structures, N=100/arm, pre-registered ΔE<2.3, seed 2026): TiO₂ hybrid 63% vs random-20 9% (McNemar b=54, c=0, p=1.1e-16); a-Si hybrid 87% vs random-20 4% (b=83, c=0, p=2.1e-25). RT(seen) vs holdout(unseen): TiO₂ 62→63, a-Si 86→87 (Fisher p=1.0) — no memorization.
- TiO₂ substrate controls: SiO₂ 62% / Al₂O₃-sub 92% / Si₃N₄-sub 96% (Δn 0.958/0.645/0.386;
  R range 0.582/0.309/0.102); curse gaps +3.59 / +1.67 / +0.75.
- McNemar paired tests: hybrid vs naive significant in all four configurations (p ≤ 2×10⁻³), 400 targets total, 0 regressions.
- Five lossless dielectrics combined: 365/500 = 73% hybrid roundtrip success.
- K sensitivity: TiO₂ 47/57/62/64 at K = 5/10/20/50; HfO₂ 68/75/82 at K = 5/10/20 (same knee).
- EVT estimate: √(2 ln 1392) ≈ 3.8 → predicted over-optimism O(3–5) ΔE₀₀, matching measured +3.59.
- Testable prediction: any lossless dielectric with Δn > 0.6 → hybrid roundtrip success > 50% at K = 20
  (grid ≥ 1000 valid structures).
