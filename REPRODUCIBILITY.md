# Reproducibility Guide

End-to-end command chain for the results reported in the manuscript
(*Forward Accuracy Does Not Predict Inverse Design Success: Optimizer's Curse and a Gamut
Cutoff in ML-Assisted Metasurface Structural Color*). Workflow: RCWA data generation ->
surrogate training -> closed-loop inverse-design validation -> figure generation ->
manuscript compilation.

> **Publication status of code and data: pending approval from the author team.**
> This guide documents the internal pipeline so the project can be re-run or released
> once the team decides on a public distribution policy. Nothing here constitutes a
> commitment to release.

## 0. Environment

- Python 3.10+; packages: numpy, scipy, matplotlib, torch, onnxruntime, grcwa, color-math
  (see `requirements.txt`).
- RCWA: `grcwa` (fancompute), nG = 65, Nxy = 256, 81 wavelength points 380-780 nm,
  TE polarization, normal incidence.
- Hardware used: Intel i7-14650HX / RTX 5060 Laptop (timings in manuscript).

## 1. RCWA data generation

Script: `rcwa_batch.py` (and `rcwa_batch_fast.py` for the fast path).

- Materials: TiO2 (anatase), GaN (wurtzite, ordinary ray), Ta2O5, HfO2, Si3N4, a-Si, Al2O3,
  each on SiO2 (primary); TiO2 also on Al2O3 and Si3N4 substrates (controls).
- Cauchy coefficients: refs [29] (GaN), [30] (Ta2O5), [31] (HfO2); a-Si uses the full complex
  index n + ik from Pierce & Spicer (1972), Phys. Rev. B 5, 3017 (60-nm film, values as
  reproduced in Palik's Handbook), loaded from data/externals/aSi_PierceSpicer1972.csv with
  linear interpolation across the RCWA band. (2026-08-06: replaces the earlier Green & Keevers
  1995 citation, which is crystalline-Si data and matched no public a-Si dataset; see
  aSi_optical_constants_audit.md. The old manuscript numbers for a-Si (0-18% roundtrip
  success, +3.68 curse gap) were artifacts of this unsourced constant set and are
  superseded by the 2026-08-07 rerun.)
- Sampling: uniform random sampling (rejection method, integer nm) over (D, H, P) in
  [80,350] x [100,600] x [200,600] nm, constraints D < P and fill factor
  pi*(D/2)^2/P^2 in [0.03, 0.70]; ~3000 structures per material/substrate; quality filter:
  lossless materials use |R+T - mean(R+T)| <= 0.05; the lossy a-Si uses a physical
  criterion (reject only R+T <= 0 or > 1.05) because strong absorption widens the
  R+T distribution (0.14-0.98) and the relative criterion would over-filter
  (2026-08-07, A1 audit). k=0 control: set env A_SI_K0=1 when running rcwa_batch.py
  to zero the a-Si extinction (fig5 no-absorption reference line; n unchanged).
- Output: `data/rcwa_*.pkl`, one record per sample with D, H, P, wavelength grid,
  R(81), T(81), sRGB, mean R+T, material, substrate.

## 2. Surrogate training

Script: `train_rcwa.py` (unified protocol, 2026-07-31+).

- 7-dim input -> Lin(256) -> 4 x ResBlock(256) -> Lin(81) -> sigmoid (553,809 params).
- 80/20 stratified holdout split (by substrate, seed 2026, frozen); +-2 nm jitter
  augmentation (x3); 3-seed ensemble (42, 123, 456); Adam (weight decay 1e-5) lr 5e-4,
  cosine to 1e-5, batch 128, full 1000 epochs, best-validation checkpoint retained
  (no early stopping). ONNX export (opset 14).
- Protocol-version note: Si3N4 models trained before 2026-07-31 used a different
  protocol and are superseded; the manuscript reports only unified-protocol models.

## 3. Closed-loop inverse design validation

Script: `closed_loop_validate.py`.

- Targets: N = 100 roundtrip targets per configuration (geometries sampled from the
  training parameter distribution — uniform random, D < P, fill factor 0.03-0.70 —
  with their RCWA-simulated colors as targets; NOT training-set structures); a-Si
  gamut-probe N = 29 (sRGB-boundary targets; fully-saturated hue sweep,
  HSV S = V = 0.95, ~10 deg hue spacing; differs from the generic generator's
  mixed-layer recipe — the archived set data/externals/probe_aSi_PS_29_targets.json
  is authoritative); TiO2 gamut-probe N = 100.
- Pipeline: ML screens 1392-candidate uniform grid, top-K = 20 by predicted CIEDE2000,
  independent RCWA re-verification, best verified design selected.
- nG re-verification: `closed_loop_validate.py --verify-nG 65,101 --input <pkl> --output <out>`
  archives per-structure cross-order color change (data/ng_verify_*.pkl); success rates at
  nG101: HfO2 80%, Si3N4 79%, Ta2O5 69%, GaN 58%, TiO2 53%, a-Si 47%, Al2O3 43%
  (nG65: 82/81/74/66/62/86/43; see nG_convergence_audit.md).
- A4 controlled comparison: `_run_holdout_baseline.py --material TiO2|a-Si --n-targets 100`
  (holdout targets = frozen-test structures never seen by the surrogate; naive / hybrid
  top-20 / random-20 arms; pre-registered DeltaE < 2.3, seed 2026).
  Output: data/controlled_*_holdout_N100.pkl.
- Output: `data/closed_loop_*_roundtrip_N100.pkl` (list of dicts; per-target fields
  include pred_de2000, ach_de2000, naive_ach_de2000, naive_pred_de2000, gap, status).
  `all_verified` stores per-candidate (ach, pred) tuples for K-sensitivity re-analysis.
- Success: hybrid achieved DeltaE00 < 2.3 (JND); Wilson 95% CIs in manuscript.

## 4. Figures

- `_gen_fig1.py` -> fig1_protocol.pdf
- `_fig2_v4_twopanel.py` -> fig2_delta_n_criterion.pdf (dual panel: R-range cutoff +
  roundtrip anti-monotonicity; colorblind-safe palette; internal asserts pin every
  published success rate and R-range value to the pkl ground truth)
- `_fig3_v2_gamut.py` -> fig3_color_gamut.pdf
- `_make_figures.py` -> fig4_curse_gap.pdf (and legacy fig2/fig3)
- `_plot_demo_v2.py` -> fig_demo_colors.pdf
- fig5_nG_convergence.pdf: `_archive/temp_scripts/_make_fig5_from_pkl.py` (reads
  data/ng_verify_*.pkl; a-Si k=0 line from the A_SI_K0=1 pool)

## 5. Manuscript

- Source: `paper_oe.tex` (article class during development; Optica/OE template switch
  pending advisor info). Compile: `pdflatex paper_oe.tex` twice.
- Figures in `figures/`, PDF output `paper_oe.pdf`, synced copy `论文.pdf`
  (MD5-verified identical after each locked revision).

## 6. Verification commands

- K-sensitivity (TiO2, HfO2): re-slice `all_verified` from the roundtrip pkls by ML-pred
  rank (ascending) and take the best RCWA-verified DeltaE among top-K. TiO2: 47/57/62/64
  at K = 5/10/20/50; HfO2: 68/75/82 at K = 5/10/20 (pkl stores top-20 verification).
- R-range and success-rate anchors are asserted inside `_fig2_v4_twopanel.py`.
