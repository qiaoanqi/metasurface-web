# Stats pack v1 — new a-Si (P&S) — seed 2026

## 1. Wilson 95% CIs (hybrid RT success)
- a-Si(PS) RT: 86/100 = 86%  CI [77.9, 91.5]
- Si3N4 RT: 81/100 = 81%  CI [72.2, 87.5]
- HfO2 RT: 82/100 = 82%  CI [73.3, 88.3]
- Ta2O5 RT: 74/100 = 74%  CI [64.6, 81.6]
- GaN RT: 66/100 = 66%  CI [56.3, 74.5]
- TiO2 RT: 62/100 = 62%  CI [52.2, 70.9]
- Al2O3 RT: 43/100 = 43%  CI [33.7, 52.8]

## 2. Probe 0/29 — exact binomial 95% upper bound
- 0/29 successes: Clopper-Pearson 95% UB = 9.8%  (rule of three: 10.3%)

## 3. McNemar hybrid vs naive, new a-Si RT (skip-None convention)
- valid pairs: 97 (b=32 hybrid-only, c=0 naive-only, both=51, neither=14)
- exact McNemar p = 4.657e-10

## 4. Curse gap bootstrap CI (new a-Si RT, median)
- N=97  median=+1.46  mean=+2.05  bootstrap95 CI of median [+1.15, +1.93]

## 5. Fisher exact (pairwise, for disclosure only)
- Si3N4 81 vs TiO2 62: p = 4.57e-03
- a-Si 86 vs TiO2 62: p = 1.74e-04
- a-Si 86 vs HfO2 82: p = 5.63e-01

## 6. Spearman forward dE vs RT success — BLOCKED
- Need forward holdout dE for GaN/Ta2O5/HfO2/Al2O3 (models exist, metrics not archived).
- Available now: a-Si 2.37/86, TiO2 2.99/62, Si3N4 13.75/81 (N=3, insufficient).
