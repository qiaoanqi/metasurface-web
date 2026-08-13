# Stats pack v2 — pre-cascade final — seed 2026

## 0. Forward per-target pkl cross-check (mean dE)
- GaN: N=493 mean=2.429 median=2.055
- Ta2O5: N=540 mean=2.707 median=2.227
- HfO2: N=545 mean=2.603 median=2.166
- Al2O3: N=450 mean=8.114 median=7.366
- aSi_PS: N=1635 mean=2.152 median=1.570

## 1. Spearman: forward dE vs RT success
- nG65, all 7 (N=7): rho = -0.500, p = 0.253
- nG65, drop a-Si (N=6): rho = -0.200, p = 0.704
- nG101, all 7 (N=7): rho = +0.071, p = 0.879
- nG101, drop a-Si (N=6): rho = -0.200, p = 0.704

## 2. Wilson 95% CIs
### nG65 main caliber
- a-Si: 86/100 = 86%  CI [77.9, 91.5]
- HfO2: 82/100 = 82%  CI [73.3, 88.3]
- Si3N4: 81/100 = 81%  CI [72.2, 87.5]
- Ta2O5: 74/100 = 74%  CI [64.6, 81.6]
- GaN: 66/100 = 66%  CI [56.3, 74.5]
- TiO2: 62/100 = 62%  CI [52.2, 70.9]
- Al2O3: 43/100 = 43%  CI [33.7, 52.8]
### nG101 recheck
- HfO2: 80/100 = 80%  CI [71.1, 86.7]
- Si3N4: 79/100 = 79%  CI [70.0, 85.8]
- Ta2O5: 69/100 = 69%  CI [59.4, 77.2]
- GaN: 58/100 = 58%  CI [48.2, 67.2]
- TiO2: 53/100 = 53%  CI [43.3, 62.5]
- a-Si: 47/100 = 47%  CI [37.5, 56.7]
- Al2O3: 43/100 = 43%  CI [33.7, 52.8]

## 3. A4 bounded controls (holdout targets, unseen by surrogate)
### TiO2/SiO2 (N=100)
- hybrid holdout: 63/100 = 63%  CI [53.2, 71.8]
- naive:          24/100 = 24%  CI [16.7, 33.2]
- random-20:      9/100 = 9%  CI [4.8, 16.2]
- McNemar hybrid vs random: pairs=100 b=54 c=0 both=9 neither=37  p=1.110e-16
- Fisher hybrid vs random (unpaired): p=4.127e-16
- McNemar hybrid vs naive: pairs=100 b=39 c=0  p=3.638e-12

### a-Si(PS)/SiO2 (N=100)
- hybrid holdout: 87/100 = 87%  CI [79.0, 92.2]
- naive:          51/96 = 53%  CI [43.2, 62.8]
- random-20:      4/100 = 4%  CI [1.6, 9.8]
- McNemar hybrid vs random: pairs=100 b=83 c=0 both=4 neither=13  p=2.068e-25
- Fisher hybrid vs random (unpaired): p=1.389e-36
- McNemar hybrid vs naive: pairs=96 b=32 c=0  p=4.657e-10

### Memorization check: RT(seen) vs holdout(unseen), hybrid
- TiO2: RT 62% vs holdout 63%  (Fisher p=1.000)
- a-Si: RT 86% (nG65) vs holdout 87%  (Fisher p=1.000)
