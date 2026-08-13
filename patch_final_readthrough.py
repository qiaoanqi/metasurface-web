# -*- coding: utf-8 -*-
import io, sys

path = "paper_oe.tex"
t = io.open(path, encoding="utf-8").read()

def sub_once(old, new, tag):
    global t
    n = t.count(old)
    if n != 1:
        print(f"FAIL {tag}: found {n} occurrences")
        sys.exit(1)
    t = t.replace(old, new)
    print(f"OK {tag}")

# 1) fig1 caption: figure is now a pure flowchart, drop the stale 'Bottom:' curse sentence
sub_once(
    r"\caption{Closed-loop inverse design protocol. The ML surrogate screens 1392 candidates ($\sim$4~s) and returns the top-$K = 20$; each undergoes independent RCWA verification ($\sim$33~s total). Bottom: ML self-claimed $\Delta E_{00}$ (0.66) vs.\ RCWA-achieved (5.83) for the na\"ive pick---the optimizer's curse.}",
    r"\caption{Closed-loop inverse design protocol. The ML surrogate screens 1392 candidates ($\sim$4~s) and returns the top-$K = 20$; each undergoes independent RCWA verification ($\sim$33~s total); the verified best is selected.}",
    "fig1 caption")

# 2) fig3 caption: corrected Si3N4 gamut (moderate, R range 0.29); describe the annotations fully
sub_once(
    r"\caption{sRGB color gamut of training sets. Left: TiO$_2$/SiO$_2$ (broad coverage). Center: a-Si/SiO$_2$ (absorption-confined). Right: Si$_3$N$_4$/SiO$_2$ (single cluster). Annotations: cross-sample sRGB standard deviations.}",
    r"\caption{sRGB color gamut of training sets. Left: TiO$_2$/SiO$_2$ (broad coverage). Center: a-Si/SiO$_2$ (absorption-confined). Right: Si$_3$N$_4$/SiO$_2$ (moderate gamut). Boxes give cross-sample sRGB standard deviations; subtitles give per-sample reflectance dynamic range ($R_{\max}-R_{\min}$, ensemble mean).}",
    "fig3 caption")

# 3) give fig:gamut its first (and only needed) reference at the a-Si std sentence
sub_once(
    r"cross-sample sRGB std is only $R = 0.21$, $G = 0.10$, $B = 0.09$, with green and blue channels locked",
    r"cross-sample sRGB std is only $R = 0.21$, $G = 0.10$, $B = 0.09$ (Fig.~\ref{fig:gamut}), with green and blue channels locked",
    "fig:gamut ref")

io.open(path, "w", encoding="utf-8").write(t)
print("paper_oe.tex written.")
