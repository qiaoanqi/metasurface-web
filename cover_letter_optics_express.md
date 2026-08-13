[Date]

The Editor
Optics Express

Dear Editor,

We are pleased to submit our manuscript entitled "Forward Accuracy Does Not Predict Inverse Design Success: Optimizer's Curse and a Gamut Cutoff in ML-Assisted Metasurface Structural Color" for consideration as a Research Article in Optics Express.

Machine-learning surrogate models have become a standard tool for the inverse design of dielectric metasurfaces, and their performance is almost universally reported through a single number: the forward holdout prediction error. The implicit assumption throughout the literature is that a model that predicts well also designs well. Our manuscript shows, with a closed-loop validation protocol, that this assumption is false, and we trace the failure to two distinct, quantifiable causes.

First, we quantify the "optimizer's curse" in this setting. When a noisy surrogate is used as the optimization objective, selecting the best-ranked candidate amplifies the prediction error rather than the true performance. Across seven dielectrics we measure a median over-optimism of +2 to +4 CIEDE2000 units for in-gamut targets: structures the surrogate rates as near-perfect perform far worse under independent rigorous coupled-wave analysis (RCWA). We show that the magnitude is not an artifact of a particular model or dataset but a generic consequence of optimizing over O(10^3) noisy predictions, and we confirm it with an extreme-value estimate, sigma*sqrt(2 ln N), that predicts the observed bias a priori. We then demonstrate hybrid re-ranking (ML screening of the top-K candidates followed by RCWA verification) as a deterministic repair that raises TiO2 roundtrip success from 19% to 62%, with the guarantee that the hybrid result is never worse than the naive pick. A controlled comparison on previously unseen structures shows the screening itself carries the value: ML top-20 candidates succeed on 63% of targets versus 9% for a random-20 baseline.

Second, we show that forward accuracy and inverse design success are decoupled. The least accurate surrogate in our study (Si3N4, forward CIEDE2000 of 13.75) attains 79% success at verified solver convergence, while a far more accurate surrogate (TiO2, 2.99) reaches only 53%; a rank-correlation test across seven dielectrics finds no significant association. We also identify a resonance cutoff at an index contrast of approximately 0.5, validated across seven dielectrics, that serves as an a priori material-screening criterion: every lossless dielectric above the threshold is active (62-82% success) while Al2O3, below it, is dead. The strongly absorbing a-Si (Δn = 2.93) is eliminated by the second-stage low-k criterion, as the two-stage rule predicts; its gamut is confined to dark hues, its in-gamut design remains tractable, and we document a solver-convergence caveat specific to such strongly absorbing high-index materials.

We believe this work is well suited to Optics Express because it speaks directly to a methodological question that affects the entire metasurface-design community, and because it replaces an unexamined reporting convention with a concrete, reproducible validation standard and a set of practical design guidelines. The core insight, that optimization over a noisy objective amplifies error, also generalizes beyond photonics to any surrogate-assisted optimization workflow.

This manuscript is original, has not been published previously, and is not under consideration elsewhere. All authors have approved the submission. [Competing-interest and funding statements are included in the manuscript.]

We suggest the following reviewers with relevant expertise: Alejandro W. Rodriguez (Princeton University, USA), Ole Sigmund (Technical University of Denmark, Denmark), Haim Suchowski (Tel Aviv University, Israel), and Tie Jun Cui (Southeast University, China). [Emails to be confirmed from source papers before entry.] We have no opposed reviewers.

Thank you for your time and consideration. We look forward to your response.

Sincerely,

Anqi Qiao
School of Physics and Electronic Science
Changsha University of Science and Technology
Changsha 410114, China
Email: qiaoanqi@stu.csust.edu.cn

[Co-author / advisor name, affiliation, and email to be added]
