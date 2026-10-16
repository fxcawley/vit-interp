# 09. Writing plan

## 1. Title candidates

1. What a JEPA Does with a Circle: Harmonic Codes, Regularizer Width, and the Reading of Distributed Physics Representations
2. Rings Are Not Failures: Circular Latents Are Represented in the Basis Their Dynamics Predict
3. The Distorted Regime of JEPA Identifiability Is a Harmonic Series

Option 1 is the safest for a main track; option 2 is the workshop title.

## 2. Abstract draft (main track, to be rewritten after results)

Video world models have been found to represent motion direction as a high-dimensional population code, which has been read as evidence that they lack compact physical state. Identifiability theory for joint-embedding predictive architectures (JEPAs) says that a latent is linearly recoverable only when the regularizer's target matches its geometry, and describes the mismatched case as distorted. We show that for circular latents these two observations are the same phenomenon and that it is predictable. In pixel-trained JEPAs across recipes, the code for an angle lies in the Fourier basis of the latent's transition operator, with harmonic power ordered by slowness. The number of harmonics is set by three knobs: the step size of the dynamics, the regularizer's demand for decorrelated unit-variance dimensions, and the embedding width. Under a Gaussian target with spare width, position acquires quadratic features before heading acquires its second harmonic, in the order the eigenvalues predict; under an EMA-teacher recipe the code stays two-dimensional. The latent is spread, not lost: it is recovered from the code through the predicted nonlinearity. Applied to released models, the analysis shows that V-JEPA 2's direction code is a compact ring multiplied by spatial redundancy, and that LeWorldModel's orientation code widens with its regularizer weight. Dimensionality counts cannot by themselves distinguish distributed computation from a compact latent, and the choice of target and width is a choice about code geometry.

## 3. Figures (main track)

1. Concept figure: latent law, eigenfunctions, predicted code for Gaussian / circle / interval; the two 2026 results placed on it.
2. Spectra: p_k vs k for SIGReg, VICReg, EMA, matched target, with random-init and time-shuffle nulls (W1).
3. Dose-response: K_5 vs s (with the exp(-k^2 s^2) fit) and K_5 vs lambda x d heatmaps; EMA flat line.
4. Kurtosis signature: per-coordinate excess kurtosis distributions, SIGReg vs VICReg, with the -1.5 / K line.
5. Filling order (W4): first-appearance rank of each eigenfunction vs theoretical rank; Kendall tau.
6. Torus distortion: decomposition of the variance unexplained by a linear angle probe into cross-harmonics vs other, Gaussian vs matched target; R^2 asymmetry reproduced and closed.
7. V-JEPA 2: direction-subspace harmonic spectrum, dimensionality before/after pooling and speed regress-out, sawtooth.
8. LeWorldModel: Push-T orientation spectrum at the released checkpoint and across lambda.
9. Cross-seed: harmonic-subspace principal angles vs coordinate matching.

## 4. Tables

1. Worlds x recipes x knobs with run counts and failure rates.
2. R^2 (both directions) and circular R^2 next to K_5 and PR for every cell, so readers can map our measurements to Klindt/Nicollier's.
3. Released-model numbers with CIs.

## 5. Section skeleton

1. Introduction: the two beliefs, the question, the classical prediction, contributions (four bullets).
2. Background: JEPA objectives and regularizers; identifiability results; the interpretability finding; SFA.
3. Theory: eigenfunctions per law, filling order, kurtosis, identifiability up to phase (compact version of `04-theory-notes.md`).
4. Experimental setup: worlds, recipes, readouts, preregistration.
5. Results in toy worlds: basis, knobs, kurtosis, filling order.
6. Results in benchmark worlds: torus distortion.
7. Released models: V-JEPA 2, LeWM.
8. Discussion: how to read dimension counts; how to choose targets; limits (predictor, depth, action conditioning); relation to neuroscience.
9. Reproducibility and preregistration statement.

## 6. Strip-down map (main track to workshop, 4-8 pages)

Keep: Fig. 2 (spectra with nulls), Fig. 3 (dose-response), one of Fig. 7 or 8 (whichever released-model result holds), Table 2 compact. Drop to appendix or omit: kurtosis, filling order, torus, cross-seed. Title option 2. The theory section becomes one paragraph plus the eigenvalue table.

## 7. Checklists before submission

Anti-slop
- [ ] Every headline number was predicted in `00-preregistration.md` before the run; deviations are flagged.
- [ ] Random-init null and time-shuffle control appear in the same figure as every trained spectrum.
- [ ] No claim about V-JEPA 2 or LeWM beyond what P6 / P7 measured.
- [ ] Optimization failure rates are reported per cell.
- [ ] The SFA lineage is credited in the introduction, not buried in related work.

Venue hygiene
- [ ] Code, configs, seeds, generators released; preregistration commit hash cited.
- [ ] Licenses of Kubric, DM Control, V-JEPA 2 weights, LeWM code, the Joseph reproduction repo recorded.
- [ ] Ethics statement: synthetic data only; no human subjects; no proprietary data.
- [ ] KLA clearance for external publication obtained even though no KLA data is used.

Writing
- [ ] Each figure caption states the prediction it tests and whether it held.
- [ ] Numbers carry CIs and seed counts.
- [ ] The word "distributed" is defined once and used consistently.
