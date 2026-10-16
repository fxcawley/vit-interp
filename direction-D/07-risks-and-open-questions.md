# 07. Risks and open questions

## 1. Scoop exposure (high)

- The Brown GalilAI group (Balestriero, Klindt, Maes) owns LeJEPA, LeWM and the identifiability theorem, and Balestriero is also a co-author of Joseph et al. (with Fel and Richards), so one group sits on both the theory and the interpretability finding; a circular-latent follow-up that connects them is a natural next paper for them. The Centre Borelli group (Nicollier, Facciolo) has the manifold theory, the heat-kernel regularizers and the pixel environments, and promised code on acceptance. A third party already proposed a Fourier auxiliary head for LeWM (arXiv:2609.04264). Four JEPA-theory papers appeared between May and September 2026.
- Mitigation: preregister with a commit hash now; post the preprint at the end of Phase 2; keep the released-model analyses (P6, P7) as the part least likely to be duplicated; consider contacting Balestriero early (he spoke at KLA) so that overlap becomes collaboration rather than a race (`08-collaborators.md`).

## 2. Optimization fragility (medium-high)

Nicollier et al. lost 5 of 10 seeds on the matched torus target and selected lambda on seed 0. Our cells with matched targets and large lambda may fail similarly. Mitigation: the preregistered success criterion that does not look at latents; >= 5 seeds in headline cells; report failure rates; treat a cell with majority failures as no-claim (K4).

## 3. Measurement confounds

- Harmonic structure is generic for smooth periodic embeddings; the random-init encoder will show it too. Claims are about concentration, ordering and dose-response relative to that null, never about presence.
- Probe capacity: harmonic regression with K_max = 8 on 64 bins has 16 regressors; fine for the bin means but over-fit risk if applied per sample with small data. Use bin means and hold-out bins.
- Aliasing: with discrete direction stimuli (Joseph et al.'s 8), harmonics above 3 fold back. Continuous stimuli in Tier 2 and 3.
- Speed and direction are entangled in motion stimuli (velocity vector); regress out speed before counting direction dimensions, and report both.
- Spatial copies: pooling must be done on probe weights per position, not on token activations, to avoid averaging away position-specific codes; report both orders.
- The predictor: a nonlinear predictor can absorb some nonlinearity that the theory assigns to the encoder; include the linear-predictor ablation in Tier 1.
- EMA recipes do not have a clean notion of lambda; the width comparison with distributional recipes is at matched d only.

## 4. Open questions to resolve in Phase 0 (reading and small checks)

1. LeWorldModel defaults: embedding dimension d, lambda, number of SIGReg projections, predictor size; which Push-T state variables are logged (T-block orientation). Source: official repo and paper appendix.
2. Does the released LeWM checkpoint exist for Push-T and under what license? Can the official code retrain in hours on our GPU?
3. Nicollier et al. code: released yet? If not, implement the circle/torus heat-kernel MMD ourselves (simple for S^1) and state the difference.
4. Joseph et al. pipeline: is the public reproduction (hafeezkhan909/vjepa-physics-interp) licensed for reuse; does it include the Kubric scene generator; which V-JEPA 2 layer and token pooling did they use for layer-8 steering.
5. V-JEPA 2 weights license terms for research re-analysis (Meta license; verify).
6. Kubric and DM Control / MuJoCo licenses (Apache-2.0 expected; verify).
7. Verify the [to verify] items in `02-lit-review.md`: Wiskott 2003 optimal free responses; Sprekeler 2011; Coifman and Lafon 2005; Schaeffer et al. 2023 ID; Gardner et al. 2022.
8. Kurtosis approximation: numerical check of -1.5 / K with resonances, for K = 1..8.
9. SIGReg details: does it test projections of the whole embedding vector per batch (yes, by the LeWM description); confirm the Epps-Pulley implementation so that our "kurtosis along random projections" matches its statistic family.
10. ICML 2027 and workshop deadlines (verify on venue sites).

## 5. Risk that the result is "obvious"

A reviewer may say the outcome was predictable from SFA. Our answer is in `03-positioning.md` objection 1. Operationally: make sure the paper leads with the consequences (reinterpretation of Joseph et al.; design rule for targets and width; the filling-order signature), not with the existence of harmonics.

## 6. Risk that Tier 3 does not transfer

V-JEPA 2 is trained on natural video with EMA; its direction code may be shaped by data statistics (dominant motion directions, camera motion) rather than by the toy mechanisms. If P6 fails, the toy results still stand but the paper's reach shrinks to workshop scale. Budget Tier 3 early (from week 4) so that this is known before the writing phase.

## 7. Author-side risks

One person, one GPU, a day job. The plan is sized for that (under 150 GPU-hours), but Tier 2 rendering and Tier 3 model loading are engineering time. A collaborator who already runs V-JEPA 2 or LeWM would halve Phase 3.
