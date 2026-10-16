# 03. Positioning

## 1. Contribution statement (draft)

We study how joint-embedding predictive architectures represent a circular latent variable such as heading, orientation or phase. Classical slow-feature theory predicts that any learner that extracts slow, decorrelated features of such a variable represents it in the Fourier basis of its transition operator, harmonics ordered by slowness. We show that this prediction holds in pixel-trained JEPAs across recipes, and we identify what sets the number of harmonics: the step size of the latent dynamics, the regularizer's demand for decorrelated unit-variance dimensions, and the embedding width. Under a Gaussian or variance target with spare width, the latent spreads across harmonics in a predictable order; under an EMA-teacher recipe it stays two-dimensional. The "distortion" that identifiability theory attributes to a mismatched target is this spreading, and the latent remains recoverable through a known nonlinearity. Applying the analysis to released models, we find that the high-dimensional direction code reported in V-JEPA 2 is a compact ring written in its canonical basis, multiplied by spatial redundancy, and that LeWorldModel's orientation code follows the same rule. Dimensionality counts therefore cannot by themselves distinguish distributed computation from a compact latent, and the choice of regularizer target and width is a choice about code geometry.

## 2. What is left after Klindt and Nicollier

| Aspect | Klindt et al. (May 2026) | Nicollier et al. (Sep 2026) | This direction |
|---|---|---|---|
| Latents | Gaussian (theorem), generalized-normal sweep, Reacher angles (conflated) | Riemannian manifolds: sphere, torus, Clifford torus | Circle, torus, OU, bounded-uniform, and their products |
| Measurement | R^2 linear recovery | R^2 both directions, invariance loss | Harmonic spectrum p_k, K_5, participation ratio, per-coordinate kurtosis, cross-seed subspace alignment, plus R^2 for comparability |
| Regularizer | SIGReg (also VICReg, InfoNCE in sweep) | Gaussian vs geometry-matched heat-kernel MMD; lambda selected, not varied | SIGReg, VICReg, matched target, EMA-teacher (no target); lambda and d as experimental knobs |
| Dynamics | rho fixed per experiment | diffusion time fixed | step size s as a knob with a quantitative decay prediction |
| Released models | none | none | V-JEPA 2 (Joseph et al. protocol, continuous directions, pooled), LeWorldModel Push-T |
| Claim about the non-identifiable regime | "broken" | "distorted" | characterized: predictable harmonic spreading; latent recoverable |
| Link to interpretability | none | none | reinterprets the distributed direction code |

The theorem is theirs. The characterization of the regime the theorem excludes, the knobs, and the transfer to interpretability are open.

## 3. Novelty relative to each neighbor, one line each

- Joseph et al.: they report the ring and the dimension count; we explain the basis, decompose the count, and test with continuous directions.
- Klindt et al.: they prove linear recovery for Gaussian latents; we measure what the code becomes when the latent is a circle, which their Appendix F frames but does not test.
- Nicollier et al.: they show mismatched targets lower R^2; we show what the mismatched code is, that its width scales with lambda and d, and that the latent is not lost.
- Franzius / Wiskott (SFA): they predicted harmonic head-direction codes from slowness; we test the prediction in modern JEPAs with distributional regularizers and add the regularizer-width mechanism.
- Dorrell et al. and the grid-cell literature: they show constraints set the harmonic content of spatial codes in trained networks; we transfer the logic to JEPA regularizers.
- Symmetry-based disentanglement (Higgins, Quessard, Marchetti, Keurti): they build equivariant (cos, sin) codes by design; we show the same code emerges, and widens into higher irreducible representations, from generic JEPA objectives.
- Spectral-Target Latent Structuring (arXiv:2609.04264): they add a supervised Fourier head to fix "laziness"; we show the Fourier structure is what the unsupervised objective already converges to when width and lambda permit, and that laziness is the small-lambda end of the same curve.
- Nelson et al. (statistical vs structural identifiability): our cross-seed result instantiates the distinction: harmonic subspaces are structurally identifiable, coordinates are not even statistically stable.

## 4. Why the answer changes what people do

- Interpretability practice: stop reading orthogonal-probe dimension counts as evidence for or against compact latents without first (a) pooling spatial copies, (b) fitting harmonics for any variable with circular topology, (c) checking the recipe's regularizer. Provide the harmonic-regression diagnostic as a drop-in tool.
- World-model design: if the state has angles (joints, headings, phases, wheel rotations), a Gaussian target with spare width spends dimensions on higher harmonics and cross-harmonics; a product-manifold target or a width matched to the latent count keeps the code compact. This bears directly on LeWM and SPHERE-JEPA users.
- Theory: a concrete, measured description of the non-Gaussian regime, with the SFA connection made explicit, which the theory papers invoke but do not test.

## 4b. Framing: relational evaluation of representations (flavor, not a rewrite)

Two 2026 positions from the community that will read this paper supply language for what D already does; neither changes the question.

- Self-consistency (Pres et al., arXiv:2608.05188, ICML 2026 Position track). Their claim, for LLM behavior, is that failures hide when behavior is specified and evaluated on single input-output pairs, and that the primary object should be the relations between outputs across related inputs, formalized as invariances f(g(x)) = f(x) and equivariances f(g(x)) = g'(f(x)). A JEPA objective is a consistency objective in exactly this formalism: the relation is the positive pair, the output relation is equality under the predictor, and the consistency function is the alignment loss; the identifiability theorems are statements about what consistency optimization yields in representation space. D's central measurement is equivariance consistency: a code on which rotating the heading by Delta acts as one linear map for every input is, for a circular latent, the same thing as a code in the harmonic basis (`04-theory-notes.md`, section 5b). Linear-probe R^2 is single-point evaluation of a representation; harmonic structure, equivariance residuals and cross-seed subspace alignment (Nelson et al.) are relational evaluations. We say this once in the introduction and once in the discussion. We do not import behavioral consistency, self-description or consistency training; those are LLM-side programs.
- "Already clean" (Cammarata, Goodfire essay, Oct 2026; contested, not peer-reviewed, and read here through secondary coverage because the host is blocked from this network). His thesis is that apparent messiness in representations has often been a measurement and training artifact, and that better-trained models are cleaner. D is a theory-backed instance: what was counted as a 40-80-dimensional distributed code is a canonical basis times redundancy, and the mess is in the count. His proposed discriminating test, watching representations form across training checkpoints, is free for us because we train the toy models; it enters as exploratory prediction E1, not as a plank.
- Partial observability (Long-WAM, Huang, Zhang et al., NVIDIA 2026). Control models need history because the current frame is not the state. D's theory, like Klindt's and Nicollier's, assumes the latent is recoverable from the observation; motion direction is recoverable only from a window of frames. This enters as the observation-window knob (E2) and as a limitation paragraph, and as a data-side explanation to reach for if P6 fails on a model pretrained on egocentric video.

## 5. Anticipated objections and responses

1. "This is slow feature analysis from 2002." Yes, the prediction is classical and we say so in the first paragraph. The open questions are empirical (do modern pixel JEPAs with these regularizers do it), mechanistic (what sets the width) and interpretive (what it means for Joseph et al.). A theory-driven prediction that holds is a strength in a field where most probing results have no prediction attached.
2. "Toy worlds." Tier 2 uses the exact environments of Nicollier et al. (reacher torus) and Joseph et al. (Kubric sphere, re-rendered with continuous directions); Tier 3 analyzes released V-JEPA 2 and LeWorldModel checkpoints. If Tier 3 fails, we say so (K3).
3. "Harmonic structure is trivial: any smooth periodic embedding has a Fourier series." Correct, and the random-init encoder also yields one. Our claims are about concentration and ordering relative to that null, and about dose-response in s, lambda and d. Those are not trivial and can fail (K1, K2).
4. "Nicollier already showed the Gaussian target distorts." They showed R^2 drops. We show the form of the distortion, its scaling, and that the information is retained. Different claim, different measurement, and it changes how their result should be read.
5. "Joseph et al.'s neurons already show sinusoidal tuning, so you add nothing." They show k = 1 tuning in single units; they did not decompose the subspace, did not pool spatial copies, used eight directions, and drew the opposite conclusion from the dimension count. Our P6 is a falsifiable test of their interpretation.
6. "Eight directions in their stimuli cannot resolve harmonics above three." Agreed; we re-render with continuous directions using the public reproduction of their pipeline.
7. "V-JEPA 2 has no Gaussian target, so the regularizer story does not apply to it." Correct, and that is why the EMA recipe is a separate arm: H1 and H2 (basis and dynamics) apply to any slow-feature learner; H3 (width from the regularizer) applies to distributional targets. For V-JEPA 2 the prediction is a compact harmonic code plus spatial copies, not regularizer-driven spreading. If V-JEPA 2 spreads for another reason, that is a finding too.
8. "Optimization sensitivity makes the results fragile." Nicollier lost 5 of 10 torus runs. We preregister an optimization-success criterion that does not look at latents, report failure rates as results, and use >= 5 seeds in key cells.
9. "Why not just prove it?" The SFA theorem covers the whitening-constrained case; the SIGReg case with finite lambda and a nonlinear predictor is not closed-form. We state the theory we rely on, derive the predictions in `04-theory-notes.md`, and test them; a proof is a separate paper and the Brown group is better placed to write it.

## 6. Slop check (Durrett's pattern: narrow question, off-the-shelf tool, exaggerated conclusion)

- Narrow question? No: it bears on how a whole class of probing results is read and on a design choice in every JEPA world model with angular state.
- Off-the-shelf tool run blindly? The tools are probes and regressions, but each prediction comes with a number derived from theory before the run, and the random-init null and time-shuffle control can falsify it.
- Exaggeration risk: the main one is overclaiming about V-JEPA 2 from a toy result. The decision rules in `01-research-question.md` keep the released-model claims tied to P6 and P7.
- Belief the field relies on: B1 and B2 are stated with citations to the papers that hold them.
- Mechanism: spectral filling order with two dose-response knobs and one capacity knob, plus the Gaussian-marginal kurtosis signature that distinguishes SIGReg from VICReg.

## 7. Venue strategy

North star: a main-track interpretability or representation-learning paper. Candidate deadlines, all to verify on the venue sites: ICML 2027 (abstract/paper late January 2027), NeurIPS 2027 (May 2027). Workshop fallbacks with the strip-down map in `09-writing-plan.md`: CVPR 2027 interpretability workshops (MIV, HOW; March 2027 if the 2026 pattern holds), ICLR 2027 workshops on world models or representational alignment (February 2027), UniReps at NeurIPS 2027. TMLR is the fallback for a careful null.

Preprint policy: post to arXiv as soon as Tier 1 plus one Tier 3 result are in hand, before the main-track deadline, because the scoop exposure is high (`07-risks-and-open-questions.md`).
