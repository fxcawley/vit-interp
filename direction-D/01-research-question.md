# 01. Research question, hypotheses, predictions, kill criteria

## 1. Beliefs attacked

B1 (interpretability of world models). A physical variable represented in a high-dimensional, distributed population code indicates the absence of a compact latent-variable representation. Source: Joseph et al. (arXiv:2602.07050) [verified 2026-10-13], Sec. 8.1: direction is "encoded through a high-dimensional population structure with circular geometry", steering it needs "many tens of directions", and the authors read this as support for "distributed, task-specific computations" over "compact, reusable latent state variables".

B2 (JEPA theory and evaluation). Representation quality is read off linear recovery R^2 of the latent; codes that are not linearly invertible are "distorted" (Nicollier et al., arXiv:2609.21656 [verified 2026-10-13]) or outside the identifiable regime (Klindt et al., arXiv:2605.26379 [verified 2026-10-13]). Nicollier et al. report, for a Gaussian target on a torus world, R^2(Z to Y) = 0.986 but R^2(Y to Z) = 0.493: the code is a deterministic function of the latent and yet "loses information required to reconstruct it linearly". Neither paper says what the code is in that regime.

Who relies on these beliefs: every probing study that counts dimensions to decide between "physics engine" and "distributed" (Joseph et al. and the cognitive-science debate they cite); every JEPA paper that reports linear-probe recovery as the measure of a world model (Klindt, Nicollier, Cui et al. arXiv:2606.27014, controlled-WM identifiability arXiv:2607.22430); every practitioner choosing SIGReg or a spherical target for a robot whose state includes headings, joint angles or phases.

## 2. The question

When a JEPA is trained on observations of a circular latent, is the geometry of its learned code, meaning its harmonic spectrum and its width, determined by (i) the latent's transition dynamics and (ii) the anti-collapse regularizer's demand for decorrelated unit-variance dimensions, in the order and amount that spectral theory predicts, or by something idiosyncratic to the model and data?

If (i) and (ii) hold, then a distributed ring code is a compact latent written in its canonical basis plus redundancy, B1 is wrong as a rule of inference, and the "distortion" in B2 is a predictable harmonic spreading from which the latent is recoverable through a known nonlinearity.

## 3. Hypotheses

H1 (basis). For a circular latent, the learned code lies in the span of Fourier harmonics of the latent, and the power in harmonic k follows the eigenvalue ordering of the latent's transition operator (slowest first).

H2 (dynamics knob). The number of retained harmonics decreases as the transition step size increases, because higher harmonics lose predictability faster: eigenvalue exp(-k^2 s^2 / 2) for Gaussian steps of SD s radians.

H3 (regularizer knob). Under a Gaussian or variance-covariance target, the number of retained harmonics increases with embedding width d (at fixed latent dimension) and with regularizer weight lambda, because the regularizer demands d decorrelated unit-variance functions of the latent and the only such functions ordered by slowness are the harmonics. Under an EMA-teacher recipe with no distributional target, the code stays near two-dimensional at the same d.

H3b (Gaussian marginals). SIGReg penalizes non-Gaussian 1-D projections, and a pure harmonic has excess kurtosis -1.5, so under SIGReg individual coordinates are mixtures of harmonics with kurtosis near -1.5/K; under VICReg-style whitening alone, coordinates can be pure harmonics and per-coordinate kurtosis is dispersed.

H4 (transfer to released models). The direction code of V-JEPA 2 at the Physics Emergence Zone and the orientation code of LeWorldModel on Push-T have harmonic spectra dominated by k = 1 after pooling across spatial positions, and the "tens of dimensions" count in V-JEPA 2 decomposes into harmonic pairs times spatial copies times speed interactions.

## 4. Predictions (numbers attached)

Notation: m(theta) is the heading-conditioned mean embedding (64 bins); p_k is the fraction of the variance of m(theta) in harmonic k; K_5 is the number of harmonics with p_k >= 0.05; PR is the participation ratio of Cov(m(theta)). Definitions and estimators in `05-methods.md`.

P1 (basis). In the heading world, >= 90% of the variance of m(theta) is explained by harmonics k <= 4 for every trained recipe, and the ordering p_1 > p_2 > p_3 holds in >= 90% of seeds. Null 1: the random-init encoder, which also yields a smooth periodic m(theta) but whose spectrum is set by the renderer; the trained spectrum must be more concentrated on k = 1 than the random one by more than the seed spread. Null 2: time-shuffled training pairs, under which the slowness ordering has no reason to hold.

P2 (dynamics). Across step sizes s in {0.1, 0.2, 0.4, 0.8} rad at fixed (recipe, d, lambda), K_5 is monotone non-increasing in s with Spearman rho(s, K_5) <= -0.8 over seeds, and the fitted decay of p_k against exp(-k^2 s^2) has slope within a factor of 2 of the theoretical one.

P3 (regularizer). Under SIGReg at fixed s, K_5 increases with d in {2, 8, 32} and with lambda over a two-decade sweep; at d = 2 the code is (cos theta, sin theta) up to rotation with circular-regression R^2 >= 0.95. Under the EMA recipe at the same d, K_5 <= 2 and the effective rank of Cov(m(theta)) is <= 3 at every d.

P3b (Gaussian marginals). Per-coordinate excess kurtosis under SIGReg is within 0.3 of -1.5/K_5 on average; under VICReg the across-coordinate SD of excess kurtosis is at least three times the SIGReg value.

P4 (what distortion is). In the torus world of Nicollier et al. with a Gaussian target, >= 50% of the variance that a linear probe to the two angles cannot explain is explained by cross-harmonics cos(k1 theta1 + k2 theta2) with k1, k2 nonzero, and the angles are recovered with circular-regression R^2 >= 0.95 from the code once harmonics up to K_5 are included as features. The latent is spread, not lost.

P5 (filling order). In a combined world with one Gaussian OU latent (rho = 0.95) and one heading latent (s = 0.3 rad), extra width under a Gaussian target is filled in the order of the interleaved eigenvalues: heading k=1 (0.956, two dims), OU He_1 (0.950), OU He_2 (0.903), OU He_3 (0.857), heading k=2 (0.835, two dims), OU He_4 (0.815), OU He_5 (0.774), heading k=3 (0.667). So the theory predicts that position acquires quadratic and cubic features before heading acquires its second harmonic. The prediction is the rank order of first appearance as d grows from 3 to 12, tested with Kendall's tau >= 0.8 against this order.

P6 (V-JEPA 2). On continuous-direction Kubric stimuli at Joseph et al.'s layer 8 (ViT-L), after pooling direction-probe weights across patches, >= 70% of the direction-subspace variance lies in harmonics k <= 2, and the orthogonal-probe dimensionality drops by a factor >= 3 relative to the unpooled count. The sawtooth period is 2.

P7 (LeWorldModel). On Push-T with the released checkpoint, the T-block orientation has K_5 >= 2 (Gaussian target, d larger than latent dimension); retraining with the official code across lambda shifts K_5 in the direction of P3.

### 4b. Exploratory predictions (reported as exploratory, never as headline)

E1 (cleanliness over training). Spectral concentration, measured as the explained-variance share of the eigenvalue-predicted leading harmonics, increases monotonically over training checkpoints and with lambda up to its plateau; early checkpoints are indistinguishable from the random-init null. Motivation: Cammarata's "already clean" thesis that better-trained models have cleaner representations; his own proposed test is the checkpoint trajectory. Labeled exploratory because the thesis is contested and the measurement is new.

E2 (observability window). In the motion-direction version of W1, where heading is visible only across frames, the harmonic heading code is absent at a one-frame observation window and appears and sharpens as the window grows (2, 4, 8 frames); a predictor with access to the latent history can carry heading the encoder cannot. Motivation: the recoverability assumption shared by the theory papers and the Long-WAM observation that the current frame is not the state.

E3 (equivariance consistency). The equivariance residual (`05-methods.md`, section 3) tracks one minus the harmonic explained variance across all cells, as the equivalence in `04-theory-notes.md` section 5b requires; a cell where they diverge indicates a measurement failure, not a new phenomenon.

## 5. Kill criteria

K1 (basis fails). In the EMA toy model, the trained spectrum is not more concentrated on low harmonics than the random-init null, or m(theta) is not periodic-smooth (for example, eight clusters rather than a ring under continuous theta). Then the spectral account is dead and the paper does not exist in this form; what remains is a replication of Nicollier's R^2 result.

K2 (knobs fail). K_5 does not move with s, lambda or d beyond seed spread in the heading world. Then the width is not set by dynamics or regularizer; report the null with the measured spectra, and the paper becomes a short note that harmonics are present but their number is not controllable, which weakens H3 but leaves H1 and the reinterpretation.

K3 (transfer fails). V-JEPA 2's pooled direction subspace is not dominated by low harmonics, or pooling does not reduce the dimensionality count. Then Joseph et al.'s distributed reading stands for the large model; the toy results publish at workshop scale as a cautionary study of regularizer effects.

K4 (optimization confound). More than half of the runs in a cell fail the preregistered optimization-success criterion (see `05-methods.md`), as happened for 5 of 10 toroidal runs in Nicollier et al. Then no claim is made from that cell; increase seeds or change the optimizer, and report the failure rate as a result.

## 6. Decision rules

- P1 and at least one of P2, P3 hold: main-track framing (mechanism with dose-response).
- P1 holds, P2 and P3 fail: workshop framing ("harmonic codes are generic; width is not controllable").
- P1 fails: stop; write up as a negative replication.
- P6 or P7 holds in addition: lead the paper with the released-model result and the reinterpretation; the toy worlds become the mechanism section.
