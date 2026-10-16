# 00. Preregistration (DRAFT; freeze with a commit hash at the end of Phase 0)

Date drafted: 2026-10-13. Status: not frozen. Thresholds marked [calibrate] are set in Phase 0 without looking at any Tier 1 result.

## Question

Is the geometry of a JEPA's code for a circular latent (harmonic spectrum and width) set by the latent's dynamics and the regularizer's rank demand in the order and amount spectral theory predicts?

## Primary predictions (confirmatory)

P1 Basis. Heading world W1, all trained recipes: >= 90% of the variance of the heading-conditioned mean embedding in harmonics k <= 4; p_1 > p_2 > p_3 in >= 90% of seeds; concentration on k = 1 exceeds the random-init encoder's by more than twice the seed SD; the time-shuffled control does not show the slowness ordering.

P2 Dynamics. R1 and R3, s in {0.1, 0.2, 0.4, 0.8} rad: Spearman rho(s, K_5) <= -0.8; fitted decay of p_k vs exp(-k^2 s^2) within a factor of 2 of theory.

P3 Regularizer. R1 at s = 0.3: K_5 increases with d in {2, 8, 32} and with lambda over two decades; at d = 2, circular R^2 >= 0.95. R3 at the same d: K_5 <= 2 and effective rank <= 3.

P3b Marginals. SIGReg per-coordinate excess kurtosis within 0.3 of -1.5 / K_5 on average; VICReg across-coordinate SD of kurtosis >= 3x SIGReg's.

## Secondary predictions (confirmatory if Gate 1 passes)

P4 Torus. Gaussian target on W5 / reacher: >= 50% of variance unexplained by the linear angle probe is in cross-harmonics; circular R^2 >= 0.95 once harmonics up to K_5 are features.

P5 Filling order. W4 (rho = 0.95, s = 0.3), d from 3 to 12: Kendall tau >= 0.8 against the order heading k=1, He_1, He_2, He_3, heading k=2, He_4, He_5, heading k=3.

P6 V-JEPA 2. Layer 8 ViT-L, continuous directions: >= 70% of pooled direction-subspace variance in k <= 2; orthogonal-probe count drops by >= 3x after pooling; sawtooth period 2.

P7 LeWorldModel. Push-T orientation: K_5 >= 2 at the released checkpoint; K_5 moves with lambda in the direction of P3 on retraining.

## Exploratory (reported separately; no confirmatory weight)

E1 Spectral concentration rises monotonically over training checkpoints and with lambda; early checkpoints match the random-init null.
E2 W1-motion: no heading code at a one-frame window; code appears and sharpens at 2, 4, 8 frames.
E3 Equivariance residual tracks one minus harmonic explained variance in every cell.

## Kill criteria

K1 P1 fails (no concentration beyond the random-init null, or non-periodic structure under continuous theta).
K2 K_5 insensitive to s, lambda and d beyond seed spread.
K3 P6 fails and P7 fails.
K4 Majority of runs in a cell fail the optimization-success criterion.

## Optimization-success criterion [calibrate]

Prediction loss within 10% of the cell's best; regularizer loss within the band of the official recipe on its benchmark; smallest covariance eigenvalue of y >= 1e-3 of the largest (distributional recipes) or variance floor [calibrate] (EMA). Failures are reported, never dropped.

## Analysis plan

Readouts and estimators exactly as in `05-methods.md`, section 3; seeds and statistics as in section 6. No new primary metric after unblinding; exploratory analyses are labeled as such.

## Decision rules

- P1 and (P2 or P3): main-track framing.
- P1 only: workshop framing.
- not P1: negative replication note.
- P6 or P7 in addition: released-model result leads.

## Freeze record

Commit hash: (to fill). Date frozen: (to fill). Changes after freeze are logged below with reasons.
