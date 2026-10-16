# 05. Methods

## 1. Worlds (latent laws and renderers), in tiers

Tier 0, no renderer (minutes per run). Latents mixed by a fixed random RealNVP or sinusoidal shear as in Klindt et al. Purpose: calibrate the pipeline against their reported R^2 and against Nicollier's torus numbers before touching pixels. Not a result.

Tier 1, sprite worlds (minutes to an hour per run). 64 x 64 grayscale or RGB, rendered on the fly (PIL or a JAX/NumPy rasterizer). All latents continuous.
- W1 heading: one asymmetric glyph (an arrow or "F" shape, no rotational symmetry) at fixed position with orientation theta; dynamics theta' = theta + N(0, s^2) wrapped. Static-orientation version uses frame pairs; a motion-direction version uses 4-frame clips of a dot translating with heading theta and fixed speed, so that direction is only visible across frames (Joseph et al.'s variable).
- W2 position: a dot with 2-D OU position, rho = 0.95 (Nicollier's value).
- W3 bounded speed or position: uniform on an interval with reflecting steps.
- W4 combined: W1 glyph with W2 position, independent dynamics (for P5).
- W5 torus: two glyph orientations or a two-link arm drawn as line segments (for P4 before Tier 2).
- Nuisances toggled per cell: background texture with random phase per frame, brightness jitter, small additive noise. Purpose: ensure the code is not a pixel artifact and give the random-init null something to encode.

Tier 2, benchmark environments (hours per run).
- Reacher-hard from DeepMind Control / MuJoCo with joint angles uniform on T^2 and a torus diffusion, matching Nicollier et al. (128 x 128), for direct comparability with their R^2 values.
- Kubric sphere videos as in Joseph et al., re-rendered with continuous directions (e.g., 72 directions x 7 speeds x 7 starts) at 256 x 256, 16 frames. Reuse the public reproduction pipeline if its license allows.

Tier 3, released models (inference only plus probes).
- V-JEPA 2 ViT-L (and H/G if compute allows) at the layers Joseph et al. identify; their circular-regression probes; our pooled and harmonic analyses.
- LeWorldModel released checkpoint on Push-T (T-block orientation is the circular latent; the state is also logged, so ground truth is available) and OGBench-Cube if an angular state exists; retraining with the official code across lambda.

## 2. Recipes

R1 SIGReg (LeWM-style): encoder + predictor, loss = prediction + lambda SIGReg, official implementation. Knobs: lambda over {0.01, 0.1, 1, 10} (to be re-centered after Phase 0 on the official default), d in {2, 8, 32}.
R2 VICReg-style: same architecture, variance + covariance penalties in place of SIGReg, matched so that the whitening strength is comparable (calibrate by the covariance off-diagonal norm at convergence).
R3 EMA-teacher / stop-gradient JEPA: same encoder, target encoder as EMA, no distributional term (I-JEPA / V-JEPA style); for clips, masked future prediction.
R4 Matched target: uniform-on-S^1 (or product) target via heat-kernel or simple MMD on the circle; for the torus world, Nicollier's product heat-kernel MMD if code is released, else our implementation.
R5 Controls: random-init encoder (same architecture, no training); time-shuffled training pairs (x, x') drawn from independent times, which destroys the dynamics while keeping the marginal and the regularizer.

Encoder: a small CNN (4-6 conv layers, 1-5M parameters) for Tier 1; EdgeNeXt-XX-Small or a small ViT for Tier 2 to match Nicollier; predictor a 2-layer MLP (linear predictor as an ablation, since the theory is cleanest with a linear predictor). Same encoder across recipes within a world.

## 3. Readouts (definitions)

Let theta be the ground-truth angle, binned into B = 64 equal bins; m_b = mean of y over samples in bin b; M the B x d matrix of bin means.

- Harmonic regression. Design matrix H with columns cos(k theta_b), sin(k theta_b) for k = 1..K_max (K_max = 8). Least squares M = H C + E. Power of harmonic k: P_k = ||C_k||_F^2 (rows of C for harmonic k). Normalized p_k = P_k / sum_j P_j. Periodic variance fraction: 1 - ||E||^2 / ||M - mean||^2. K_5 = #{k : p_k >= 0.05}. Also report the spectrum p_k itself.
- Width. Participation ratio PR = (sum_i lambda_i)^2 / sum_i lambda_i^2 of the eigenvalues of Cov over bins of M; effective rank at 95% variance; and Joseph et al.'s orthogonal probe sequence (iteratively train a circular-regression probe, project out, retrain, until chance) for comparability.
- Linear identifiability. R^2(z to y) and R^2(y to z) with ridge, as in Klindt and Nicollier; circular-regression R^2 from y to (cos theta, sin theta).
- Kurtosis. Excess kurtosis of each coordinate of y over the data distribution; excess kurtosis along 100 random unit projections (SIGReg's own statistic family).
- Cross-seed alignment. For each harmonic k, the subspace spanned by C_k rows across seeds: principal angles; report the mean cosine. Coordinate-level: maximum absolute correlation matching across seeds (Hungarian), compared with the random-rotation reference sqrt(2 ln d / d).
- Behavioral use. Next-step prediction error of the predictor; for Tier 2/3, planning success or surprise metrics where the environment supports them, to show the code is used, not decorative.
- Pooling (Tier 3). Direction probes per patch position; average probe weight vectors across positions before orthogonalization; report dimensionality before and after pooling and after regressing out speed.

## 4. Preregistered optimization-success criterion (no peeking at latents)

A run counts as converged if, at the end of training, (a) the alignment / prediction loss is within 10% of the best value in its cell across seeds, (b) the regularizer loss is within the band observed for the official recipe on its own benchmark, and (c) the covariance of y has no eigenvalue below 1e-3 of the largest, unless the recipe is EMA (where collapse is checked instead by the variance of y exceeding a preregistered floor). Runs that fail are reported as failures, not dropped silently. Threshold values are fixed in `00-preregistration.md` after Phase 0 calibration.

## 5. Sweeps

- Dynamics (P2): s in {0.1, 0.2, 0.4, 0.8} rad, at R1 with (lambda, d) = default, and at R3. 3 seeds each.
- Regularizer (P3, P3b): R1 and R2 over lambda x d = 4 x 3 at s = 0.3; R3 over d; 3 seeds each, 5 for the (default lambda, d = 32) cell.
- Filling order (P5): W4 with d in {3, 4, 5, 6, 8, 10, 12} at R1 default lambda; 5 seeds.
- Torus (P4): W5 and Tier 2 reacher at R1 Gaussian target and R4 matched target; 5 seeds (expect optimization failures; see K4).
- Controls: random-init and time-shuffled at every world, 3 seeds.

Approximate run count for Tier 1: dynamics 2 x 4 x 3 = 24; regularizer 2 x 12 x 3 + extras about 80; EMA width 3 x 3 = 9; filling order 7 x 5 = 35; torus 2 x 5 = 10; controls about 30. Total about 190 runs at 10-20 minutes each: 30-60 GPU-hours. Tier 2: about 20 runs at 2-4 hours: 40-80 GPU-hours. Tier 3: inference over a few thousand clips plus probes: under 10 GPU-hours. Total under 150 GPU-hours on one 24-48 GB card; wall-clock two to three weeks with small models run in parallel.

## 6. Statistics

- Seeds: >= 3 everywhere, >= 5 for the cells that carry a headline (P1 default cell, P3 widest cell, P5, Tier 2 torus).
- Report mean and SD across seeds; paired bootstrap CIs for differences between recipes within a world; Spearman rho with bootstrap CI for monotonic predictions (P2, P3); Kendall's tau for the filling order (P5).
- The random-init null and the time-shuffle control are reported next to every trained spectrum.
- No metric is defined after seeing results; new metrics go in an exploratory section.

## 7. Software and data

- PyTorch; LeWorldModel official code (github.com/lucas-maes/le-wm) for R1 and for Tier 3 retraining; V-JEPA 2 official weights and loaders; Kubric (Apache-2.0, to verify) for Tier 2 videos; DeepMind Control / MuJoCo (Apache-2.0, to verify) for reacher; the public reproduction of Joseph et al.'s probing (license to check) or our reimplementation.
- All worlds are synthetic; no proprietary data. KLA data is not used anywhere.
- Release: code, configs, seeds, rendered datasets or generators, trained toy checkpoints, the preregistration file with its commit hash.
