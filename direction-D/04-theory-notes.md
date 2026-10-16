# 04. Theory notes: where each prediction comes from

These are working derivations, not proofs. Each ends with what would falsify it. Notation: latent z (or theta for a circle), observation x = g(z), encoder f, code y = f(x) in R^d, transition operator (T h)(z) = E[h(z') | z] for one step of the latent dynamics.

## 1. The working theory: a JEPA fills its width with the slowest eigenfunctions

Alignment loss E||y' - y||^2 (or with a linear predictor, E||P y - y'||^2) rewards features that change little between positive pairs. An anti-collapse constraint of the whitening type (Cov(y) = I) forces d decorrelated unit-variance features. The constrained minimum of the alignment loss over functions of z is the span of the top-d eigenfunctions of T (equivalently, the slowest features): this is the slow-feature-analysis theorem (Wiskott and Sejnowski 2002) and the spectral-embedding view of SSL (HaoChen et al. 2021; Balestriero and LeCun 2022; Cui et al. 2026). Klindt et al. 2026 use the same decomposition (Hermite basis for Gaussian latents, Appendix F for the general case).

Two caveats we test rather than assume: (a) SIGReg is not a hard whitening constraint; at finite lambda the learner can leave some dimensions under-dispersed ("laziness", arXiv:2609.04264) instead of paying alignment cost for fast features. (b) An EMA-teacher recipe has no explicit rank demand; its effective width is set by implicit dynamics, and the prediction is that it stays near the minimum needed.

Falsified by: K1 (codes not in the harmonic span beyond the random-init null) or by a trained spectrum that is flatter than the eigenvalue ordering under the time-shuffle control's null.

## 2. Eigenfunctions and eigenvalues per latent law

| Latent law (stationary) | Transition | Eigenfunctions | Eigenvalues | Slowest nontrivial code |
|---|---|---|---|---|
| Gaussian AR(1)/OU: z ~ N(0,1), z' = rho z + sqrt(1-rho^2) eps | Mehler kernel | probabilists' Hermite He_n(z) | rho^n | He_1 = z, linear, 1 dim |
| Circle, uniform theta, theta' = theta + eta mod 2pi, eta ~ N(0, s^2) | wrapped-Gaussian convolution | cos k theta, sin k theta | exp(-k^2 s^2 / 2), each doubly degenerate | (cos theta, sin theta), 2 dims, nonlinear in theta |
| Circle with von Mises steps of concentration kappa | convolution | same | I_k(kappa) / I_0(kappa) | same |
| Interval [0, L], uniform, reflected Gaussian steps of SD sigma | reflected kernel | cos(n pi x / L) | approx exp(-n^2 pi^2 sigma^2 / (2 L^2)) for small steps | cos(pi x / L): monotone, nonlinear, compressed at the ends; 1 dim |
| Torus T^2 (two independent circles) | product | cos/sin(k1 theta1 + k2 theta2) | product of the two circle eigenvalues | two (cos, sin) pairs, then cross terms |

Reading: the Gaussian case is the only one where the slowest eigenfunction is linear (this is Klindt's theorem). The interval case gives Klindt's "monotone, not linear" general result. The circle gives a two-dimensional first eigenspace, which no monotone 1-D function can represent: Klindt's Appendix F argument does not cover it because it presumes a Sturm-Liouville problem on an interval with a simple first eigenvalue.

Numbers used in P2 and P5: with s = 0.3 rad, harmonics k = 1..4 have eigenvalues 0.956, 0.835, 0.667, 0.487; with rho = 0.95, He_1..He_5 have 0.950, 0.903, 0.857, 0.815, 0.774. Interleaving gives the filling order stated in P5.

Falsified by: a measured spectrum whose decay in k does not track exp(-k^2 s^2) across the s sweep (P2).

## 3. Why spare width becomes higher harmonics (H3)

If the regularizer enforces d decorrelated unit-variance coordinates and the world has only one circular latent, then the code must consist of d orthonormal functions of theta under the uniform measure. The alignment loss prefers slow ones. The ordered orthonormal system of L^2(S^1) by slowness is the Fourier basis, so the code spans harmonics 1..K with 2K = d (if d is odd one half-pair, or a partially collapsed dimension). Increasing d or lambda (which enforces the rank demand more strictly) increases K; decreasing lambda lets the learner collapse dimensions instead, giving K_5 < d/2 and "laziness".

For a world with several latents the same argument fills width in the interleaved eigenvalue order (P5), which predicts that a Gaussian position latent acquires He_2 and He_3 features before heading acquires k = 2 at the stated parameters. This is the sharpest signature in the plan because it is a specific rank order with no free parameter once rho and s are set.

Under the EMA-teacher recipe there is no rank demand; the predicted code is the minimum slow subspace, 2 dims per circle and 1 per Gaussian, with remaining dimensions low-variance or redundant copies.

Falsified by: K2 (K_5 does not move with d or lambda), or by EMA codes with K_5 > 2 at matched d (which would mean width arises from something other than the regularizer, e.g., optimization dynamics or augmentation, and would be its own result).

## 4. Why SIGReg coordinates are mixtures: kurtosis of harmonic sums (H3b)

A single harmonic x = cos theta with theta uniform has E[x^2] = 1/2, E[x^4] = 3/8, so excess kurtosis = (3/8)/(1/4) - 3 = -1.5 (arcsine marginal). SIGReg penalizes any 1-D projection whose marginal departs from N(0,1), so a coordinate equal to a pure harmonic is penalized. A unit-variance mixture of K harmonics with equal amplitudes and random phases, w . phi(theta) = sum_k r cos(k theta - psi_k), has excess kurtosis approximately -1.5 / K: the fourth cumulants add across harmonics (each -3/8 r^4) while the variance squared scales as (K r^2 / 2)^2, giving (-3 K r^4 / 8) / (K^2 r^4 / 4) = -1.5 / K; resonance terms between harmonics (k1 + k2 = k3, etc.) vanish on average over random phases but not per realization, so this is an approximation to be checked numerically in Phase 0.

Consequences: (i) under SIGReg, the learner prefers rotations of the harmonic span that mix many harmonics into every coordinate, so per-coordinate excess kurtosis should sit near -1.5 / K_5 with small spread; (ii) under VICReg (variance and covariance penalties only) there is no marginal-shape penalty, so coordinates may be pure or nearly pure harmonics and per-coordinate kurtosis is dispersed between about -1.5 and 0; (iii) a Gaussian-marginal target can only be approached by increasing K, which is a second, independent pressure toward width under SIGReg beyond the rank demand. Note that SIGReg tests random projections of the whole vector, not coordinates, so the relevant quantity is kurtosis along random projections; per-coordinate kurtosis is the stronger, more specific test, and both are reported.

Falsified by: P3b failing, i.e., SIGReg and VICReg coordinates having the same kurtosis distribution.

## 5. Identifiability up to harmonic phase, and what is stable across seeds

For d = 2 and a circular latent, the optimum is y = R (cos theta, sin theta) with R in O(2): recovery up to phase shift and orientation flip. This is the circular analogue of Klindt's "up to rotation" and of Nicollier's spherical result restricted to S^1 (consistent with their matched-target torus runs reaching R^2 = 0.996).

For d = 2K, the alignment objective's value depends only on the span, and the span of harmonics 1..K is canonical because each harmonic pair is an eigenspace with a distinct eigenvalue. The coordinates, however, are free up to O(2K) (and SIGReg actively prefers mixing). Prediction: across seeds, the harmonic-k subspaces recovered by regression align (principal angles near zero), while individual coordinates do not match (coordinate correlation near the random-rotation value, about sqrt(2 ln d / d)). This is the concrete form of Nelson et al.'s structural-vs-statistical distinction and of the "basis" gap noted in the JEA review.

Falsified by: harmonic subspaces failing to align across seeds while R^2 to (cos, sin) is high (which would indicate the regression is fitting a non-canonical decomposition, i.e., eigenvalues not well separated at that s).

## 5b. Equivariance consistency is the harmonic basis

Say a code m(theta) in R^d is equivariance-consistent under rotation if for every shift Delta there is one linear map R(Delta) with m(theta + Delta) = R(Delta) m(theta) for all theta. Then the span of {m(theta)} is a finite-dimensional subspace of functions on the circle that is invariant under all shifts, and every shift-invariant finite-dimensional subspace of L^2(S^1) is a direct sum of Fourier modes (the irreducible representations of SO(2) are the pairs (cos k theta, sin k theta)). So an equivariance-consistent code is a harmonic code, with R(Delta) block-diagonal in the harmonic basis, rotating the k-th block by k Delta. Conversely a harmonic code is equivariance-consistent. The two readouts, harmonic explained variance and equivariance residual, therefore measure one property from two sides (E3), and the harmonic regression is the relational evaluation that Pres et al.'s framework asks for, applied to a representation rather than to behavior.

Two cautions for the estimator. If the code is sampled at B angle bins and d >= B, a linear R(Delta) exists for any injective code (define it on the span), so the test is empty; use B >= 4d or restrict R(Delta) to be orthogonal, which additionally requires stationarity of inner products <m(theta), m(theta')> depending only on theta - theta'. And an equivariance fit per Delta does not by itself check that the maps compose, R(Delta_1 + Delta_2) = R(Delta_2) R(Delta_1); report the composition error as the stronger test.

Falsified by: E3 failing, which would mean the harmonic regression is fitting something other than a shift-invariant subspace (for example a code that is periodic in theta but with angle-dependent speed around the ring).

## 6. What the torus "distortion" should consist of (P4)

On T^2 with a Gaussian target and width d > 4, the ordered eigenfunctions include cross-harmonics cos(k1 theta1 + k2 theta2) with eigenvalue exp(-(k1^2 + k2^2) s^2 / 2). A linear probe from the code to (cos theta1, sin theta1, cos theta2, sin theta2) uses only the k = (1,0) and (0,1) pairs; variance in cross terms is "unexplained" to that probe, which lowers R^2(Y to Z) while R^2(Z to Y) stays high because the code is a deterministic function of (theta1, theta2). This reproduces Nicollier's asymmetry (0.493 vs 0.986) with a mechanism and predicts that adding cross-harmonic features as regressors closes the gap.

Falsified by: the unexplained variance not being in cross-harmonics (e.g., in non-periodic nuisance features), which would point to the renderer or encoder rather than the objective.

## 7. Why Joseph et al.'s count can be large without a distributed computation (P6)

Three multiplicative sources of orthogonal probe directions for direction in a patch-token transformer: (a) harmonic pairs, 2 per retained harmonic (sawtooth period 2); (b) spatial copies, one per patch position that carries the variable (they report direction becomes spatially redundant post-PEZ); (c) interactions with speed and start position in their stimulus grid (7 speeds x 7 starts), which make the velocity vector rather than the angle the natural variable, adding dimensions that co-vary with direction. Pooling probe weights across patches removes (b); fitting speed as a separate regressor removes most of (c); the remainder should be a few harmonic pairs. With 8 training directions, harmonics above k = 3 are aliased, which is why continuous-direction stimuli are required.

Falsified by: K3.

## 8. Things the theory does not predict and we will not claim

- Which layer of a deep encoder carries the code (the Physics Emergence Zone is an empirical depth phenomenon outside this framework).
- Behavior of the predictor network (we probe encoders; the predictor is left for later).
- Anything about non-stationary or action-conditioned dynamics except through the released LeWM analysis, where actions are present and the theory is approximate.
- Anything under partial observability. The theory assumes the latent is recoverable from the observation the encoder sees. When it is not (heading from a single frame; task progress from a single egocentric view, the case Long-WAM makes for long context), the slowest eigenfunctions are functions of the belief state, which lives in the history, and a frame encoder cannot carry them. The observation-window knob (E2) marks where the assumption starts to hold; nothing here predicts what happens below it.
