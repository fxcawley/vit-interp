# JEA Field Pulse: Joint-Embedding Architectures

Oct 9, 2026 · @liam cawley

## Pulse

In the past twelve months JEA research split into two programs: Meta FAIR scaling V-JEPA for video and robots, and LeCun's new AMI Labs turning JEPA into a heuristic-free, provable world-model recipe. LeCun left Meta in November 2025, and AMI Labs announced a $1.03B seed round in March 2026 ([Wikipedia](https://en.wikipedia.org/wiki/Advanced_Machine_Intelligence_Labs)). Interpretability of JEPAs started only this year. Scope here: JEA means joint-embedding architectures broadly (DINO family included), with the JEPA line at the center.

| Date | Paper | Result in one line |
| --- | --- | --- |
| 2026-05 | [When Does LeJEPA Learn a World Model? (Klindt, LeCun, Balestriero)](https://arxiv.org/abs/2605.26379) | Proves LeJEPA recovers latent variables up to a rotation for stationary, additive-noise worlds, and that Gaussian latents are the only law with this guarantee |
| 2026-03 | [V-JEPA 2.1 (Mur-Labadia et al.)](https://arxiv.org/abs/2603.14482) | Loss on visible and masked tokens plus deep self-supervision fixes weak dense features; 20-point real-robot grasping gain over V-JEPA 2-AC |
| 2026-03 | [LeWorldModel (Maes et al.)](https://arxiv.org/abs/2603.19312) | First stable end-to-end JEPA from pixels with two loss terms; \~15M params, one GPU, a few hours; plans up to 48x faster than foundation-model world models |
| 2026-02 | [Interpreting Physics in Video World Models (Joseph et al.)](https://arxiv.org/abs/2602.07050) | First interp study of V-JEPA 2 internals: physics becomes decodable at about one-third depth, encoded in distributed, task-specific codes |
| 2025-11 | [LeJEPA (Balestriero, LeCun)](https://arxiv.org/abs/2511.08544) | Isotropic Gaussian embeddings claimed optimal; SIGReg enforces it; no stop-gradient or teacher; 79% ImageNet linear probe with ViT-H/14 pretrained on ImageNet-1k |
| 2025-08 | [DINOv3](https://arxiv.org/abs/2508.10104) | Gram anchoring stops dense features degrading in long training of large image JEAs |
| 2025-06 | [V-JEPA 2 (Assran et al.)](https://arxiv.org/abs/2506.09985) | 1M+ hours of video; 77.3 SSv2 top-1, 39.7 EK-100 recall@5; action-conditioned version plans zero-shot on Franka arms from under 62 hours of robot video |
| 2025-02 | [Intuitive physics from V-JEPA (Garrido et al.)](https://arxiv.org/abs/2502.11831) | Latent-prediction video models show surprise at physically impossible events on IntPhys, GRASP and InfLevel |
| 2024-07 | [How JEPA Avoids Noisy Features (Littwin et al.)](https://arxiv.org/abs/2407.03475) | In deep linear nets, JEPA prefers high-influence features while MAE prefers high-variance ones |

The center of gravity moved from "JEPA is a better self-supervised objective" to "JEPA is a provable world model". That second claim is where the beliefs below are weakest and most testable.

## Belief 1: latent prediction learns more semantic features than pixel reconstruction

**Claim.** Predicting in representation space (JEPA) yields more semantic features than reconstructing pixels (MAE), because it can ignore unpredictable detail.

**Background.** Balestriero and LeCun (ICML 2024, from memory) argued that reconstruction spends capacity on high-variance pixel detail, while perception-relevant information sits in low-variance subspaces. [Littwin et al. (NeurIPS 2024)](https://arxiv.org/abs/2407.03475) gave the mechanism in deep linear networks.

**Method.** Training dynamics of deep linear encoders under a JEPA objective and an MAE objective, on data where feature variance and feature predictiveness can be set separately.

**Result.** JEPA learns high-influence features first (large regression coefficients between views); MAE prioritizes high-covariance features even when they are noise. The bias appears only with deep encoders and fades with shallow ones. Empirically the picture is mixed: a 2026 lung-ultrasound study found V-JEPA near chance on one dataset (35.1 ± 4.9 balanced accuracy) where contrastive MoCo led (62.7), with the ranking reversed on another dataset ([Sep 2026 preprint](https://arxiv.org/abs/2609.16551)).

**Interpretation.** The theory is clean but lives in a linear Gaussian regime, and real comparisons rarely control variance and influence separately.

- Falsifiable question: at matched architecture, data and probe, does JEPA beat MAE exactly when label-relevant features have lower variance than nuisance features, and lose when the order flips?
- Control: a synthetic image set where you dial nuisance-texture variance up while holding label signal fixed, plus a random-init encoder under the same probe.
- Mechanism: track which principal directions of the data each objective learns first, the observable that Littwin et al. predict.

## Belief 2: collapse needs heuristics, or an isotropic Gaussian

**Claim.** Old view: joint-embedding training needs an EMA teacher, stop-gradient and a predictor to avoid collapse. New view ([LeJEPA](https://arxiv.org/abs/2511.08544)): embeddings should follow an isotropic Gaussian, which uniquely minimizes downstream prediction risk, and a regularizer can enforce it directly.

**Background.** BYOL, DINO and I-JEPA rely on teacher-student asymmetry; VICReg used variance-covariance penalties. LeJEPA replaces all of it with SIGReg, which matches the embedding distribution to N(0, I) along random 1-D projections using a characteristic-function test.

**Method.** A risk argument over task families, then experiments across ResNets, ConvNets and ViTs up to 1.8B parameters.

**Result.** One trade-off hyperparameter, stable training without stop-gradient or teacher, 79% ImageNet linear accuracy with ViT-H/14 pretrained only on ImageNet-1k. [LeWorldModel](https://arxiv.org/abs/2603.19312) carries the recipe to world models with two loss terms, and [Klindt et al.](https://arxiv.org/abs/2605.26379) prove linear identifiability, but only up to rotation and only for Gaussian latents.

**Interpretation.** Three things deserve scrutiny.

First, "optimal" depends on the probe. An unregularized linear probe cannot tell z from any invertible linear transform of z:

```
features z, any invertible A, probe weights w
probe on Az with w' = inverse(A)^T w:
  w'^T (A z) = w^T inverse(A) A z = w^T z
=> identical predictions, so isotropy can only matter through
   finite samples, regularization (ridge) or nonlinear probes (kNN)
```

Second, an isotropic Gaussian has no privileged basis, so individual coordinates are not interpretable even when the latents are recovered perfectly:

```
two seeds recover the same latents up to rotation: z2 = Q z1
best single-coordinate match ~ largest entry of a random unit vector
  ~ sqrt(2 ln d / d); for d = 256: sqrt(2 x 5.55 / 256) ~= 0.21
=> coordinate correlations near 0.2 even with identical latents,
   while an orthogonal Procrustes fit gives R^2 near 1
```

Third, if SIGReg constrains patch tokens, DINOv2-style high-norm outliers become impossible:

```
x ~ N(0, I_d): ||x|| ~= sqrt(d) +/- 0.71
d = 1024: ||x|| ~= 32.0 +/- 0.71
a token at 2x typical norm (64) sits (64 - 32) / 0.71 ~= 45 sd out
```

- Falsifiable question: do LeJEPA-trained ViTs grow register-like outlier tokens? If SIGReg acts only on pooled embeddings, outliers may still form inside; if it acts on patch tokens, they should vanish, and the computation registers absorb in DINOv2 must move somewhere else.
- Control: matched ViT trained with an EMA-teacher recipe on the same data; per-coordinate excess kurtosis (0 for a Gaussian) as the readout.
- Mechanism: locate where global information lives in a LeJEPA ViT using the register-neuron recipe from [Jiang et al.](https://arxiv.org/abs/2506.08010).

## Belief 3: video JEPAs learn world models

**Claim.** Predicting masked or future video in latent space yields a world model: the model "knows" intuitive physics and can plan.

**Background.** [Garrido et al. (2025)](https://arxiv.org/abs/2502.11831) tested V-JEPA with violation-of-expectation; [V-JEPA 2](https://arxiv.org/abs/2506.09985) added action-conditioned planning; [V-JEPA 2.1](https://arxiv.org/abs/2603.14482) and [LeWorldModel](https://arxiv.org/abs/2603.19312) pushed dense features and cheap end-to-end training.

**Method.** Surprise (latent prediction error) on matched possible and impossible videos; planning toward image goals in latent space; layerwise probing and attention ablations inside the encoder ([Joseph et al.](https://arxiv.org/abs/2602.07050)).

**Result.**

- Behavior holds up: V-JEPA is more surprised by impossible videos on IntPhys, GRASP and InfLevel, and V-JEPA 2-AC picks and places zero-shot on two labs' Franka arms.
- Data matters: V-JEPA pretrained on the infant-headcam corpus SAYCam does poorly on the same benchmarks ([Dec 2025 preprint](https://arxiv.org/abs/2512.06232)).
- Internals are not a physics engine: probe accuracy on possible-vs-impossible jumps from \~50% to \~85-95% at about one-third depth, then decays toward the output. Motion direction needs a 40-80 dimensional ring-shaped code, and steering it takes \~20 orthogonal probe directions (angular error 82.9° to 11.9°). Suppressing local temporal attention at that depth drops IntPhys accuracy from 78.3% to 51.9% while ImageNet barely moves (33.7% to 30.3%).

How to read their subspace-overlap numbers, since chance is not zero:

```
expected overlap of a random k-dim subspace in d dims = k / d
V-JEPA 2-L, d = 1024, direction subspace k = 136: 136 / 1024 = 13.3%
observed direction -> IntPhys overlap: 12.9%
=> indistinguishable from chance, i.e. no shared latent variable
```

**Interpretation.** The behavioral reading of "world model" is supported; the representational reading (compact, factorized physical state) is contradicted. The 2026 identifiability theory offers a sharper explanation: LeJEPA-style learners recover latents linearly only when the latents are Gaussian. Direction of motion is circular, so no linear one-to-two-dimensional code is guaranteed.

- Falsifiable question: is the distributed code a failure of the model, or a consequence of the latent's distribution?
- Control: toy worlds with Gaussian (Ornstein-Uhlenbeck) latents versus circular latents, same renderer and model; LeWorldModel trains in hours on one GPU.
- Mechanism: for a circular latent the transition operator's eigenfunctions are Fourier modes (cos kθ, sin kθ), so a spectral-alignment learner should store direction as harmonic pairs. Joseph et al.'s sine-cosine "sawtooth" is consistent with this; regress the embedding on harmonics k = 1, 2, 3 to test it.

## Belief 4: frozen-probe scores measure representation quality

**Claim.** Accuracy of a probe on a frozen encoder ranks representations, so "JEPA beats X" can be read off a probe table.

**Background.** Image JEAs were judged by linear probes on the CLS token. Video JEPAs switched to attentive probes, which pool patch tokens with learned attention ([V-JEPA](https://arxiv.org/abs/2404.08471)); [Psomas et al. (ICLR 2026)](https://arxiv.org/abs/2506.10178) show linear probes understate models whose information is spread across patches.

**Method.** Same frozen encoder, different readouts and layers.

**Result.**

- V-JEPA ViT-L/16, linear vs attentive probe: 56.7 vs 80.8 on Kinetics-400, 50.1 vs 69.5 on SSv2, gaps of 24.1 and 19.4 points.
- Physics information in V-JEPA 2 peaks mid-depth and degrades toward the output ([Joseph et al.](https://arxiv.org/abs/2602.07050)), so last-layer probes miss it.
- LeJEPA reports its training loss tracks downstream accuracy, which would allow label-free model selection; this is a claim worth replicating.

The attentive probe is not a small readout:

```
linear probe, d = 1024, 400 classes: 1024 x 400 ~= 0.41M params
attentive block, assuming standard widths:
  attention projections 4 d^2 = 4.19M
  MLP with 4x expansion 8 d^2 = 8.39M
  total ~= 13.0M, about 32x the linear probe
```

**Interpretation.** Part of an attentive-probe score is learned by the probe. Any JEPA-vs-MAE-vs-DINO conclusion should report the probe family and the layer, or it is a statement about the protocol.

- Falsifiable question: across a zoo of image and video JEAs and MIM models, is Kendall's tau between linear-probe and attentive-probe rankings above 0.5?
- Control: a randomly initialized encoder read by the same attentive probe, which sets the floor the probe reaches on its own.
- Mechanism: inspect what the attentive probe's query attends to, and whether it reassembles information the linear probe cannot reach.

## Belief 5: joint-embedding models converge on a shared representation

**Claim.** The Platonic Representation Hypothesis: as models scale, representations from different objectives and modalities converge toward one statistical model of reality.

**Background.** Huh et al. (ICML 2024) measured rising mutual-kNN alignment between vision and language models. Two 2026 papers push back on how much converges, and at what scale of structure.

**Method.** Paired alignment metrics (CKA, mutual kNN), now with permutation-calibrated nulls; and, newest, unpaired matching as a stricter test.

**Result.**

- [Gröger, Wen, Brbić (ICML 2026)](https://arxiv.org/abs/2602.14486): depth and width alone inflate similarity scores. After calibration the global spectral (CKA) convergence trend largely disappears, while local neighborhood agreement survives, which they call the Aristotelian Representation Hypothesis.
- [Schnaus et al. (Oct 2026)](https://arxiv.org/abs/2610.09411): one orthogonal map aligns independently trained image embeddings (7 vision models, e.g. DINOv2 ViT-B/14) with language-model caption embeddings without pairs, at average FOSCTTM 0.154 (0.5 is chance; CLIP trained on 400M pairs reaches 0.0006). CKA explains 73% of the variance in alignment quality, and coarse clusters align better than local neighborhoods.

What 0.154 means in plain terms, using a crude model where alignment sorts data perfectly into K equal groups and is random within each:

```
FOSCTTM = (1/K) x 0.5 = 1 / (2K)  =>  K_eff = 1 / (2 x FOSCTTM)
unpaired, COCO:                    1 / (2 x 0.154)  ~= 3.2 groups
unpaired, COCO images + SPC text:  1 / (2 x 0.236)  ~= 2.1 groups
CLIP ViT-L/14, 400M pairs:         1 / (2 x 0.0006) ~= 833 groups
```

**Interpretation.** The two papers disagree on granularity: one finds local neighborhoods are what survive, the other finds coarse clusters align best. Both are compatible with a hump-shaped dependence on scale, which nobody has measured with calibration.

- Falsifiable question: on fixed model pairs, does calibrated alignment peak at an intermediate neighborhood size (k/n), and does the peak move with model scale as PRH would require?
- Control: width- and depth-matched random-init encoders through the same calibration, plus image-text pairs from disjoint datasets.
- JEA angle: at matched ImageNet probe accuracy, do JEPA, DINO, MAE and CLIP encoders align equally well with the same language model? PRH predicts the objective washes out at scale.

## Interpretability inside JEPAs

One serious interpretability study of a video JEPA exists, and it examined the encoder, not the predictor that actually produces the surprise signal and the plans. Most of the open ground below is cheap, because LeWorldModel trains on one GPU in hours.

**What has been examined**

- [Joseph et al. (Feb 2026)](https://arxiv.org/abs/2602.07050): the one-third-depth physics transition appears in V-JEPA 2 Large, Huge and Giant, and also in VideoMAE-v2-G, so it is not specific to latent prediction.
- Image JEAs: high-norm outlier tokens and registers ([Jiang et al.](https://arxiv.org/abs/2506.08010)), SAE concept maps of DINOv2 ([Rabbit Hull, ICLR 2026](https://arxiv.org/abs/2510.08638)), and SAEs on register versus outlier tokens ([Varma, Oct 2026](https://arxiv.org/abs/2610.03698)). DINOv3 ships with registers.

**Open gaps**

1. The predictor. Nobody has localized which predictor layers and tokens carry the violation signal. Patch predictor activations from a possible clip into its matched impossible clip, layer by layer.
2. Basis. Under LeJEPA's rotation non-identifiability, neuron-level findings such as direction-tuned MLP units may not transfer between seeds or recipes. Test whether units match across seeds or only subspaces do.
3. Outlier tokens. Whether V-JEPA 2 or LeJEPA encoders grow high-norm tokens is unreported (Belief 2 gives a sharp prediction).
4. SAE controls. I found no random-model or frozen-decoder SAE controls on any video encoder.
5. The latent-law test from Belief 3, which decides whether distributed physics codes are a model failure or a geometric necessity.

Gaps 4 and 5 are written up as directions, with methods and decision rules, in Vision Interp Lit Review: Defect Detection, ViTs, VLMs, VLAs.
