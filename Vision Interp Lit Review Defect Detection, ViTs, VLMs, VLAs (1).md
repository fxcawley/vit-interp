# Vision Interp Lit Review: Defect Detection, ViTs, VLMs, VLAs

Oct 8, 2026 · @liam cawley

## How to read this

Sections 2 to 7 each take one belief the vision field leans on and lay it out in your four tenets: who relies on it, what evidence already cuts against it, an open falsifiable question, the control that could overturn it, and candidate mechanisms. Sections 8 and 9 are the toolkit and the semiconductor-specific data. Section 10 lays out four research directions, each as claim, background, method, result and interpretation.

| Tenet | What to pull from each paper |
| --- | --- |
| 1. Belief | Who builds on it, and what breaks downstream if it is wrong |
| 2. Falsifiable question | A prediction with a number attached, e.g. "AUROC drops more than 5 points when X is removed" |
| 3. Overturning control | Random-init model, shuffled labels, blind or blank input, a matched non-learned baseline, a second seed |
| 4. Mechanism | A causal intervention (ablate, patch, steer) that both reproduces the effect and removes it |

One constraint shapes every project below: KLA data is proprietary, so each idea needs a version that runs on public data (MVTec AD 2, VisA, Real-IAD, WM-811K, MIIC and Carinthia-S for SEM). Your KLA system then becomes the private validation, not the paper's evidence.

## Belief 1: DINOv2 patch features are clean local descriptors

This is the belief your DINOv2 system most likely rests on, and it already has the most direct evidence against it. Every patch-matching detector (PatchCore, AnomalyDINO) assumes a patch's feature describes that patch's content, not global image state or its grid position.

**Evidence already against it**

- [Darcet et al., Vision Transformers Need Registers (ICLR 2024)](https://arxiv.org/abs/2309.16588): large ViTs including DINOv2 produce high-norm outlier tokens in low-information, redundant patches, only after long enough training of a large enough model. Learned register tokens remove them. Their reading of these tokens as global memory is contested, see the next two bullets.
- [SINDER (ECCV 2024)](https://arxiv.org/abs/2407.16826): the defective tokens in an image share almost one direction and carry little input information, local or global; in DINOv2-g they become evident only after layer 15.
- [Varma (Oct 2, 2026)](https://arxiv.org/abs/2610.03698): SAEs on DINOv2 tokens find outlier-token features mostly low-level structural, background and texture patterns, register-token features more semantic. Ablating top register features drops representation cosine similarity by 48.17%, outlier features by 0.31%.
- [Baxevanakis et al. (Mar 2026)](https://arxiv.org/abs/2603.25803): outliers are high-norm in keys but low-norm in queries and values. In DINOv2-S most vanish before the final output; in DINOv2-L a small subset persists, and that is enough to badly harm the feature map. Their block-wise maps (App. A.2) trace where artifacts first appear.
- [Jiang, Dravid, Efros, Gandelsman (NeurIPS 2025)](https://arxiv.org/abs/2506.08010): a sparse set of register neurons creates the outliers; in DINOv2-L they appear after the MLP around layer 17 of 24. A test-time register matches trained registers and lifts unsupervised object localization by about 21 points. Steering the same neurons moves the outlier onto a chosen patch, which disrupts that patch's content (shown on text). [Code](https://github.com/nickjiang2378/test-time-registers).
- [PH-Reg (NeurIPS 2025)](https://arxiv.org/abs/2505.21501) and the training-free CLIP dense methods it benchmarks (MaskCLIP, SCLIP, ClearCLIP, NACLIP): the attention-side mechanism, why value or self-self attention cleans patch features, is already worked out and tied to artifact tokens. In anomaly detection it is unapplied, not unknown.
- [Zero-ablation overstates register dependence (CVPR 2026 workshop)](https://arxiv.org/abs/2604.14433): zeroing registers costs up to 36.6 points of classification and 30.9 of segmentation, yet plausible replacement values preserve every task. Ablate by replacement, not zeros.
- Other artifacts and fixes: positional grid artifacts that persist in DINOv2-reg ([DVT, ECCV 2024](https://arxiv.org/abs/2401.02957)); dense-feature collapse in long training, fixed by Gram anchoring ([DINOv3](https://arxiv.org/abs/2508.10104)); inference-time attention edits that restore DINOv2 clustering ([Kioxia, 2024](https://arxiv.org/abs/2410.04801)).

**Where outliers land, and a miss mechanism**

Outliers form in patches that are near-duplicates of their neighbors right after tokenization ([2025 analysis](https://arxiv.org/abs/2509.19687)). Repetitive, low-texture images should therefore host more of them; for SEM metal-layer images such as MIIC that is a hypothesis to test, not a known fact. The new question is what happens when one lands on a defect. A co-location test gives the decision rule:

```
null: an outlier sits on a given defect patch with probability p0
  (estimate p0 on matched non-defect patches of similar local variance)
example: 50 missed defects, p0 = 0.02 -> expected 1.0 co-located
Poisson(1): P(X >= 6) = 1 - e^-1 (1 + 1 + 1/2 + 1/6 + 1/24 + 1/120)
                      = 1 - 0.3679 x 2.7167 ~= 0.0006
=> 6 of 50 misses sitting on outliers is a p < 0.001 signal
```

Jiang et al.'s steering result makes this causal. Place an outlier on a detected defect patch and the defect should fade from the anomaly map; evacuate outliers with a test-time register and co-located misses should recover. Any false-alarm fix built on registers changes this miss channel too, so both must be reported.

The effect could still be null. Outlier tokens resemble each other, so a memory bank of normal images may already hold near-matches that neutralize them.

**Falsifiable questions**

1. Does a test-time register on the same weights change pixel AUPRO by more than 2 points on MVTec AD 2 and MIIC, and is the change larger for small defects?
2. Are missed defects hosted by outlier tokens more often than matched control patches?
3. Does a backbone-size inversion in plain-DINOv2 kNN pipelines vanish with registers at a fixed pipeline? (Direction A.)

**Overturning controls**

- Same weights with and without a test-time register: one variable changed, nothing retrained.
- Replacement ablations (mean or resampled register content), never zeros.
- Sub-patch translation sweep on clean images: anything that moves the score there is not about content.
- Randomly initialized ViT with the same pipeline as a floor.

**Mechanism to look for**

Steer register neurons to put the outlier on a chosen defect patch and measure the score drop (sufficiency for a miss); remove it with a test-time register (necessity). For positional artifacts, project out a linear (row, col) probe subspace and re-measure AUPRO.

## Belief 2: anomaly maps and saliency explain what the model detected

Inspection teams treat the heatmap as the explanation, and the literature says that habit fails silently. The strongest prior result is that unsupervised and anomaly-detection models routinely score well for the wrong reasons, including on industrial data.

**Evidence already against it**

- [Adebayo et al., Sanity Checks for Saliency Maps (NeurIPS 2018)](https://arxiv.org/abs/1810.03292): several popular saliency methods give nearly the same maps after the weights or labels are randomized, so they behave like edge detectors. From memory; the canonical tenet-3 control.
- [Geirhos et al., Don't Trust Your Eyes (ICML 2024)](https://proceedings.mlr.press/v235/geirhos24a.html): a hidden circuit can make feature visualizations show arbitrary patterns with ImageNet accuracy unchanged. Even unmanipulated networks process visualizations along different paths than natural images. Their conclusion: use them to generate hypotheses, not to confirm them.
- [Kauffmann et al., The Clever Hans Effect in Anomaly Detection (2020)](https://arxiv.org/abs/2006.10609): Clever Hans behavior is widespread in anomaly detectors, and it comes mainly from model structure, not the data, so more data does not fix it.
- [Kauffmann et al., Nature Machine Intelligence (2025)](https://arxiv.org/abs/2408.08041): extends this to unsupervised representation learning and anomaly detection on medical and industrial inspection data. The Clever Hans strategies cause large performance drops under plausible dataset shift or subgroup reweighting, and theory points to the learner's inductive biases.

**The resolution problem specific to patch-kNN maps**

A DINOv2-kNN anomaly map is distance to the nearest normal patch, computed on a 14 px grid and then upsampled and blurred. A small defect barely occupies its patch:

```
3 x 3 px defect / (14 x 14 px patch) = 9 / 196 ~= 4.6% of patch area
```

Whether 4.6% of the pixels moves the embedding past the normal nearest-neighbor distance distribution is an empirical number nobody publishes for SEM-scale defects. It decides your detection floor.

**Falsifiable questions**

1. Do anomaly maps track the defect, or the acquisition context that co-occurs with defects (illumination, focus, position in the field of view)?
2. What is the minimum defect area fraction per patch at which pixel AUROC exceeds 0.9? Predict it, then measure it with synthetic defects of controlled size.

**Overturning controls**

- Model randomization applied to anomaly maps: re-initialize the last k blocks of DINOv2. If AUPRO barely falls, the map reflects low-level image statistics, not learned features.
- Counterfactual pasting: move a defect crop onto a clean image from a different acquisition session, and cover the defect with a clean crop. If the image score follows the session rather than the defect, that is Clever Hans.
- Subgroup reweighting as in Kauffmann et al.: re-score after balancing by tool or acquisition date.

**Mechanism to look for**

The Kauffmann and Montavon group "neuralizes" one-class and kNN scorers so LRP can attribute an anomaly score to pixels and feature channels. Use that to find which feature directions drive the distance, then ablate those directions and check whether the shortcut breaks while true defects survive.

## Belief 3: MVTec-style benchmarks reflect industrial readiness

The benchmark the field optimizes is saturated, and the newer one shows the same methods losing about 35 AU-PRO points. That gap is the opening: within the anomaly-detection literature, nobody has explained mechanistically why these methods fail on the hard cases, though the relevant patch-feature mechanisms are known from CLIP dense inference (Belief 1).

**Where the belief comes from**

- [PatchCore (Roth et al., CVPR 2022)](https://arxiv.org/abs/2106.08265): the template. Store nominal patch features in a memory bank, score each test patch by distance to its nearest neighbor. From memory.
- [AnomalyDINO (Damm et al., WACV 2025 oral)](https://arxiv.org/abs/2405.14529): the same recipe on DINOv2 patches, vision only. One-shot MVTec AD image AUROC goes from 93.1% to 96.6%, beating language-based methods. Its starting question was whether language is needed at all.

**Evidence against it**

- [MVTec AD 2 (Heckler-Kram et al., 2025; IJCV 2026)](https://arxiv.org/abs/2503.21622): on MVTec AD and VisA, top methods now differ by under one point. Methods above 90% AU-PRO on MVTec AD score at most 58.7% on AD 2. The new scenarios include extremely small defects, dark-field and back-light illumination, and lighting-shift test sets. Thresholded results also fail to track threshold-free metrics. Test labels sit on a [hidden evaluation server](https://benchmark.mvtec.com/).

**Is a one-point gap even real?** Hanley-McNeil standard error for AUROC A, with nA defective and nN good images:

```
Q1 = A/(2-A),  Q2 = 2A^2/(1+A)
SE^2 = [A(1-A) + (nA-1)(Q1-A^2) + (nN-1)(Q2-A^2)] / (nA*nN)

Illustrative MVTec category: A=0.97, nA=60, nN=40
Q1-A^2 = 0.0008,  Q2-A^2 = 0.0143
SE^2 = [0.0291 + 59(0.0008) + 39(0.0143)] / 2400 = 0.000264
SE ~= 0.016  -> 1.6 points per category
Mean over 15 categories: 1.6/sqrt(15) ~= 0.4 points
```

So a sub-one-point mean gap is about 2 standard errors before seed variance. On MIIC's test split (116 anomalous, 1,274 normal) at A = 0.97: SE^2 = \[0.0291 + 115(0.0008) + 1273(0.0143)\] / (116 x 1274) \~= 0.000124, so SE \~= 0.011 and the 95% interval is about ±2.2 AUROC points; about ±3.1 if, as in the benchmark, half the test set is held out for validation.

**Falsifiable questions**

1. Does a method's rank on MVTec AD predict its rank on AD 2 or on MIIC, where rank flips already appear (Section 9)? Report Kendall's tau with a bootstrap interval. Prediction worth testing: tau is near zero on the small-defect and dark-field scenarios.
2. On highly registered, repetitive imagery, does DINOv2-kNN beat a classical aligned die-to-die difference? Die-to-die comparison is the inspection industry's own baseline, and vision-foundation-model papers almost never run it.

**Overturning controls**

- Aligned die-to-die or template differencing as the non-learned baseline.
- Randomly initialized ViT with the same kNN pipeline, to separate learned features from pipeline effects.
- Bootstrap intervals and at least 3 seeds on every number.

**Mechanism to look for**

Bin AD 2 failures by defect area per patch and by illumination type. Then test whether the failing bins match the patch-area floor from Belief 2 or the artifact tokens from Belief 1. If they do, the benchmark gap has a mechanistic explanation, which is the paper.

## Belief 4: SAE features are real, stable concepts

Vision SAE work now treats dictionary latents as the model's concepts, but the two strongest controls against that view were run on language models, not vision. The generic result is anticipated, since random-direction SAEs already match trained ones on CLIP ViT-B/32. What is unpublished is the random-model control in vision, how it depends on depth, and which concept categories survive it.

**Who relies on it**

- [Stevens et al., SAEs for Scientifically Rigorous Interpretation of Vision Models (2025)](https://arxiv.org/abs/2502.06755): SAEs on ViTs, with feature suppression as the experimental test. Reports different abstractions in CLIP vs DINOv2. Code: [saev](https://github.com/osu-nlp-group/saev).
- [Fel, Wang et al., Into the Rabbit Hull (ICLR 2026)](https://arxiv.org/abs/2510.08638): about 32,000 concepts from DINOv2 using an Archetypal SAE, a stable variant whose atoms stay in the convex hull of real activations. Some concepts fire only on register tokens and encode global properties such as motion blur and illumination. [Blog part I](https://kempnerinstitute.harvard.edu/research/deeper-learning/into-the-rabbit-hull-part-i/), [part II](https://kempnerinstitute.harvard.edu/research/deeper-learning/into-the-rabbit-hull-part-ii/).

**Evidence against it**

- [Paulo and Belrose (ICLR 2026)](https://arxiv.org/abs/2501.16615): two SAEs that differ only in seed share about 30% of features (131K latents, Llama 3 8B). TopK SAEs are more seed-dependent than ReLU plus L1.
- [Heap et al. (2025)](https://arxiv.org/abs/2501.17727): SAEs trained on random-weight transformers look about as interpretable as on trained ones, because the input data is already superposed. They recommend random-model controls; this was language, not vision.
- [Korznikov et al. (ICML 2026)](https://arxiv.org/abs/2602.14111): SAEs with frozen random decoders or encoders match fully trained SAEs on interpretability (0.87 vs 0.90), sparse probing (0.69 vs 0.72) and causal editing (0.73 vs 0.72), with a CLIP ViT-B/32 replication. On synthetic data with known features, SAEs recover only 7-9% of them at about 71% explained variance.
- Rabbit Hull itself: DINOv2 is partly dense, with universally active positional features, antipodal pairs and convex clusters rather than near-orthogonal directions. The authors propose concepts as convex regions (the Minkowski Representation Hypothesis), not linear directions.

**Why texture-heavy images make the random-model control sharper**

Random convolutional features are already decent texture descriptors (Ustyuzhaninov et al., ICLR 2017, from memory). Texture-dominated domains such as SEM are where a random DINOv2 is most likely to yield "defect-like" latents from image statistics alone.

**Falsifiable questions**

1. On ImageNet and on MIIC, what fraction of DINOv2-SAE latents have a match (Hungarian-matched decoder cosine above 0.7) in an SAE on randomly initialized DINOv2? Above 50% means most "concepts" come from data and architecture.
2. Do the latents that best separate defect from normal survive a seed change?
3. Does an anomaly score built on SAE latents beat raw-feature kNN? In language models, SAE probes often do not beat plain linear probes (Kantamneni et al. 2025, from memory).

**Compute is not the constraint**

```
ViT-B/14: d = 768; SAE at 16x -> 12,288 latents
params ~= 2 x 768 x 12,288 ~= 18.9M
tokens: 10k images x 1369 patches ~= 13.7M
FLOPs ~= 6 x 18.9M x 13.7M ~= 1.6e15 per epoch
at ~50 TFLOP/s effective: ~31 s per epoch
activations in fp16: 13.7M x 768 x 2 B ~= 21 GB
```

Five seeds times two models (trained, random) is ten SAEs, minutes each on one GPU.

**Mechanism to look for**

For latents with no random-init match, suppress them and measure the change in defect scores. Separately, find Rabbit Hull-style illumination and blur latents and test whether they drive false positives, which ties back to the Clever Hans question in Belief 2.

## Belief 5: VLMs ground their answers in the image

The best current evidence says VLMs often answer from language priors while the needed visual information sits unused in their own encoder. Industrial "inspection copilot" work assumes the opposite, and nobody has localized where defect information is lost inside a VLM.

**Evidence against it**

- [Fu et al., Hidden in Plain Sight (2025)](https://arxiv.org/abs/2506.08008): on vision-centric tasks such as depth and correspondence, VLMs score far below a direct readout of their own vision encoder, often near chance. Encoder strength does not predict VLM strength, and VLM answers largely match their no-image answers. The bottleneck is the language model's use of visual tokens, not the encoder.
- [Neo et al. (ICLR 2025)](https://arxiv.org/abs/2410.07149): in LLaVA, object information sits in a subset of visual tokens; removing them drops object identification accuracy by over 70%. Attention knockout shows the answer position pulls that information in middle to late layers.
- [Kaduri et al., What's in the Image? (CVPR 2025)](https://arxiv.org/abs/2411.17491): text query tokens compress global image information, while fine details are fetched directly from image tokens through spatially local attention in mid layers.
- [MMAD (Jiang et al., ICLR 2025)](https://arxiv.org/abs/2410.09453): 39,672 questions on 8,366 industrial images. The best model, GPT-4o, averages 74.9%, which the authors call far short of industrial requirements. [Code and data](https://github.com/jam-cc/MMAD).
- Test-time registers (Belief 1) also clean up LLaVA's cross-modal attention maps, so artifact tokens contaminate VLM attributions too.

**How much of 74.9% is the image?** An illustrative decomposition for a binary defect question, assuming the model uses the image correctly on a fraction u of items and otherwise falls back to a prior with accuracy b:

```
accuracy = u + (1 - u) * b
with b = 0.5 (balanced yes/no) and accuracy = 0.749:
0.749 = u + 0.5 - 0.5u  ->  u = 0.498
```

That is, about half the items. The real b comes from running the blind baseline, which is the first experiment.

**Falsifiable questions**

1. On MMAD anomaly discrimination, does a kNN or linear readout of the VLM's own vision encoder beat the VLM by more than 10 points? Fu et al. predicts yes.
2. Where does "defect present" stop being linearly decodable: encoder output, projector output, or a specific LLM layer?
3. Does the answer change when attention from defect-region visual tokens is knocked out? If not, the model is not using the defect.

**Overturning controls**

- Blind baseline: same question with a blank or mean image.
- Counterfactual image: the defect inpainted away, everything else identical.
- Shuffled image-question pairs.
- Encoder readout given the same number of labeled examples as the VLM's few-shot prompt.

**Mechanism to look for**

A layer-wise probe for "defect present" along the LLM residual stream, plus attention knockout from defect tokens to the answer position. The paper-worthy result is the specific layer and pathway where decodable defect information fails to reach the answer.

## Belief 6: VLAs condition on both vision and language

VLA benchmark scores above 95% now look like vision-to-action replay: several groups find language barely matters, while camera viewpoint breaks the policy. This is furthest from defect detection, but your DINOv2 expertise transfers directly, because OpenVLA's vision stream fuses DINOv2 and SigLIP features (from memory, verify in the OpenVLA paper).

**Evidence against it**

- [LIBERO-Plus (Fei et al., CVPR 2026)](https://arxiv.org/abs/2510.13626): perturbations along seven axes across ten models. Modest camera or initial-state changes drop success from 95% to below 30%, while language changes barely matter and models often ignore the instruction entirely. [Code](https://github.com/sylvestf/LIBERO-plus).
- [LangGap (2026)](https://arxiv.org/abs/2603.00592): summarizes LIBERO-PRO as finding identical trajectories under meaningless instructions, and builds a semantic-perturbation benchmark showing language deficits in π0.5.
- [A Mechanistic Study of VLA Models (2026)](https://arxiv.org/abs/2603.19233): five VLAs plus a language-free ACT policy as a control. Language sensitivity depends on whether the scene alone pins down the task. Interactive [Action Atlas](https://action-atlas.com).

**Who has looked inside**

- [Häon et al., Mechanistic Interpretability for Steering VLAs (CoRL 2025)](https://arxiv.org/abs/2509.00328): projects FFN activations of OpenVLA and π0 onto the token embedding basis and finds sparse semantic directions, such as speed and direction, that causally shift actions. Steering works zero-shot in LIBERO and on a UR5 arm.

**Falsifiable question**

Is OpenVLA's viewpoint fragility carried by the DINOv2 stream? Prediction: ablating or mean-replacing the DINOv2 features hurts in-distribution success more than ablating SigLIP, and the DINOv2 stream's positional components (Belief 1) are what break under viewpoint shift.

**Sample size matters here too.** With a success rate p and n episodes, SE = sqrt(p(1-p)/n):

```
p = 0.3, n = 500 (suite level):  SE = sqrt(0.21/500) ~= 0.020 -> +/-4 points
p = 0.3, n = 50 (one task):       SE = sqrt(0.21/50)  ~= 0.065 -> +/-13 points
```

Per-task claims need several hundred episodes each, which is simulator time, not GPU time.

**Overturning controls**

- Language-free policy (ACT) and a meaningless-instruction condition as floors for language use.
- Stream ablation: zero, mean, or shuffle-across-episodes replacement for DINOv2 and SigLIP features separately.

**Mechanism to look for**

Activation patching from a canonical-viewpoint episode into a shifted-viewpoint one, per stream and per layer. The layer where patching restores success localizes the fragility.

## Mechanism toolkit (tenet 4)

Every mechanism claim above reduces to a handful of causal tools, and all of them run on DINOv2 today. Pick the tool by the question, not the other way round.

| Method | Question it answers | Where it applies above | Source |
| --- | --- | --- | --- |
| Activation patching and ablation | Is this component necessary, or sufficient, for the behavior? | Register neurons; DINOv2 vs SigLIP streams in OpenVLA | Standard practice |
| Register-neuron ablation, test-time registers | Does the high-norm pathway change the output? | Belief 1 | [Jiang et al. code](https://github.com/nickjiang2378/test-time-registers) |
| Attention knockout | Does information flow from token set A to token B? | Defect tokens to a VLM's answer | [Neo et al.](https://arxiv.org/abs/2410.07149), [Kaduri et al.](https://arxiv.org/abs/2411.17491) |
| Linear probes with control tasks | Is the information decodable beyond what a probe could memorize? | "Defect present" per layer | Hewitt and Liang 2019, from memory |
| Sparse autoencoders (TopK, Archetypal) | Which directions decompose the activations? | Belief 4 | [saev](https://github.com/osu-nlp-group/saev), [Prisma](https://github.com/Prisma-Multimodal/ViT-Prisma) |
| LRP for kNN and one-class scorers | Which pixels and channels drive an anomaly score? | Belief 2 | [Kauffmann et al.](https://arxiv.org/abs/2408.08041) |
| Text decomposition of attention heads | What does each CLIP head write into the residual stream? | CLIP or SigLIP encoders | Gandelsman et al., ICLR 2024, from memory |
| Logit lens on visual tokens | Do visual tokens become word-like across LLM layers? | VLMs | [Neo et al.](https://arxiv.org/abs/2410.07149) |

[Prisma (Joseph et al., CVPR 2025 MIV workshop oral)](https://arxiv.org/abs/2504.19475) is the closest thing to TransformerLens for vision: 75+ vision and video transformers with TransformerLens-style hooks, SAE, transcoder and crosscoder training, and 80+ pretrained SAEs. It also reports vision SAEs working at much lower sparsity than language SAEs, and SAE reconstructions sometimes lowering model loss, both oddities worth knowing before you train your own.

**Control checklist to run on every claim**

- [ ] Randomly initialized weights, same architecture and pipeline
- [ ] Second and third seed, for both the model probe and any SAE
- [ ] Shuffled labels or shuffled image-text pairs
- [ ] Blank, mean, or counterfactual (defect-removed) input
- [ ] A non-learned baseline (template or die-to-die difference)
- [ ] Confidence intervals sized from the actual positive count

## Semiconductor landscape and public data

The public semiconductor record is thin: the one large public SEM anomaly benchmark tested only ImageNet CNN backbones, no DINOv2, no ViT. A careful DINOv2 interpretability study on it would be close to first of its kind, on a set with pixel masks for 116 anomalies among 25,276 images.

**What industry has published**

- [IBM Albany (Huang et al., ASMC 2025)](https://arxiv.org/abs/2506.03345): DINOv2 transfer plus semi-supervised learning for SEM defect classification, 11 defect types, over 7,400 images from their 300 mm fab. Private data.
- [Kioxia (Nakamura et al., 2024)](https://arxiv.org/abs/2410.04801): attenuating DINOv2 artifacts at inference time (see Belief 1).
- [Bosch and KIT reality check (Baeuerle et al., 2025)](https://arxiv.org/abs/2509.20479): on scanning acoustic tomography of electronic packages, four foundation-model segmentation pipelines score IoU 0.00, against up to 0.52 on MVTec AD. Gemini 2.5 Pro reaches F2 0.37, against 0.82 for a supervised ResNet classifier, while scoring 0.99 on MVTec AD. Their aligned reference-image difference also failed, because recording variation and mechanical tolerances swamped the defects. So treat die-to-die differencing (Belief 3) as a control that can lose, not a straw man.
- [Barusco et al., MIIC benchmark (2025)](https://arxiv.org/abs/2505.07576): nine anomaly methods on SEM images of IC metal layers. Rank order flips relative to MVTec: PaDiM drops to 84.9% image AUROC, and STFPM beats RD4AD. A 2021 inpainting-reconstruction model (99.3%) rivals the best feature method, CFA (99.0%). PatchCore's pixel PRO is 93.0% while its pixel AUPRC is only 44.9%, so PRO hides false positives.

**Public data you can publish on**

| Dataset | Modality | Size | Labels | Why it matters for you |
| --- | --- | --- | --- | --- |
| [MIIC](https://arxiv.org/abs/2505.07576) | SEM, IC metal layer, 512x512 grayscale | 25,276 images: 25,160 normal, 116 anomalous | Pixel masks | Dataset from Huang et al. (ICIP 2021); 2505.07576 benchmarks it, CNN backbones only, no ViT |
| [Carinthia-S](https://zenodo.org/records/16895427) | SEM, unstructured wafers (Infineon) | 4,591 defect images, 6 classes | Expert segmentation masks | Mask-grounded checks of attributions and SAE latents |
| [Semi-AD](https://ieee-dataport.org/documents/semi-ad-semiconductor-anomaly-detection-dataset) | Patterned wafer and IC substrate crops | Aligned reference-test pairs | Pixel ground truth | Direct test of comparison-based vs kNN detection; check which subsets are downloadable |
| [MVTec AD 2](https://www.mvtec.com/company/research/datasets/mvtec-ad-2) | Optical industrial | 8,000+ high-res images, 8 scenarios | Hidden test server | Small defects, dark-field, lighting shift |
| WM-811K (from memory) | Wafer bin maps | About 811k maps | Pattern classes | Wafer-level spatial patterns, not pixel defects |

**Resolution is a free variable the MIIC benchmark fixed badly for ViTs.** It resized 512 px images to 224 px.

```
linear scale: 224/512 = 0.4375
DINOv2 at 224: 224/14 = 16 x 16 = 256 patches
each patch spans 14/0.4375 = 32 native px
a 6 x 6 native-px defect: 36/(32 x 32) = 3.5% of its patch

DINOv2 at 518 (pad 512 by 3 px each side): 37 x 37 = 1369 patches
each patch spans 14 native px
same defect: 36/196 = 18% of its patch  (~5x larger share)
cost: 1369/256 = 5.3x tokens, ~5.3^2 = 28x attention FLOPs
```

That 5x jump in patch share is a clean, cheap independent variable for the Belief 2 patch-area question.

## Directions

Four directions, each built on a claim you could overturn: one in anomaly detection, one in interpretability method, and two in representation science that grow out of the JEA review. Revised after your review: Direction A is now a known mechanism applied to a new failure mode, not a new mechanism, and the VLM-on-wafer idea is dropped.

### A. Outlier tokens as a miss mechanism in patch-kNN anomaly detection

**Claim.** In plain-DINOv2 patch-kNN detectors, a high-norm outlier that lands on a defect patch overwrites its content, and the defect drops out of the anomaly map.

**Background.** The mechanism is known outside anomaly detection: register neurons create the outliers, and CLIP dense-inference work explains attention-side cleaning of patch features (Belief 1). Token-level analysis is already taken ([Varma, Oct 2026](https://arxiv.org/abs/2610.03698)), so the contribution has to be the anomaly-specific causal chain. The natural experiment you flagged points the same way: registered or newer backbones scale monotonically in anomaly detection ([VisionAD with DINOv2-R](https://arxiv.org/abs/2504.11895); [Dinomaly2 with DINOv3 S to L](https://arxiv.org/abs/2510.17611)), while plain-DINOv2 kNN results are dataset-dependent ([AnomalyDINO](https://arxiv.org/abs/2405.14529) favors ViT-S on MVTec AD) or invert (the MuRF DINOv2-L drop, which its authors attribute to nearest-neighbor search in high dimension). The [MuRF version I fetched](https://arxiv.org/abs/2603.25744) (v1, 26 Mar 2026) uses ViT-B/14 throughout, so I could not verify the L number itself; it also acknowledges KLA funding.

**Method.**

1. Fix one pipeline (AnomalyDINO or MuRF code) and sweep S, B, L, g × {plain, registers, plain + test-time register}, reading the same relative depth in every size. Data: MVTec AD 2 and MIIC.
2. Swap MuRF's approximate Faiss IVF search (32 of 512 lists probed) for exact search.
3. Co-location: are missed defects hosted by outliers more often than matched control patches (test in Belief 1)?
4. Sufficiency and necessity: steer an outlier onto a detected defect; evacuate outliers from missed ones.

**Result that would count.** Two checks come before any experiment:

```
"NN degrades in high dimension" predicts the wrong place for a drop
iid distance contrast ~ 1 / sqrt(d)
S -> B: sqrt(384 / 768)  = 0.71  (29% less contrast)
B -> L: sqrt(768 / 1024) = 0.87  (13% less contrast)
=> S -> B should hurt more than B -> L; a drop only at L does not fit

depth: MuRF reads layers 7, 9, 11 of the 12-layer ViT-B
  relative depth 7/12 to 11/12 = 0.58 to 0.92
same indices in the 24-layer ViT-L: 7/24 to 11/24 = 0.29 to 0.46
DINOv2-L outliers appear near layer 17: 17/24 = 0.71
=> if those indices were reused, the L features predate the outliers,
   and an inversion would be a depth artifact, not an outlier effect
```

Decision rule: the L inversion vanishes with registers at a fixed pipeline and depth, so the outlier account holds. It persists with registers but vanishes with exact search, so it is a search effect. It follows the layer indices, so it is a protocol artifact.

**Interpretation.** A known mechanism with a new, operational failure mode: a register fix that cuts false alarms also changes misses, so both must be reported. Venue: VAND, or MIV, whose 2025 organizers include the test-time-register authors.

### B. Factorial random controls for vision SAEs

**Claim.** Vision SAE concepts reflect what the network learned, not image statistics plus near-random directions.

**Background.** [Heap et al.](https://arxiv.org/abs/2501.17727) (random models, language) and [Korznikov et al.](https://arxiv.org/abs/2602.14111) (random SAEs, including CLIP ViT-B/32) make the generic negative result expected. Nobody has crossed the two controls in vision, measured how the gap depends on depth, or checked which concept categories survive. This is the closer-look stance, not throwing SAEs at a new model.

**Method.** A 2 × 2 of {trained, random-init} DINOv2-B × {trained, frozen-decoder} SAE, at layers 3, 6, 9 and 12, three seeds each, on ImageNet and MIIC. Readouts: auto-interp score, Hungarian-matched decoder cosine across cells and seeds, and the concept-category mix (object, part, texture, position, register-only), labeled with segmentation masks.

```
cells: 2 models x 2 SAE types x 4 layers x 3 seeds = 48 SAEs
per SAE: ~10 epochs x ~31 s (Belief 4 estimate) ~= 5 min
total: 48 x 5 min = 4 GPU-hours, plus activation caching
```

**Result that would count.** Trained-model SAEs beat random-model SAEs by more than the seed spread at deep layers but not early ones: depth-dependent learning, which contradicts Heap's result carrying over to vision. No separation at any depth: current vision SAE metrics cannot certify learned concepts. The headline case: Rabbit Hull's Elsewhere or register-only concepts appearing in a random network.

**Interpretation.** Both outcomes publish at workshop scale. Venue: MIV, or the NeurIPS and ICLR interpretability workshops.

### C. The granularity of convergence

**Claim.** Representations converge across models and modalities (PRH), and the convergence lives at a specific scale of structure.

**Background.** Calibrated global CKA trends vanish while local neighborhoods survive ([Gröger et al.](https://arxiv.org/abs/2602.14486)); unpaired alignment works only coarsely, with coarse clusters aligning best ([Schnaus et al.](https://arxiv.org/abs/2610.09411)). The two findings pull in opposite directions on granularity (JEA review, Belief 5).

**Method.** Fixed pairs of vision encoders (DINOv2 and DINOv3 from S to g, I-JEPA, MAE, CLIP) and language models of several sizes. Permutation-calibrated alignment at neighborhood sizes k/n from 0.001 to 0.5, plus unpaired FOSCTTM. Controls: width- and depth-matched random-init encoders through the same calibration, images and captions from disjoint datasets, and objectives compared at matched ImageNet probe accuracy.

**Result that would count.** A hump at intermediate k/n reconciles both papers. A peak that sharpens or moves to finer scales with model size is the PRH signature; a peak that stays put as size grows argues against convergence. Gaps between objectives larger than the seed spread, at matched probe accuracy, mean the objective does not wash out.

**Interpretation.** It turns PRH into a curve with a location, from embeddings alone, so compute is small. Venue: UniReps or the ICLR Representational Alignment workshop, both cited in the Rosetta paper.

### D. Does a latent's distribution decide how a JEPA encodes it?

**Claim.** Video JEPAs store physics in distributed, task-specific codes rather than compact state variables ([Joseph et al.](https://arxiv.org/abs/2602.07050)).

**Background.** [Klindt et al.](https://arxiv.org/abs/2605.26379) prove LeJEPA recovers latents linearly only when they are Gaussian. Motion direction is circular, and Joseph et al. found it in a 40-80 dimensional ring code. SIGReg pushes embeddings toward Gaussian marginals, which a compact angle code cannot satisfy:

```
theta uniform on the circle, x = cos(theta)
E[x^2] = 1/2, E[x^4] = 3/8
kurtosis = (3/8) / (1/2)^2 = 1.5; excess = 1.5 - 3 = -1.5  (Gaussian: 0)
=> a 2-D (cos, sin) code violates a Gaussian target

random-walk heading, step ~ N(0, s^2):
  Fourier mode k has eigenvalue exp(-k^2 s^2 / 2)
s = 0.3 rad: k = 1..4 -> 0.96, 0.84, 0.67, 0.49
=> four harmonics (8 dims) stay slow; a spectral learner
   should spread one angle across them
```

**Method.** Kubric toy worlds like Joseph et al.'s, with three latent laws: Gaussian Ornstein-Uhlenbeck position, random-walk heading, bounded uniform speed. Train LeWorldModel (SIGReg, one GPU, hours) and a matched EMA-teacher JEPA. Readouts: linear identifiability R² per latent, dimensionality from the orthogonal-probe sequence, and regression of embeddings on cos kθ and sin kθ for k = 1 to 4. Controls: random-init encoder, time-shuffled training that destroys the transition structure, three seeds.

**Result that would count.** Gaussian latents recovered at R² near 1 in about one dimension each, as the theory says. Heading spread over harmonics k = 1 to 3, and over more dimensions under SIGReg than under the EMA recipe. If heading gets a compact code anyway, Joseph et al.'s distributed code reflects V-JEPA's data or training, not geometry.

**Interpretation.** It tests 2026 theory against a 2026 interpretability finding, needs no proprietary data, and runs on one GPU. Venue: MIV, or NeurIPS and ICLR world-model and interpretability workshops.

**Dropped.** Fine-tuned VLMs detecting wafer defects: a cookbook claim with a near-certain headline, not a belief the field relies on.

**Timing.** VAND 4.0 at CVPR 2026 took papers until March 2, 2026 ([site](https://sites.google.com/view/vand4-cvpr2026)). If CVPR 2027 follows the pattern, which is an inference, that is about 20 weeks out. A and B share feature-extraction and register-intervention code; C and D need no proprietary data at all.

## Reading order

Read these ten in sequence; the first four are enough to start Direction A this week.

1. [Darcet et al., Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588): the artifact your system probably has.
2. [Jiang et al., Vision Transformers Don't Need Trained Registers](https://arxiv.org/abs/2506.08010): the mechanism and the training-free intervention.
3. [AnomalyDINO](https://arxiv.org/abs/2405.14529): the baseline pipeline you will reproduce.
4. [Barusco et al., MIIC benchmark](https://arxiv.org/abs/2505.07576): your public dataset and its gaps.
5. [Kauffmann et al., Clever Hans in unsupervised learning](https://arxiv.org/abs/2408.08041): the strongest prior that anomaly maps mislead.
6. [Adebayo et al., Sanity Checks for Saliency Maps](https://arxiv.org/abs/1810.03292): the template for a randomization control.
7. [Heap et al., SAEs on randomly initialized transformers](https://arxiv.org/abs/2501.17727): the control Direction B imports into vision.
8. [Paulo and Belrose, SAE seed instability](https://arxiv.org/abs/2501.16615): the second control for project 2.
9. [Fel, Wang et al., Into the Rabbit Hull](https://arxiv.org/abs/2510.08638): the most complete map of DINOv2's concepts and geometry.
10. [Fu et al., Hidden in Plain Sight](https://arxiv.org/abs/2506.08008): the encoder-vs-VLM control, if you return to VLM grounding.

Next tier: [MVTec AD 2](https://arxiv.org/abs/2503.21622), [Denoising ViTs](https://arxiv.org/abs/2401.02957), [Neo et al.](https://arxiv.org/abs/2410.07149), [LIBERO-Plus](https://arxiv.org/abs/2510.13626) and [Häon et al.](https://arxiv.org/abs/2509.00328).

Added after your review, in reading order for the directions: [Baxevanakis et al.](https://arxiv.org/abs/2603.25803), [Varma](https://arxiv.org/abs/2610.03698), [PH-Reg](https://arxiv.org/abs/2505.21501), [Korznikov et al.](https://arxiv.org/abs/2602.14111), [Gröger et al.](https://arxiv.org/abs/2602.14486), [Klindt et al.](https://arxiv.org/abs/2605.26379) and [Joseph et al.](https://arxiv.org/abs/2602.07050).
