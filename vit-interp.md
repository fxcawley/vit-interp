# Vision Interp Lit Review: Defect Detection, ViTs, VLMs, VLAs

Oct 8, 2026 · @liam cawley

## How to read this

Sections 2 to 7 each take one belief the vision field leans on and lay it out in your four tenets: who relies on it, what evidence already cuts against it, an open falsifiable question, the control that could overturn it, and candidate mechanisms. Sections 8 and 9 are the toolkit and the semiconductor-specific data. Section 10 ranks projects.

| Tenet | What to pull from each paper |
| --- | --- |
| 1. Belief | Who builds on it, and what breaks downstream if it is wrong |
| 2. Falsifiable question | A prediction with a number attached, e.g. "AUROC drops more than 5 points when X is removed" |
| 3. Overturning control | Random-init model, shuffled labels, blind or blank input, a matched non-learned baseline, a second seed |
| 4. Mechanism | A causal intervention (ablate, patch, steer) that both reproduces the effect and removes it |

One constraint shapes every project below: KLA data is proprietary, so each idea needs a version that runs on public data (MVTec AD 2, VisA, Real-IAD, WM-811K, public SEM sets). Your KLA system then becomes the private validation, not the paper's evidence.

## Belief 1: DINOv2 patch features are clean local descriptors

This is the belief your DINOv2 system most likely rests on, and it already has the most direct evidence against it. Every patch-matching detector (PatchCore, AnomalyDINO) assumes a patch's feature describes that patch's content, not global image state or its grid position.

**Evidence already against it**

- [Darcet et al., Vision Transformers Need Registers (ICLR 2024)](https://arxiv.org/abs/2309.16588): large ViTs including DINOv2 repurpose low-information background patches as high-norm tokens that carry global information. Learned register tokens absorb this role.
- [Jiang, Dravid, Efros, Gandelsman, Vision Transformers Don't Need Trained Registers (NeurIPS 2025)](https://arxiv.org/abs/2506.08010): a sparse set of "register neurons" creates the outliers. In DINOv2 ViT-L/14 they appear after the MLP around layer 17. Moving those activations into one appended, untrained token at test time matches trained registers, and lifts DINOv2's unsupervised object localization by about 21 points. [Code](https://github.com/nickjiang2378/test-time-registers).
- [Yang et al., Denoising Vision Transformers (ECCV 2024 oral)](https://arxiv.org/abs/2401.02957): a second, separate artifact. Grid-like patterns in feature maps trace to positional embeddings, and they persist in DINOv2-reg.
- [DINOv3 (Aug 2025)](https://arxiv.org/abs/2508.10104): Meta reports that dense features collapse with large models and long schedules, and adds Gram anchoring to hold the patch-similarity structure to an earlier teacher.
- [Nakamura et al. (Kioxia, 2024)](https://arxiv.org/abs/2410.04801): a memory-chip maker found the DINOv2 artifacts hurt zero-shot image clustering and fixed them with inference-time attention edits. Semiconductor industry is already touching this.

**Why wafer and SEM imagery is the stress case**

Artifacts land in redundant, low-information regions. Semiconductor images are mostly that: repeated die patterns, line-space arrays, flat background. Back-of-envelope, with the \~2% outlier rate Darcet et al. report for DINOv2 (from memory, verify in their Fig. 3):

```
518 px input / 14 px patch = 37 x 37 = 1369 patch tokens
0.02 x 1369 ~= 27 artifact tokens per image
a small defect covers ~1-4 patches
=> artifact tokens outnumber defect tokens ~7x to ~27x per image
```

With your class balance (\~59 positives in tens of thousands), even a small artifact-driven false-alarm rate dominates: 1% image-level FPR on 30,000 negatives is 300 false alarms against 59 true defects.

The direction of the effect is not obvious, which is what makes it a real question. Artifact tokens resemble each other, so a memory bank built from normal images may already contain near-matches and neutralize them.

**Falsifiable questions**

1. Does removing high-norm tokens (DINOv2-reg, or test-time registers on the same weights) change pixel-level AUPRO by more than 2 points on texture-like categories, and by less on object categories?
2. Does a defect-free image's anomaly map change when you shift it by 1 to 13 px? If positional artifacts matter, the per-patch score variance under shift should be a sizable fraction of the defect-vs-normal score gap.

**Overturning controls**

- Same weights with and without a test-time register: one variable changed, nothing retrained.
- Sub-patch translation sweep on clean images: anything that moves the score here is not about content.
- Randomly initialized DINOv2 architecture as a floor for how much of the anomaly signal is architecture and resolution alone.

**Mechanism to look for**

Check whether false-positive hot spots co-locate with high-norm tokens, then ablate the register neurons and see if those false positives vanish. For positional artifacts, fit a linear probe from patch features to (row, col), project that subspace out, and measure the AUPRO change.

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

1. Do anomaly maps track the defect, or the acquisition context that co-occurs with defects (lot, tool, focus, die edge)? With 59 positives, a lot signature is a plausible shortcut.
2. What is the minimum defect area fraction per patch at which pixel AUROC exceeds 0.9? Predict it, then measure it with synthetic defects of controlled size.

**Overturning controls**

- Model randomization applied to anomaly maps: re-initialize the last k blocks of DINOv2. If AUPRO barely falls, the map reflects low-level image statistics, not learned features.
- Counterfactual pasting: move a defect crop onto a clean image from a different lot, and cover the defect with a clean crop. If the image score follows the lot rather than the defect, that is Clever Hans.
- Subgroup reweighting as in Kauffmann et al.: re-score after balancing by tool or acquisition date.

**Mechanism to look for**

The Kauffmann and Montavon group "neuralizes" one-class and kNN scorers so LRP can attribute an anomaly score to pixels and feature channels. Use that to find which feature directions drive the distance, then ablate those directions and check whether the shortcut breaks while true defects survive.

## Belief 3: MVTec-style benchmarks reflect industrial readiness

The benchmark the field optimizes is saturated, and the newer one shows the same methods losing about 35 AU-PRO points. That gap is the opening: nobody has explained mechanistically why these methods fail on the hard cases.

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

So a sub-one-point mean gap is about 2 standard errors before seed variance. For your data (A=0.97, nA=59, nN=30,000) the nN term dominates: SE \~= sqrt(0.0143/59) \~= 0.016, a 95% interval of about ±3 AUROC points. You cannot rank two methods closer than that on 59 positives.

**Falsifiable questions**

1. Does a method's rank on MVTec AD predict its rank on AD 2 or on wafer-like images? Report Kendall's tau with a bootstrap interval. Prediction worth testing: tau is near zero on the small-defect and dark-field scenarios.
2. On highly registered, repetitive imagery, does DINOv2-kNN beat a classical aligned die-to-die difference? Die-to-die comparison is the inspection industry's own baseline, and vision-foundation-model papers almost never run it.

**Overturning controls**

- Aligned die-to-die or template differencing as the non-learned baseline.
- Randomly initialized ViT with the same kNN pipeline, to separate learned features from pipeline effects.
- Bootstrap intervals and at least 3 seeds on every number.

**Mechanism to look for**

Bin AD 2 failures by defect area per patch and by illumination type. Then test whether the failing bins match the patch-area floor from Belief 2 or the artifact tokens from Belief 1. If they do, the benchmark gap has a mechanistic explanation, which is the paper.

## Belief 4: SAE features are real, stable concepts

Vision SAE work now treats dictionary latents as the model's concepts, but the two strongest controls against that view were run on language models, not vision. Running them on DINOv2 with texture-like industrial images is cheap and, as far as I found, unpublished.

**Who relies on it**

- [Stevens et al., SAEs for Scientifically Rigorous Interpretation of Vision Models (2025)](https://arxiv.org/abs/2502.06755): SAEs on ViTs, with feature suppression as the experimental test. Reports different abstractions in CLIP vs DINOv2. Code: [saev](https://github.com/osu-nlp-group/saev).
- [Fel, Wang et al., Into the Rabbit Hull (ICLR 2026)](https://arxiv.org/abs/2510.08638): about 32,000 concepts from DINOv2 using an Archetypal SAE, a stable variant whose atoms stay in the convex hull of real activations. Some concepts fire only on register tokens and encode global properties such as motion blur and illumination. [Blog part I](https://kempnerinstitute.harvard.edu/research/deeper-learning/into-the-rabbit-hull-part-i/), [part II](https://kempnerinstitute.harvard.edu/research/deeper-learning/into-the-rabbit-hull-part-ii/).

**Evidence against it**

- [Paulo and Belrose (ICLR 2026)](https://arxiv.org/abs/2501.16615): two SAEs that differ only in seed share about 30% of features (131K latents, Llama 3 8B). TopK SAEs are more seed-dependent than ReLU plus L1.
- [Heap et al., SAEs Can Interpret Randomly Initialized Transformers (2025)](https://arxiv.org/abs/2501.17727): SAE latents on random-weight transformers look about as interpretable as on trained ones, because the input data is already superposed. They call for random-init controls.
- Rabbit Hull itself: DINOv2 is partly dense, with universally active positional features, antipodal pairs and convex clusters rather than near-orthogonal directions. The authors propose concepts as convex regions (the Minkowski Representation Hypothesis), not linear directions.

**Why industrial images make the random-init control urgent**

Random convolutional features are already decent texture descriptors (Ustyuzhaninov et al., ICLR 2017, from memory). SEM and wafer images are mostly texture and periodicity. So a random DINOv2 may yield "defect-like" latents from image statistics alone.

**Falsifiable questions**

1. On wafer or SEM-like images, what fraction of DINOv2-SAE latents have a match (Hungarian-matched decoder cosine above 0.7) in an SAE on randomly initialized DINOv2? Above 50% means most "concepts" come from data and architecture.
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

The public semiconductor record is thin: the one large public SEM anomaly benchmark tested only ImageNet CNN backbones, no DINOv2, no ViT. A careful DINOv2 interpretability study on it would be close to first of its kind, and its class balance (116 defects in 25,276 images) mirrors yours.

**What industry has published**

- [IBM Albany (Huang et al., ASMC 2025)](https://arxiv.org/abs/2506.03345): DINOv2 transfer plus semi-supervised learning for SEM defect classification, 11 defect types, over 7,400 images from their 300 mm fab. Private data.
- [Kioxia (Nakamura et al., 2024)](https://arxiv.org/abs/2410.04801): attenuating DINOv2 artifacts at inference time (see Belief 1).
- [Bosch and KIT reality check (Baeuerle et al., 2025)](https://arxiv.org/abs/2509.20479): on scanning acoustic tomography of electronic packages, four foundation-model segmentation pipelines score IoU 0.00, against up to 0.52 on MVTec AD. Gemini 2.5 Pro reaches F2 0.37, against 0.82 for a supervised ResNet classifier, while scoring 0.99 on MVTec AD. Their aligned reference-image difference also failed, because recording variation and mechanical tolerances swamped the defects. So treat die-to-die differencing (Belief 3) as a control that can lose, not a straw man.
- [Barusco et al., MIIC benchmark (2025)](https://arxiv.org/abs/2505.07576): nine anomaly methods on SEM images of IC metal layers. Rank order flips relative to MVTec: PaDiM drops to 84.9% image AUROC, and STFPM beats RD4AD. A 2021 inpainting-reconstruction model (99.3%) rivals the best feature method, CFA (99.0%). PatchCore's pixel PRO is 93.0% while its pixel AUPRC is only 44.9%, so PRO hides false positives.

**Public data you can publish on**

| Dataset | Modality | Size | Labels | Why it matters for you |
| --- | --- | --- | --- | --- |
| [MIIC](https://arxiv.org/abs/2505.07576) | SEM, IC metal layer, 512x512 grayscale | 25,276 images: 25,160 normal, 116 anomalous | Pixel masks | Imbalance close to yours; no ViT results yet |
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

## Candidate projects, ranked

Start with project 1: it sits on your DINOv2 system, runs on public SEM data with your class balance, and its audience is the exact group that built the tool it tests. Project 2 reaches the widest interpretability audience, and can share most of project 1's code.

| Rank | Project, in four-tenet form | Belief | Public data | Compute | Venue fit | Main risk |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Register neurons as a false-alarm mechanism in DINOv2 anomaly detection. Belief: patch features are local. Question: do test-time registers cut false alarms on SEM by a measurable amount? Control: same weights with and without the register, plus a sub-patch shift sweep and random init. Mechanism: register-neuron ablation and hot-spot co-location. | 1, 2, 3 | MIIC, MVTec AD 2, Carinthia-S | 1 GPU, days | MIV (its 2025 organizers include Gandelsman and Dravid, test-time-register authors) or VAND | A clean null if the memory bank already neutralizes artifacts; still reportable with controls |
| 2 | Do vision SAEs find defects or image statistics? Belief: SAE latents are learned concepts. Question: what share of DINOv2-SAE latents on SEM match an SAE on random-init DINOv2? Control: random init and three seeds. Mechanism: suppress unmatched latents, measure defect-score change. | 4 | MIIC, Carinthia-S | 1 GPU, about a week | MIV, or NeurIPS and ICLR interpretability workshops | Low; every outcome is informative |
| 3 | The patch-area floor. Question: at what defect-area share does pixel AUROC pass 0.9? Control: synthetic defects of fixed size, resolution 224 vs 518. Mechanism: tie failures to Belief 1 or 2 effects. | 2, 3 | MIIC normals with pasted defects | 1 GPU, days | VAND; also a strong section inside project 1 | Reads as engineering unless the mechanism lands |
| 4 | Where VLMs lose the defect. Question: does an encoder readout beat the VLM by over 10 points on industrial defects? Control: blind, counterfactual and shuffled inputs. Mechanism: layer-wise probe plus attention knockout. | 5 | MMAD, MIIC | One 7B VLM, 24 to 80 GB GPU | MIV or the VAND VLM track | Needs VLM infrastructure; crowded topic |
| 5 | Is OpenVLA's viewpoint fragility in the DINOv2 stream? Control: per-stream ablation, language-free policy. Mechanism: viewpoint activation patching. | 6 | LIBERO-Plus | Simulator time, hundreds of episodes per task | Robot-learning workshops | Furthest from your day job |

**Timing.** VAND 4.0 at CVPR 2026 took papers until March 2, 2026 ([site](https://sites.google.com/view/vand4-cvpr2026)), and [MIV ran at CVPR 2025](https://cvpr.thecvf.com/virtual/2025/workshop/32373). If CVPR 2027 workshops follow that pattern, which is an inference, a deadline lands roughly 20 weeks out: enough for project 1 with project 3 folded in. Also read [Baitieva et al. (VAND 2025)](https://openaccess.thecvf.com/CVPR2025_workshops/VAND), a critical look at academic anomaly benchmarks, before framing project 1.

## Reading order

Read these ten in sequence; the first four are enough to start project 1 this week.

1. [Darcet et al., Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588): the artifact your system probably has.
2. [Jiang et al., Vision Transformers Don't Need Trained Registers](https://arxiv.org/abs/2506.08010): the mechanism and the training-free intervention.
3. [AnomalyDINO](https://arxiv.org/abs/2405.14529): the baseline pipeline you will reproduce.
4. [Barusco et al., MIIC benchmark](https://arxiv.org/abs/2505.07576): your public dataset and its gaps.
5. [Kauffmann et al., Clever Hans in unsupervised learning](https://arxiv.org/abs/2408.08041): the strongest prior that anomaly maps mislead.
6. [Adebayo et al., Sanity Checks for Saliency Maps](https://arxiv.org/abs/1810.03292): the template for a randomization control.
7. [Heap et al., SAEs on randomly initialized transformers](https://arxiv.org/abs/2501.17727): the control project 2 imports into vision.
8. [Paulo and Belrose, SAE seed instability](https://arxiv.org/abs/2501.16615): the second control for project 2.
9. [Fel, Wang et al., Into the Rabbit Hull](https://arxiv.org/abs/2510.08638): the most complete map of DINOv2's concepts and geometry.
10. [Fu et al., Hidden in Plain Sight](https://arxiv.org/abs/2506.08008): the encoder-vs-VLM control, if you move toward project 4.

Next tier: [MVTec AD 2](https://arxiv.org/abs/2503.21622), [Denoising ViTs](https://arxiv.org/abs/2401.02957), [Neo et al.](https://arxiv.org/abs/2410.07149), [LIBERO-Plus](https://arxiv.org/abs/2510.13626) and [Häon et al.](https://arxiv.org/abs/2509.00328).

