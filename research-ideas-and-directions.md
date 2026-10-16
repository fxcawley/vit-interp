# Research ideas and directions: interpretability of foundation-model features for industrial defect detection



Status: working document. Started 2026-10-08. Companion files: `lit_review_registers_x_patch_knn_anomaly_detection_20261008.md` (verified facts with sources), `raw/` (page captures of the primary sources).



Conventions

- Each project follows the template: belief -> falsifiable question -> control -> mechanism.

- "Kill criterion" = a pre-specified observation that ends the hypothesis. Write it down before running anything.

- "Gate" = a go/no-go measurement that decides whether the next step is worth running.

- Claims marked [verified] were checked against the primary source on 2026-10-08. Claims marked [to check] were not.

- Effect sizes are reported with paired bootstrap confidence intervals over test images (and over seeds where a random reference sample is drawn). A result "beyond noise" means the CI of the paired difference excludes zero and the point estimate exceeds 2x the seed standard deviation.



---



## 0. Selection criteria for projects



1. The belief is held by practitioners, not only by one paper. Evidence: it is a default in released code or a stated assumption in method papers.

2. The question has a kill criterion that can trigger within the first experiment.

3. The control isolates one mechanism. "Method A vs method B" is not a control.

4. A mechanism-level readout exists (neurons, attention, probes), so the result is an explanation rather than a benchmark delta.

5. Training-free or near training-free; single GPU; public data.

6. The window is open: no published result on the specific question, and the adjacent field is active enough that the result will be read.



---



## 1. Background facts the projects depend on



Compact list; sources and quotes are in the companion lit review.



Artifact tokens

- F1 [verified] Darcet et al. (ICLR 2024): high-norm tokens appear in DINOv2 for ViT-L and larger in their size sweep, after ~1/3 of training, around layer 15 of 40 in ViT-g, on patches with high cosine similarity to their neighbours (redundant patches). ~2.4% of tokens in ViT-g exceed norm 150. They probe worse for position and pixel content and better for image class than normal tokens.

- F2 [verified] Baxevanakis et al. (arXiv:2603.25803, Mar 2026): on released checkpoints, DINOv2-S and DINOv2-B show no high-norm feature-map outliers; DINOv2-L shows attention-map artifacts that are not high-norm tokens in the feature map; DINOv2-g shows both. OpenCLIP-B has as many or more high-norm tokens as OpenCLIP-H. DeiT3-S also has outliers. Model size alone does not determine outlier formation.

- F3 [verified] Jiang et al. (NeurIPS 2025, arXiv:2506.08010): a small set of MLP neurons ("register neurons") produces the outliers (OpenCLIP ViT-B/16: layer-6 MLP, ~5 neurons; DINOv2-L/14 in appendix). Copying their activation into an appended zero token and zeroing it elsewhere ("test-time register") removes outliers from patches. Zeroing instead of moving costs 15% zero-shot ImageNet accuracy. Dense gains are small (ADE20k 48.3 -> 49.1 mIoU; NYUd RMSE 0.388 -> 0.378); object discovery with LOST gains ~20 corloc points. One register suffices. Also removes outlier leakage into LLaVA's text-to-image attention.

- F4 [verified] Fel et al. (ICLR 2026, "Into the Rabbit Hull"): in DINOv2-R, concepts that fire exclusively on register tokens encode global scene properties, named examples: motion blur and illumination.

- F5 [verified] Wu et al. (arXiv:2605.05206, May 2026, DiTs): masking high-norm tokens does not help generation; the problem is "corrupted local patch semantics", not only extreme values.

- F6 [verified] Varma et al. (arXiv:2610.03698, Oct 2026): SAE features on DINOv2 outlier tokens are low-level/texture/background; ablating them changes representation cosine by 0.31% vs 48% for register-token features.

- F7 [verified] Other register papers in 2026: zero-ablation critique (arXiv:2604.14433), test-time registers as global priors for image generation (arXiv:2607.16824), registers for pixel-space DiTs. The area is active; no anomaly-detection paper exists in it.

- F8 [to check] Whether Jiang et al.'s neuron-detection algorithm finds register neurons in DINOv2-S/B even though their output norms are not extreme.

- F9 [to check] Whether outliers exist in the intermediate layers of released DINOv2-L (the layers AD methods consume), as opposed to the final layer analysed by F2.



Anomaly-detection pipelines

- F10 [verified] AnomalyDINO (WACV 2025): DINOv2 ViT-S/14 default, 448 or 672 px, cosine distance to a nominal patch memory bank, image score = mean of the top 1% patch distances, PCA-based background masking for object categories (not textures), rotation augmentation of reference images. Ablation over S/B/L: "no considerable differences"; S best on MVTec AD, larger better on VisA. Appendix documents false positives from background noise and a "right for the wrong reason" detection (VisA Capsules); masking was introduced to suppress them.

- F11 [verified] Dinomaly (CVPR 2025): DINOv2-R ViT-B/14, middle 8 of 12 layers, reconstruction-based. Backbone ablation at ViT-B, MVTec AD: DINOv2 vs DINOv2-R I-AUROC 99.55/99.60, P-AUROC 98.26/98.35, P-AP 68.35/69.29, AUPRO 94.83/94.79. ViTAD: DINOv2 vs DINOv2-R AUPRO 92.7/92.8, P-AP 55.3/54.5.

- F12 [verified] INP-Former (CVPR 2025) and RoBiS (MVTec AD 2 challenge, arXiv:2505.21152) use DINOv2-R ViT-B/14. VisionAD (ACM MM 2025, patch-kNN) uses DINOv2-R ViT-L/14 and gains from S -> B -> L (1-shot MVTec I-AUROC 96.3 / 97.6 / 97.7). AD-DINOv3 uses DINOv3, which ships with registers.

- F13 [verified] CLIP-based zero/few-shot AD (WinCLIP, AnomalyCLIP, AdaCLIP, AA-CLIP, APRIL-GAN) uses OpenCLIP ViT-B/16 or CLIP ViT-L/14-336 without registers. AnomalyCLIP replaces Q-K attention with V-V attention ("DPAM") in late layers to clean patch tokens; no mechanistic account is given.

- F14 [verified] Web search for "test-time register(s)" together with anomaly/defect/industrial inspection returns nothing (2026-10-08).



Benchmarks and statistics

- F15 [verified] MVTec AD 2 (arXiv:2503.21622, IJCV 2026): 8 objects, >8000 images, every scene captured under at least four lighting conditions; training set regular lighting only. TEST_pub has pixel ground truth for all lighting conditions (per object 80-162 images, ~1084 total). TEST_priv and TEST_priv,mix: images only, scoring via evaluation server. SOTA AU-PRO_0.30 < 60%, AU-PRO_0.05 < 31%. Lighting robustness differs across methods (RD -1.4 points, MSFlow -12.4 points on priv,mix). The paper itself states that MVTec AD / VisA SOTA differences are below one point and inside stochastic noise.

- F16 [verified] MVTec AD test set: 467 good / 1258 defective across 15 categories. The minority class per category is the good class (12 for toothbrush, 20-40 typical).

- F17 [computed] Hanley-McNeil SE at AUROC 0.97: ~0.014 for a typical category (84 bad / 31 good), 0.024 for toothbrush, ~0.004 for the 15-category mean under independence. Few-shot methods add reference-sample variance (AnomalyDINO 1-shot SD 0.4-0.8 points over 3 seeds).



Secondary topics

- F18 [verified] Heap et al. (arXiv:2501.17727): random and trained text transformers yield similarly interpretable SAE latents under auto-interp metrics. Korznikov et al. (arXiv:2602.14111, Feb 2026): random-direction / random-activation SAE baselines match trained SAEs on interpretability, sparse probing and causal editing; includes CLIP ViT-B/32; the model is not re-initialised. No paper found that trains an SAE on a randomly initialised vision foundation model.

- F19 [verified] Fu et al. (COLM 2025, arXiv:2506.08008): VLMs score near chance on vision-centric tasks their encoders solve; the bottleneck is visual-linguistic integration; probes at encoder, projector and LLM layers.

- F20 [verified] NVIDIA Cosmos Reason wafer-map fine-tune: 96.8% accuracy (zero-shot 14.37%). Small supervised ViT/Swin models on wafer maps report 96.8-97.5%. IBM (arXiv:2506.03345): >90% SEM defect classification with <15 images per class, 11 classes, ViT transfer.

- F21 [verified] Public SEM data with pixel ground truth: none found. imec LS-ADI / LS-AEI / HEXCH-DSA are internal. WaferDC (EAAI 2025) is SEM with boxes. Wafer-map datasets: WM-811K (811k maps, ~173k labelled, 9 classes), MixedWM38 (38k maps, 38 classes).



---



## 2. Project A (primary): locality violations in patch features and their cost for dense anomaly scoring



### 2.1 Belief attacked



Patch-level AD (kNN memory banks, feature reconstruction, patch-text similarity) assumes each patch token describes its own patch. Call this the locality assumption. High-norm/outlier tokens violate it by construction (F1): they discard local content and carry global content. The field's working position is either (i) the artifact is irrelevant to AD (AnomalyDINO ships ViT-S without registers; Dinomaly's ablation shows nothing), or (ii) it was fixed by registers (Dinomaly, INP-Former, VisionAD, DINOv3 adopt register backbones without stating why). Neither position has been tested where the artifact exists.



The original framing ("registers fixed the artifact; does it still hurt AD?") fails on F2: the backbones in the standard pipelines have no feature-map outliers, so a test on ViT-S/B would hit the kill criterion for the wrong reason. The locality framing predicts where the effect is (L/g, CLIP) and gives graded outcomes.



### 2.2 Falsifiable predictions



P1 Co-location. On defect-free test images, the top-1% anomaly-score patches fall on artifact tokens more often than on non-artifact tokens matched for redundancy (neighbour cosine similarity) and position.



P2 Action at a distance. Pasting a defect at location A changes the embedding of artifact tokens at a distant location B more than that of matched non-artifact tokens at B, and creates or raises anomaly peaks at B. With a test-time register the drift of the former artifact tokens at B falls to the level of normal tokens.



P3 Lighting. On MVTec AD 2 TEST_pub, the increase in false-positive mass under a lighting change is concentrated on artifact tokens, and test-time registers or DINOv2-R remove more of the lighting penalty than brightness/exposure augmentation of the memory bank does.



P4 Scale dependence. The effect in P1-P3 is ~0 for DINOv2-S/B and >0 for L/g, and present for OpenCLIP ViT-B/16 and ViT-L/14. (From F2; a direct test of "registers do not matter for AD" being a scale-conditional statement.)



P5 Shot dependence. The P1 effect shrinks as the memory bank grows (more artifact tokens in the bank to match against) and is largest at 1-shot.



P6 Spatial signature. On images where every patch is redundant (periodic patterns: MVTec grid, MVTec AD 2 fabric, synthetic line/space), artifact positions are driven by position embeddings and recur at fixed positions across images; artifact-driven false positives therefore recur at fixed positions, unlike defect-driven detections. (Relevant to production: a fixed-position false-alarm signature is detectable in the field.)



### 2.3 Kill criteria



K1 Gate 0 fails: no model/layer combination used by any AD pipeline has >= 0.5% artifact tokens on MVTec AD / MVTec AD 2 images -> the DINOv2 half of the project is dead; move to the CLIP variant (A2).

K2 P1 enrichment CI includes 1 after redundancy matching, on every model that passes Gate 0.

K3 Oracle bound (see 2.5) below 2x seed SD for pixel AUROC and AU-PRO on every model that passes Gate 0 -> any register method is bounded to a null effect on the metrics; the paper becomes a tight null plus the mechanism of why (artifact tokens match each other in the bank).

K4 Test-time registers change pixel AUROC / AU-PRO by less than the paired-bootstrap noise and the masking arm shows the same -> no causal role.



K2-K4 can each kill a prediction but not the project: a tight null with the oracle bound and the matching mechanism is still a result. K1 redirects the project.



### 2.4 Gate 0: artifact census



Measure before anything else.



Models

- DINOv2: ViT-S/14, B/14, L/14, g/14, and the `_reg` variant of each (torch hub).

- DINOv3: distilled ViT-S/B/L/H+ (all with registers) [to check: availability and license].

- OpenCLIP: ViT-B/16 (LAION-2B), ViT-L/14 (LAION-2B); CLIP ViT-L/14-336 (OpenAI) as used by AnomalyCLIP.

- DINO ViT-B/16 as the artifact-free reference (F1).

- Optional: SINDER-repaired DINOv2 (Wang et al., ECCV 2024, fine-tuning repair) if weights are public [to check]; DeiT-III.



Features

- Per layer, pre-final-LayerNorm token norms (the convention in F1/F2). Also post-LN, because AnomalyDINO consumes `x_norm_patchtokens` [to check in the repo].

- Register-neuron detection with Jiang et al.'s algorithm, to get an artifact mask even where norms are not extreme (addresses F8).

- Attention from patch queries to the top-norm keys in the last four layers (attention-sink mass), per F2's point that attention artifacts and norm artifacts are not the same thing.



Statistics per model x layer x dataset

- Fraction of tokens above a bimodality-derived threshold (fit a two-component mixture to log-norm; report both the mixture criterion and a fixed k x median criterion).

- Neighbour cosine similarity at the patch-embedding layer for artifact vs normal tokens (Darcet's redundancy statistic).

- Spatial recurrence: entropy of artifact-position distribution across images of a category (for P6).

- Resolution dependence: 224, 448, 672 px (patch count changes the outlier fraction [to check]).

- Tiling: MVTec AD 2 images are 2448 x 2048; RoBiS tiles at 518. Each tile gets its own outliers, so tiling multiplies artifact count per image. Record both whole-image-downscaled and tiled regimes.



Datasets for the census: MVTec AD train (good) all 15 categories; MVTec AD 2 train all 8 objects; VisA; 1000 ImageNet-val images as the natural-image reference.



Gate decision: proceed on the DINOv2 track with every model/layer where the artifact fraction >= 0.5% on at least the texture categories; otherwise A2 only.



### 2.5 Measurements for P1 (co-location) and the oracle bound



Pipelines

- AnomalyDINO as released (cosine, top-1% statistic), run unmasked and masked; textures are unmasked by default.

- PatchCore-style L2 kNN on the same features (norm enters the distance directly; separates "direction" from "norm" mechanisms).

- VisionAD (multi-layer fusion, DINOv2-R ViT-L default; swap in non-R ViT-L).

- Dinomaly / INP-Former at ViT-L with and without registers (first informative test of their ablation; the ViT-B ablation in F11 is uninformative under F2).



Definitions

- False-positive peak: on a good test image, (a) the single max patch, (b) the top-1% patches, (c) connected components above the threshold that yields a fixed image-level FPR on the category's good images.

- Artifact mask: (a) norm criterion at the artifact layer, (b) register-neuron activation criterion. Report both; they will disagree on some tokens.

- Enrichment: P(peak on artifact) / P(artifact). Null 1: permute artifact masks across images of the same category (preserves spatial priors of artifacts). Null 2: for each artifact token draw a non-artifact token from the same image matched on neighbour-cosine bin and position bin; compare peak rates. Report Null 2 as the primary test; Null 1 as secondary.

- Misattributed detection rate: fraction of defective images classified correctly at a fixed image-level threshold whose top peak lies outside the GT mask (the "right for the wrong reason" rate of F10). Report overall and the share whose top peak is an artifact token.

- Oracle bound: replace anomaly scores at artifact tokens by the median of their non-artifact 8-neighbours and recompute pixel AUROC, AU-PRO_0.30, AU-PRO_0.05, pixel F1-max. This is the maximum metric gain any outlier-removal method can deliver without changing non-artifact tokens. Compute before running any intervention.



### 2.6 Intervention arms



1. Baseline model.

2. Test-time register, 1 token (Jiang et al. code). Sweep: number of register neurons, 1 vs 4 registers.

3. Artifact-token exclusion: drop artifact tokens from the query side (scores interpolated from neighbours) and from the memory bank. Three sub-arms: query only, bank only, both. This separates "stop scoring artifacts" from "stop matching against artifacts".

4. Trained registers: DINOv2-R of the same size. An independently trained model, so differences include training variance (Jiang et al. make the same caveat); report as a reference, not as an ablation.

5. Zeroed register neurons (negative control; expected to damage features per F3). Shows the fix is "move", not "remove".

6. Random-neuron shift (specificity control): apply the test-time-register edit to an equal number of random MLP neurons in the same layer.

7. SINDER or other repaired checkpoints if available.

8. For P3 only: brightness/exposure/gamma augmentation of the memory bank (the practitioner's fix), alone and combined with arm 2.



Primary outcomes per arm: enrichment (Null 2), misattributed detection rate, pixel AUROC, AU-PRO_0.30, AU-PRO_0.05, pixel F1-max, image AUROC, all with paired bootstrap CIs over images and >= 5 reference-sample seeds for few-shot.



### 2.7 Mechanism experiments



M1 Distance decomposition. For each artifact token in a test image: nearest-neighbour identity in the bank (artifact or normal token), cosine distance, and the same for its matched normal token. Hypothesis: artifact tokens match other artifact tokens; their distance depends on global-content differences between test and reference images (lighting, pose, defect presence), not on local content. Test by regressing artifact-token distance on image-level descriptors (global brightness, [CLS] cosine between test and reference, presence of a defect elsewhere).



M2 Local-information probe. Linear probes for patch position and pixel reconstruction trained on normal tokens, evaluated on artifact tokens, before and after the test-time register (Darcet's probes). Prediction: with the register, the former artifact positions regain local information, which is what makes them matchable.



M3 Register neurons -> outlier -> distance. Trace: activation of register neurons at the artifact layer -> norm at later layers -> kNN distance at the output. Intervene on the neuron activation scale (0.25x, 0.5x, 1x, 2x) and record the dose-response of the kNN distance.



M4 Paste-defect protocol (P2). Source: defect crops from defective images of the same category using the GT mask (cut-paste, with and without Poisson blending). Controls: paste a crop from another good image at the same location (pasting artifact control); paste nothing. Measure per-token drift ||f_B(pasted) - f_B(clean)|| for tokens at Chebyshev distance >= r patches from A (r = 4, 8), separately for artifact and matched normal tokens; measure new peaks at B. Repeat under arm 2. Also record drift of the [CLS] token as the global-content reference.



M5 Lighting protocol (P3). MVTec AD 2 TEST_pub: for each scene with >= 4 lightings, bank from train (regular lighting). Score each lighting variant. Per-token score delta vs regular lighting; share of lighting-induced false-positive mass on artifact tokens; AU-PRO per lighting; arms 1, 2, 4, 8. [to check: TEST_pub folder layout pairs the same scene across lighting conditions.] If pairing is unavailable, use synthetic exposure/gamma changes on MVTec AD textures as a weaker substitute and say so.



M6 Attention-sink accounting. Per layer, fraction of attention mass from patch queries to artifact keys; correlation across categories with the misattributed detection rate. Links the norm mechanism to the attention mechanism (F2 distinguishes them).



M7 Spatial recurrence (P6). On grid / fabric / synthetic line-space images: artifact positions across images vs false-positive positions across good images; mutual information between the two position distributions. Distinguish from positional-embedding artifacts (Denoising Vision Transformers, Yang et al., ECCV 2024) which also produce fixed spatial patterns but are not high-norm tokens: PE artifacts persist under test-time registers, outlier artifacts do not.



### 2.8 Nuisance-variable tests carried over from the original #3



N1 Intensity-matched synthetic defects: insert defects whose local mean/variance match the background (histogram-matched crops, or structural edits such as line breaks in periodic patterns). Question: does the anomaly score track structure or local contrast? Report detection rate as a function of the contrast gap.

N2 Score vs local contrast on good images: correlation between per-patch anomaly score and local gradient energy / variance, per model and per arm.



### 2.9 Statistical protocol



- Paired bootstrap over test images (10k resamples) for all metric differences; DeLong for paired image-level AUROC.

- >= 5 reference-sample seeds in few-shot settings; report mean and SD; "beyond noise" as defined in Conventions.

- AU-PRO at both 0.30 and 0.05 integration limits (MVTec AD 2 convention); pixel F1-max; the MVTec AD 2 threshold rule (validation mean + 3 SD) for threshold-dependent metrics.

- Per-category results in the appendix; no averaging across categories without per-category CIs.

- Preregister predictions P1-P6, kill criteria K1-K4, and the gate threshold in a dated file before running Gate 0.



### 2.10 Compute and code



- Single 24-48 GB GPU. ViT-g/14 at 448 px: 1024 tokens; MVTec AD test 1725 images; MVTec AD 2 TEST_pub ~1084 images; VisA test ~2.2k images. Census + interventions: tens of GPU-hours.

- Code: AnomalyDINO (github.com/dammsi/AnomalyDINO), test-time registers (linked from arXiv:2506.08010), Dinomaly (github.com/guojiajeremy/Dinomaly), VisionAD (github.com/Qiqigeww/VisionAD), anomalib (MVTec AD 2 datamodule; Dinomaly implementation), DINOv2 hub models including `_reg`, MVTec AD 2 evaluation scripts.



### 2.11 Risks and pre-specified contingencies



- R1 Gate 0 fails for DINOv2 (F2 suggests it may for S/B; L is uncertain per F9). Contingency: A2 (CLIP) becomes the primary; the census itself is a short note ("artifacts are scale- and training-regime-dependent in released checkpoints; implications for backbone choice in AD").

- R2 Artifacts exist but enrichment is at chance and the oracle bound is below noise. Contingency: publish the bound and M1 (artifact tokens match each other in the bank) as the explanation of why registers do not matter for kNN AD; this is also a result for the field's unexplained adoption of register backbones.

- R3 Test-time registers change non-artifact tokens too (the edit is mid-network). Attribution via arm 3 (exclusion) and M2.

- R4 TEST_pub too small per lighting condition. Contingency: CIs over scenes; synthetic exposure changes as a secondary test.

- R5 Concurrent work. Post the census and P1/P2 as a preprint early; the AD-specific mechanism is the part others are not doing.



### 2.12 Deliverables



- Census table (model x layer x dataset x resolution).

- Enrichment, misattributed-detection and oracle-bound tables per model and pipeline.

- Intervention table (arms 1-8) with CIs.

- M1-M7 figures: distance decomposition, probe recovery, dose-response, paste-defect drift, lighting penalty decomposition, attention-sink mass, spatial recurrence.

- Code and preregistration file.



### 2.13 Variant A2: CLIP-based zero-shot anomaly detection



Why: the artifact is certain in the backbones used (F2: OpenCLIP-B has many outliers), Jiang et al.'s register neurons were found on OpenCLIP ViT-B/16, and the field has ad hoc fixes (DPAM / V-V attention, CLIP Surgery-style modifications) without mechanism.



Belief: patch-token-to-text similarity localises anomalies.

Predictions:

- A2-P1 Zero-shot anomaly maps (WinCLIP, APRIL-GAN-style, AnomalyCLIP without DPAM) have peaks on outlier tokens at above-matched-chance rates; the misattributed detection rate is high on defective images (the outlier token, carrying global content, aligns with "damaged" text when the image is anomalous, so the image-level answer is right and the location wrong).

- A2-P2 Applying a test-time register to the stock encoder recovers a measurable share of the localisation gain attributed to DPAM / V-V attention; the share is the quantity to report.

- A2-P3 Learned-prompt methods (AnomalyCLIP, AdaCLIP) learn prompts whose similarity to outlier tokens is suppressed relative to normal tokens (probe the learned text embeddings against outlier vs normal patch tokens).

- A2-P4 Attention-sink mass from patch queries to outlier keys in the last layers correlates with per-category localisation error.

Kill: test-time registers change WinCLIP pixel AUROC / AU-PRO by less than implementation noise and no enrichment.

Arms: stock encoder; test-time register; V-V attention only; both; DPAM as released.

Data: MVTec AD, VisA, MVTec AD 2 TEST_pub (lighting), BTAD, MPDD.



### 2.14 Variant A3: VLM-based anomaly detection



AnomalyGPT / Myriad-style systems pass vision-encoder tokens to an LLM. F3 shows outlier visual tokens leak into LLaVA's answer-token attention. Questions: does the LLM's visual attention on defect questions sit on artifact tokens; do test-time registers change the textual verdicts or the described locations; is there a defect-hallucination rate attributable to artifacts? Lower priority: requires VLM inference and the AD VLMs' code; bridges to Project C.



### 2.15 Side results that fall out of Gate 0



- S1 Does distillation suppress outlier formation? DINOv2 S/B/L are distilled from g; released S/B lack outliers (F2). Compare register-neuron existence and activation across the family; compare with DINOv3's distilled models (teacher has registers).

- S2 Register tokens as nuisance readouts (method idea, not falsification): if register / test-time-register tokens encode illumination (F4), read illumination from them and use it to normalise anomaly scores per image. Flag as a follow-up method paper, separate from the interpretability result.



---



## 3. Project B: random-initialisation control for SAE concepts in DINOv2



### 3.1 Belief attacked



SAE/dictionary concepts extracted from a vision foundation model (Fel et al., 32k atoms; Stevens et al.; others) reflect what the model learned. Industrial reading: "defect concepts" found in DINOv2 features are evidence that the model represents defects.



### 3.2 Question



Do the same concepts, including the defect-like ones, appear when the same SAE recipe is trained on a randomly initialised network of the same architecture? If yes, the concept reflects input statistics (which random networks preserve, per Heap et al.), not learning.



### 3.3 Design



Models

- Trained DINOv2 ViT-B/14 (and L/14); DINOv2-R for register-token concepts.

- Random init, same architecture, >= 3 seeds.

- Shuffled-weights control: permute entries within each trained weight matrix (preserves weight statistics, destroys structure).

- Optional training curve: a ViT trained from scratch on ImageNet-100 or similar with checkpoints, to see when concepts appear. Not available for DINOv2 itself.

- Layer sweep: patch-embedding output, early, middle, late layers. Prediction: differences between random and trained grow with depth; at the patch-embedding layer both are linear functions of pixels plus position.



SAE

- Same recipe for all (TopK or Archetypal), same token budget, same images (ImageNet subset plus MVTec AD / VisA / MVTec AD 2), same dictionary size. Report reconstruction and sparsity for all models so interpretability comparisons are at matched operating points.



Evaluations

- E1 Auto-interp score (LLM explanation of top-activating patches, then held-out prediction), as in Heap et al. Expectation: similar between random and trained (the Heap result), which is the point.

- E2 Concept category mix: fraction of latents an annotator/LLM assigns to colour, edge/orientation, texture, position, object part, scene. Prediction: random = colour/edge/position; trained adds part/scene. Report the mix, not only the mean score.

- E3 Defect relevance: sparse probing of pixel-level defect labels from SAE latents (MVTec AD / VisA GT), for random vs trained; number of latents that are selective for defective patches at matched precision.

- E4 Causal relevance (trained only has a task; for random use the kNN AD task): ablate "defect latents" and measure the change in patch-kNN anomaly scores for a kNN detector built on the same features. Also run patch-kNN AD on random-init ViT features outright (baseline: how much of AnomalyDINO's AUROC survives random weights). Cite prior results on random-feature anomaly detection before claiming novelty [to check].

- E5 Match to Fel et al.'s dictionary (if released): fraction of trained-model atoms with a cosine-matched counterpart in the random-model dictionary after Procrustes alignment, by concept category.



### 3.4 Predictions and kill criteria



- B-P1 Auto-interp scores do not separate random from trained (replicates Heap in vision).

- B-P2 Low-level defect-like latents (scratch, hole, dark spot, edge discontinuity) appear in random models with comparable auto-interp scores; semantic latents do not.

- B-P3 Defect probing accuracy from random-model latents is a large fraction of the trained-model value on textures and a smaller fraction on objects.

Kill: random-model latents are uninterpretable under the same pipeline (then SAE metrics do discriminate in vision, which contradicts Heap's transfer and is itself a result), or defect probing from random latents is near chance.



### 3.5 Confounds



- Random ViTs with LayerNorm and attention still mix tokens; "random" is not "pixel". State what the random network computes.

- Positional embeddings dominate random-model latents unless removed or controlled; report with and without PE.

- Token budget and dictionary size change auto-interp scores; match them.

- The auto-interp judge's bias toward low-level descriptions; use two judges and a human subset.



### 3.6 Cost



SAE training on ~10-50M patch tokens per model per layer; one GPU-day per model/layer at ViT-B scale [to check]. Public data only.



---



## 4. Project C: VLM accuracy on wafer maps / SEM vs the encoder and a blind baseline



### 4.1 Belief attacked



A fine-tuned VLM "sees" wafer defects (F20: 96.8% after fine-tuning Cosmos Reason). F19 says the LLM often fails to use information the encoder has.



### 4.2 Design



Data: WM-811K (9 classes incl. none; severe imbalance, "none" dominates labelled maps), MixedWM38 (38 classes, mixed-type defects). Metrics: macro-F1 and per-class recall, not accuracy.

Readouts (Fu et al. decomposition):

- C1 Blind baseline: fine-tune / prompt the LLM with no image (or with a constant image); reports the prior the VLM can reach from label distribution and prompt wording.

- C2 Linear probe on the VLM's frozen vision encoder (identify the encoder in the Cosmos Reason release [to check]); also on DINOv2 and a small supervised CNN for reference.

- C3 Probe after the projector.

- C4 Probes on LLM hidden states at several layers.

- C5 Image-dependence tests: swap images across samples with fixed prompts; answer-change rate. Shuffle-pixel control.

- C6 Language-prior test: paraphrase class names (e.g., replace "scratch" with a neutral token) and measure the change.

Predictions: C2 >= fine-tuned VLM on macro-F1; VLM zero-shot ~ blind baseline; mixed-type classes (MixedWM38) show the largest VLM-vs-probe gap; C5 shows a non-trivial fraction of answers unchanged under image swap.

Kill: fine-tuned VLM exceeds the encoder probe by more than CI width on macro-F1 and passes C5 — then the VLM adds something; characterise what via C4.

SEM version: requires internal data; a public substitute does not exist (F21). Plan the wafer-map version for publication and the SEM version as an internal appendix if cleared.



Note on value: the headline comparison is near-certain (small supervised models already match the VLM number). The content is the decomposition and the mixed-type failure analysis.



---



## 5. Project D: what does domain adaptation change inside a vision foundation model?



Tension: IBM gets >90% SEM classification with <15 images/class by transfer from DINOv2 (F20); NVIDIA states general foundation models lack detail for e-beam/optical microscopy and need adaptation.



Design

- Base DINOv2 vs adapted (LoRA / full fine-tune on SEM or a public microscopy/texture proxy).

- Shared dictionary across both (Universal SAE, Thasarathan et al. 2025) to align concepts; report concepts gained, lost, re-weighted, by layer.

- Representation change by layer (CKA, linear probe transfer in both directions).

- Specific hypotheses: adaptation changes early-layer statistics (SEM noise, charging, edge profiles) more than late semantics; outlier-token behaviour changes under adaptation (does fine-tuning on redundant imagery increase artifact fraction? connects to Project A).

Data: internal SEM for the real question; public proxies (NEU steel, Severstal, DAGM textures, MVTec textures) for the method. Lower publication priority without SEM release.



---



## 6. Project E: statistics of saturated AD benchmarks (tool, or appendix to A)



- Paired bootstrap / DeLong CIs for published methods on MVTec AD and VisA using released anomaly maps where available; probability of rank flips under test-set resampling.

- The minority-class observation (F16): the good class limits precision; report per-category n_good next to every AUROC.

- AU-PRO variance: units are anomalous regions; estimate via region-level bootstrap.

- Seed and reference-sample variance for few-shot methods as a function of shots.

Not a standalone paper unless it comes with a service (a script and a table the community adopts). Keep as the statistical backbone of A.



---



## 7. Project F (lower priority, not KLA-tied): VLA interpretability transfer



Two 2026 findings complicate "LLM interp transfers to VLAs": SAE features in VLAs correspond to memorised training sequences; action features are poorly linearly separable in OpenVLA but recoverable in pi0.5. Possible question: is memorisation in VLA SAE features a property of the policy or of the SAE recipe (apply Project B's random/shuffled controls to a VLA)? Park here; revisit if A and B conclude.



---



## 8. Cross-cutting inventory



Datasets

- MVTec AD (15 categories, CC BY-NC-SA): textures carpet, grid, leather, tile, wood are the redundancy-heavy public proxies for wafer/SEM backgrounds.

- MVTec AD 2 (8 objects, CC BY-NC-SA): fabric, sheet metal, rice, wall plugs for redundancy; TEST_pub for lighting with GT; private splits via server.

- VisA (12 categories, CC BY 4.0): PCB1-4 for electronics-like structure.

- MVTec LOCO (logical anomalies), Real-IAD (30 objects, multi-view), BTAD, MPDD, DAGM (synthetic textures), KolektorSDD2.

- Wafer maps: WM-811K, MixedWM38.

- SEM: none public with pixel GT; WaferDC (boxes); imec sets private; KLA internal (publication requires clearance).



Metrics

- Image AUROC; pixel AUROC; AU-PRO at 0.30 and 0.05; pixel AP and F1-max; MVTec AD 2 threshold rule for F1.

- New: artifact enrichment (matched null), misattributed detection rate, oracle-removal bound, lighting-penalty share on artifacts, spatial recurrence MI.



Known artifact sources to keep separate

- High-norm / register-neuron outliers (this project).

- Positional-embedding artifacts (Denoising Vision Transformers): fixed spatial pattern, not high-norm, survive registers.

- Attention-only artifacts without norm outliers (F2, DINOv2-L).

- Image-level nuisance (lighting, contrast, background noise) acting on all tokens.



---



## 9. Open questions to resolve first



1. F9: artifact fraction in released DINOv2-L at intermediate layers, on MVTec textures.

2. F8: does the register-neuron detector fire in DINOv2-S/B?

3. MVTec AD 2 TEST_pub: are the same scenes present under each lighting condition with matching IDs?

4. AnomalyDINO: which output tensor is used (pre- or post-LN), which layer, and whether `_reg` models are a drop-in (they are in torch hub; the repo may not expose them).

5. Fel et al. dictionary: code and atoms released?

6. Cosmos Reason vision encoder identity and whether hidden states are accessible for probes.

7. Prior work on random-weight features for anomaly detection (for Project B E4).

8. SINDER weights availability.

9. DINOv3 licence terms for research use.



---



## 10. Suggested order with gates



1. Preregister A (P1-P6, K1-K4, gate threshold). Write Open Questions 1-4 answers into the file.

2. Gate 0 census (A 2.4). Decide DINOv2 track vs A2-only.

3. P1 + oracle bound + misattributed detection on MVTec AD textures and VisA, unmasked, with every model passing Gate 0. Gate: enrichment CI excludes 1 or oracle bound beyond 2x seed SD; otherwise write the null with M1.

4. Interventions (arms 1-8), M1-M3.

5. M4 paste-defect, M5 lighting, M6 attention accounting, M7 spatial recurrence.

6. Write-up; post preprint; A2 in parallel if resources allow or if Gate 0 redirected.

7. Project B after A's census, reusing the feature extraction code; Project C independently (different stack); D and E as appendices or internal reports.



---



## 11. Sources



- Darcet, Oquab, Mairal, Bojanowski. Vision Transformers Need Registers. ICLR 2024. arXiv:2309.16588.

- Jiang, Dravid, Efros, Gandelsman. Vision Transformers Don't Need Trained Registers. NeurIPS 2025. arXiv:2506.08010.

- Baxevanakis, Karageorgis, Dravilas, Szewczyk. Do All Vision Transformers Need Registers? A Cross-Architectural Reassessment. arXiv:2603.25803 (TMLR submission).

- Zero-Ablation Overstates Register Content Dependence in DINO Vision Transformers. arXiv:2604.14433.

- Wu et al. Taming Outlier Tokens in Diffusion Transformers. arXiv:2605.05206.

- Test-Time Registers as Global Priors for Tokenized Image Generation. arXiv:2607.16824.

- Varma, Rufail, Khullar, Sharma. Decoding the Functional Roles of Register and High-Norm Patch Tokens in Vision Transformers. arXiv:2610.03698.

- Leveraging Registers in Vision Transformers for Robust Adaptation. arXiv:2501.04784.

- Sun et al. Massive Activations in Large Language Models (attention biases, includes ViTs). 2024.

- Yang et al. Denoising Vision Transformers. ECCV 2024.

- Wang et al. SINDER: Repairing the Singular Defects of DINOv2. ECCV 2024.

- Damm, Laszkiewicz, Lederer, Fischer. AnomalyDINO. WACV 2025. arXiv:2405.14529.

- Guo et al. Dinomaly. CVPR 2025. arXiv:2405.14325.

- Luo et al. INP-Former. CVPR 2025. RoBiS, arXiv:2505.21152.

- Wang, Guo et al. Search is All You Need for Few-shot Anomaly Detection (VisionAD). ACM MM 2025. arXiv:2504.11895.

- AD-DINOv3. arXiv:2509.14084. DINOv3, arXiv:2508.10104.

- Jeong et al. WinCLIP. CVPR 2023. Zhou et al. AnomalyCLIP. ICLR 2024. Cao et al. AdaCLIP. ECCV 2024. AA-CLIP. CVPR 2025.

- Roth et al. PatchCore. CVPR 2022.

- Heckler-Kram et al. The MVTec AD 2 Dataset. arXiv:2503.21622; IJCV 2026. https://benchmark.mvtec.com

- Bergmann et al. MVTec AD. CVPR 2019 / IJCV 2021. Zou et al. VisA. ECCV 2022.

- Fel et al. Into the Rabbit Hull: From Task-Relevant Concepts in DINO to Minkowski Geometry. ICLR 2026.

- Thasarathan et al. Universal Sparse Autoencoders. 2025.

- Stevens et al. Interpretable and Testable Vision Features via Sparse Autoencoders. arXiv:2502.06755.

- Heap et al. Sparse Autoencoders Can Interpret Randomly Initialized Transformers / Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers. arXiv:2501.17727.

- Korznikov et al. Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines? arXiv:2602.14111.

- Fu, Bonnen, Guillory, Darrell. Hidden in plain sight: VLMs overlook their visual representations. COLM 2025. arXiv:2506.08008.

- IBM. Semiconductor SEM Image Defect Classification Using Supervised and Semi-Supervised Learning with Vision Transformers. ASMC 2025. arXiv:2506.03345.

- NVIDIA Cosmos Cookbook, wafer-map anomaly classification with Cosmos Reason 1 (96.8% fine-tuned, 14.37% zero-shot).

- Häon et al. Steering VLAs through internal representations. CoRL 2025. (VLA SAE memorisation and pi0.5 separability papers: 2026, citations to be added from the 17-page review.)


