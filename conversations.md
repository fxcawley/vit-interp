What I verified

1. The artifact is largely absent in the models AD pipelines actually run.



Darcet et al. (Sec. 2.1, Fig. 4c): across Tiny→giant, "only the three largest models exhibit outliers" (ViT-L and up).

A March 2026 reproducibility study (Baxevanakis et al., arXiv:2603.25803) on the released checkpoints: released DINOv2-S and DINOv2-B show no high-norm feature-map outliers; DINOv2-L shows attention-map artifacts that "do not translate to a high-norm feature representation"; only DINOv2-g has both. Meanwhile OpenCLIP ViT-B/16 has as many or more outliers than OpenCLIP-H.

AnomalyDINO's default is ViT-S/14 (their S/B/L ablation: "no considerable differences," ViT-S best on MVTec-AD). So on the canonical patch-kNN setup there may be nothing for a register to remove.

2. The "nobody tested registers in AD" gap is partly closed — and the result is null.



Dinomaly (CVPR 2025), INP-Former (CVPR 2025), RoBiS (MVTec AD 2 challenge), VisionAD (ACM MM 2025, a patch-kNN method) and AD-DINOv3 all already use DINOv2-R / DINOv3 (registers built in).

Dinomaly Table A1, ViT-B, MVTec-AD: DINOv2 vs DINOv2-R → P-AUROC 98.26/98.35, P-AP 68.35/69.29, AUPRO 94.83/94.79. Within noise. Consistent with (1): no outliers at ViT-B, so registers do nothing there.

Nobody, however, has asked where false-positive peaks land or why, and nobody has applied Jiang et al.'s test-time register to AD. That is the open part.



3. Useful facts for your design



Jiang et al.'s register-neuron analysis was done primarily on OpenCLIP ViT-B/16 (5 neurons after the layer-6 MLP), with DINOv2-L/14 in the appendix. Dense gains were small (ADE20k +0.8 mIoU).

MVTec AD 2: "every single scene was captured under at least four different lighting conditions," and TEST_pub has pixel GT for all lighting conditions (~1084 images). That is exactly your "hold the defect fixed, vary only lighting" control, with GT, no evaluation-server dependency. The two private splits need the server.

Fel et al. (ICLR 2026, the 32k-concept DINOv2 paper) find concepts "that activate exclusively on register tokens ... encode global scene properties like motion blur and illumination." That hands you a mechanism linking (1) to the lighting control: in a no-register model the same global illumination info sits in the outlier patch tokens, so a lighting shift should move exactly those tokens away from the nominal bank.

AnomalyDINO already documents background false positives ("right for the wrong reason" on VisA Capsules) and introduced PCA background masking to suppress them — your test must run unmasked / on textures or the effect is pre-filtered.

Your Hanley–McNeil check: direction right, details off. The minority class per MVTec category is good (12–40 images), not anomalous. Proper HM at A=0.97 gives SE ≈ 1.4 pts for a typical category (2.4 for toothbrush), ≈ 0.4 pts for the 15-category mean. Same conclusion; use paired bootstrap/DeLong over images plus seeds. Note the MVTec AD 2 paper already makes the sub-1-point argument in its abstract, so #3 is a tool, not a contribution.

My take: is it attackable?

Yes, but attack a different belief. Not "registers fixed the artifact" — attack "foundation-model patch features are locally faithful descriptors of their patch," the locality assumption every patch-kNN and reconstruction AD method rests on. Outlier tokens are the known-mechanism violation (they discard local content and carry global content). That framing survives the ViT-S/B facts because it predicts where the effect lives and gives graded results:



On good images, top-1% anomaly peaks land on artifact tokens more often than on redundancy-matched non-artifact patches (uniform-chance is the wrong null — ordinary background mismatches also prefer uniform patches; match on Darcet's neighbour-cosine statistic).

Paste a defect at location A; artifact tokens at location B shift more than non-artifact tokens at B ("the background knows there's a defect"), producing displaced peaks. This is the cleanest mechanistic demo and it's novel.

Lighting shift (MVTec AD 2 TEST_pub): the penalty concentrates on artifact tokens and is removed by test-time registers / DINOv2-R, not by brightness augmentation.

Effect ≈ 0 at S/B, > 0 at L/g, shrinking with shots (1-shot banks contain few outlier tokens to match against).

Mechanism: register-neuron activations → outlier formation → local-info loss (probe) → inflated cosine kNN distance; move the activations into a test-time register and show the distance collapses. Also run the trivial baseline of simply excluding outlier tokens from the kNN, so you can separate "remove the outliers" from "fix the downstream tokens."



Pre-registered step 0 (one afternoon): measure outlier fraction in the exact features each pipeline consumes (S/B/L/g, final vs intermediate layers, with/without registers) on MVTec AD textures and MVTec AD 2. If S/B are ~0%, the DINOv2 half becomes "artifacts are a scale-dependent liability for AD," and the strongest venue for the mechanism is CLIP-based zero-shot AD (WinCLIP/AnomalyCLIP/AdaCLIP on OpenCLIP B/16, L/14): the artifact is certain there, the test-time-register code was built on that exact model, and the field's V-V-attention "DPAM" fixes have no mechanistic account. That might be the better first paper.



Publishability: workshop-grade is near-certain either way if you report effect sizes with paired CIs. Main-track needs predictions 2–3 to hold. Velocity warning: five-plus register papers in 2026 alone (reassessment, zero-ablation, DiT registers, image-generation priors, SAE-on-register-tokens two days ago). The AD-specific mechanism is the moat; a benchmark delta is not.



On #2 as the second paper: still open. Heap et al. is text-only; Korznikov et al. (Feb 2026) do random-SAE baselines on CLIP ViT-B/32 but never re-initialise the model. A random-init DINOv2 SAE through Fel et al.'s pipeline is cheap, and "your defect concepts are image statistics" is a result the industrial side can't ignore. On #4: the NVIDIA 96.8% wafer-map number sits at the level tiny ViT/Swin baselines already report, so the blind-baseline/encoder-probe decomposition would very likely land — but wafer maps are bitmaps; the SEM version needs data you'd have to clear internally. Public SEM data with pixel GT essentially doesn't exist (imec sets are private, WaferDC is boxes only), so plan on MVTec textures / MVTec AD 2 fabric–sheet-metal–rice as the public proxy and KLA data as an appendix if approved.
