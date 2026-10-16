# Locality pilot: goal tree and branch-matching contract

Version `locality-pilot-v1`, specified before the first edited-image model run.
This is an **exploratory pilot**. It can justify another experiment, not a paper.
The terminal research goal is a reproducible, consequential localization failure,
an isolating explanation, and a useful intervention at matched true-defect recall.

## Goal tree

```text
Terminal goal: change a justified model/evaluation/deployment decision
  |
  +-- G0: a consequential question, distinct from the closest prior work
  |       Known neighbors: AnomalyDINO, F2PAD, Reimagining Anomalies.
  |       A novel-looking heatmap is not enough. Novelty remains provisional.
  |
  +-- G1: legitimate public assets + pinned code/checkpoint + runnable environment
  |       unavailable / policy blocked -> BLOCKED (not a scientific null)
  |       procedural fixtures -> engineering path only; never pass G5
  |
  +-- G2: reproduce the released detector's actual operations
  |       compare preprocessing, final features, exact scores, production map
  |       disagreement above tolerance -> INVALID; repair before looking at effects
  |
  +-- G3: valid paired intervention and independent sampling units
  |       no data-role overlap or content duplicates; unique donors per receiver
  |       outside-A pixels identical; identity repeatability; guard-band control
  |       inspect ALL paired-input galleries + first and largest-effect maps
  |       invalid edits / unreviewed provenance -> INVALID / PENDING REVIEW
  |
  +-- G4: calibration fixed before edited inputs are scored
  |       nominal bank and calibration images disjoint from receivers/donors
  |       calibration and threshold never fitted on the edited evaluation set
  |
  +-- G5: primary raw endpoint, after G1–G4 and inspection pass
  |       local positive control uninformative -> INCONCLUSIVE
  |       interval straddles practical margin -> INCONCLUSIVE
  |       interval wholly below margin -> NO LARGE EFFECT IN THIS SETTING
  |       interval wholly above margin -> CANDIDATE FOR INDEPENDENT REPLICATION
  |
  +-- G6: independent category, second edit construction, additional nominal bank,
  |       and accessible current backbone; natural-data relevance
  |       failure -> narrow or abandon general claim; do not select another winner
  |
  +-- G7: mechanism + matched intervention + retained true-defect recall
  |       diagnostic correlation / globally damped scores -> insufficient
  |
  +-- G8: held-out replication + nearest-prior comparison + external criticism
          evidence supports a useful claim -> choose a publication format
```

Only G1–G5 are implemented by this first runner. G6–G8 are not automatically
awarded by an effect-size threshold. The decision memo sets a two-engineering-day,
four-GPU-hour initial spending cap. The runner enforces a conservative four-hour
wall-time cap while the model is resident; it records actual runtime and memory.

## Fixed first setting

- Released AnomalyDINO, **informed** preprocessing, seed-0 five-shot bank,
  no masking or rotation on `grid` and `tile`, 448-pixel input, DINOv2 ViT-S/14.
  This is a documented setting, not the release's default agnostic augmentation.
- `get_intermediate_layers(x)[0]`, default `n=1, norm=True`: final block's
  post-LayerNorm patch tokens, excluding CLS/register tokens.
- L2-normalize feature vectors; exact nearest normalized-L2 distance / 2 equals
  `1 - max cosine`. Chunking changes memory use, not the search result.
- Production map: OpenCV bilinear resize, SciPy Gaussian `sigma=4`, reflect
  padding, default truncation 4. The paired stimuli are 448×448; their production
  maps are also 448×448. Native parity additionally checks the original image size.
- Official source commits and checkpoint SHA256 are in `assets.json`.
  No model is trained; no random-weight fallback exists.

## Stimuli and splits

Bank = first five sorted nominal training images, as in released seed-0 sampling.
Shuffle the remaining training images with the recorded category-specific RNG:
first 64 are calibration; the next N are unique clean donors. All nominal test
images are receivers. Select N unique defective test images using a fixed
permutation and **GT-only** crop eligibility. No score-dependent selection.
If unique donors are insufficient, fail rather than silently reuse donors.

Resize the source images with the released bicubic preprocessing before editing.
Use a fixed 56×56 crop at the GT-densest location in each donor. The sham uses the
same crop coordinates in a distinct nominal donor. GT must occupy at least 16
pixels in the feathered center. Exclusions are logged before inference.

Paste at patch coordinates (8,8) and (20,20). Both locations are fixed before
scoring. Compare hard and seven-pixel feathered edges, with the same alpha mask
for defect and sham. The identity arm leaves the receiver unchanged. All
operations outside A must be byte-identical, including after normalization.

B is **all patches outside the edit rectangle plus a four-patch Chebyshev guard**.
It is fixed geometrically, never chosen by the largest score. Record each trial,
including true-defect visibility, identity error and raw/processed remote alarms.
A control that changes only A's raw scores must not spread into B through the
production postprocessing. Paste plausibility remains a visual/scientific check:
perfect byte invariants alone do not prove that an edit represents a real defect.

## Endpoint and uncertainty

Primary = `grid/hard`. Tile and feather are explicitly secondary checks.
Do not promote whichever cell happens to look strongest into the primary claim.

Calibrate **separate raw and processed thresholds** using whole-image maxima on
64 disjoint nominal images: the `ceil((n+1)*(1-0.05))` order statistic.
This is a nominal image-level alarm target under exchangeability, not a promise
of 5% FPR on edited images or under dataset shift.

Per trial, a remote alarm means `max(score in B) > threshold`. For each receiver,
average `defect alarm - sham alarm` across its two locations. Receivers are the
sampling units. A unique defect donor and clean donor are assigned to each
receiver; repeated locations/constructions are not independent observations.

The pilot relevance margin is **0.10**, i.e. ten additional remotely flagged
images per hundred at this operating point. This is an explicit spending
criterion, not a theoretical prediction or a universal industrial tolerance.

Report the paired mean and a receiver-cluster percentile bootstrap interval.
**Branch selection uses a conservative Hoeffding interval** for receiver values
in [-1,1]. This prevents all-zero paired observations from producing a falsely
precise null. Small datasets will often be inconclusive; the pilot does not
promise power to resolve a ten-point difference. Before a confirmatory study,
justify the margin against an application cost and design for adequate precision.
Intervals condition on the fixed bank, calibration data and donor assignment;
they do not include bank-selection uncertainty or unknown acquisition dependence.

The local positive control is the same paired threshold test at transplanted
GT patches. Its interval must be above zero before the remote result can choose
a scientific branch. Weak detection of the inserted anomalies makes the assay
uninformative. This criterion is deliberately conservative.

## How much inspection is required?

1. Every machine-checkable invariant runs on every pair. Any failure stops the
   run; no malformed condition is quietly dropped after scoring.
2. Before branch matching, inspect the full paired-input gallery, the fixed
   first example, the explicitly labeled largest-effect example, and all
   quantitative conditions. Common color scales prevent visual amplification.
3. Verify provenance, check that edits remain plausible, and consider seams,
   nuisance changes, threshold effects and postprocessing. Record the review
   against the exact manifest hash, including inspected artifact names.
4. If the result and the controls disagree, retain `pending` or `inconclusive`.
   Visual persuasiveness never overrides a failed quantitative or integrity gate.
5. A procedural run always returns `engineering_only`, even if its numbers
   satisfy the mathematical inequality. A positive real-data pilot authorizes
   replication; it is not proof that registers, outliers or global attention
   caused the effect.

The current independent-backbone step is not complete: the DINOv3 Hugging Face
endpoint returned a policy block in this environment. The public MVTec download
was blocked as well. These are access outcomes, recorded in `results/access.json`.
