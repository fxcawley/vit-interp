# First execution record — 2026-10-16

**Real-data pilot: blocked. Engineering validation: completed.**
Neither a positive nor a negative scientific conclusion has been drawn.

## What ran

`engineering-v1/` preserves the exported numerical record and inspection figures
from `pilot/runs/procedural-v2`. It uses the actual official pretrained DINOv2
ViT-S/14 checkpoint, at 448 pixels, on generated texture fixtures:

- Two fixture families, 12 independent receivers per family.
- Two fixed locations and two edit constructions per receiver: **96 trials**.
- Identity, matched normal-crop sham, and defect-crop arms.
- Separate bank, calibration and donor roles with duplicate-content checks.
- **28.1 seconds** reported runner wall time, excluding environment installation
  and fixture generation. PyTorch peak allocated GPU tensor memory was
  **153,247,232 bytes (~146 MiB)**; this is not total device/driver memory use.

The immutable run manifest identifies implementation commit `96d1f81`, the
configuration and source hashes, checkpoint hash, environment and all input
hashes. Raw patch maps remain in the ignored local run directory; every paired
metric and its source identities are preserved in `pairs.jsonl`. The generator
and one-command runner reproduce the experiment without proprietary data.

## What was checked

The unmodified released AnomalyDINO detection code was executed on deterministic
reference subsets, independently constructing its bank and using FAISS.

| Comparison | Maximum absolute difference |
|---|---:|
| Preprocessed tensor | 0 |
| Extracted final patch features | 0 |
| Exact nearest-neighbor patch scores | 1.014e-6 |
| Production anomaly map | 6.588e-7 |
| Repeated identity forward/scoring | 0 |
| Direct interpolation/smoothing spread into guarded region B | 0 |

All numeric comparisons were below the prespecified `2e-5` tolerance.
Outside-edit pixel/normalized-tensor invariants passed on every pair.
Unit tests separately exercise distance equivalence, guard-band failures,
split/content leakage, image-level calibration, receiver-level uncertainty,
false-precision protection, unsafe/block-page archives and decision gating.

## What inspection changed

All input galleries, fixed first examples, largest-effect examples and the
all-condition plot were inspected. The first rendering had overlapping titles;
`pilot.render` regenerated readable figures from stored scores and input hashes
without rerunning the encoder or changing numerical results. Both copies of
`summary.json` have SHA256:

`1038af0a05a4b8542e9bf2986ef74f735bc3c4fa58033bb2a63282effedce4f4`

The sham edits themselves often attract strong local responses. In these
procedural fixtures, the paired local-positive-control interval does not pass
the prespecified criterion. Phase/boundary discontinuities are a live alternative
explanation. Do not turn the strongest remote-response picture into an anomaly
detection result. No construction, margin, threshold or model was retuned after
viewing these outcomes.

`review.json` records the inspection and explicitly does **not** approve the
edits as evidence about industrial defects. The assessor returns
`engineering_only` even if someone marks all procedural-review fields true.

## Failures retained

- `initial-attempt-status.json`: the first run stopped before model inference
  because Windows CRLF checkouts appeared dirty to Linux Git. The source content
  was unchanged (the whitespace-only diff was checked). Matching the external
  repositories' local EOL settings to their existing checkout resolved it.
  New clone commands explicitly use LF.
- `access.json` and `real-data-attempt-status.json`: MVTec's official MyDrive
  archive returned a KLA policy-block page; no archive was extracted. The
  independent DINOv3 Hugging Face endpoint was also blocked. No alternate
  transport or mirror was used to evade the policy.

## Next condition

Supply an approved local public MVTec AD root containing both grid and tile, or
obtain approved access to the official download. Then run the command in
`../README.md`. The scientific gate remains pending until the real paired
stimuli, controls, uncertainty and full inspection record pass. Independent
backbone relevance, mechanism and a remedy at matched recall remain later
requirements even if that pilot is positive.
