# Frozen anomaly-map locality pilot

Implements the bounded experiment in `../research-decision-2026-10-16.md`.
Read `PROTOCOL.md` for the goal tree, the fixed decisions, and what counts as a
match. `protocol.json` is hashed into each run before inference.

## Environment

Python 3.12 + PyTorch/CUDA. Use WSL on this workspace. An isolated environment
was created at `/home/lcawley/.cache/vit-interp/pilot-venv` with
`--system-site-packages` to reuse the existing CUDA-enabled PyTorch.

```bash
python -m pip install -r pilot/requirements.txt
python -m unittest discover -s pilot/tests -v
```

An actual run records installed package versions, source hashes, checkpoint
hash, Git revision, split membership, every input image/mask hash, numeric
tolerances, runtime and peak GPU allocation in its artifacts.

## Assets

The external repositories and checkpoint are intentionally not committed.
Clone the official repositories, check out the revisions in `assets.json`, and
place the checkpoint at `pilot/checkpoints/dinov2_vits14_pretrain.pth`.

```bash
git clone https://github.com/dammsi/AnomalyDINO.git pilot/external/AnomalyDINO
git -C pilot/external/AnomalyDINO checkout b9d1c2648e3a5247437d4d953d907a8f3d994457
git clone https://github.com/facebookresearch/dinov2.git pilot/external/dinov2
git -C pilot/external/dinov2 checkout 7764ea0f912e53c92e82eb78a2a1631e92725fc8
```

The model URL and expected SHA256 are in `assets.json`. In this environment the
official model downloaded successfully; MVTec's official MyDrive download
returned an HTTP-200 HTML security block. Do not treat an HTTP status as proof
of a valid archive, or substitute a mirror to evade that block.

Supply **approved public MVTec AD data**, with the root containing
`grid/train/good`, `grid/test`, `grid/ground_truth` and the equivalent `tile`
directories. Do not use KLA data. A standalone validator/extractor for an
approved local category archive is available:

```bash
python -m pilot.prepare --archive /approved/grid.tar.xz --output /tmp/grid-validated
```

It rejects block pages, wrong formats, traversal paths and archive links.
The extractor uses a fresh destination; assemble the approved category
directories into the common MVTec root before running. Dataset license:
CC BY-NC-SA 4.0, copyright MVTec Software GmbH; preserve attribution and terms
when sharing images/derivatives.

## Reproducible engineering demonstration

```bash
python -m pilot.generate_demo --output pilot/data/procedural
python -m pilot.run --data-root pilot/data/procedural --kind procedural --output pilot/runs/procedural-v1
python -m pilot.assess --run pilot/runs/procedural-v1
```

The explicit procedural marker cannot be labeled `mvtec`. This exercises the
real pretrained DINOv2 encoder and the unmodified released AnomalyDINO scoring
path on generated fixtures. It establishes software fidelity, **not the
presence or absence of the proposed failure on public industrial images**.

From Windows Git Bash, prevent MSYS path rewriting:

```bash
MSYS_NO_PATHCONV=1 wsl -d Ubuntu --cd /mnt/c/Users/lcawley/projects/vit-interp \
  -- /home/lcawley/.cache/vit-interp/pilot-venv/bin/python -m pilot.run \
  --data-root pilot/data/procedural --kind procedural --output pilot/runs/procedural-v1
```

## Real pilot, once the public data are available

```bash
python -m pilot.run --data-root /approved/mvtec --kind mvtec --output pilot/runs/mvtec-v1
```

The output directory must be new; no run can overwrite earlier evidence.
There is no automatic grid search, no automatic threshold retuning and no
automatic change of backbone. A missing dataset yields a `blocked` status.

Outputs include:

- `manifest.json`: the immutable study/input/code specification.
- `pairs.jsonl`: every paired observation, including controls and exclusions
  documented separately in the prescore manifest.
- `raw/*.npz`: patch maps and edit/remote/GT patch masks for every pair.
- `calibration.json`: the unedited calibration observations and thresholds.
- `summary.json`: all conditions, uncertainty and released-pipeline agreement.
- `*-all-pairs.png`: every receiver and all paired image edits.
- `*-first.png`, `*-largest.png`: fixed and explicitly score-selected diagnostics.
- `effects.png`: all raw endpoint estimates using a common scale.
- `review-template.json`: review fields bound to this manifest's hash.

Fill a copy of the review template only after inspecting the artifacts:

```bash
python -m pilot.assess --run pilot/runs/mvtec-v1 --review /path/to/review.json
```

The assessor stays `pending_visual_and_provenance_review` until every review
requirement is recorded. Small or ambiguous effects stay `inconclusive`; no
outcome is automatically called a publication success. Model/source changes or
protocol revisions require a new config and run, preserved alongside the old one.
