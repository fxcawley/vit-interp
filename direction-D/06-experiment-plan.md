# 06. Experiment plan with gates

Clock: formalization 2026-10-13. Main-track target: ICML 2027 (deadline to verify; assumed late January 2027, about 15 weeks). Workshop fallback: CVPR 2027 interpretability workshops or ICLR 2027 workshops (February-March 2027, to verify). Preprint as early as Phase 2 permits.

## Phase 0: reading, tooling, calibration (week 1-2)

Goals
- Resolve the open questions in `07-risks-and-open-questions.md`, section 4, that gate design choices (LeWM defaults; whether Nicollier's code is out; licenses; Joseph's stimuli).
- Run Tier 0: replicate Klindt's 2-D Gaussian recovery and Nicollier's torus asymmetry (R^2 0.986 / 0.493) with a Gaussian target. If we cannot reproduce the asymmetry qualitatively, stop and find out why before Tier 1.
- Numerically check the kurtosis approximation (-1.5 / K) for harmonic mixtures with resonances.
- Fix the optimization-success thresholds and freeze `00-preregistration.md` with a commit hash.

Gate 0: Tier 0 reproduces; preregistration frozen. Deliverable: a one-page calibration note.

## Phase 1: Tier 1 heading world, basis and knobs (week 2-5)

Runs: P1 (basis) with random-init and time-shuffle nulls; P2 (s sweep) on R1 and R3; P3 and P3b (lambda x d on R1 and R2; d on R3).

Gate 1 (decides the paper's shape, see decision rules in `01-research-question.md`):
- P1 holds and at least one of P2 / P3 holds: continue to Phase 2 with main-track framing.
- P1 holds, P2 and P3 fail: continue with workshop framing; skip P5.
- P1 fails: stop; write the negative replication.

Deliverables: spectra figure (p_k vs k per recipe with nulls), dose-response figure (K_5 vs s; K_5 vs lambda and d), kurtosis figure (SIGReg vs VICReg).

## Phase 2: combined and torus worlds, Tier 1 then Tier 2 (week 5-8)

Runs: P5 filling order (W4); P4 torus distortion (W5, then reacher with R1 vs R4).

Gate 2: P5 Kendall tau >= 0.8 or a clear, reportable deviation; P4 cross-harmonic share measured either way.

Deliverables: filling-order figure (first-appearance rank vs theoretical eigenvalue rank), torus decomposition figure (unexplained variance split into cross-harmonics vs other). Post the arXiv preprint at the end of Phase 2 if Gate 1 passed, with Tier 3 marked as in progress or included if ready.

## Phase 3: released models, in parallel from week 4 (week 4-10)

- V-JEPA 2: reproduce Joseph et al.'s direction probing at layer 8 with their 8-direction stimuli (sanity), then with continuous-direction stimuli; harmonic decomposition; pooling; speed regress-out; orthogonal-probe counts before and after. P6.
- LeWorldModel: load the released Push-T checkpoint; probe T-block orientation; harmonic spectrum and K_5; retrain across lambda with the official code (each run hours). P7.

Gate 3: P6 or P7 holds: lead the paper with it. Neither holds: report as K3 and keep the toy results as the paper.

## Phase 4: writing (week 10-14)

Main-track draft per `09-writing-plan.md`; strip-down version prepared in parallel. Internal review against the objections list in `03-positioning.md`. Code and data release prepared with the submission.

## Compute budget summary

Under 150 GPU-hours total (see `05-methods.md`, section 5). One 24-48 GB GPU suffices; Tier 1 runs can be packed several per GPU. CPU time for Kubric rendering is the only non-GPU cost of note (hours for a few thousand short clips).

## What to do if scooped mid-way

If a paper appears that measures harmonic spectra of JEPA codes with regularizer knobs: pivot the headline to whichever of P4-P7 it does not cover (most likely the released-model reinterpretation, P6/P7, or the filling order, P5), cite it, and post immediately. The preregistration file timestamps our predictions.
