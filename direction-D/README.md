# Direction D: what a JEPA does with a circle

Working title candidates, abstract draft and figure list: `09-writing-plan.md`.

Status: formalization, 2026-10-13. Nothing has been run. The preregistration file is a draft until Phase 0 closes (see `06-experiment-plan.md`).

## One-paragraph pitch

Video world models represent motion direction as a high-dimensional ring (Joseph et al., Feb 2026), which has been read as evidence that they do not hold compact physical state. JEPA identifiability theory (Klindt, LeCun, Balestriero, May 2026; Nicollier et al., Sep 2026) says linear recovery of a latent holds only when the regularizer's target matches the latent's geometry, and calls the mismatched case "distorted". Classical slow-feature theory (Wiskott and Sejnowski 2002; Franzius, Sprekeler, Wiskott 2007) predicts exactly what the mismatched, "distorted", "distributed" code should look like for a circular latent: the eigenfunctions of the latent's transition operator, which are Fourier harmonics ordered by slowness, with the number of harmonics set by how many decorrelated dimensions the anti-collapse regularizer demands. Nobody has measured the harmonic spectrum of a JEPA's code, shown which design knobs set its width, or used it to reread the distributed-code result or the released world models. We do those four things with preregistered dose-response predictions, a random-init null and a time-shuffle control, in toy worlds, in the exact environments the theory papers used, and in released V-JEPA 2 and LeWorldModel checkpoints.

## The one question

Is the geometry of a JEPA's code for a circular latent (its harmonic spectrum and width) set by the latent's dynamics and the regularizer's rank demand, in the order and amount spectral theory predicts, or by something idiosyncratic to model and data?

## Files

| File | What it is |
|---|---|
| `00-preregistration.md` | Frozen predictions, kill criteria, decision rules. Draft until Phase 0 closes. |
| `01-research-question.md` | Belief attacked, the question, hypotheses H1-H4, numbered predictions P1-P7, kill criteria K1-K4. |
| `02-lit-review.md` | Annotated literature with verification marks, grouped by role in the argument. |
| `03-positioning.md` | Contribution statement, novelty relative to each neighbor, anticipated objections, slop check, venue strategy. |
| `04-theory-notes.md` | Derivations behind the predictions: eigenfunctions per latent law, filling order, kurtosis of harmonic mixtures, identifiability up to harmonic phase. |
| `05-methods.md` | Worlds, renderers, recipes, readouts with formulas, controls, sweeps, statistics, software, compute. |
| `06-experiment-plan.md` | Phases with gates, timeline, GPU budget, deliverables per phase. |
| `07-risks-and-open-questions.md` | Scoop exposure, optimization failure modes, measurement confounds, open questions to resolve in Phase 0. |
| `08-collaborators.md` | Verified names in Ann Arbor and outside, why each, what to ask, when to contact. |
| `09-writing-plan.md` | Titles, abstract draft, figure and table list, main-track to workshop strip-down map, checklists. |
| `references.md` | Every citation with arXiv ID and verification status. `references.bib` is generated from arXiv. |

## Conventions

- [verified 2026-10-13] means read in the primary source (arXiv abstract, HTML or PDF) on that date. [to verify] means stated from memory or secondary source.
- Every prediction carries a number and a null. "Beyond noise" means the CI of the paired difference across seeds excludes zero and the point estimate exceeds twice the seed SD.
- Kill criteria are written before any run. If one triggers, the plan says what the paper becomes.
- The four tenets, applied here: belief = "distributed code implies no compact latent" and "non-linear-recoverable implies distorted"; question = the one above; controls = random-init encoder, time-shuffled pairs, matched target, EMA recipe at matched width; mechanism = spectral filling order with two dose-response knobs (step size, regularizer weight) and one capacity knob (width).

## Honest scope note

The theory plank of the original idea (identifiability on circular latents) was taken in September 2026 by Nicollier et al. (arXiv:2609.21656), who extend Klindt et al. to Riemannian latent manifolds and run pixel experiments on a torus world. This direction is therefore not "test a 2026 theorem"; it is a mechanistic and interpretive paper that characterizes the regime the theorems leave as "distorted", with predictions those papers do not make and measurements they do not take. See `03-positioning.md`, section "What is left after Klindt and Nicollier".
