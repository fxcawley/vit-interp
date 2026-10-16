# 08. Potential collaborators

Principle agreed on 2026-10-13: do Phase 0 and the first Tier 1 results before contacting anyone outside KLA, so that the approach is "here is a preregistered result, would you like to be involved", not "here is an idea".

Verification: [verified 2026-10-13] = affiliation and relevance confirmed from a current web page today. [to verify] = from memory; confirm before contacting.

## Ann Arbor / University of Michigan

| Person | Unit | Why relevant to D | What to ask | Status |
|---|---|---|---|---|
| Daniel S. Weller | KLA, AI and Modeling Center of Excellence, Ann Arbor (since Dec 2020); formerly UVA faculty; UMich postdoc with Fessler | Already your collaborator; signal processing and computational imaging background suits the spectral framing; internal champion for the patent/internal-conference track | Review of `04-theory-notes.md`; KLA process for external publication of non-KLA-data work | [verified 2026-10-13] |
| Qing Qu | UMich EECS (ECE), DeepThink Lab | Low-dimensional structure in deep representations; NeurIPS'26 tutorial on principles of deep representation learning via low-dimensional models (with Yi Ma, Peng Wang); COLM'26 work on spectral subspace steering. The filling-order and width results are low-dimensional-structure results | Feedback on the spectral filling argument; possible student interest; seminar slot | [verified 2026-10-13] |
| Yixin Wang | UMich Statistics | Causal representation learning and identifiability; Blei PhD, Jordan postdoc | The identifiability-up-to-harmonic-phase statement and its relation to Nelson et al.'s structural/statistical distinction | [verified 2026-10-13] |
| Honglak Lee | UMich CSE (also LG AI) | Co-author of PlaNet (latent dynamics for planning from pixels); long record in world models and representation learning | World-model framing; Tier 3 planning readouts; student collaboration | [verified publication page 2026-10-13; current activity to verify] |
| Laura Balzano | UMich EECS | Subspace learning, low-rank models; principal-angle machinery used in cross-seed alignment | Methodological review of the alignment statistics | [to verify current fit] |
| Wei Hu | UMich CSE | Deep learning theory, feature learning dynamics; relevant to the Littwin-style training-dynamics question (which harmonics are learned first in time) | Theory discussion on learning order | [to verify current fit] |
| Andrew Owens, Stella Yu, Justin Johnson | UMich CSE / EECS vision | Self-supervised and video representation learning; Kubric-style rendering experience likely in their groups | Tier 2 rendering help; student collaboration on the V-JEPA 2 analysis | [to verify current fit] |
| Jeffrey Fessler | UMich EECS | Weller's postdoc advisor; imaging | Introduction network only | [verified 2026-10-13] |

Also worth a look: the UMich faculty listing shows a Neurosurgery / Computational Medicine affiliate working on computer vision and self-supervised learning (name not captured by the search; check cse.engin.umich.edu/people/faculty).

## Outside Ann Arbor, by role in the project

| Person / group | Why | What to ask | Status |
|---|---|---|---|
| Randall Balestriero (Brown, GalilAI; visiting FAIR) | LeJEPA, LeWM, Klindt theorem, and a co-author of Joseph et al.; spoke at KLA last year. The single most important contact: he is on both sides of the question | After Tier 1: share the preregistration and spectra; ask whether they have a circular-latent result in progress; propose coordination or co-authorship rather than a race | [verified 2026-10-13] |
| David Klindt (Brown, GalilAI, co-advises with Balestriero) | First author of the identifiability theorem; Appendix F is the theory hook | Same as above; specifically the degenerate-eigenspace case | [verified via Brown alphaXiv page 2026-10-13] |
| Lucas Maes (LeWM first author) | Owns the LeWM code and the stable-worldmodel-v1 suite | Practical questions for P7; Push-T checkpoint | [verified 2026-10-13] |
| Sonia Joseph (Mila / McGill; Prisma), Quentin Garrido (FAIR), Thomas Fel, Blake Richards (Mila) | Authors of the interpretability result we reinterpret; Prisma hosts video-model hooks; Richards brings the neuroscience ring-code framing; Fel is the Archetypal-SAE author relevant to Direction B | After P6 is in hand: offer the pooled/harmonic re-analysis; co-authorship is the respectful route if we use their pipeline | [author list verified 2026-10-13; current affiliations to verify] |
| Leo Nicollier, Gabriele Facciolo (Centre Borelli, ENS Paris-Saclay) | Beyond Gaussian Worlds; SPHERE-JEPA; the torus environment | Code and the exact torus setup for P4; whether they measured anything beyond R^2 | [verified 2026-10-13] |
| Will Dorrell, Rylan Schaeffer, Ila Fiete (grid-cell / periodic-code theory) | Harmonic codes from constraints in trained networks | Sanity check on the constraints-set-harmonics argument; possible citation of unpublished follow-ups | [to verify affiliations] |
| MIV / HOW workshop organizers (CVPR interpretability workshops) | Venue; 2025 organizers included the test-time-register authors | Submission; not a research contact | [to verify 2027 organizers] |

## Sequencing

1. Now: Weller (internal), for the KLA publication process and the spectral-framing review.
2. After Gate 0: one UMich conversation (Qu or Wang) for a theory sanity check; low cost, high value if the filling-order derivation has a hole.
3. After Gate 1 with positive P1-P3: Balestriero and Klindt, with the preregistration and figures; this is the point where overlap turns into collaboration.
4. After P6 preliminary: Joseph and Garrido.
5. Never: anyone needing KLA data, since none will be used.
