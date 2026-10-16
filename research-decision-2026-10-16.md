# Autoresearch simulation: the next evidence purchase

Decision date: 2026-10-16. This is a project-selection memo, not a preregistration or an experimental result. No model training, dataset download, or outreach was performed.

## Objective and decision

Produce a reproducible finding that changes how people evaluate, interpret, or deploy a vision model. A useful failure demonstration, an explanation that predicts its occurrence, and a control that prevents misinterpretation are worth more than another catalog of features.

**Allocate the next experimental budget to a local anomaly-map fidelity pilot.** Keep the corrected Direction D assay audit as a secondary option, rather than expanding it into a spectral-learning program. This ranking reflects consequence, access to a native task output, the user's inspection experience, and the cost of learning whether the effect exists. It is not a claim that either project is novel or will succeed.

The workshop version should be the smallest complete scientific claim. A position paper would require evidence for a distinct recommendation; it is not a fallback for experiments that failed to establish anything.

## What the literature currently supports

The existing review is sufficient to reject the original D experiment grid and to choose cheap tests. It is not sufficient to certify a new paper's novelty.

Primary sources read in this selection pass:

| Source | What matters for this decision |
|---|---|
| [Durrett, AI submissions at COLM](https://gregdurrett.github.io/colm2026-blog/ai-papers.html) | Technical competence can hide a weak research question. He also identifies useful released artifacts as a genuine contribution. Narrow scope and ordinary tools are not themselves disqualifying. |
| [AnomalyDINO](https://arxiv.org/html/2405.14529v3) | Explicitly values both local and global information and documents background false positives. Do not claim that its authors assume strictly local features or that background sensitivity is undiscovered. |
| [F2PAD](https://arxiv.org/html/2407.06519) | Already addresses feature-to-pixel localization errors from reduced resolution and mixing of adjacent normal/anomalous pixels. Ordinary boundary leakage is established prior work. |
| [Reimagining Anomalies](https://arxiv.org/html/2402.14469v2) | Already uses image counterfactuals to explain anomaly detectors, including spurious cues and industrial defects. “Counterfactual explanations for AD” is not our novelty claim. |
| [Klotz et al., Evaluating the Interpretability of SAEs with Concept Annotations](https://arxiv.org/html/2606.24716) | Uses paired attribute edits, CLIP/DINOv2 SAEs, concept coalitions, and a perturbation-alignment score. Generic paired-image evaluation of vision SAEs is already occupied. Its image-to-feature response evaluation must be distinguished from a claim about a model's native downstream computation. |
| [Joshi et al., Causality is Key for Interpretability Claims to Generalise](https://arxiv.org/html/2602.16698) | Explicitly argues for matching causal claims to evidence. A position paper merely repeating “we need causal evaluation” would add little. |
| [Detecting Defects that Matter](https://arxiv.org/html/2610.04392) | Reports continuing localization and distribution-shift problems on a hidden-test benchmark; DINOv3 backbones dominate its industrial track. Any proposed deployment recommendation needs an early check on a current backbone. |

The immediately preceding review independently checked Joseph's pooling, Klindt's parameter sweeps, Nicollier's probe setup, and CTWM's spectral/planning measurements. Those corrections remain prerequisites for any D continuation.

## Three candidate themes

| Theme | Decisive result to seek | Main reason it might not merit a paper |
|---|---|---|
| **Where dense-model evidence points** | A controlled local change induces operationally meaningful false alarms at unchanged, distant locations; a specific intervention removes those errors while retaining local detection. | Global context sensitivity, ordinary smoothing, and bad localization are already known. A small score drift or a legacy-backbone-only example is insufficient. |
| **What a dimensionality assay actually measures** | The exact published assay supports an inference contradicted by a validated native-computation test on a released model. | Covariance-sensitive erasure is known; a compact conditional mean or successful steering of a new probe does not establish a compact causal state. |
| **Whether feature explanations predict edits across contexts** | A feature or coalition predicts native task effects under held-out counterfactuals, and an existing evaluation systematically selects the wrong explanation. | Existing causal-abstraction, SAE-evaluation, and paired-image work already covers much of the general agenda. We need a particular consequential failure first. |

Shared insight: feature identity, representation geometry, and functional sufficiency are different objects. Their disagreement is a source of testable questions, not automatically a contribution.

## The selected pilot

**Question:** In a frozen patch-feature anomaly detector for local structural defects, does adding a defect at A create excess false alarms in a predetermined, unchanged region B beyond matched editing controls?

This is a question about localization reliability. Global features may be useful, and context-dependent logical anomalies are outside this first test.

1. Reproduce one released pipeline and record its actual feature tensor, preprocessing, scoring, and postprocessing. Start with an accessible public texture dataset and one documented checkpoint. Use a fixed nominal bank and exact, chunked nearest-neighbor scoring.
2. Construct paired images with a defect insertion at A, a normal-crop insertion at A using the same procedure, and an identity control. Fix B and a guard band before looking at scores. Verify that B's pixels remain identical after preprocessing. Use multiple source images and edit realizations; compare within source image.
3. Measure raw patch scores before interpolation/smoothing, then the production anomaly map. Use thresholds fixed on separate calibration data. The endpoint is excess remote false alarms per image or region, not merely embedding drift.
4. Check an independent category/edit construction and an accessible current register-equipped backbone before treating the effect as a current design problem. A synthetic paste artifact alone is insufficient evidence for deployment claims.
5. Only after a reproducible effect exists, investigate its source. Registers, global-context pathways, feature matching, and postprocessing are competing explanations. Do not presuppose an outlier-token mechanism. A later remedy must be evaluated at matched true-defect recall and against matched sham interventions.

**Budget:** two engineering days and at most four GPU-hours for the initial feasibility/paired-control pass. These are spending caps, not throughput estimates. Data or checkpoint access failure is an implementation blocker, not a scientific null.

**Stop/defer conditions:** an effect explained by smoothing or the edit artifact; no practically meaningful excess beyond controls at adequate precision; effect restricted to obsolete/unrepresentative settings; or inability to establish the endpoint within the budget. A wide interval is inconclusive. Freeze a practical effect margin and power plan before a confirmatory evaluation; do not invent a numerical prediction from the theory.

**Required artifact:** one figure with the paired images, changed region, unchanged region, raw scores, matched controls, and uncertainty over independent scenes; one reproducible command; one paragraph stating the strongest alternative explanation. That artifact decides whether to spend another week.

## Evaluation rule and safeguards

Use Durrett's warning as the relevance gate, then apply separate gates:

- **Consequence:** who changes a decision, and what exactly changes?
- **Validity:** does the measurement discriminate the explanations at issue?
- **Distinctiveness:** what does the closest prior work already show?
- **Feasibility:** can the next observation settle something on the available machine?
- **Transfer:** what evidence separates a general issue from the chosen construction?

Do not combine these into a numerical score that lets elegance compensate for invalidity. Preserve four evidence labels: source fact, mathematical result under explicit assumptions, empirical hypothesis, and research judgment.

Keep one active question and a fixed spending cap. Log all attempted conditions, negative results, and protocol changes. Separate calibration and exploratory work from held-out confirmation. Test mathematical claims against counterexamples before preregistering them. Measure rank as an outcome when studying rank; do not exclude it as “failed optimization.” Report uncertainty at the independent sampling unit.

Use consistency only where the task supplies a justified relation. “Unchanged pixels” does not always imply “unchanged desired output”; physical defects and logical anomalies require different relations. Cammarata's essay is a hypothesis source, and observability is an assumption to test, not a universal explanation.

The Bitter Lesson motivates scalable search and testing. It does not provide a scientific-interest metric. Scale cheap refutations and reproducible experiments while keeping the research objective fixed. Automation should shorten the path to evidence, not multiply unsupported narratives.

## Local research conversations

These are potential fits based on public work, not claims of interest or availability. No messages were sent.

| Person | Verified connection | A focused question to bring with the first artifact |
|---|---|---|
| **Daniel Weller** | Existing relationship supplied by the user. | Which localization error and operating-point tradeoff would actually change an inspection decision, using public data? |
| **Qing Qu, Michigan EECS** | [Current publications](https://qingqu.engin.umich.edu/publications/) include subspace steering and evaluation of representation geometry. | Is the proposed measurement invariant to irrelevant coordinate choices, and does the intervention isolate a mechanism? |
| **Laura Balzano, Michigan EECS** | [Faculty page](https://web.eecs.umich.edu/~girasole/) confirms SPADA, subspace/statistical methods, and current work on low-rank training dynamics. | Which part of a measured rank or probe count is signal geometry versus covariance and estimation? |
| **Yixin Wang, Michigan Statistics** | [Current papers](https://yixinwang.github.io/papers.html) include causal representation learning, interventional state-space models, and identifiability. | What does our intervention identify, and which nuisance paths remain uncontrolled? |
| **Stella Yu, Michigan EECS** | [Official profile](https://eecs.engin.umich.edu/people/yu-stella/) confirms computer/robotic visual perception. | Does the controlled failure survive a realistic visual setting and change a meaningful vision task? |

Correction: [Andrew Owens's own page](https://andrewowens.com/) now lists **Cornell Tech**, so the earlier Ann Arbor shortlist is outdated for him. Itamar Pres and the existing Balestriero connection are useful external paths; do not infer a collaboration from the connection. Approach one well-matched researcher with a reproducible observation and a precise request for criticism.

## Search limits

Brave failed on the first query but worked on the retry and subsequent targeted searches. Fetch MCP failed; accessible primary pages were read with curl-backed readers. Two Google Scholar queries and one OpenAlex query supplemented discovery. Search snippets were not treated as primary evidence.

Four attempted arXiv API queries returned invalid responses or timed out; a fifth chained query consequently did not run. These are failed searches, not zero-result novelty checks. The pilot is not novelty-certified.

Targeted successful search examples: `vision model interpretability counterfactual causal feature interventions benchmark "2026" "SAE"`; `"anomaly localization" ("non-local" OR "counterfactual" OR "context leakage" OR "spurious")`; Scholar `"DINOv2" "anomaly" "context" "localization"` and `"anomaly localization" "counterfactual" "patch"`.

The next literature pass should follow the specific observed failure, if any, into its nearest baselines and causal explanation. Stop expanding the general survey merely to feel better prepared.
