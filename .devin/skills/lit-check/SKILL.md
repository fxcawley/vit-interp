---
name: lit-check
description: Scoop check, primary-source verification and idea-sink guard for a research idea or claim, using arXiv/Scholar/OpenAlex/Brave via curl-backed scripts
argument-hint: "<idea, claim, or arXiv id to check>"
allowed-tools:
  - read
  - grep
  - glob
  - exec
permissions:
  allow:
    - Exec(node)
    - Exec(curl)
---

Run a literature check on what the user named. The purpose is to stop work on idea-sinks before they consume a week: lines of work with an unverifiable premise, no kill criterion, a question the field already settled or already knows is hard, or a tooling rabbit hole. Produce the report in section 5 and nothing else unless asked.

## 1. Tools and their quirks (read first)

Scripts live in `.devin/skills/lit-check/scripts/` and use `curl` for transport because the `fetch` MCP server fails behind the corporate proxy. Run them with `node <script> ...` from that directory. Pace Scholar and OpenAlex (one call per ~10 s); never fire searches in parallel.

| Script | Use | Notes |
|---|---|---|
| `arx.js "<query>" [max] [submittedDate]` | arXiv API search. Fields: `ti:`, `abs:`, `au:`, `all:`; quote phrases; combine with AND/OR. | Prints totalResults; weak ranking, so run several phrasings. |
| `abs.js <id> [<id>...]` | Abstract, date, comment for arXiv IDs. | Use for verification marks. |
| `oa.js "<query>" [n] [fromYear]` | OpenAlex works search with citation counts and venues. | Loose relevance; good for non-arXiv venues and counts. |
| `gs.js "<query>" [ylo]` | Google Scholar results with snippets. | Breaks after ~10 fast calls; keep it for the decisive queries. |
| `txt.js <url> [start] [len] [regex]` | Page text; with a regex, prints matching lines with context. Works on `arxiv.org/html/<id>`. | GitHub raw code files are quarantined by Zscaler; use `git clone` or the GitHub API tree instead. |
| `pdftxt.js <file.pdf> [regex] [ctx]` | Crude text from a downloaded PDF (for papers with no HTML version). Download with `curl -sL -o f.pdf https://arxiv.org/pdf/<id>`. | Ligatures and math are mangled; good enough for grep. |
| `bib.js <id>...` | BibTeX from arXiv. | Append to the project's `.bib`. |

The `brave-search` MCP tool works for general web search when it is up; the `fetch` MCP tool does not. DuckDuckGo lite and no-JS Bing are unreliable from this network. Semantic Scholar's API rate-limits without a key.

## 2. Scoop check (do this before anything is built)

1. Write the idea as one sentence and extract 3-6 search terms plus synonyms (e.g., register/high-norm/outlier/artifact tokens).
2. arXiv API: at least four phrasings over `abs:` and `all:`, sorted by `submittedDate`, plus one sorted by relevance. Record totalResults for each.
3. Scholar: two decisive queries with quoted phrases and a year floor. OpenAlex: one query for counts and venues. Brave: one or two queries for blog posts, code and workshop pages.
4. For the five nearest neighbors, read the abstract (`abs.js`) and, for the two nearest, grep the full text (`txt.js` with a regex or `pdftxt.js`) for the specific measurement or claim at issue. "They mention X" is not the same as "they measured X".
5. State the verdict in one of three forms: clean (nothing within the claim), adjacent (list what each neighbor did and did not do), or taken (name the paper and the date, and say what remains).

## 3. Verification of load-bearing facts

For every fact the plan depends on (a number, a model size, a dataset property, what a baseline does), fetch the primary source and mark it `[verified YYYY-MM-DD]` with where it was read (abstract, HTML, PDF). Facts from memory stay `[to verify]` until read. Prefer the full text over the abstract when the fact is a detail (layer index, split size, which checkpoint). Record discrepancies between secondary summaries and the source explicitly.

## 4. Idea-sink guard

Answer each in writing; a "no" on any of the first four is a stop or a redesign.

1. Belief: is there a belief practitioners rely on (a default in released code, an assumption stated in method papers)? Name who holds it.
2. One question: is there exactly one falsifiable question with a number attached? If the plan has six predictions and eight arms, pick the headline before running.
3. Kill criterion: is there an observation, reachable in the first experiment, that ends the idea? If nothing can kill it, it is not a question.
4. Control: is there a control that could overturn the result (random init, shuffle, matched null, same weights plus one intervention), not just a method-vs-method comparison?
5. Mechanism: is there a causal readout with a dose-response knob, or only a benchmark delta?
6. Known dead end: search "<approach> negative results", "<approach> does not work", "<approach> replication", "sanity check <method>" on arXiv and Brave. If the field already tried it, say what changed.
7. Known hard: is the sub-problem an open problem in its own field (search "<sub-problem> open problem")? If so, the idea depends on solving it, which is a sink unless that is the project.
8. Who is positioned to scoop: name the groups that own the tools, the theory or the data, and whether contacting them is better than racing.
9. Time-box: write the maximum hours for the first experiment and what "no result by then" means.
10. Consequence: say in one sentence what someone would do differently if the answer came out either way. If nothing, stop.

## 5. Report format

```
# Lit check: <idea>  (date)

## Verdict: clean | adjacent | taken
<two sentences>

## Nearest neighbors
| Paper (arXiv, date) | Did | Did not | Read (abs/HTML/PDF) |

## Load-bearing facts
| Fact | Source | Status [verified date / to verify] | Note |

## Idea-sink guard
1. Belief: ...
2. One question: ...
3. Kill criterion: ...
4. Control: ...
5. Mechanism: ...
6. Known dead end: <queries run, result>
7. Known hard: ...
8. Scoop exposure: ...
9. Time-box: ...
10. Consequence: ...

## Recommendation
go | redesign (what) | stop (why)
```

Keep the report under one page unless the user asks for the full trail. Save it next to the project's planning documents if the user is keeping a research directory.
