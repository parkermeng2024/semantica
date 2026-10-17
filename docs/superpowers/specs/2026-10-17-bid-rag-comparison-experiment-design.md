# Spec: Bid RAG comparison — Vector vs Graph-RAG vs Hyper-RAG

**Date:** 2026-10-17
**Status:** Proposed — awaiting approval

## Objective

Decide which retrieval architecture the bid assistant should use for
**requirement → evidence** grounding: plain vector RAG, Graph-RAG, or
Hyper-RAG (hypergraph). The experiment is designed to be decidable in one
pass, with pre-registered rules for choosing between them (and for a routed
hybrid if the evidence supports one).

The scenario is the existing bid workflow: a requirement from the tender
document (e.g. `kind=mandatory`, `required_evidence_types=[capability]`) must
be mapped to evidence in our capability corpus and produce a draft section with
`evidence_ids` and `evidence_status` (`grounded` / `ungrounded`).

Because that workflow already emits evidence references, **retrieval quality is
directly measurable** — the evaluation does not have to rely only on
subjective grading of final answers.

## Hypotheses

| ID | Hypothesis | Why |
|---|---|---|
| H1 | Hypergraph wins on **n-ary constraints** | One requirement binding several entities (qualification type + grade + validity window + attached certificate) loses its "same requirement" semantics when split into pairwise triples |
| H2 | Graph-RAG wins on **global coverage** questions | "List every mandatory item and the gap" needs cross-document aggregation, which community summaries are built for |
| H3 | Vector RAG is the best value on **single-hop facts** | Differences should concentrate in multi-hop / n-ary groups |
| H0 | No significant difference | If it holds, always pick the cheapest arm |

Differences are only expected in the hard buckets, so **every metric is
reported per bucket** — never as a single overall average.

## Dataset

### Corpora

- **Tender side:** 30–50 publicly available government / SOE tender documents
  (PDF plus attachment spreadsheets).
- **Capability side:** qualifications, certificates, case studies, product
  documentation, financial statements — 500–2 000 documents, 20k–100k chunks.

### Annotation unit

One sample per `requirement`:

- `gold_evidence`: document ID + page + span (must resolve to source text).
- `type`:
  - `single_hop` — one passage suffices ("do we hold ISO 9001?").
  - `multi_hop` — needs several passages combined ("≥3 comparable projects in
    the last three years, each ≥¥5M").
  - `n_ary` — one fact bound to several entities ("ISO 9001 **and** valid on
    the bid date **and** scanned certificate supplied").
  - `global` — needs full-corpus coverage ("list all mandatory items").
- `difficulty` (1–3) and `category` (logistics / capability / finance /
  compliance / …).

### Size and split

- 200–500 requirements; hard buckets (`multi_hop` + `n_ary` + `global`) must be
  **at least 50%**, otherwise the comparison cannot resolve the difference.
- Two annotators, Cohen's κ ≥ 0.8, third annotator adjudicates.
- Split **by tender document** (dev 60% / test 40%) to avoid template leakage;
  if one procurement platform dominates, group again by template family.

## Arms

| Arm | Structure | Implementation |
|---|---|---|
| **A. Vector** | Chunk embeddings + rerank | chunk 512 / overlap 50, embedder fixed to Ollama `bge-m3` (1024-d), top-k = 20 → rerank → 8 passages |
| **B. Graph** | Entity/relation graph + community summaries | Microsoft GraphRAG or LightRAG; local (entity neighbourhood) and global (community summary) modes |
| **C. Hyper** | Hyperedges + entity/relation vector stores | iMoonLab Hyper-RAG, two sub-arms: `hyper` and `hyper-lite` |

**Controls (must match exactly, or the comparison is void)**

- Same LLM, `temperature=0`, same prompt template, same reranker.
- Same context budget in tokens (e.g. 8k), truncated by score — not by item count.
- Same embedding model and dimension.
- Index / retrieve / generate time and cost measured separately.

## Metrics

### Retrieval

- `Recall@k`, `MRR`, `nDCG@10`.
- **Evidence Coverage** — share of cited `evidence_ids` that hit gold evidence;
  this is the metric that matches the audit contract.
- `Context Precision` — share of irrelevant passages in the prompt (cost/noise).

### Generation

- **Compliance F1** — `grounded` / `ungrounded` agreement with human adjudication.
- **Citation Precision / Recall** — is each cited span actually supporting the claim?
- **Hallucination rate** — unsupported claims / total claims.
- Human 1–5 scores: completeness, accuracy, reviewability (blind, arm names masked).

### Cost and efficiency

- Indexing: LLM calls, tokens, wall time, storage.
- Retrieval: p50 / p95 latency (local retrieval + prompt assembly only).
- Cost per requirement (¥).

### Stratification

Report `single_hop` / `multi_hop` / `n_ary` / `global` separately, each with
confidence intervals.

## Procedure

1. **M0 — Data.** Clean, deduplicate, annotate gold, freeze a snapshot (hash it).
2. **M1 — Index.** Build all arms, record cost, archive each index artifact.
3. **M2 — Run.** Batch inference on the test split at `temperature=0`,
   three repetitions to report variance.
4. **M3 — Statistics.** Paired bootstrap (10k) 95% CIs; McNemar for binary
   outcomes; Wilcoxon signed-rank for continuous metrics; Holm–Bonferroni for
   multiple comparisons.
5. **M4 — Error analysis.** 50 sampled cases per arm and bucket, attributed to
   `extraction / retrieval / generation / annotation` error.
6. **M5 — Decision.** Cost–effect Pareto curve plus a recommendation.

## Pre-registered decision rules

Written before seeing results, to prevent metric shopping:

- If **C (`hyper`)** beats A and B on `n_ary` nDCG@10 by ≥5 points
  (p < 0.05) and p95 latency ≤ 2× A → adopt Hyper-RAG, preferring `hyper-lite`.
- If **B** leads on `global` → use Graph-RAG community summaries for the
  global view.
- If **A** is non-inferior on `single_hop` (difference CI contains 0) → keep
  vector retrieval for single-hop requirements.
- Cost ceiling: any arm whose per-requirement cost exceeds 3× A cannot be the
  default path.

The expected landing zone is a **routed hybrid**: single-hop → vectors,
n-ary → hypergraph, global → community summaries.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Template leakage inflates scores | Split by tender document and by template family |
| Unequal context budgets | Fixed token budget, score-based truncation |
| LLM-as-judge bias | Two judge models plus 20% human spot-check |
| Hypergraph-DB deployment / ecosystem risk | Containerise and validate offline in the PoC first |
| Confidential tender material | Fully local pipeline (Ollama + local embedder); de-identify before it enters the experiment store |
| Annotation subjectivity | Two annotators, adjudication, published annotation guide |

## Deliverables

- Frozen test-snapshot plus annotation guide.
- Index artifacts and cost sheet per arm.
- Stratified metric tables (4 buckets × 3–4 arms).
- Error-analysis report.
- Recommendation plus Pareto curve.
- Reproducible scripts: fixed seeds, locked dependencies, recorded commit and
  versioned prompts.

**Rough effort:** M0 3–5 days, M1 5–8 days (the hypergraph arm needs upstream
integration), M2 2 days, M3–M4 2–3 days.
