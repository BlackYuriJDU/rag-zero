# RAG Zero

[![CI](https://github.com/BlackYuriJDU/rag-zero/actions/workflows/ci.yml/badge.svg)](https://github.com/BlackYuriJDU/rag-zero/actions/workflows/ci.yml)

🇬🇧 English · 🇧🇷 [Português](README.pt-BR.md)

**Grounded question answering over Brazilian public procurement (PNCP) — with citations, honest refusal, and measured retrieval.**

> Status: **alpha (0.1.0).** P0/P1 are done. The real corpus has 2,714 procurements
> (PE, electronic auction, 90 days) and 2,721 chunks. Every number below comes from
> `rag-zero evals run` and is stored in `evals/results/latest.json`.
> The documentation is mostly in Portuguese; this page is the English summary.

## Why

Brazil publishes every public purchase on [PNCP](https://pncp.gov.br) — hundreds of
thousands of notices a year. The portal's search matches exact words; real questions
("which school-meal opportunities are open in PE?") do not.

RAG Zero answers those questions **with verifiable citations** — or **refuses** when the
corpus does not contain the answer. Evaluation is a first-class component: retrieval and
answers are measured separately, and which mode to use is decided by numbers, not trends.

## Measured results

Golden set with 72 items: 24 structural (lookup by ID), 30 verbatim (the question repeats
the notice text), 12 curated paraphrases (synonyms, no repeated text) and 6 unanswerable.
`k=10`, CPU, local embedding model.

| mode | recall@10 | MRR@10 | citation precision | structural | refusal | p95 |
|---|---|---|---|---|---|---|
| **bm25** | **0.905** | **0.758** | **0.690** | 1.000 | 0.667 | **68 ms** |
| dense | 0.405 | 0.262 | 0.190 | 1.000 | 0.667 | 901 ms |
| hybrid (RRF) | 0.762 | 0.487 | 0.381 | 1.000 | 0.667 | 778 ms |

p95 comes from a single run and varies about 2× between runs on the author's machine
(68–152 ms for BM25); the order of magnitude is stable: BM25 is ~10× faster.

What the numbers say:

1. **BM25 dominates on this corpus**, so it is the CLI default.
2. **Dense retrieval does not pay for itself here.** The corpus is full of near-identical
   objects ("price registration for future acquisition of…") and the embedding model does
   not separate municipality/agency as sharply as BM25 does. It is a model/corpus limit,
   not an index bug.
3. **RRF can hurt.** Fusing a good ranker with a weak one dilutes the good one.
4. **Refusal went from 3/6 to 4/6** with a temporal-scope rule. The two remaining failures
   are compositional ("bullet trains", "nuclear submarines"): each term exists in the
   corpus, the combination does not.
5. **Dense latency is a real CPU cost** (~0.6 s per query warm, ~46 s cold start).

> Honesty about the golden set: the 12 paraphrases name the municipality, which gives BM25
> a strong lexical anchor. A future set needs questions without proper nouns to isolate
> real synonymy.

## Install

```bash
git clone https://github.com/BlackYuriJDU/rag-zero && cd rag-zero
uv sync                  # default install: BM25 only, lightweight
uv sync --extra dense    # adds fastembed for dense / hybrid retrieval
```

Dense and hybrid modes are optional on purpose. Without the extra, asking for them prints
how to install it instead of failing with a traceback.

## Quickstart

```bash
# 1. Ingest 90 days of electronic auctions in PE (idempotent)
uv run rag-zero ingest --uf PE --dias 90 --modalidade 6

# 2. Generate embeddings — optional, needs the dense extra (local, CPU)
uv run rag-zero index

# 3. Ask (questions in Portuguese)
uv run rag-zero ask "Quais contratações de merenda escolar em Recife?"

# 4. Measure
uv run rag-zero evals build
uv run rag-zero evals run --modes bm25,dense,hybrid --k 10
```

`ask` ends with `confidence=… · N ms · mode=…` and prints `RECUSA: reason` when the
evidence is insufficient. `search` shows the ranking without generating an answer.

## Architecture

```
PNCP API (public, no key)
   │  paginated ingest, throttled, with retry
   ▼
SQLite ── contratacoes (structured metadata)
   │
   ├── notice record ──▶ chunks (~180 words, 25 overlap)
   │                        ├── FTS5 (BM25, Portuguese stemming)
   │                        └── 384-d embeddings (float32 BLOB, optional)
   ▼
retrieval (BM25 | dense | RRF) ──▶ extractive answer
                                   ├── citations (PNCP ID + link)
                                   └── refusal by IDF coverage / temporal scope

evals/ golden set ──▶ metrics ──▶ CI
```

No Docker, no Postgres, no external service. The decision is recorded in
[`docs/adr/0001-sqlite-first.md`](docs/adr/0001-sqlite-first.md), with objective migration
triggers (>100k chunks, concurrent writes, or dense p95 >200 ms).

## Roadmap

- [x] **P0** — PNCP client (quirks documented), idempotent ingestion by fingerprint
- [x] **P1** — chunking with correct Portuguese stemming, FTS5/BM25, local embeddings, RRF
- [x] **P2 (partial)** — extractive answers with citations, refusal by evidence and temporal
      scope, stratified golden set, metrics with no LLM judge
- [ ] **P2 (rest)** — tender PDFs, extraction/OCR, span-level citations
- [ ] **P3** — Portuguese cross-encoder reranker
- [ ] **P4** — FastAPI + inspection UI
- [ ] **P5** — prompt-injection tests, v1.0 release

## Limitations

- **No tender PDFs.** The indexed document is the notice record (object, agency, values,
  dates). Attachments and OCR are P2.
- **Compositional refusal is unsolved.** Terms that exist alone but never together pass
  the IDF coverage check. A co-occurrence verifier on the top-k is the next experiment.
- **Small golden set.** 72 items give direction, not statistical significance.
- **Dense on CPU** is slow; BM25 currently wins anyway.
- **Corpus scope.** PE, electronic auction, 90 days. The system does **not** filter by
  state or modality: a question about another state may retrieve PE contracts that share
  vocabulary. Geographic-scope refusal is pending work — today the limit is declared, not
  enforced.
- Independent project; not affiliated with the Brazilian government.

## Contributing

Contributions are welcome — in English or Portuguese. Start with
[CONTRIBUTING.md](CONTRIBUTING.md); issues labelled `good first issue` are a good entry
point. Please read the [Limitations](#limitations) first: several of them are open problems
with a clear definition of done.

## License

MIT — see [LICENSE](LICENSE).

Author: Arthur Araújo ([@BlackYuriJDU](https://github.com/BlackYuriJDU)).
