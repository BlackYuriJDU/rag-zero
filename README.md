# RAG Zero

**Grounded question answering over Brazilian public procurement (PNCP) — a production-style RAG system with real evaluations.**

> Status: 🚧 **P0 — foundations under construction.** The results table below will be filled with measured numbers as each phase lands. No screenshots of happy paths: every claim in this README will have a script that reproduces it.

## Why

Brazil publishes every public purchase on [PNCP](https://pncp.gov.br) — hundreds of thousands of notices a year, in messy PDFs and semi-structured JSON. The portal's search works for exact keywords, not for real questions like *"quais oportunidades abertas de merenda escolar em PE?"*.

RAG Zero aims to answer questions over that corpus with **verifiable citations** — or to **refuse** honestly when the answer is not in the corpus. Just as important: it treats **evaluation as a first-class component**, not an afterthought. Retrieval quality and answer quality are measured separately, every change is scored against a golden set, and a regression gate runs in CI.

## What it will do (roadmap P0–P5)

- [x] **P0** — Repo foundation, PNCP API client, smoke tests, golden-set template
- [ ] **P0** — Ingest ~5k notices (PE, Pregão Eletrônico), naive baseline measured
- [ ] **P1** — Hybrid retrieval: Portuguese FTS (BM25) + dense embeddings, fused with RRF; structural chunking; ablation #1
- [ ] **P2** — Grounded generation: structured answers with span-verified citations; refusal path; faithfulness eval
- [ ] **P3** — Reranker: API baseline, then a fine-tuned PT-BR cross-encoder (ONNX int8); ablation #2
- [ ] **P4** — FastAPI service + minimal UI with a retrieval viewer, MCP server, deploy, cost & latency tracking
- [ ] **P5** — Prompt-injection tests, bilingual docs, v1.0 release

## Results (to be filled — this table is the point of the project)

| Configuration | recall@10 | MRR@10 | faithfulness | citation precision | $/query | p95 latency |
|---|---|---|---|---|---|---|
| Baseline (dense-only, naive chunks) | _pending_ | | | | | |
| + hybrid (BM25 + dense, RRF) | _pending_ | | | | | |
| + structural chunking | _pending_ | | | | | |
| + reranking | _pending_ | | | | | |

## Architecture

```
PNCP API (public, no auth)
   │  ingest (paginated, polite, retry-aware)
   ▼
Postgres ── contratacoes (structured metadata)
   │
   ├── edital PDFs ──▶ text extraction ──▶ structural chunks ──▶ embeddings
   ▼
pgvector (dense) + tsvector (Portuguese FTS)
   │
   ▼
hybrid retrieve (BM25 + dense, RRF) ──▶ rerank ──▶ grounded answer
                                                     ├── citation verification (spans)
                                                     └── refusal when evidence is missing

evals/ golden set ──▶ metrics ──▶ CI regression gate
```

## Quickstart

```bash
uv sync
uv run rag-zero pncp smoke --uf PE --dias 7
uv run rag-zero pncp dump  --uf PE --dias 30 --max-paginas 20
```

The `smoke` command queries the live PNCP API (read-only public data) and prints a summary. Raw responses are saved under `data/raw/` (git-ignored).

Local database (optional in P0 — used from indexing onward):

```bash
docker compose up -d        # Postgres 16 + pgvector, schema applied automatically
```

## Evaluations

The harness lives in `evals/`. The golden set follows three rules:

1. Questions are written the way a real buyer/supplier/journalist would ask.
2. Every expected answer is verified **by hand** against the indexed corpus (no LLM-graded ground truth without human review).
3. ~10–15% of items are **unanswerable** by design — a correct refusal is a feature, not a failure.

See [`evals/golden_set.template.yaml`](evals/golden_set.template.yaml) for the schema and curation rules.

## Development

```bash
uv sync --dev
uv run ruff check .
uv run pytest
```

CI runs lint + tests on every push (see `.github/workflows/ci.yml`).

## Limitations (honest section)

- Scanned/image-only PDFs are out of scope for v1 (no OCR) — detected and reported, not silently ignored.
- PNCP API quirks are real: `204` means "no results", `pagina` starts at 1, page size caps at 50 for contracting endpoints. The client handles and documents each of them.
- This is an independent open-source project; it is not affiliated with the Brazilian government.

## License

MIT — see [LICENSE](LICENSE).

Author: Arthur Araújo ([@BlackYuriJDU](https://github.com/BlackYuriJDU)).
