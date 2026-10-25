# ADR-0001 — SQLite + FTS5 + NumPy antes de Postgres/pgvector

- **Status:** aceito
- **Data:** 2026-10-23
- **Contexto:** P0/P1 do RAG Zero, corpus inicial de ~2,7 mil avisos do PNCP (PE, Pregão Eletrônico, 90 dias).

## Contexto

O desenho original previa Postgres 16 + pgvector via Docker. Na máquina de
desenvolvimento não há Docker, Postgres nem WSL — e o CI precisa rodar sem serviço
externo. Ao mesmo tempo, o corpus desta fase cabe folgado em memória:

- 2.721 chunks × 384 dimensões × 4 bytes ≈ **4,2 MB** de vetores;
- uma busca densa é uma multiplicação matriz-vetor de ~1 milhão de operações, que
  NumPy resolve em poucos milissegundos;
- o FTS5 do SQLite já entrega BM25, tokenização Unicode e remoção de acentos.

## Decisão

Usar **SQLite** como armazenamento e índice:

- metadados em tabelas relacionais normais;
- texto para BM25 em tabela virtual FTS5;
- embeddings em `BLOB` (`float32` little-endian), carregados em uma matriz NumPy;
- similaridade de cosseno por produto escalar (vetores normalizados na escrita).

## Consequências

**Ganhos**

- zero infraestrutura: `uv sync && rag-zero ingest` funciona em qualquer máquina;
- CI determinístico e rápido, sem `services:`;
- um único arquivo (`data/rag_zero.db`) para inspecionar, copiar e versionar
  snapshots;
- menos código de infraestrutura para manter enquanto o produto ainda não existe.

**Custos**

- busca densa é O(N) — aceitável até ~10⁵ chunks; acima disso, índice aproximado;
- sem concorrência de escrita robusta (SQLite serializa escritas);
- sem `tsvector`/`pgvector` nativos: a troca de backend exige reindexação.

## Limite explícito

Este ADR vale para **P0/P1**. O gatilho para migrar é objetivo, não estético:

1. corpus > 100 mil chunks, ou
2. necessidade de escrita concorrente (API multiusuário), ou
3. p95 de retrieval denso > 200 ms.

Enquanto nenhum gatilho disparar, trocar por Postgres é custo sem benefício medido.

## Onde a troca acontece

Toda a persistência está em `src/rag_zero/db.py` e a busca em
`src/rag_zero/retrieval.py`. O restante do pipeline (ingestão, chunking, resposta,
avaliação) depende dessas interfaces, não do SQLite.
