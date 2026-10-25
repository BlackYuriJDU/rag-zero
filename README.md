# RAG Zero

[![CI](https://github.com/BlackYuriJDU/rag-zero/actions/workflows/ci.yml/badge.svg)](https://github.com/BlackYuriJDU/rag-zero/actions/workflows/ci.yml)

**Grounded question answering over Brazilian public procurement (PNCP) — with citations, honest refusal, and measured retrieval.**

> Status: **P0/P1 concluídos.** O corpus real tem 2.714 contratações (PE, Pregão
> Eletrônico, 90 dias) e 2.721 chunks. Todos os números abaixo saem de
> `rag-zero evals run` e ficam gravados em `evals/results/latest.json`.

## Por que

O Brasil publica toda compra pública no [PNCP](https://pncp.gov.br) — centenas de
milhares de avisos por ano. A busca do portal funciona por palavra exata; perguntas
reais ("quais oportunidades de merenda escolar em PE?") não.

O RAG Zero responde essas perguntas **com citações verificáveis** — ou **recusa**
quando o corpus não contém a resposta. E trata avaliação como componente de primeira
classe: retrieval e resposta são medidos separadamente, e a decisão de qual modo usar
é tomada por número, não por moda.

## Resultados medidos

Golden set com 72 itens: 24 estruturais (lookup por ID), 30 verbatim (a pergunta
repete o texto do aviso), 12 paráfrases curadas (sinônimos, sem repetir o texto) e 6
sem resposta. `k=10`, CPU, modelo de embedding local.

| modo | recall@10 | MRR@10 | citation precision | estrutural | recusa | p95 |
|---|---|---|---|---|---|---|
| **bm25** | **0.905** | **0.758** | **0.690** | 1.000 | 0.667 | **68 ms** |
| dense | 0.405 | 0.262 | 0.190 | 1.000 | 0.667 | 901 ms |
| hybrid (RRF) | 0.762 | 0.487 | 0.381 | 1.000 | 0.667 | 778 ms |

> p95 de uma execução (`evals/results/latest.json`); varia ~2× entre execuções nesta
> máquina (68–152 ms no BM25). A ordem de grandeza é estável: BM25 é ~10× mais rápido.
> A CLI em processo frio soma ~0,7 s de import + carga do DF; o cold start de 46 s é só
> do modelo denso, que o modo padrão não carrega.

Por subtipo (é aqui que o resultado fica legível):

| subtipo | n | bm25 recall / MRR | dense recall / MRR | hybrid recall / MRR |
|---|---|---|---|---|
| verbatim | 30 | **0.867 / 0.712** | 0.267 / 0.134 | 0.700 / 0.370 |
| paraphrase | 12 | **1.000 / 0.873** | 0.750 / 0.581 | 0.917 / 0.778 |
| unanswerable | 6 | 0.667 (4/6 recusas corretas) | 0.667 | 0.667 |

### O que os números dizem (e o que eu fiz com isso)

1. **BM25 domina neste corpus.** Ganha em recall, MRR, precisão de citação e latência,
   nos dois subtipos. Por isso o modo padrão da CLI passou a ser `bm25` — manter
   `hybrid` como default seria preferir a estética da arquitetura ao resultado medido.
2. **O modo denso não se paga.** O corpus é cheio de objetos quase idênticos
   ("registro de preços para futura aquisição de…"), e o embedding não separa
   município/órgão com a nitidez que o BM25 separa. `cos(stored, fresh) = 1.0`
   confirmou que não é bug de índice: é limite de modelo/corpus.
3. **RRF pode piorar.** Fundir um ranker bom com um fraco dilui o bom. Hybrid ficou
   abaixo do BM25 em todos os estratos — um lembrete de que fusão pressupõe rankers
   complementares, não apenas diferentes.
4. **Recusa subiu de 3/6 para 4/6** com uma regra de escopo temporal: o corpus cobre
   anos de compra de 2022 a 2027, então perguntar por 2031 (ou por 2019) é recusado
   antes mesmo do retrieval. As duas falhas restantes são **composicionais**
   ("trens-bala", "submarinos nucleares"): cada termo existe no corpus, a combinação
   não. Cobertura IDF não captura isso.
5. **Latência densa é custo real em CPU**: ~0,6 s por consulta em regime quente e ~46 s
   de cold start para carregar o ONNX. BM25 responde em ~45–55 ms (p50).

> Honestidade sobre o golden set: as 12 paráfrases nomeiam o município, o que dá ao
> BM25 uma âncora lexical forte. Um conjunto futuro precisa de perguntas sem nome
> próprio ("quem comprou leite especial?") para isolar sinonímia de verdade.

## Arquitetura

```
API PNCP (pública, sem chave)
   │  ingest paginado, com throttle e retry
   ▼
SQLite ── contratacoes (metadados estruturados)
   │
   ├── ficha do aviso ──▶ chunks (~180 palavras, overlap 25)
   │                          ├── FTS5 (BM25, stemming PT)
   │                          └── embeddings 384d (BLOB float32)
   ▼
retrieval (BM25 | denso | RRF) ──▶ resposta extrativa
                                    ├── citações (ID PNCP + link)
                                    └── recusa por cobertura IDF / escopo temporal

evals/ golden set ──▶ métricas ──▶ CI
```

Sem Docker, sem Postgres, sem serviço externo: `uv sync` e pronto. A decisão está em
[`docs/adr/0001-sqlite-first.md`](docs/adr/0001-sqlite-first.md), com gatilhos objetivos
de migração (>100 mil chunks, escrita concorrente ou p95 denso >200 ms).

## Quickstart

```bash
uv sync

# 1. Ingerir 90 dias de Pregão Eletrônico em PE (idempotente)
uv run rag-zero ingest --uf PE --dias 90 --modalidade 6

# 2. Gerar embeddings (local, CPU; ~2 min para 2,7 mil chunks)
uv run rag-zero index

# 3. Perguntar
uv run rag-zero ask "Quais contratações de merenda escolar em Recife?"
uv run rag-zero ask "Qual o valor total estimado da contratação 11361854000110-1-000088/2026?"

# 4. Medir
uv run rag-zero evals build
uv run rag-zero evals run --modes bm25,dense,hybrid --k 10
```

`ask` termina com `confiança=… · N ms · modo=…`, e imprime `RECUSA: motivo` quando não
há evidência suficiente. `search` mostra o ranking sem gerar resposta.

## O que está implementado

- [x] **P0** — cliente PNCP (quirks documentados), ingestão idempotente por fingerprint
- [x] **P1** — chunking com stemming PT correto, FTS5/BM25, embeddings locais, RRF
- [x] **P2 (parcial)** — resposta extrativa com citação, recusa por evidência e escopo
      temporal, golden set estratificado, métricas sem juiz LLM
- [ ] **P2 (resto)** — PDFs de edital, extração/OCR, citações por span
- [ ] **P3** — reranker cross-encoder PT-BR
- [ ] **P4** — API FastAPI + UI de inspeção
- [ ] **P5** — testes de prompt injection, release v1.0

## Avaliações

O harness vive em `src/rag_zero/evals/` e o golden set em `evals/golden_set.yaml`.

1. Itens **estruturais** são derivados do banco e conferidos por SQL (valor, prazo, órgão).
2. Itens **verbatim** usam um trecho real do objeto — medem o piso do índice lexical.
3. Itens de **paráfrase** são escritos à mão e verificados contra o corpus — medem
   generalização semântica.
4. Itens **sem resposta** medem recusa; recusar corretamente é acerto, não falha.

Métricas são definidas matematicamente (`recall@k`, `MRR@k`, precisão de citação,
acurácia estrutural, acurácia de recusa) — nenhum juiz LLM no caminho. O relatório
inclui `by_subtipo`, então um número agregado nunca esconde um estrato ruim.

## Desenvolvimento

```bash
uv sync --dev
uv run ruff check .
uv run ruff format --check .
uv run pytest        # 33 testes, sem rede e sem modelo
```

CI roda lint, format e testes em todo push (`.github/workflows/ci.yml`).

## Limitações (seção honesta)

- **Sem PDFs de edital.** O documento indexado é a ficha do aviso (objeto, órgão,
  valores, datas). Anexos e OCR são P2.
- **Recusa composicional não resolvida.** Termos que existem isoladamente mas nunca
  juntos ("submarinos nucleares") passam pela cobertura IDF. Um verificador de
  co-ocorrência no top-k é o próximo experimento — não uma regra inventada.
- **Golden set pequeno.** 72 itens dão direção, não significância estatística. Cada
  número tem intervalo de confiança largo; trate como bússola, não como veredito.
- **Denso em CPU.** ~0,6 s/consulta e 46 s de cold start. Em produção, cache do modelo
  ou GPU — ou simplesmente BM25, que hoje ganha.
- **Escopo do corpus.** PE, Pregão Eletrônico, 90 dias de publicação. O sistema **não**
  filtra por UF nem modalidade: uma pergunta sobre outro estado pode recuperar
  contratos de PE se compartilhar vocabulário. Detectar e recusar escopo geográfico é
  trabalho pendente — hoje a limitação é declarada, não aplicada.
- Projeto independente; não afiliado ao governo brasileiro.

## Licença

MIT — veja [LICENSE](LICENSE).

Autor: Arthur Araújo ([@BlackYuriJDU](https://github.com/BlackYuriJDU)).
