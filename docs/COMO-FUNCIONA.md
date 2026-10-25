# Como o RAG Zero funciona (guia de estudo)

Este documento explica **o que** cada peça faz, **por que** ela existe e **como**
verificar que está funcionando. A ordem segue o fluxo de um dado: da API do PNCP até
uma resposta com citação.

## 1. O problema, em uma frase

O PNCP publica centenas de milhares de avisos de compra por ano. A busca do portal
funciona por palavra exata; perguntas reais ("quais oportunidades abertas de merenda
escolar em PE?") não. O RAG Zero responde essas perguntas **com citações
verificáveis** — ou recusa quando o corpus não contém a resposta.

## 2. Pipeline

```
API PNCP ──▶ ingest ──▶ SQLite (contratações)
                  │
                  └──▶ ficha do aviso ──▶ chunks ──▶ embeddings (BLOB)
                                                      │
pergunta ──▶ normalização PT ──▶ BM25 (FTS5) ─┐       │
                          └──▶ denso (NumPy) ─┴─▶ RRF ──▶ resposta extrativa
                                                           ├── citações
                                                           └── recusa por cobertura
```

### 2.1 Ingestão (`ingest.py`, `pncp/client.py`)

- A API é pública e não exige chave. O cliente documenta e trata os *quirks*:
  `204` = sem resultados, `pagina` começa em 1, `tamanhoPagina` mínimo 10 e máximo
  50 em contratações.
- Throttle de 0,7 s entre requisições: educação com API pública, não arrancamento.
- Idempotência: a chave é `numeroControlePNCP`; um *fingerprint* SHA-256 do texto da
  ficha decide se o item precisa ser reindexado. Rodar duas vezes não duplica nada.

**Como verificar:** `rag-zero ingest` duas vezes seguidas — a segunda deve reportar
`novos=0` e `inalterados=N`.

### 2.2 Ficha e chunking (`chunking.py`)

O "documento" desta fase é a **ficha do aviso**: objeto + órgão + unidade + valores +
datas + modalidade. Não é o PDF do edital (isso é P2, com extração e OCR).

- Chunk = janela de ~180 palavras com 25 de sobreposição.
- A maioria das fichas vira **um** chunk — e isso é correto: chunking agressivo em
  documento curto dilui o sinal.
- O texto exibido e o texto indexado são diferentes: o índice lexical guarda tokens
  sem acento e steamizados; o texto original permanece intacto para citação.

**Por que stemmer:** `licitação` e `licitações` precisam colapsar no mesmo radical.
O Snowball PT só faz isso corretamente quando recebe a palavra **com acento** — daí a
ordem `stem → strip_accents`, e não o contrário.

**Como verificar:** `tests/test_chunking.py::test_stemming_unifica_plural_e_acento`.

### 2.3 Embeddings (`embeddings.py`)

- Modelo: `paraphrase-multilingual-MiniLM-L12-v2` (384 dims, ONNX, CPU).
- Vetores normalizados na escrita → cosseno vira produto escalar.
- O modelo é *lazy*: `search --mode bm25` nunca baixa 220 MB de modelo.

**Por que não um LLM:** embedding é recuperação, não geração. Um modelo pequeno e
local torna o custo por consulta **zero** e o eval reproduzível offline.

### 2.4 Retrieval (`retrieval.py`)

Três modos, propositalmente comparáveis:

| modo | sinal | força | fraqueza |
|---|---|---|---|
| `bm25` | lexical (FTS5) | termos raros, IDs, nomes | sinonímia |
| `dense` | semântico | paráfrase | detalhes exatos |
| `hybrid` | RRF dos dois | robusto | precisa dos dois índices |

**O padrão é `bm25`** — não por preferência, mas porque venceu em todos os estratos
do golden set (ver §3.1 e `docs/adr/0002-bm25-default.md`). `dense` e `hybrid`
continuam disponíveis como experimento medido.

**RRF** combina *rankings*, não scores:

```
score(d) = Σ 1 / (k + rank_i(d))     k = 60
```

Isso evita calibrar escalas incompatíveis (BM25 não tem limite superior; cosseno tem).

Resultados são **por contratação**, não por chunk: um aviso aparece no máximo uma vez.

**Como verificar:** `rag-zero search "merenda escolar" --mode bm25` e repita com
`--mode hybrid`; compare as ordens.

### 2.5 Resposta e recusa (`answer.py`)

Duas rotas:

1. **ID explícito na pergunta** → lookup estruturado. Não é RAG; é a resposta certa
   para uma pergunta que já contém a chave.
2. **Linguagem natural** → retrieval + resposta extrativa (campos e trechos), sempre
   com citação.

A recusa usa **cobertura IDF**, não "confiança do modelo":

```
cobertura = Σ idf(termos presentes) / Σ idf(todos os termos)
```

Termo que não existe no corpus tem IDF máximo e derruba a cobertura. Se a pergunta
fala de algo que o corpus nunca menciona, o sistema recusa em vez de alucinar.

Duas regras adicionais, ambas nascidas de falhas medidas:

- **Escopo temporal.** Se a pergunta menciona um ano maior que o maior ano de compra do
  corpus (ex.: 2031 num corpus cujo maior ano é 2027), recusa antes do retrieval. Os
  substantivos podem existir; o período não.
- **Citação proporcional à pergunta.** Pergunta singular ("qual contratação…") cita
  **uma** fonte; pergunta plural ("quais contratações…") cita até quatro. Citar quatro
  fontes numa pergunta singular infla a precisão de citação com fontes que não
  respondem à pergunta.

**Limite honesto:** sinônimo ausente do corpus também derruba a cobertura. É um
trade-off calibrável: mais recusa = menos alucinação; menos recusa = mais cobertura.
O golden set existe para medir esse trade-off, não para escondê-lo. Recusa
**composicional** ("submarinos nucleares": cada termo existe, a combinação não) ainda
não é capturada — é o próximo experimento, não uma regra inventada.

### 2.6 Avaliação (`evals/`)

O golden set tem três tipos e quatro subtipos:

- **estrutural** (`lookup`) — resposta é um campo exato (valor, prazo, órgão). Mede o lookup.
- **retrieval** — exige achar o aviso certo. Divide-se em:
  - `verbatim` — a pergunta repete um trecho real do objeto (piso do índice lexical);
  - `paraphrase` — sinônimos escritos à mão, sem repetir o texto (generalização).
- **sem_resposta** (`unanswerable`) — o corpus não contém a resposta. Mede recusa correta.

Métricas são calculadas por definição matemática, sem juiz LLM:

- `recall@k` = fração das fontes esperadas recuperadas no top-k;
- `MRR@k` = média do inverso do rank da primeira fonte correta;
- `citation precision` = fração das citações que estão nas fontes esperadas;
- `structural accuracy` = resposta contém o valor esperado (normalizado);
- `refusal accuracy` = fração dos itens impossíveis corretamente recusados.

**Como verificar:** `rag-zero evals run --modes bm25,dense,hybrid` grava
`evals/results/latest.json` com os números e a configuração usada.

## 3. Resultados medidos (k=10, 72 itens)

| modo | recall@10 | MRR@10 | cit. precision | estrutural | recusa | p95 |
|---|---|---|---|---|---|---|
| **bm25** | **0.905** | **0.758** | **0.690** | 1.000 | 0.667 | **68 ms** |
| dense | 0.405 | 0.262 | 0.190 | 1.000 | 0.667 | 901 ms |
| hybrid | 0.762 | 0.487 | 0.381 | 1.000 | 0.667 | 778 ms |

Por subtipo:

| subtipo | n | bm25 | dense | hybrid |
|---|---|---|---|---|
| verbatim | 30 | **0.867 / 0.712** | 0.267 / 0.134 | 0.700 / 0.370 |
| paraphrase | 12 | **1.000 / 0.873** | 0.750 / 0.581 | 0.917 / 0.778 |
| unanswerable | 6 | 0.667 | 0.667 | 0.667 |

(valores = recall@10 / MRR@10; recusa = fração de itens impossíveis corretamente recusados)

**Como ler isso sem se enganar:**

- BM25 ganha nos dois subtipos. O denso não venceu nenhum recorte — e `cos(stored,
  fresh) = 1.0` provou que o índice está correto. É limite de modelo/corpus, não bug.
- RRF pode **piorar** o resultado: fundir um ranker bom com um fraco dilui o bom.
- A recusa subiu de 3/6 para 4/6 com a regra de escopo temporal. As duas falhas
  restantes são composicionais ("trens-bala", "submarinos nucleares"): cada palavra
  existe, a combinação não. Cobertura IDF não vê isso.
- As 12 paráfrases nomeiam o município — o que dá âncora lexical ao BM25. É uma
  limitação do conjunto, declarada, não um efeito escondido.

## 4. Decisões que você pode questionar

- **SQLite em vez de Postgres** — ver `docs/adr/0001-sqlite-first.md`. O gatilho de
  migração é medido, não estético.
- **Resposta extrativa em vez de LLM** — permite avaliar retrieval e recusa sem
  confundir com qualidade de geração. O contrato de citações já está pronto para um
  gerador LLM em P2.
- **Golden set parcialmente sintético** — os itens mecânicos são derivados do banco
  (verificáveis); os de paráfrase e os impossíveis são manuais. Nenhum item entra sem
  conferência contra o corpus.

## 5. O que ainda não existe (e por que)

- **PDFs de edital** — exige download, extração e OCR; o corpus atual já responde
  perguntas de triagem. P2.
- **Reranker** — só faz sentido depois de medir o teto do retrieval híbrido. P3.
- **API/UI** — P4.
- **Testes de prompt injection** — P5.
