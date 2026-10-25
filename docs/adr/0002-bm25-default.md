# ADR-0002 — BM25 como modo padrão; denso fica experimental

- **Status:** aceito
- **Data:** 2026-10-25
- **Contexto:** primeiro golden set estratificado (72 itens) sobre 2.721 chunks reais do PNCP.

## Contexto

O desenho inicial tratava `hybrid` (BM25 + denso via RRF) como modo padrão — a escolha
"de manual". A primeira medição completa contradisse o manual:

| modo | recall@10 | MRR@10 | cit. precision | p95 |
|---|---|---|---|---|
| bm25 | 0.905 | 0.758 | 0.690 | 68 ms |
| dense | 0.405 | 0.262 | 0.190 | 901 ms |
| hybrid | 0.762 | 0.487 | 0.381 | 778 ms |

O padrão se repete em **todos** os estratos: verbatim (0.867 vs 0.700 vs 0.267 de
recall) e paráfrase (1.000 vs 0.917 vs 0.750). O denso não venceu nenhum recorte.

## Diagnóstico (antes de decidir)

- `cos(embedding armazenado, recalculado) = 1.0` → não é corrupção de índice.
- O corpus é dominado por objetos quase idênticos ("registro de preços para futura
  aquisição de…"); o modelo de 384 dimensões não separa município/órgão com a nitidez
  que um termo raro separa.
- RRF funde *rankings*: quando um ranker é quase aleatório no domínio, ele dilui o
  ranking bom. Fusão pressupõe complementaridade, não apenas diferença.

## Decisão

1. `bm25` é o modo padrão em `search`, `ask`, `retrieval.search` e `answer_question`.
2. `dense` e `hybrid` continuam disponíveis e medidos — são o experimento, não o produto.
3. O default só muda com evidência: um novo golden set (sem âncora de município) ou um
   modelo de embedding que vença o BM25 em algum estrato.

## Consequências

**Ganhos**

- p95 cai de ~0,8–1,0 s (hybrid) para ~0,07–0,15 s (bm25) no harness; cold start de
  46 s desaparece do caminho padrão. A latência varia ~2× entre execuções (68–152 ms
  observados para o BM25) — o número é direção, não constante. A CLI em processo frio
  soma ~0,7 s de import + carga do índice de DF, por isso o `ask` ponta a ponta fica
  perto de 0,9 s.
- Precisão de citação sobe de 0.381 para 0.690.
- O produto não depende de download de modelo para funcionar.

**Custos**

- Consultas sem nenhuma âncora lexical (sinônimo puro, sem nome próprio) continuam
  dependendo de `--mode hybrid` — e hoje o ganho é medido, não presumido.
- A decisão é datada e ligada a este corpus; por isso o golden set e o `latest.json`
  ficam versionados junto do código.

## Como reverter

Trocar o default de volta é uma linha em quatro assinaturas. O gatilho para reverter
é objetivo: dense ou hybrid superar bm25 em recall@10 **e** MRR@10 no mesmo golden set,
com p95 ≤ 2× o do BM25.
