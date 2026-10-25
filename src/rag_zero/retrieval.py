"""Recuperação: BM25 (FTS5), denso (cosseno) e fusão por Reciprocal Rank Fusion.

Por que híbrido: BM25 é ótimo para termos raros e exatos (nomes de órgão, "merenda",
números de processo) e péssimo para sinonímia; embeddings capturam paráfrase e erram
detalhes. RRF combina *rankings*, não scores — evita calibrar escalas incompatíveis:

    score(d) = Σ 1 / (k + rank_i(d)),  k = 60

É o mesmo mecanismo usado em busca de produção por ser barato e robusto.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from rag_zero import db
from rag_zero.chunking import content_terms
from rag_zero.config import settings
from rag_zero.embeddings import Embedder, from_bytes


@dataclass
class SearchHit:
    chunk_id: int
    numero_controle_pncp: str
    seq: int
    secao: str
    conteudo: str
    uf: str | None
    municipio: str | None
    orgao: str | None
    objeto: str | None
    valor: float | None
    encerramento: str | None
    modalidade: str | None
    link: str | None
    score: float = 0.0
    bm25: float | None = None
    dense: float | None = None
    ranks: dict[str, int] = field(default_factory=dict)


class EmbeddingStore:
    """Matriz de vetores em memória, invalidada quando o índice muda."""

    def __init__(self, conn) -> None:
        self._conn = conn
        self._version: tuple[int, int] | None = None
        self.ids: np.ndarray = np.empty(0, dtype=np.int64)
        self.matrix: np.ndarray = np.empty((0, settings.embedding_dim), dtype=np.float32)
        self._position: dict[int, int] = {}

    def _current_version(self) -> tuple[int, int]:
        total = db.count_chunks(self._conn)
        embedded = int(
            self._conn.execute(
                "SELECT count(*) FROM chunks WHERE embedding IS NOT NULL"
            ).fetchone()[0]
        )
        return total, embedded

    def load(self) -> EmbeddingStore:
        version = self._current_version()
        if version == self._version and len(self.ids):
            return self
        rows = db.embedding_rows(self._conn)
        self.ids = np.array([row["id"] for row in rows], dtype=np.int64)
        if rows:
            self.matrix = np.vstack([from_bytes(row["embedding"]) for row in rows]).astype(
                np.float32
            )
        else:
            self.matrix = np.empty((0, settings.embedding_dim), dtype=np.float32)
        self._position = {int(chunk_id): i for i, chunk_id in enumerate(self.ids)}
        self._version = version
        return self

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        self.load()
        if not len(self.ids):
            return []
        scores = self.matrix @ np.asarray(query_vector, dtype=np.float32)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(self.ids[i]), float(scores[i])) for i in top]


class DocumentFrequency:
    """df por termo steamizado, para cobertura IDF. Barato no nosso volume."""

    def __init__(self, conn) -> None:
        self.n = 0
        self.df: dict[str, int] = {}
        self._version = -1
        self._conn = conn

    def load(self) -> DocumentFrequency:
        total = db.count_chunks(self._conn)
        if total == self._version:
            return self
        df: dict[str, int] = {}
        for row in self._conn.execute("SELECT fts_text FROM chunks_fts"):
            for term in set(row["fts_text"].split()):
                df[term] = df.get(term, 0) + 1
        self.df, self.n = df, total
        self._version = total
        return self

    def idf(self, term: str) -> float:
        return math.log(1 + (self.n + 1) / (self.df.get(term, 0) + 1))


def evidence_coverage(query: str, df: DocumentFrequency) -> tuple[float, list[str]]:
    """Fração (ponderada por IDF) dos termos da pergunta que existem no corpus.

    Termos desconhecidos são raríssimos -> IDF alto -> derrubam a cobertura. É o
    sinal usado para recusar: se a pergunta fala de algo que o corpus nunca menciona,
    nenhuma similaridade deve ser tratada como evidência.
    """
    df.load()
    terms = content_terms(query)
    if not terms:
        return 0.0, []
    total = sum(df.idf(term) for term in terms)
    present = sum(df.idf(term) for term in terms if term in df.df)
    unknown = [term for term in terms if term not in df.df]
    return (present / total if total else 0.0), unknown


def _fts_query(query: str) -> str:
    terms = list(dict.fromkeys(content_terms(query)))
    return " OR ".join(f'"{term}"' for term in terms)


def bm25_search(conn, query: str, k: int) -> list[tuple[int, float]]:
    match = _fts_query(query)
    if not match:
        return []
    rows = conn.execute(
        """SELECT chunk_id, bm25(chunks_fts) AS rank
           FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?""",
        (match, k),
    ).fetchall()
    return [(int(row["chunk_id"]), -float(row["rank"])) for row in rows]


def rrf_fuse(rankings: Sequence[Sequence[int]], k: int = 60) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for position, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + position)
    return dict(sorted(scores.items(), key=lambda pair: pair[1], reverse=True))


def search(
    conn,
    query: str,
    *,
    mode: str = "bm25",
    k: int = 10,
    embedder: Embedder | None = None,
    store: EmbeddingStore | None = None,
) -> list[SearchHit]:
    """Busca por contrato (não por chunk): um aviso aparece no máximo uma vez."""
    if mode not in {"bm25", "dense", "hybrid"}:
        raise ValueError(f"modo inválido: {mode}")

    candidate_k = max(k * 5, 50)
    lexical = bm25_search(conn, query, candidate_k)
    dense: list[tuple[int, float]] = []
    if mode in {"dense", "hybrid"}:
        if embedder is None:
            raise ValueError("modo denso exige um embedder")
        store = (store or EmbeddingStore(conn)).load()
        dense = store.search(embedder.embed_query(query), candidate_k)

    lexical_rank = {chunk_id: rank for rank, (chunk_id, _) in enumerate(lexical, start=1)}
    dense_rank = {chunk_id: rank for rank, (chunk_id, _) in enumerate(dense, start=1)}
    lexical_score = dict(lexical)
    dense_score = dict(dense)

    if mode == "bm25":
        ordered = [chunk_id for chunk_id, _ in lexical]
        fused: dict[int, float] = {}
    elif mode == "dense":
        ordered = [chunk_id for chunk_id, _ in dense]
        fused = {}
    else:
        fused = rrf_fuse([list(lexical_rank), list(dense_rank)], k=settings.rrf_k)
        ordered = list(fused)

    hits: list[SearchHit] = []
    seen_contracts: set[str] = set()
    for chunk_id in ordered:
        row = db.chunk_row(conn, chunk_id)
        if row is None or row["numero_controle_pncp"] in seen_contracts:
            continue
        seen_contracts.add(row["numero_controle_pncp"])
        hits.append(
            SearchHit(
                chunk_id=chunk_id,
                numero_controle_pncp=row["numero_controle_pncp"],
                seq=row["seq"],
                secao=row["secao"],
                conteudo=row["conteudo"],
                uf=row["uf"],
                municipio=row["municipio"],
                orgao=row["orgao_razao_social"],
                objeto=row["objeto_compra"],
                valor=row["valor_total_estimado"],
                encerramento=row["data_encerramento_proposta"],
                modalidade=row["modalidade_nome"],
                link=row["link_sistema_origem"],
                score=(
                    lexical_score.get(chunk_id, 0.0)
                    if mode == "bm25"
                    else dense_score.get(chunk_id, 0.0)
                    if mode == "dense"
                    else fused.get(chunk_id, 0.0)
                ),
                bm25=lexical_score.get(chunk_id),
                dense=dense_score.get(chunk_id),
                ranks={
                    name: rank
                    for name, rank in (
                        ("bm25", lexical_rank.get(chunk_id)),
                        ("dense", dense_rank.get(chunk_id)),
                    )
                    if rank is not None
                },
            )
        )
        if len(hits) >= k:
            break
    return hits
