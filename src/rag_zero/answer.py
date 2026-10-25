"""Resposta grounded e recusa honesta.

O gerador desta fase é **extrativo e determinístico**: ele não inventa texto livre,
seleciona campos estruturados e trechos recuperados, e cita a fonte de cada
afirmação. Isso torna a avaliação verificável sem juiz LLM — e deixa o caminho aberto
para trocar o gerador por um LLM depois, mantendo o mesmo contrato de citações.

Duas portas de entrada:

- a pergunta contém um ID PNCP -> resposta estruturada exata (não é RAG; é lookup);
- pergunta em linguagem natural -> retrieval híbrido + resposta extrativa.

A recusa é um **sinal de evidência**, não de confiança do modelo: medimos a cobertura
IDF dos termos da pergunta contra o corpus. Pergunta sobre algo que o corpus nunca
menciona tem cobertura baixa e deve ser recusada.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from rag_zero import db
from rag_zero.chunking import format_brl, format_datetime
from rag_zero.config import settings
from rag_zero.embeddings import Embedder
from rag_zero.retrieval import (
    DocumentFrequency,
    EmbeddingStore,
    SearchHit,
    evidence_coverage,
    search,
)

CONTROL_NUMBER_RE = re.compile(r"\b\d{14}-\d{1,2}-\d{1,6}/\d{4}\b")
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
PLURAL_RE = re.compile(
    r"\b(quais|quantos|quantas|liste|listar|relacione|todas|todos)\b"
    r"|\b(contratações|licitações|avisos)\b",
    re.I,
)

FIELD_INTENTS: list[tuple[str, re.Pattern[str]]] = [
    ("encerramento", re.compile(r"encerr|prazo\s+(final|de|para).*propost|limite.*propost", re.I)),
    ("abertura", re.compile(r"abertura|in[ií]cio.*propost|come[cç]a.*propost", re.I)),
    ("valor", re.compile(r"\bvalor\b|\bquanto\b|\bpre[cç]o\b|or[cç]amento", re.I)),
    ("orgao", re.compile(r"[oó]rg[aã]o|quem\s+(publicou|contratou)|entidade", re.I)),
    ("modalidade", re.compile(r"modalidade|preg[aã]o|dispensa|concorr[eê]ncia|leil[aã]o", re.I)),
    ("municipio", re.compile(r"munic[ií]pio|cidade|\bonde\b", re.I)),
    ("processo", re.compile(r"\bprocesso\b", re.I)),
    ("objeto", re.compile(r"\bobjeto\b|do\s+que\s+se\s+trata|sobre\s+o\s+que", re.I)),
]

FIELDS: dict[str, tuple[str, callable]] = {
    "valor": ("valor total estimado", lambda row: format_brl(row["valor_total_estimado"])),
    "encerramento": (
        "encerramento de propostas",
        lambda row: format_datetime(row["data_encerramento_proposta"]),
    ),
    "abertura": (
        "abertura de propostas",
        lambda row: format_datetime(row["data_abertura_proposta"]),
    ),
    "orgao": ("órgão", lambda row: row["orgao_razao_social"] or "não informado"),
    "modalidade": ("modalidade", lambda row: row["modalidade_nome"] or "não informada"),
    "municipio": ("município", lambda row: row["municipio"] or "não informado"),
    "processo": ("processo", lambda row: row["processo"] or "não informado"),
    "objeto": ("objeto", lambda row: row["objeto_compra"] or "não informado"),
}


@dataclass(frozen=True)
class Citation:
    numero_controle_pncp: str
    secao: str = "ficha"
    link: str | None = None


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False
    reason: str | None = None
    confidence: float = 0.0
    latency_ms: float = 0.0
    hits: list[SearchHit] = field(default_factory=list)


def detect_field(question: str) -> str | None:
    for name, pattern in FIELD_INTENTS:
        if pattern.search(question):
            return name
    return None


def is_plural_question(question: str) -> bool:
    """A pergunta pede uma lista? Se não, citar 4 fontes é ruído.

    Citação é afirmação: citar o 2º ao 4º resultado numa pergunta singular
    ("qual contratação...") infla a resposta com fontes que não a respondem.
    """
    return bool(PLURAL_RE.search(question))


def _truncate(text: str | None, limit: int = 180) -> str:
    if not text:
        return "não informado"
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _citation_from_row(row, link: str | None = None) -> Citation:
    return Citation(
        numero_controle_pncp=row["numero_controle_pncp"],
        secao="ficha",
        link=link if link is not None else row["link_sistema_origem"],
    )


def _summary(row) -> str:
    orgao = row["orgao_razao_social"] or "órgão não informado"
    return (
        f"Contratação {row['numero_controle_pncp']} — {orgao}\n"
        f"Objeto: {_truncate(row['objeto_compra'], 400)}\n"
        f"Local: {row['municipio'] or 'não informado'}/{row['uf'] or '—'} · "
        f"Modalidade: {row['modalidade_nome'] or 'não informada'}\n"
        f"Valor estimado: {format_brl(row['valor_total_estimado'])} · "
        f"Encerramento: {format_datetime(row['data_encerramento_proposta'])}"
    )


def _answer_from_row(question: str, row) -> Answer:
    intent = detect_field(question)
    citation = _citation_from_row(row)
    if intent:
        label, getter = FIELDS[intent]
        value = getter(row)
        return Answer(
            text=f"{label.capitalize()} da contratação {row['numero_controle_pncp']}: {value}.",
            citations=[citation],
            confidence=1.0,
        )
    return Answer(text=_summary(row), citations=[citation], confidence=1.0)


def answer_question(
    conn,
    question: str,
    *,
    mode: str = "bm25",
    k: int = 8,
    embedder: Embedder | None = None,
    store: EmbeddingStore | None = None,
    df: DocumentFrequency | None = None,
) -> Answer:
    started = time.perf_counter()

    # 1) ID explícito: lookup exato. RAG aqui só adicionaria ruído.
    control = CONTROL_NUMBER_RE.search(question)
    if control:
        row = db.get_contratacao(conn, control.group(0))
        if row is None:
            return Answer(
                text=f"A contratação {control.group(0)} não está no corpus indexado.",
                refused=True,
                reason="id_ausente_no_corpus",
                confidence=1.0,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        answer = _answer_from_row(question, row)
        answer.latency_ms = (time.perf_counter() - started) * 1000
        return answer

    # 2) Escopo temporal: o corpus cobre uma janela finita de anos. Perguntar por
    #    2031 quando o corpus termina em 2026 é uma pergunta que ele não pode responder,
    #    mesmo que os substantivos existam.
    years = {int(year) for year in YEAR_RE.findall(question)}
    if years:
        span = db.corpus_year_range(conn)
        if span:
            fora = [year for year in sorted(years) if year < span[0] or year > span[1]]
            if fora:
                return Answer(
                    text=(
                        f"O corpus indexado cobre compras de {span[0]} a {span[1]}. "
                        f"A pergunta menciona {fora[0]}, fora desse intervalo."
                    ),
                    refused=True,
                    reason="fora_do_escopo_temporal",
                    confidence=1.0,
                    latency_ms=(time.perf_counter() - started) * 1000,
                )

    # 3) Portão de evidência: antes de buscar, pergunte se o corpus fala disso.
    df = df or DocumentFrequency(conn)
    coverage, unknown = evidence_coverage(question, df)
    if coverage < settings.min_evidence_coverage:
        missing = ", ".join(unknown[:5]) if unknown else "termos centrais da pergunta"
        return Answer(
            text=(
                "Não encontrei evidência suficiente no corpus para responder. "
                f"Termos sem ocorrência: {missing}."
            ),
            refused=True,
            reason="evidencia_insuficiente",
            confidence=coverage,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    # 4) Retrieval.
    hits = search(conn, question, mode=mode, k=k, embedder=embedder, store=store)
    if not hits:
        return Answer(
            text="Não encontrei contratações relevantes no corpus para essa pergunta.",
            refused=True,
            reason="sem_resultados",
            confidence=coverage,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
    best = hits[0]
    if mode in {"dense", "hybrid"} and best.dense is not None:
        if best.dense < settings.min_dense_similarity:
            return Answer(
                text=(
                    "O melhor resultado não tem similaridade suficiente "
                    "para ser tratado como evidência."
                ),
                refused=True,
                reason="similaridade_baixa",
                confidence=best.dense,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

    # 5) Resposta extrativa.
    intent = detect_field(question)
    singular = not is_plural_question(question)
    if intent:
        label, _ = FIELDS[intent]
        row = db.get_contratacao(conn, best.numero_controle_pncp)
        value = FIELDS[intent][1](row) if row else "não informado"
        text = f"{label.capitalize()} da contratação {best.numero_controle_pncp}: {value}."
        citations = [_citation_from_row(row, best.link)] if row else []
    elif singular:
        row = db.get_contratacao(conn, best.numero_controle_pncp)
        text = _summary(row) if row else f"Contratação {best.numero_controle_pncp}."
        citations = [_citation_from_row(row, best.link)] if row else []
    else:
        linhas = [f"Encontrei {len(hits)} contratação(ões) relevante(s). As mais próximas:"]
        for position, hit in enumerate(hits[:4], start=1):
            linhas.append(
                f"\n[{position}] {hit.numero_controle_pncp} — {hit.orgao or 'órgão não informado'}"
                f" ({hit.municipio or '—'}/{hit.uf or '—'})\n"
                f"    Objeto: {_truncate(hit.objeto)}\n"
                f"    Valor estimado: {format_brl(hit.valor)} · "
                f"Encerramento: {format_datetime(hit.encerramento)}\n"
                f"    Fonte: {hit.link or 'link não informado'}"
            )
        text = "\n".join(linhas)
        citations = [
            Citation(numero_controle_pncp=hit.numero_controle_pncp, secao=hit.secao, link=hit.link)
            for hit in hits[:4]
        ]

    return Answer(
        text=text,
        citations=citations,
        confidence=coverage,
        latency_ms=(time.perf_counter() - started) * 1000,
        hits=hits,
    )
