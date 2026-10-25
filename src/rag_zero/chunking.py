"""Transformação de uma contratação em texto recuperável e chunks.

Duas ideias centrais:

1. **O que é um documento?** Nesta fase, a unidade recuperável é a *ficha do aviso*
   (objeto + metadados estruturados), não o PDF do edital. É o menor artefato que já
   responde perguntas reais ("quais compras de merenda em PE?") sem depender de OCR.
2. **Normalização é parte do retrieval.** FTS5 não tem stemmer de português; sem
   steamizar, "licitação" não casa com "licitações". Normalizamos (minúsculas, sem
   acento, tokens alfanuméricos) e aplicamos Snowball PT só no índice lexical — o
   texto exibido e o embedding continuam intactos.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

import snowballstemmer

from rag_zero.db import Chunk
from rag_zero.pncp.models import Contratacao

_STEMMER = snowballstemmer.stemmer("portuguese")
# `[^\W_]+` = letras/dígitos Unicode, preservando acentos (o stemmer PT depende deles).
_SURFACE_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_WORD_RE = re.compile(r"[a-z0-9]+")

# Stopwords usadas no sinal de cobertura (não no índice): palavras funcionais não
# devem fazer uma pergunta parecer "coberta" pelo corpus.
STOPWORDS = frozenset(
    """
    a o as os um uma uns umas de do da dos das em no na nos nas por para com sem
    e ou que qual quais quando onde como quanto quantos quantas quem é sao
    foi ser seu sua seus suas the of and to in for on at is are
    contratacao contratacoes licitacao licitacoes
    """.split()
)


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def tokenize(text: str) -> list[str]:
    """Tokens minúsculos, sem acento, alfanuméricos (números preservados)."""
    return _WORD_RE.findall(strip_accents(text).lower())


def stem_tokens(text: str) -> list[str]:
    """Steamiza com acento (regra do Snowball PT) e só então remove o acento.

    Ordem importa: ``licitação`` e ``licitações`` só colapsam no mesmo radical
    (``licit``) quando o stemmer vê a forma acentuada.
    """
    surface = _SURFACE_WORD_RE.findall(text.lower())
    return [strip_accents(token) for token in _STEMMER.stemWords(surface)]


def fts_text(text: str) -> str:
    return " ".join(stem_tokens(text))


def content_terms(text: str) -> list[str]:
    """Termos de conteúdo (sem stopwords), já steamizados — para cobertura/IDF.

    A decisão de stopword acontece na palavra original, antes do stemming: se
    filtrarmos depois, ``quais`` vira ``qua`` e escapa da lista.
    """
    terms: list[str] = []
    for surface in _SURFACE_WORD_RE.findall(text.lower()):
        if surface in STOPWORDS:
            continue
        stem = strip_accents(_STEMMER.stemWords([surface])[0])
        if len(stem) > 2:
            terms.append(stem)
    return terms


def format_brl(value: float | None) -> str:
    if value is None:
        return "não informado"
    formatted = f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {formatted}"


def format_datetime(value: str | None) -> str:
    """ISO do PNCP -> dd/mm/aaaa (hora quando existir)."""
    if not value:
        return "não informado"
    try:
        date_part, _, time_part = value.partition("T")
        year, month, day = date_part.split("-")
        if time_part:
            return f"{day}/{month}/{year} {time_part[:5]}"
        return f"{day}/{month}/{year}"
    except ValueError:
        return value


def notice_text(item: Contratacao) -> str:
    """Ficha canônica do aviso — é isso que vira chunk, embedding e citação."""
    orgao = item.orgao_entidade
    unidade = item.unidade_orgao
    linhas = [
        f"Objeto: {item.objeto_compra or 'não informado'}",
    ]
    if item.informacao_complementar:
        linhas.append(f"Informação complementar: {item.informacao_complementar}")
    if orgao:
        cnpj = f" (CNPJ {orgao.cnpj})" if orgao.cnpj else ""
        linhas.append(f"Órgão: {orgao.razao_social or 'não informado'}{cnpj}")
    if unidade:
        local = " — ".join(filter(None, [unidade.nome_unidade, unidade.municipio_nome]))
        uf = f"/{unidade.uf_sigla}" if unidade.uf_sigla else ""
        linhas.append(f"Unidade: {local}{uf}")
    linhas.append(
        "Modalidade: "
        + (item.modalidade_nome or "não informada")
        + f" · Modo de disputa: {item.modo_disputa_nome or 'não informado'}"
        + f" · Situação: {item.situacao_compra_nome or 'não informada'}"
    )
    linhas.append(f"Valor total estimado: {format_brl(item.valor_total_estimado)}")
    if item.valor_total_homologado is not None:
        linhas.append(f"Valor total homologado: {format_brl(item.valor_total_homologado)}")
    publicacao = format_datetime(
        item.data_publicacao_pncp.isoformat() if item.data_publicacao_pncp else None
    )
    abertura = format_datetime(
        item.data_abertura_proposta.isoformat() if item.data_abertura_proposta else None
    )
    encerramento = format_datetime(
        item.data_encerramento_proposta.isoformat() if item.data_encerramento_proposta else None
    )
    linhas.append(
        f"Publicação: {publicacao} · Abertura de propostas: {abertura}"
        f" · Encerramento de propostas: {encerramento}"
    )
    complemento = [
        f"Processo: {item.processo}" if item.processo else "",
        f"Número da compra: {item.numero_compra}" if item.numero_compra else "",
        f"SRP: {'sim' if item.srp else 'não'}",
        f"ID PNCP: {item.numero_controle_pncp}" if item.numero_controle_pncp else "",
    ]
    linhas.append(" · ".join(part for part in complemento if part))
    return "\n".join(linhas)


def chunk_text(text: str, *, max_words: int = 180, overlap: int = 25) -> list[str]:
    """Empacota linhas em janelas de ~``max_words`` palavras, com sobreposição.

    A sobreposição existe para não cortar uma resposta no meio entre dois chunks.
    Para fichas curtas (a maioria), o resultado é um único chunk — e isso é correto:
    chunking agressivo em documento curto só dilui o sinal.
    """
    if max_words <= overlap or overlap < 0:
        raise ValueError("max_words deve ser maior que overlap (>= 0)")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    def flush() -> None:
        nonlocal current, current_words
        if current:
            chunks.append("\n".join(current))
            current = []
            current_words = 0

    for line in lines:
        words = line.split()
        if len(words) > max_words:
            flush()
            step = max_words - overlap
            for start in range(0, len(words), step):
                window = words[start : start + max_words]
                if window:
                    chunks.append(" ".join(window))
                if start + max_words >= len(words):
                    break
            continue
        if current and current_words + len(words) > max_words:
            flush()
        current.append(line)
        current_words += len(words)

    flush()
    return [chunk for chunk in chunks if chunk.strip()]


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_chunks(
    item: Contratacao, *, max_words: int = 180, overlap: int = 25
) -> tuple[str, list[Chunk]]:
    """Retorna (fingerprint, chunks prontos para o banco)."""
    text = notice_text(item)
    pieces = chunk_text(text, max_words=max_words, overlap=overlap)
    chunks = [
        Chunk(
            seq=index,
            secao="ficha",
            conteudo=piece,
            fts_text=fts_text(piece),
            tokens=len(piece.split()),
        )
        for index, piece in enumerate(pieces)
    ]
    return fingerprint(text), chunks
