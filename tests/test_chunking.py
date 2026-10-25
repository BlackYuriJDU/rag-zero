"""Testes de normalização, ficha e chunking."""

from rag_zero.chunking import (
    build_chunks,
    chunk_text,
    content_terms,
    fts_text,
    notice_text,
)
from tests.conftest import make_item


def test_stemming_unifica_plural_e_acento() -> None:
    assert fts_text("licitação") == fts_text("licitações")
    assert fts_text("contratação") == fts_text("contratações")
    assert "merend" in fts_text("merenda")


def test_content_terms_remove_stopwords() -> None:
    terms = content_terms("quais as contratações de merenda escolar em Recife?")
    assert "merend" in terms and "recif" in terms
    assert "quais" not in terms and "de" not in terms


def test_notice_text_contains_core_fields() -> None:
    text = notice_text(make_item())
    assert "merenda escolar" in text
    assert "Prefeitura Municipal do Recife" in text
    assert "R$ 120.000,50" in text
    assert "12345678000196-1-000123/2026" in text


def test_short_notice_is_one_chunk() -> None:
    fp, chunks = build_chunks(make_item())
    assert len(chunks) == 1
    assert len(fp) == 64


def test_long_text_is_split_with_overlap() -> None:
    text = " ".join(f"palavra{i}" for i in range(500))
    chunks = chunk_text(text, max_words=100, overlap=20)
    assert len(chunks) == 6
    assert chunks[0].split()[-1] == "palavra99"
    assert chunks[1].split()[0] == "palavra80"


def test_invalid_window_raises() -> None:
    try:
        chunk_text("a b c", max_words=10, overlap=10)
    except ValueError as exc:
        assert "max_words" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("esperava ValueError")
