"""Testes do caminho de resposta: lookup exato, resposta extrativa e recusa."""

from rag_zero.answer import answer_question
from tests.conftest import insert, make_item

NUMERO = "12345678000196-1-000123/2026"


def seed(conn) -> None:
    insert(conn, make_item())
    insert(
        conn,
        make_item(
            numero="22222222000102-1-000002/2026",
            objeto="Serviços de limpeza predial",
            municipio="Olinda",
            valor=50000.0,
        ),
    )


def test_explicit_id_uses_structured_lookup(conn) -> None:
    seed(conn)
    answer = answer_question(
        conn, f"Qual o valor total estimado da contratação {NUMERO}?", mode="bm25"
    )
    assert not answer.refused
    assert "R$ 120.000,50" in answer.text
    assert answer.citations[0].numero_controle_pncp == NUMERO


def test_explicit_id_outside_corpus_is_refused(conn) -> None:
    seed(conn)
    answer = answer_question(
        conn, "Qual o prazo de propostas da contratação 00000000000000-1-000000/2031?"
    )
    assert answer.refused
    assert answer.reason == "id_ausente_no_corpus"


def test_natural_question_returns_citations(conn) -> None:
    seed(conn)
    answer = answer_question(conn, "Quais contratações de merenda escolar em Recife?", mode="bm25")
    assert not answer.refused
    assert answer.citations
    assert answer.citations[0].numero_controle_pncp == NUMERO


def test_unknown_subject_is_refused(conn) -> None:
    seed(conn)
    answer = answer_question(conn, "Há contratação para compra de submarinos nucleares em Olinda?")
    assert answer.refused
    assert answer.reason == "evidencia_insuficiente"


def test_future_year_is_refused_by_temporal_scope(conn) -> None:
    seed(conn)
    answer = answer_question(conn, "Qual o valor da compra de tratores autônomos em 2031?")
    assert answer.refused
    assert answer.reason == "fora_do_escopo_temporal"


def test_past_year_before_corpus_is_refused_too(conn) -> None:
    """Simetria do portão: 2019 é tão fora do corpus quanto 2031."""
    seed(conn)
    answer = answer_question(conn, "Quais contratos de merenda escolar existiam em 2019?")
    assert answer.refused
    assert answer.reason == "fora_do_escopo_temporal"


def test_singular_question_cites_only_the_best_source(conn) -> None:
    seed(conn)
    answer = answer_question(conn, "Qual contratação de merenda escolar em Recife?", mode="bm25")
    assert not answer.refused
    assert len(answer.citations) == 1
    assert answer.citations[0].numero_controle_pncp == NUMERO


def test_deadline_field(conn) -> None:
    seed(conn)
    answer = answer_question(conn, f"Quando encerra o prazo de propostas da contratação {NUMERO}?")
    assert "10/11/2026" in answer.text
