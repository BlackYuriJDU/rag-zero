"""Testes das métricas — a base de qualquer número no README."""

from rag_zero.evals.metrics import (
    citation_precision,
    contains_answer,
    normalize_answer,
    percentile,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_and_mrr() -> None:
    retrieved = ["a", "b", "c"]
    assert recall_at_k(retrieved, {"c", "d"}, 3) == 0.5
    assert reciprocal_rank(retrieved, {"b"}) == 0.5
    assert reciprocal_rank(["x"], {"b"}) == 0.0


def test_citation_precision() -> None:
    assert citation_precision(["a", "b"], {"a", "b"}) == 1.0
    assert citation_precision(["a", "x"], {"a"}) == 0.5
    assert citation_precision([], {"a"}) is None


def test_normalized_containment_ignores_accents_and_currency_format() -> None:
    assert normalize_answer("R$ 120.000,50") == "r 120 000 50"
    assert contains_answer("O valor é R$ 120.000,50.", "r$ 120.000,50")
    assert contains_answer("Órgão: Prefeitura de Recife", "prefeitura de recife")
    assert not contains_answer("R$ 1,00", "R$ 2,00")


def test_percentile_is_nearest_rank() -> None:
    values = [10, 20, 30, 40, 50]
    assert percentile(values, 50) == 30
    assert percentile(values, 95) == 50
    assert percentile([], 95) == 0.0
