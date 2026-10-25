"""Testes do runner de avaliação — agregação e contagem por subtipo.

Regressão real: `n` somava o tamanho das listas de métricas (recall + MRR +
precisão), reportando 90 itens onde havia 30.
"""

from __future__ import annotations

from rag_zero.answer import Answer, Citation
from rag_zero.evals import runner
from rag_zero.evals.golden import GoldenItem


class _Hit:
    def __init__(self, numero: str) -> None:
        self.numero_controle_pncp = numero
        self.secao = "ficha"
        self.link = None


def test_by_subtipo_counts_items_not_metric_values(monkeypatch, conn) -> None:
    items = [
        GoldenItem(
            id="a",
            tipo="retrieval",
            pergunta="q1",
            resposta_esperada="X",
            fontes_esperadas=[{"numero_controle_pncp": "X"}],
            subtipo="verbatim",
        ),
        GoldenItem(
            id="b",
            tipo="retrieval",
            pergunta="q2",
            resposta_esperada="Y",
            fontes_esperadas=[{"numero_controle_pncp": "Y"}],
            subtipo="verbatim",
        ),
    ]

    def fake_answer(conn, question, **kwargs):
        numero = "X" if question == "q1" else "Y"
        return Answer(
            text="ok",
            citations=[Citation(numero_controle_pncp=numero)],
            hits=[_Hit(numero)],
            latency_ms=1.0,
        )

    monkeypatch.setattr(runner, "answer_question", fake_answer)
    report = runner.evaluate(conn, items, mode="bm25", k=10)

    assert report.by_subtipo["verbatim"]["n"] == 2
    assert report.by_subtipo["verbatim"]["recall@10"] == 1.0
    assert report.by_subtipo["verbatim"]["citation_precision"] == 1.0


def test_metric_lists_stay_empty_for_other_types(monkeypatch, conn) -> None:
    items = [
        GoldenItem(
            id="u",
            tipo="sem_resposta",
            pergunta="pergunta impossível",
            resposta_esperada=None,
            subtipo="unanswerable",
        )
    ]

    def fake_answer(conn, question, **kwargs):
        return Answer(text="recuso", refused=True, reason="evidencia_insuficiente", latency_ms=1.0)

    monkeypatch.setattr(runner, "answer_question", fake_answer)
    report = runner.evaluate(conn, items, mode="bm25", k=10)

    bucket = report.by_subtipo["unanswerable"]
    assert bucket["n"] == 1
    assert bucket["refusal_accuracy"] == 1.0
    assert bucket["recall@10"] is None
