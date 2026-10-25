"""Métricas de avaliação — pequenas, explícitas e testáveis.

Nada de "LLM as judge" nesta camada: recall, MRR, precisão de citação e recusa têm
definição matemática. Juiz LLM entra depois, para faithfulness de texto livre, e
sempre com amostra revisada por humano.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence

from rag_zero.chunking import tokenize


def normalize_answer(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(tokenize(text))


def contains_answer(prediction: str | None, expected: str | None) -> bool:
    """Compara por continência normalizada (a resposta pode citar o contexto)."""
    expected_norm = normalize_answer(expected)
    if not expected_norm:
        return False
    return expected_norm in normalize_answer(prediction)


def recall_at_k(retrieved: Sequence[str], expected: Iterable[str], k: int) -> float:
    expected_set = set(expected)
    if not expected_set:
        return 0.0
    return len(set(retrieved[:k]) & expected_set) / len(expected_set)


def reciprocal_rank(retrieved: Sequence[str], expected: Iterable[str], k: int = 10) -> float:
    expected_set = set(expected)
    for position, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in expected_set:
            return 1.0 / position
    return 0.0


def citation_precision(cited: Sequence[str], expected: Iterable[str]) -> float | None:
    cited_set = set(cited)
    if not cited_set:
        return None
    return len(cited_set & set(expected)) / len(cited_set)


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, math.ceil(p / 100 * len(ordered)) - 1)
    return ordered[rank]


def mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None
