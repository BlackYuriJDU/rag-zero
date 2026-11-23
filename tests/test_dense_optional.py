"""O modo denso é um extra opcional: sem fastembed, a falha tem de ser clara."""

from __future__ import annotations

import importlib.util

import pytest

from rag_zero.embeddings import DenseUnavailableError, FastEmbedder


def test_fastembedder_explains_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, *args, **kwargs):
        return None if name == "fastembed" else real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    with pytest.raises(DenseUnavailableError, match=r"rag-zero\[dense\]"):
        FastEmbedder()
