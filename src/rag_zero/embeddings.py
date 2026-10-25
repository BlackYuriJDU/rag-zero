"""Embeddings densos, locais e normalizados.

Escolha do modelo: ``paraphrase-multilingual-MiniLM-L12-v2`` (384 dims, ONNX via
fastembed). É multilíngue o suficiente para PT-BR, roda em CPU e baixa ~220 MB —
não exige GPU nem torch. O modelo é *lazy*: comandos puramente lexicais nunca o
carregam.

Normalizamos os vetores na escrita e na consulta. Assim similaridade de cosseno vira
um produto escalar, e a matriz inteira cabe em memória (5k × 384 × 4 B ≈ 7,7 MB).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from rag_zero import db
from rag_zero.config import settings


class Embedder(Protocol):
    dim: int

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.clip(norms, 1e-12, None)).astype(np.float32)


def to_bytes(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype="<f4").tobytes()


def from_bytes(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype="<f4")


class FastEmbedder:
    """Wrapper fino sobre o fastembed, com carregamento lazy."""

    def __init__(
        self,
        model_name: str | None = None,
        dim: int | None = None,
        threads: int | None = None,
    ) -> None:
        self.model_name = model_name or settings.embedding_model
        self.dim = dim or settings.embedding_dim
        self.threads = threads
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name, threads=self.threads)
        return self._model

    def embed_passages(self, texts: Sequence[str], *, batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        vectors = list(self._ensure_model().embed(list(texts), batch_size=batch_size))
        matrix = np.vstack(vectors)
        if matrix.shape[1] != self.dim:
            raise ValueError(f"dimensão inesperada: {matrix.shape[1]} != {self.dim}")
        return normalize(matrix)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_passages([text], batch_size=1)[0]


def index_pending(
    conn,
    embedder: Embedder,
    *,
    batch_size: int = 32,
    limit: int | None = None,
    progress: callable | None = None,
) -> int:
    """Embedda todos os chunks sem vetor. Retorna quantos foram processados."""
    total = 0
    while True:
        rows = db.pending_embedding_chunks(
            conn, limit=batch_size if limit is None else min(batch_size, limit)
        )
        if not rows:
            break
        vectors = embedder.embed_passages([row["conteudo"] for row in rows], batch_size=batch_size)
        db.set_embeddings(
            conn, [(row["id"], to_bytes(vec)) for row, vec in zip(rows, vectors, strict=True)]
        )
        conn.commit()
        total += len(rows)
        if progress:
            progress(total)
        if limit is not None:
            limit -= len(rows)
            if limit <= 0:
                break
    return total
