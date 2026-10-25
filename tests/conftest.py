"""Fixtures compartilhadas — corpus sintético, sem rede e sem modelo."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from rag_zero import db
from rag_zero.chunking import build_chunks
from rag_zero.pncp.models import Contratacao


def make_item(
    numero: str = "12345678000196-1-000123/2026",
    objeto: str = "Aquisição de gêneros alimentícios para merenda escolar",
    valor: float | None = 120000.50,
    municipio: str = "Recife",
    uf: str = "PE",
    modalidade: str = "Pregão - Eletrônico",
    orgao: str = "Prefeitura Municipal do Recife",
    encerramento: str = "2026-11-10T09:00:00",
    processo: str = "0001.000123/2026",
) -> Contratacao:
    return Contratacao.model_validate(
        {
            "numeroControlePNCP": numero,
            "anoCompra": 2026,
            "sequencialCompra": 123,
            "numeroCompra": "001",
            "processo": processo,
            "objetoCompra": objeto,
            "valorTotalEstimado": valor,
            "modalidadeNome": modalidade,
            "modoDisputaNome": "Aberto",
            "situacaoCompraNome": "Divulgada no PNCP",
            "srp": False,
            "dataPublicacaoPncp": "2026-10-20T10:00:00",
            "dataAberturaProposta": "2026-11-01T09:00:00",
            "dataEncerramentoProposta": encerramento,
            "linkSistemaOrigem": "https://example.test/compras/1",
            "orgaoEntidade": {"cnpj": "12345678000196", "razaoSocial": orgao},
            "unidadeOrgao": {
                "ufSigla": uf,
                "municipioNome": municipio,
                "nomeUnidade": "Secretaria de Educação",
            },
        }
    )


def insert(conn, item: Contratacao) -> None:
    fp, chunks = build_chunks(item)
    db.upsert_contratacao(conn, item, fingerprint=fp, chunks=chunks, existed=False)


@pytest.fixture()
def conn():
    connection = db.connect(":memory:")
    db.init_db(connection)
    yield connection
    connection.close()


class HashingEmbedder:
    """Embedder determinístico para testes: bag-of-words hash, sem modelo."""

    dim = 384

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.lower().split():
                matrix[row, hash(token) % self.dim] += 1.0
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return (matrix / np.clip(norms, 1e-12, None)).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_passages([text])[0]
