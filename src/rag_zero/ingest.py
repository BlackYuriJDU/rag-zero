"""Ingestão idempotente: API pública do PNCP -> SQLite.

Idempotência é requisito, não detalhe: rodar a ingestão duas vezes não pode duplicar
avisos nem re-embeddar tudo. Usamos a chave canônica ``numeroControlePNCP`` e um
fingerprint do texto da ficha; só reindexamos o que realmente mudou.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from rag_zero import db
from rag_zero.chunking import build_chunks
from rag_zero.pncp.client import PNCPClient


@dataclass
class IngestStats:
    pages: int = 0
    fetched: int = 0
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    skipped: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "pages": self.pages,
            "fetched": self.fetched,
            "inserted": self.inserted,
            "updated": self.updated,
            "unchanged": self.unchanged,
            "skipped": self.skipped,
        }


def ingest_period(
    conn,
    *,
    data_inicial: date,
    data_final: date,
    modalidade: int,
    uf: str | None = None,
    max_paginas: int = 100,
    client: PNCPClient | None = None,
) -> IngestStats:
    """Baixa todas as páginas do período e grava/atualiza as contratações."""
    stats = IngestStats()
    known = db.existing_fingerprints(conn)
    owns_client = client is None
    client = client or PNCPClient()
    try:
        pagina = 1
        while pagina <= max_paginas:
            page = client.contratacoes_por_publicacao(
                data_inicial, data_final, modalidade, pagina=pagina, uf=uf
            )
            stats.pages += 1
            stats.fetched += len(page.data)

            for item in page.data:
                numero = item.numero_controle_pncp
                if not numero:
                    stats.skipped += 1
                    continue
                fp, chunks = build_chunks(item)
                if known.get(numero) == fp:
                    stats.unchanged += 1
                    continue
                existed = numero in known
                db.upsert_contratacao(conn, item, fingerprint=fp, chunks=chunks, existed=existed)
                known[numero] = fp
                if existed:
                    stats.updated += 1
                else:
                    stats.inserted += 1

            conn.commit()
            if page.paginas_restantes <= 0 or not page.data:
                break
            pagina += 1
    finally:
        if owns_client:
            client.close()
    return stats
