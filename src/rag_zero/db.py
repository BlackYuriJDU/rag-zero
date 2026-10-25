"""Camada de armazenamento: SQLite + FTS5 (ver docs/adr/0001-sqlite-first.md).

Por que SQLite e não Postgres/pgvector nesta fase:

- o corpus alvo (milhares de avisos) cabe inteiro em memória: 5k chunks × 384 dims
  em float32 = ~7,7 MB. Distância de cosseno por produto escalar em NumPy resolve em
  poucos milissegundos;
- zero infraestrutura: funciona na máquina de desenvolvimento e no CI sem serviço;
- FTS5 já entrega BM25 com tokenização unicode e remoção de acentos.

A interface (`search`/`get`) é desacoplada: trocar por Postgres+pgvector é substituir
este módulo, não o pipeline.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from rag_zero.config import settings
from rag_zero.pncp.models import Contratacao

SCHEMA = """
CREATE TABLE IF NOT EXISTS contratacoes (
    numero_controle_pncp       TEXT PRIMARY KEY,
    ano_compra                 INTEGER,
    sequencial_compra          INTEGER,
    numero_compra              TEXT,
    processo                   TEXT,
    objeto_compra              TEXT,
    informacao_complementar    TEXT,
    valor_total_estimado       REAL,
    valor_total_homologado     REAL,
    modalidade_id              INTEGER,
    modalidade_nome            TEXT,
    modo_disputa_nome          TEXT,
    situacao_compra_nome       TEXT,
    srp                        INTEGER,
    uf                         TEXT,
    municipio                  TEXT,
    orgao_cnpj                 TEXT,
    orgao_razao_social         TEXT,
    unidade_nome               TEXT,
    data_publicacao            TEXT,
    data_abertura_proposta     TEXT,
    data_encerramento_proposta TEXT,
    link_sistema_origem        TEXT,
    raw                        TEXT NOT NULL,
    fingerprint                TEXT NOT NULL,
    coletado_em                TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE IF NOT EXISTS chunks (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    numero_controle_pncp  TEXT NOT NULL
                          REFERENCES contratacoes (numero_controle_pncp) ON DELETE CASCADE,
    seq                   INTEGER NOT NULL,
    secao                 TEXT NOT NULL DEFAULT 'ficha',
    conteudo              TEXT NOT NULL,
    fts_text              TEXT NOT NULL,
    tokens                INTEGER NOT NULL,
    embedding             BLOB,
    UNIQUE (numero_controle_pncp, seq)
);

CREATE INDEX IF NOT EXISTS idx_chunks_contratacao ON chunks (numero_controle_pncp);
CREATE INDEX IF NOT EXISTS idx_chunks_sem_embedding ON chunks (id) WHERE embedding IS NULL;
CREATE INDEX IF NOT EXISTS idx_contratacoes_uf ON contratacoes (uf, data_publicacao DESC);

-- Tabela FTS independente (não external content): simples de invalidar por chunk_id
-- e barata no nosso volume. `fts_text` é o texto normalizado/steamizado em Python.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    fts_text,
    tokenize = 'unicode61 remove_diacritics 2'
);
"""


@dataclass(frozen=True)
class Chunk:
    seq: int
    secao: str
    conteudo: str
    fts_text: str
    tokens: int


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    if str(db_path) == ":memory:":
        conn = sqlite3.connect(":memory:")
    else:
        path = Path(db_path) if db_path else settings.db_file
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def count_contratacoes(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT count(*) FROM contratacoes").fetchone()[0])


def count_chunks(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT count(*) FROM chunks").fetchone()[0])


def count_pending_embeddings(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT count(*) FROM chunks WHERE embedding IS NULL").fetchone()
    return int(row[0])


def existing_fingerprints(conn: sqlite3.Connection) -> dict[str, str]:
    rows = conn.execute("SELECT numero_controle_pncp, fingerprint FROM contratacoes")
    return {row["numero_controle_pncp"]: row["fingerprint"] for row in rows}


def get_contratacao(conn: sqlite3.Connection, numero_controle_pncp: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM contratacoes WHERE numero_controle_pncp = ?", (numero_controle_pncp,)
    ).fetchone()


def corpus_year_range(conn: sqlite3.Connection) -> tuple[int, int] | None:
    """Menor e maior ano de compra do corpus — base do portão de escopo temporal."""
    row = conn.execute(
        "SELECT min(ano_compra), max(ano_compra) FROM contratacoes WHERE ano_compra IS NOT NULL"
    ).fetchone()
    if row is None or row[0] is None:
        return None
    return int(row[0]), int(row[1])


def _contratacao_values(item: Contratacao, fingerprint: str) -> tuple[object, ...]:
    orgao = item.orgao_entidade
    unidade = item.unidade_orgao
    return (
        item.numero_controle_pncp,
        item.ano_compra,
        item.sequencial_compra,
        item.numero_compra,
        item.processo,
        item.objeto_compra,
        item.informacao_complementar,
        item.valor_total_estimado,
        item.valor_total_homologado,
        item.modalidade_nome,
        item.modalidade_id,
        item.modo_disputa_nome,
        item.situacao_compra_nome,
        item.srp,
        unidade.uf_sigla if unidade else None,
        unidade.municipio_nome if unidade else None,
        orgao.cnpj if orgao else None,
        orgao.razao_social if orgao else None,
        unidade.nome_unidade if unidade else None,
        item.data_publicacao_pncp.isoformat() if item.data_publicacao_pncp else None,
        item.data_abertura_proposta.isoformat() if item.data_abertura_proposta else None,
        item.data_encerramento_proposta.isoformat() if item.data_encerramento_proposta else None,
        item.link_sistema_origem,
        json.dumps(item.model_dump(by_alias=True, mode="json"), ensure_ascii=False),
        fingerprint,
    )


UPSERT = """
INSERT INTO contratacoes (
    numero_controle_pncp, ano_compra, sequencial_compra, numero_compra, processo,
    objeto_compra, informacao_complementar, valor_total_estimado, valor_total_homologado,
    modalidade_nome, modalidade_id, modo_disputa_nome, situacao_compra_nome, srp, uf, municipio,
    orgao_cnpj, orgao_razao_social, unidade_nome, data_publicacao, data_abertura_proposta,
    data_encerramento_proposta, link_sistema_origem, raw, fingerprint
) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
ON CONFLICT (numero_controle_pncp) DO UPDATE SET
    ano_compra = excluded.ano_compra,
    sequencial_compra = excluded.sequencial_compra,
    numero_compra = excluded.numero_compra,
    processo = excluded.processo,
    objeto_compra = excluded.objeto_compra,
    informacao_complementar = excluded.informacao_complementar,
    valor_total_estimado = excluded.valor_total_estimado,
    valor_total_homologado = excluded.valor_total_homologado,
    modalidade_nome = excluded.modalidade_nome,
    modalidade_id = excluded.modalidade_id,
    modo_disputa_nome = excluded.modo_disputa_nome,
    situacao_compra_nome = excluded.situacao_compra_nome,
    srp = excluded.srp,
    uf = excluded.uf,
    municipio = excluded.municipio,
    orgao_cnpj = excluded.orgao_cnpj,
    orgao_razao_social = excluded.orgao_razao_social,
    unidade_nome = excluded.unidade_nome,
    data_publicacao = excluded.data_publicacao,
    data_abertura_proposta = excluded.data_abertura_proposta,
    data_encerramento_proposta = excluded.data_encerramento_proposta,
    link_sistema_origem = excluded.link_sistema_origem,
    raw = excluded.raw,
    fingerprint = excluded.fingerprint,
    coletado_em = strftime('%Y-%m-%dT%H:%M:%fZ','now')
"""


def replace_chunks(
    conn: sqlite3.Connection, numero_controle_pncp: str, chunks: Sequence[Chunk]
) -> None:
    """Troca todos os chunks de uma contratação (idempotente).

    Os embeddings antigos são descartados de propósito: se o texto mudou, o vetor
    antigo não representa mais o conteúdo. O comando `index` re-embedda o que falta.
    """
    old_ids = [
        row[0]
        for row in conn.execute(
            "SELECT id FROM chunks WHERE numero_controle_pncp = ?", (numero_controle_pncp,)
        )
    ]
    for chunk_id in old_ids:
        conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk_id,))
    conn.execute("DELETE FROM chunks WHERE numero_controle_pncp = ?", (numero_controle_pncp,))

    for chunk in chunks:
        cur = conn.execute(
            """INSERT INTO chunks (numero_controle_pncp, seq, secao, conteudo, fts_text, tokens)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                numero_controle_pncp,
                chunk.seq,
                chunk.secao,
                chunk.conteudo,
                chunk.fts_text,
                chunk.tokens,
            ),
        )
        conn.execute(
            "INSERT INTO chunks_fts (chunk_id, fts_text) VALUES (?, ?)",
            (cur.lastrowid, chunk.fts_text),
        )


def upsert_contratacao(
    conn: sqlite3.Connection,
    item: Contratacao,
    *,
    fingerprint: str,
    chunks: Sequence[Chunk],
    existed: bool,
) -> str:
    """Grava a contratação e, se o conteúdo mudou, refaz seus chunks.

    Retorna: ``inserted`` | ``updated``. Quem decide se o texto mudou é o chamador
    (comparando fingerprints); esta função só é chamada para itens novos ou alterados.
    """
    conn.execute(UPSERT, _contratacao_values(item, fingerprint))
    replace_chunks(conn, item.numero_controle_pncp, chunks)
    return "updated" if existed else "inserted"


def iter_contratacoes(conn: sqlite3.Connection) -> Iterable[sqlite3.Row]:
    yield from conn.execute("SELECT * FROM contratacoes ORDER BY numero_controle_pncp")


def pending_embedding_chunks(
    conn: sqlite3.Connection, limit: int | None = None
) -> list[sqlite3.Row]:
    sql = "SELECT id, conteudo FROM chunks WHERE embedding IS NULL ORDER BY id"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return list(conn.execute(sql))


def set_embeddings(conn: sqlite3.Connection, pairs: Sequence[tuple[int, bytes]]) -> None:
    conn.executemany("UPDATE chunks SET embedding = ? WHERE id = ?", [(b, i) for i, b in pairs])


def embedding_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(conn.execute("SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL"))


def chunk_row(conn: sqlite3.Connection, chunk_id: int) -> sqlite3.Row | None:
    return conn.execute(
        """SELECT c.*, k.uf, k.municipio, k.orgao_razao_social, k.objeto_compra,
                  k.valor_total_estimado, k.data_encerramento_proposta,
                  k.modalidade_nome, k.link_sistema_origem
           FROM chunks c JOIN contratacoes k USING (numero_controle_pncp)
           WHERE c.id = ?""",
        (chunk_id,),
    ).fetchone()
