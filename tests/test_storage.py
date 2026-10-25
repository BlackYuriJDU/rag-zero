"""Testes de armazenamento e idempotência da ingestão."""

from rag_zero import db
from rag_zero.chunking import build_chunks, fts_text
from tests.conftest import insert, make_item


def test_upsert_creates_contract_and_chunk(conn) -> None:
    item = make_item()
    insert(conn, item)
    assert db.count_contratacoes(conn) == 1
    assert db.count_chunks(conn) == 1
    row = db.get_contratacao(conn, "12345678000196-1-000123/2026")
    assert row is not None and row["uf"] == "PE"


def test_fts_row_is_searchable_with_stemming(conn) -> None:
    insert(conn, make_item())
    match = fts_text("merenda escolar")
    hits = conn.execute(
        "SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ?", (match,)
    ).fetchall()
    assert len(hits) == 1


def test_reinsert_same_contract_replaces_chunks(conn) -> None:
    item = make_item()
    insert(conn, item)
    first = conn.execute("SELECT id FROM chunks").fetchone()[0]

    fp, chunks = build_chunks(item)
    db.upsert_contratacao(conn, item, fingerprint=fp, chunks=chunks, existed=True)
    assert db.count_chunks(conn) == 1
    assert conn.execute("SELECT count(*) FROM chunks_fts").fetchone()[0] == 1
    assert conn.execute("SELECT id FROM chunks").fetchone()[0] != first


def test_embedding_roundtrip(conn) -> None:
    insert(conn, make_item())
    chunk_id = conn.execute("SELECT id FROM chunks").fetchone()[0]
    blob = b"\x00\x00\x80\x3f"  # 1.0 em float32 little-endian
    db.set_embeddings(conn, [(chunk_id, blob)])
    rows = db.embedding_rows(conn)
    assert len(rows) == 1 and rows[0]["embedding"] == blob
    assert db.count_pending_embeddings(conn) == 0
