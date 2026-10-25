"""Testes de recuperação: BM25, fusão RRF e sinal de evidência."""

from rag_zero.retrieval import (
    DocumentFrequency,
    EmbeddingStore,
    bm25_search,
    evidence_coverage,
    rrf_fuse,
    search,
)
from tests.conftest import HashingEmbedder, insert, make_item

ITEMS = [
    make_item(
        numero="11111111000101-1-000001/2026",
        objeto="Aquisição de gêneros alimentícios para merenda escolar",
        municipio="Recife",
    ),
    make_item(
        numero="22222222000102-1-000002/2026",
        objeto="Contratação de transporte escolar rural",
        municipio="Caruaru",
        valor=90000.0,
    ),
    make_item(
        numero="33333333000103-1-000003/2026",
        objeto="Serviços de limpeza predial e conservação",
        municipio="Olinda",
        valor=50000.0,
    ),
]


def seed(conn) -> None:
    for item in ITEMS:
        insert(conn, item)


def test_bm25_finds_exact_domain_term(conn) -> None:
    seed(conn)
    top = bm25_search(conn, "merenda escolar", k=5)
    assert top
    row = conn.execute(
        "SELECT numero_controle_pncp FROM chunks WHERE id = ?", (top[0][0],)
    ).fetchone()
    assert row[0] == ITEMS[0].numero_controle_pncp


def test_bm25_requires_stemmed_query_consistency(conn) -> None:
    seed(conn)
    assert bm25_search(conn, "merendas", k=5) == bm25_search(conn, "merenda", k=5)


def test_rrf_rewards_agreement(conn) -> None:
    assert rrf_fuse([[1, 2], [1, 3]])[1] > rrf_fuse([[1, 2], [1, 3]])[2]


def test_hybrid_returns_one_hit_per_contract(conn) -> None:
    seed(conn)
    hits = search(conn, "merenda escolar recife", mode="hybrid", k=10, embedder=HashingEmbedder())
    numbers = [hit.numero_controle_pncp for hit in hits]
    assert len(numbers) == len(set(numbers))
    assert numbers[0] == ITEMS[0].numero_controle_pncp


def test_embedding_store_loads_and_searches(conn) -> None:
    seed(conn)
    store = EmbeddingStore(conn)
    embedder = HashingEmbedder()
    from rag_zero.embeddings import to_bytes

    vectors = embedder.embed_passages(
        [row["conteudo"] for row in conn.execute("SELECT conteudo FROM chunks")]
    )
    ids = [row["id"] for row in conn.execute("SELECT id FROM chunks")]
    from rag_zero import db

    db.set_embeddings(
        conn, [(chunk_id, to_bytes(vec)) for chunk_id, vec in zip(ids, vectors, strict=True)]
    )
    assert len(store.load().ids) == 3
    hits = store.search(embedder.embed_query("merenda"), k=2)
    assert len(hits) == 2


def test_coverage_drops_for_unknown_subject(conn) -> None:
    seed(conn)
    df = DocumentFrequency(conn)
    known, _ = evidence_coverage("merenda escolar em Recife", df)
    unknown, missing = evidence_coverage("submarinos nucleares em Olinda", df)
    assert known > unknown
    assert "submarin" in missing
