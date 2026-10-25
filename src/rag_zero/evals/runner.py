"""Executa o golden set e produz métricas por configuração de retrieval."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from rag_zero.answer import answer_question
from rag_zero.config import settings
from rag_zero.embeddings import Embedder
from rag_zero.evals.golden import GoldenItem
from rag_zero.evals.metrics import (
    citation_precision,
    contains_answer,
    mean,
    percentile,
    recall_at_k,
    reciprocal_rank,
)
from rag_zero.retrieval import DocumentFrequency, EmbeddingStore


def _aggregate(
    recall: list[float],
    mrr: list[float],
    precision: list[float],
    structural: list[float],
    refusals: list[float],
    *,
    k: int,
) -> dict[str, float | None]:
    return {
        f"recall@{k}": mean(recall),
        f"mrr@{k}": mean(mrr),
        "citation_precision": mean(precision),
        "structural_accuracy": mean(structural),
        "refusal_accuracy": mean(refusals),
    }


@dataclass
class EvalReport:
    mode: str
    k: int
    counts: dict[str, int]
    metrics: dict[str, float | None]
    latencies_ms: dict[str, float] | None = None
    by_subtipo: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "k": self.k,
            "counts": self.counts,
            "metrics": self.metrics,
            "latencies_ms": self.latencies_ms,
            "by_subtipo": self.by_subtipo,
        }


def evaluate(
    conn,
    items: list[GoldenItem],
    *,
    mode: str = "hybrid",
    k: int = 10,
    embedder: Embedder | None = None,
) -> EvalReport:
    store = EmbeddingStore(conn) if mode in {"dense", "hybrid"} else None
    df = DocumentFrequency(conn)

    buckets: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"recall": [], "mrr": [], "precision": [], "structural": [], "refusal": []}
    )
    latencies: list[float] = []
    false_refusals = 0
    bucket_counts: dict[str, int] = defaultdict(int)

    for item in items:
        answer = answer_question(
            conn,
            item.pergunta,
            mode=mode,
            k=k,
            embedder=embedder,
            store=store,
            df=df,
        )
        latencies.append(answer.latency_ms)
        bucket = buckets[item.subtipo or item.tipo]
        bucket_counts[item.subtipo or item.tipo] += 1
        expected_ids = [f["numero_controle_pncp"] for f in item.fontes_esperadas]

        if item.tipo == "sem_resposta":
            bucket["refusal"].append(1.0 if answer.refused else 0.0)
            continue

        if item.tipo == "estrutural":
            bucket["structural"].append(
                1.0
                if (not answer.refused and contains_answer(answer.text, item.resposta_esperada))
                else 0.0
            )
            if answer.refused:
                false_refusals += 1
            continue

        if item.tipo == "retrieval":
            retrieved = [hit.numero_controle_pncp for hit in answer.hits]
            bucket["recall"].append(recall_at_k(retrieved, expected_ids, k))
            bucket["mrr"].append(reciprocal_rank(retrieved, expected_ids, k))
            prec = citation_precision(
                [c.numero_controle_pncp for c in answer.citations], expected_ids
            )
            if prec is not None:
                bucket["precision"].append(prec)
            if answer.refused:
                false_refusals += 1

    metric_names = ("recall", "mrr", "precision", "structural", "refusal")
    merged = {
        name: [value for values in buckets.values() for value in values[name]]
        for name in metric_names
    }

    by_subtipo = {}
    for name, values in buckets.items():
        by_subtipo[name] = {
            "n": bucket_counts[name],
            **_aggregate(
                values["recall"],
                values["mrr"],
                values["precision"],
                values["structural"],
                values["refusal"],
                k=k,
            ),
        }

    return EvalReport(
        mode=mode,
        k=k,
        counts={
            "estrutural": sum(1 for i in items if i.tipo == "estrutural"),
            "retrieval": sum(1 for i in items if i.tipo == "retrieval"),
            "sem_resposta": sum(1 for i in items if i.tipo == "sem_resposta"),
        },
        metrics={
            **_aggregate(
                merged["recall"],
                merged["mrr"],
                merged["precision"],
                merged["structural"],
                merged["refusal"],
                k=k,
            ),
            "false_refusals": float(false_refusals),
        },
        latencies_ms={
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "mean": mean(latencies) or 0.0,
        },
        by_subtipo=by_subtipo,
    )


def evaluate_all(
    conn,
    items: list[GoldenItem],
    *,
    modes: list[str] | None = None,
    k: int = 10,
    embedder: Embedder | None = None,
) -> list[EvalReport]:
    modes = modes or ["bm25", "dense", "hybrid"]
    reports = []
    for mode in modes:
        if mode in {"dense", "hybrid"} and embedder is None:
            raise ValueError(f"modo {mode} exige embedder")
        reports.append(evaluate(conn, items, mode=mode, k=k, embedder=embedder))
    return reports


def save_reports(
    reports: list[EvalReport], out_dir: str | Path, *, meta: dict | None = None
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": {
            "embedding_model": settings.embedding_model,
            "rrf_k": settings.rrf_k,
            "min_evidence_coverage": settings.min_evidence_coverage,
            "min_dense_similarity": settings.min_dense_similarity,
            **(meta or {}),
        },
        "reports": [report.to_dict() for report in reports],
    }
    path = out_dir / "latest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
