"""Interface de linha de comando do RAG Zero."""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from rag_zero import __version__, db, retrieval
from rag_zero.answer import answer_question
from rag_zero.chunking import format_brl
from rag_zero.config import settings
from rag_zero.embeddings import FastEmbedder, index_pending
from rag_zero.evals.golden import build_golden, load_golden, save_golden, validate_golden
from rag_zero.evals.runner import evaluate_all, save_reports
from rag_zero.ingest import ingest_period
from rag_zero.pncp.client import MODALIDADES, PNCPClient, PNCPError

# Windows: força UTF-8 na saída — órgãos e objetos de licitação vivem cheios de acento.
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8", errors="replace")

app = typer.Typer(
    help="RAG Zero — busca inteligente em licitações públicas (PNCP).",
    no_args_is_help=True,
)
pncp_app = typer.Typer(help="Comandos da fonte PNCP.", no_args_is_help=True)
evals_app = typer.Typer(help="Golden set e avaliação.", no_args_is_help=True)
app.add_typer(pncp_app, name="pncp")
app.add_typer(evals_app, name="evals")

console = Console()


def _open_db(db_path: str | None):
    conn = db.connect(db_path)
    db.init_db(conn)
    return conn


@app.command()
def version() -> None:
    """Mostra a versão."""
    console.print(f"rag-zero {__version__}")


@app.command()
def status(db_path: str | None = typer.Option(None, help="Caminho do SQLite.")) -> None:
    """Resumo do corpus indexado."""
    conn = _open_db(db_path)
    total = db.count_contratacoes(conn)
    chunks = db.count_chunks(conn)
    pending = db.count_pending_embeddings(conn)
    console.print(
        f"contratações: [bold]{total}[/bold] · chunks: [bold]{chunks}[/bold] · "
        f"sem embedding: [bold]{pending}[/bold]"
    )
    if total:
        rows = conn.execute(
            "SELECT uf, count(*) AS n FROM contratacoes GROUP BY uf ORDER BY n DESC LIMIT 8"
        )
        console.print(" · ".join(f"{row['uf'] or '—'}: {row['n']}" for row in rows))


@pncp_app.command()
def smoke(
    uf: str = typer.Option("PE", help="UF (ex.: PE, SP)."),
    dias: int = typer.Option(7, help="Janela de publicação em dias, terminando hoje."),
    modalidade: int = typer.Option(6, help="Código da modalidade (6 = Pregão Eletrônico)."),
) -> None:
    """Teste de fumaça: uma página real do PNCP + resumo na tela."""
    hoje = date.today()
    inicio = hoje - timedelta(days=dias)

    try:
        with PNCPClient() as client:
            page = client.contratacoes_por_publicacao(inicio, hoje, modalidade, uf=uf)
    except PNCPError as exc:
        console.print(f"[red]Erro PNCP: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    nome_modalidade = MODALIDADES.get(modalidade, "?")
    console.print(
        f"PNCP respondeu: [bold]{page.total_registros}[/bold] contratações — "
        f"{uf.upper()} · modalidade {modalidade} ({nome_modalidade}) · {inicio} a {hoje}."
    )

    table = Table()
    table.add_column("Publicação", style="dim")
    table.add_column("Órgão", max_width=30)
    table.add_column("Objeto", max_width=60)
    table.add_column("Valor estimado (R$)", justify="right")
    for item in page.data[:10]:
        orgao = item.orgao_entidade.razao_social if item.orgao_entidade else "?"
        valor = format_brl(item.valor_total_estimado).replace("R$ ", "")
        pub = item.data_publicacao_pncp.strftime("%d/%m/%Y") if item.data_publicacao_pncp else "—"
        table.add_row(
            pub,
            escape((orgao or "?")[:30]),
            escape((item.objeto_compra or "—")[:60]),
            valor,
        )
    console.print(table)

    out_dir = Path(settings.data_dir) / "raw" / "pncp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (
        f"contratacoes_publicacao_{uf.upper()}_{modalidade}_{inicio:%Y%m%d}_{hoje:%Y%m%d}_p1.json"
    )
    out.write_text(
        json.dumps(
            [i.model_dump(by_alias=True, mode="json") for i in page.data],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    console.print(f"[green]Resposta bruta salva em[/green] {out}")


@pncp_app.command()
def dump(
    uf: str = typer.Option("PE", help="UF."),
    dias: int = typer.Option(30, help="Janela de publicação em dias, terminando hoje."),
    modalidade: int = typer.Option(6, help="Código da modalidade (6 = Pregão Eletrônico)."),
    max_paginas: int = typer.Option(20, help="Teto de páginas a percorrer (proteção)."),
) -> None:
    """Baixa todas as páginas do período e salva em JSONL (uma contratação por linha)."""
    hoje = date.today()
    inicio = hoje - timedelta(days=dias)

    out_dir = Path(settings.data_dir) / "raw" / "pncp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"contratacoes_{uf.upper()}_{modalidade}_{inicio:%Y%m%d}_{hoje:%Y%m%d}.jsonl"

    total = 0
    try:
        with PNCPClient() as client, out.open("w", encoding="utf-8") as fh:
            for item in client.iter_contratacoes_por_publicacao(
                inicio, hoje, modalidade, uf=uf, max_paginas=max_paginas
            ):
                fh.write(
                    json.dumps(item.model_dump(by_alias=True, mode="json"), ensure_ascii=False)
                    + "\n"
                )
                total += 1
    except PNCPError as exc:
        console.print(f"[red]Erro PNCP: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]{total}[/green] contratações salvas em {out}")


@app.command()
def ingest(
    uf: str = typer.Option("PE", help="UF."),
    dias: int = typer.Option(90, help="Janela de publicação em dias, terminando hoje."),
    modalidade: int = typer.Option(6, help="Código da modalidade (6 = Pregão Eletrônico)."),
    max_paginas: int = typer.Option(100, help="Teto de páginas (proteção)."),
    db_path: str | None = typer.Option(None, help="Caminho do SQLite."),
) -> None:
    """Ingere o período no banco (idempotente)."""
    hoje = date.today()
    inicio = hoje - timedelta(days=dias)
    conn = _open_db(db_path)
    with console.status(f"Ingerindo {uf.upper()} · modalidade {modalidade} · {inicio} a {hoje}…"):
        stats = ingest_period(
            conn,
            data_inicial=inicio,
            data_final=hoje,
            modalidade=modalidade,
            uf=uf,
            max_paginas=max_paginas,
        )
    console.print(
        f"páginas={stats.pages} · recebidos={stats.fetched} · novos={stats.inserted} · "
        f"atualizados={stats.updated} · inalterados={stats.unchanged} · ignorados={stats.skipped}"
    )
    console.print(
        f"total no banco: [bold]{db.count_contratacoes(conn)}[/bold] contratações · "
        f"[bold]{db.count_chunks(conn)}[/bold] chunks"
    )


@app.command()
def index(
    batch_size: int = typer.Option(32, help="Chunks por lote de embedding."),
    limit: int | None = typer.Option(None, help="Teto de chunks (para testar)."),
    db_path: str | None = typer.Option(None, help="Caminho do SQLite."),
) -> None:
    """Gera embeddings para os chunks pendentes (local, CPU)."""
    conn = _open_db(db_path)
    pending = db.count_pending_embeddings(conn)
    if not pending:
        console.print("[green]Nada a fazer: todos os chunks já têm embedding.[/green]")
        return
    console.print(f"Embeddando [bold]{pending}[/bold] chunks com {settings.embedding_model}…")
    embedder = FastEmbedder()
    total = index_pending(
        conn,
        embedder,
        batch_size=batch_size,
        limit=limit,
        progress=lambda n: console.print(f"  {n} chunks", end="\r"),
    )
    console.print(f"\n[green]{total}[/green] chunks embeddados.")


@app.command()
def search(
    query: str = typer.Argument(..., help="Consulta em linguagem natural."),
    mode: str = typer.Option("bm25", help="bm25 (padrão, medido) | dense | hybrid."),
    k: int = typer.Option(10, help="Quantidade de contratações."),
    db_path: str | None = typer.Option(None, help="Caminho do SQLite."),
) -> None:
    """Busca contratações (sem gerar resposta)."""
    conn = _open_db(db_path)
    embedder = FastEmbedder() if mode in {"dense", "hybrid"} else None
    hits = retrieval.search(conn, query, mode=mode, k=k, embedder=embedder)

    table = Table(title=f"{len(hits)} resultado(s) · modo {mode}")
    table.add_column("#", justify="right", style="dim")
    table.add_column("ID PNCP", style="cyan", no_wrap=True)
    table.add_column("Órgão", max_width=28)
    table.add_column("Local", no_wrap=True)
    table.add_column("Objeto", max_width=52)
    table.add_column("Valor", justify="right", no_wrap=True)
    for position, hit in enumerate(hits, start=1):
        table.add_row(
            str(position),
            hit.numero_controle_pncp or "—",
            escape((hit.orgao or "—")[:28]),
            f"{hit.municipio or '—'}/{hit.uf or '—'}",
            escape((hit.objeto or "—")[:52]),
            format_brl(hit.valor).replace("R$ ", ""),
        )
    console.print(table)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Pergunta em linguagem natural."),
    mode: str = typer.Option("bm25", help="bm25 (padrão, medido) | dense | hybrid."),
    k: int = typer.Option(8, help="Candidatos recuperados."),
    db_path: str | None = typer.Option(None, help="Caminho do SQLite."),
) -> None:
    """Responde com evidências e citações — ou recusa."""
    conn = _open_db(db_path)
    embedder = FastEmbedder() if mode in {"dense", "hybrid"} else None
    answer = answer_question(conn, question, mode=mode, k=k, embedder=embedder)

    style = "yellow" if answer.refused else "green"
    console.print(Text(answer.text, style=style))
    if answer.citations:
        console.print("\n[dim]Fontes:[/dim]")
        for citation in answer.citations:
            console.print(
                f"  · {citation.numero_controle_pncp} · {citation.link or 'sem link'}",
                markup=False,
            )
    console.print(
        f"\n[dim]confiança={answer.confidence:.2f} · {answer.latency_ms:.0f} ms · "
        f"modo={mode}{' · RECUSA: ' + answer.reason if answer.reason else ''}[/dim]"
    )


@evals_app.command("build")
def evals_build(
    n_estrutural: int = typer.Option(24, help="Itens estruturais."),
    n_retrieval: int = typer.Option(30, help="Itens de retrieval."),
    seed: int = typer.Option(42, help="Semente da amostragem."),
    out: str = typer.Option("evals/golden_set.yaml", help="Arquivo de saída."),
    db_path: str | None = typer.Option(None, help="Caminho do SQLite."),
) -> None:
    """Gera o golden set a partir do corpus (itens verificáveis)."""
    conn = _open_db(db_path)
    if db.count_contratacoes(conn) == 0:
        console.print("[red]Corpus vazio: rode `rag-zero ingest` antes.[/red]")
        raise typer.Exit(code=1)
    items = build_golden(conn, n_estrutural=n_estrutural, n_retrieval=n_retrieval, seed=seed)
    path = save_golden(items, out)
    console.print(f"[green]{len(items)} itens salvos em {path}[/green]")


@evals_app.command("run")
def evals_run(
    golden: str = typer.Option("evals/golden_set.yaml", help="Golden set."),
    modes: str = typer.Option("bm25,dense,hybrid", help="Modos separados por vírgula."),
    k: int = typer.Option(10, help="Profundidade de avaliação."),
    out: str = typer.Option("evals/results", help="Diretório de resultados."),
    db_path: str | None = typer.Option(None, help="Caminho do SQLite."),
) -> None:
    """Roda o golden set nos modos escolhidos e grava as métricas."""
    conn = _open_db(db_path)
    items = load_golden(golden)
    missing = validate_golden(conn, items)
    if missing:
        console.print(
            f"[yellow]Aviso: {len(missing)} fonte(s) esperada(s) não estão no corpus atual "
            f"(ex.: {missing[0]}). O golden set está defasado em relação à ingestão.[/yellow]"
        )
    mode_list = [mode.strip() for mode in modes.split(",") if mode.strip()]
    embedder = FastEmbedder() if any(m in {"dense", "hybrid"} for m in mode_list) else None

    reports = []
    for mode in mode_list:
        with console.status(f"Avaliando modo {mode} ({len(items)} itens)…"):
            report = evaluate_all(conn, items, modes=[mode], k=k, embedder=embedder)[0]
        reports.append(report)

    table = Table(title=f"Golden set · k={k}")
    table.add_column("modo")
    for column in (f"recall@{k}", f"MRR@{k}", "cit. precision", "estrutural", "recusa", "p95 ms"):
        table.add_column(column, justify="right")
    for report in reports:
        metrics = report.metrics
        table.add_row(
            report.mode,
            f"{(metrics[f'recall@{k}'] or 0):.3f}",
            f"{(metrics[f'mrr@{k}'] or 0):.3f}",
            f"{(metrics['citation_precision'] or 0):.3f}",
            f"{(metrics['structural_accuracy'] or 0):.3f}",
            f"{(metrics['refusal_accuracy'] or 0):.3f}",
            f"{report.latencies_ms['p95']:.0f}" if report.latencies_ms else "—",
        )
    console.print(table)
    path = save_reports(reports, out, meta={"k": k, "golden": golden})
    console.print(f"[green]Resultados salvos em {path}[/green]")
