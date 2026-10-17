"""Interface de linha de comando do RAG Zero."""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from rag_zero import __version__
from rag_zero.config import settings
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
app.add_typer(pncp_app, name="pncp")

console = Console()


@app.command()
def version() -> None:
    """Mostra a versão."""
    console.print(f"rag-zero {__version__}")


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
        valor = f"{item.valor_total_estimado:,.2f}" if item.valor_total_estimado else "—"
        pub = item.data_publicacao_pncp.strftime("%d/%m/%Y") if item.data_publicacao_pncp else "—"
        table.add_row(pub, (orgao or "?")[:30], (item.objeto_compra or "—")[:60], valor)
    console.print(table)

    if page.data and page.data[0].numero_controle_pncp is None:
        console.print(
            "[yellow]Aviso: 'numeroControlePNCP' não veio no payload — "
            "usar o JSON bruto abaixo para calibrar os modelos.[/yellow]"
        )

    salvar = settings.data_dir
    out_dir = Path(salvar) / "raw" / "pncp"
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
