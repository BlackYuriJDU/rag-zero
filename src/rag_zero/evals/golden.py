"""Construção e leitura do golden set.

Regra do projeto: nenhum item entra sem ser verificável contra o corpus. Os itens
mecânicos são derivados do banco (a resposta é um campo real, o ID é real); os itens
de paráfrase e os impossíveis são escritos à mão e conferidos por consulta.

Formato (YAML):

    - id: gs-0001
      tipo: estrutural | retrieval | sem_resposta
      pergunta: "..."
      resposta_esperada: "..." | null
      fontes_esperadas:
        - numero_controle_pncp: "..."
          secao: "ficha"
      notas: "..."
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from rag_zero.chunking import format_brl, format_datetime


@dataclass
class GoldenItem:
    id: str
    tipo: str
    pergunta: str
    resposta_esperada: str | None
    fontes_esperadas: list[dict] = field(default_factory=list)
    notas: str = ""
    subtipo: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


STRUCTURAL_TEMPLATES = {
    "valor": "Qual o valor total estimado da contratação {numero}?",
    "encerramento": "Quando encerra o prazo de propostas da contratação {numero}?",
    "orgao": "Qual órgão publicou a contratação {numero}?",
    "modalidade": "Qual a modalidade da contratação {numero}?",
}


def _value_for(field_name: str, row) -> str | None:
    if field_name == "valor":
        return format_brl(row["valor_total_estimado"]) if row["valor_total_estimado"] else None
    if field_name == "encerramento":
        return (
            format_datetime(row["data_encerramento_proposta"])
            if row["data_encerramento_proposta"]
            else None
        )
    if field_name == "orgao":
        return row["orgao_razao_social"] or None
    if field_name == "modalidade":
        return row["modalidade_nome"] or None
    return None


def build_golden(
    conn,
    *,
    n_estrutural: int = 24,
    n_retrieval: int = 30,
    seed: int = 42,
) -> list[GoldenItem]:
    rows = list(conn.execute("SELECT * FROM contratacoes ORDER BY numero_controle_pncp"))
    rng = random.Random(seed)
    rng.shuffle(rows)
    items: list[GoldenItem] = []

    per_field = max(1, n_estrutural // len(STRUCTURAL_TEMPLATES))
    cursor = 0
    for field_name, template in STRUCTURAL_TEMPLATES.items():
        taken = 0
        while taken < per_field and cursor < len(rows):
            row = rows[cursor]
            cursor += 1
            value = _value_for(field_name, row)
            if not value or value == "não informado":
                continue
            items.append(
                GoldenItem(
                    id=f"gs-est-{len(items) + 1:04d}",
                    tipo="estrutural",
                    pergunta=template.format(numero=row["numero_controle_pncp"]),
                    resposta_esperada=value,
                    fontes_esperadas=[
                        {"numero_controle_pncp": row["numero_controle_pncp"], "secao": "ficha"}
                    ],
                    notas=f"campo={field_name}; derivado do banco (verificável)",
                    subtipo="lookup",
                )
            )
            taken += 1

    for row in rows:
        if len([i for i in items if i.tipo == "retrieval"]) >= n_retrieval:
            break
        objeto = (row["objeto_compra"] or "").strip()
        words = objeto.split()
        if len(words) < 8:
            continue
        snippet = " ".join(words[:14])
        items.append(
            GoldenItem(
                id=f"gs-ret-{len(items) + 1:04d}",
                tipo="retrieval",
                pergunta=f'Qual contratação tem como objeto o trecho "{snippet}"?',
                resposta_esperada=row["numero_controle_pncp"],
                fontes_esperadas=[
                    {"numero_controle_pncp": row["numero_controle_pncp"], "secao": "ficha"}
                ],
                notas="consulta verbatim; mede o piso do índice lexical",
                subtipo="verbatim",
            )
        )

    items.extend(curated_items())
    items.extend(unanswerable_items())
    return items


def curated_items() -> list[GoldenItem]:
    """Itens escritos à mão, conferidos por SQL contra o snapshot do corpus.

    Diferente dos itens mecânicos, aqui a pergunta NÃO repete o texto do objeto:
    ela usa sinônimos ("levar estudantes", "calçamento", "remédios"). É o que mede
    generalização semântica de verdade — e o que um golden set sintético não dá.
    """
    definicoes = [
        (
            "gs-cur-0001",
            "Qual contratação em Barra de Guabiraba é para levar estudantes até a escola?",
            "10120962000138-1-000011/2026",
            "paráfrase: 'levar estudantes' vs 'transporte escolar'",
        ),
        (
            "gs-cur-0002",
            "Existe obra de calçamento de ruas em Chã de Alegria?",
            "11049798000182-1-000026/2026",
            "paráfrase: 'calçamento' vs 'pavimentação em paralelepípedos'",
        ),
        (
            "gs-cur-0003",
            "O município de Belém do São Francisco comprou remédios?",
            "10113728000183-1-000033/2026",
            "paráfrase: 'remédios' vs 'medicamentos'",
        ),
        (
            "gs-cur-0004",
            "O que a Secretaria de Saúde de Mirandiba comprou para limpeza?",
            "11043312000107-1-000073/2026",
            "paráfrase: 'para limpeza' vs 'materiais de limpeza'",
        ),
        (
            "gs-cur-0005",
            "Joaquim Nabuco comprou uniforme para os alunos da rede municipal?",
            "20599336000120-1-000009/2026",
            "paráfrase: 'uniforme' vs 'fardamento/uniforme escolar'",
        ),
        (
            "gs-cur-0006",
            "Aliança instalou ar-condicionado e ventiladores nas escolas?",
            "46168650000141-1-000010/2026",
            "paráfrase: 'instalou' vs 'aquisição de equipamentos de climatização'",
        ),
        (
            "gs-cur-0007",
            "Garanhuns comprou notebooks e computadores de mesa?",
            "35872812000101-1-000099/2026",
            "paráfrase: 'notebooks e computadores de mesa' vs 'microcomputadores desktop'",
        ),
        (
            "gs-cur-0008",
            "Arcoverde vai comprar alimentos para as escolas?",
            "19470663000106-1-000010/2026",
            "paráfrase: 'alimentos' vs 'gêneros alimentícios'",
        ),
        (
            "gs-cur-0009",
            "Bezerros contratou roupas padronizadas para os servidores?",
            "10091510000175-1-000057/2026",
            "paráfrase: 'roupas padronizadas' vs 'peças de vestuário padronizadas'",
        ),
        (
            "gs-cur-0010",
            "Ipojuca alugou aparelhos de ar-condicionado?",
            "11294386000108-1-000139/2026",
            "paráfrase: 'alugou' vs 'locação de aparelhos de ar condicionado'",
        ),
        (
            "gs-cur-0011",
            "Amaraji comprou leite especial e suplementos alimentares?",
            "11607836000175-1-000030/2026",
            "paráfrase: 'suplementos alimentares' vs 'leites especiais e suplementos'",
        ),
        (
            "gs-cur-0012",
            "Afogados da Ingazeira comprou uniformes e equipamentos de proteção para eletricistas?",
            "10346096000106-1-000056/2026",
            "paráfrase: 'equipamentos de proteção' vs 'EPI's'",
        ),
    ]
    return [
        GoldenItem(
            id=item_id,
            tipo="retrieval",
            pergunta=pergunta,
            resposta_esperada=numero,
            fontes_esperadas=[{"numero_controle_pncp": numero, "secao": "ficha"}],
            notas=f"curadoria manual; {nota}",
            subtipo="paraphrase",
        )
        for item_id, pergunta, numero, nota in definicoes
    ]


def unanswerable_items() -> list[GoldenItem]:
    return [GoldenItem(**item) for item in UNANSWERABLE]


def _item(id_: str, pergunta: str, nota: str) -> dict:
    return {
        "id": id_,
        "tipo": "sem_resposta",
        "pergunta": pergunta,
        "resposta_esperada": None,
        "fontes_esperadas": [],
        "notas": nota,
        "subtipo": "unanswerable",
    }


UNANSWERABLE = [
    _item(
        "gs-nan-0001",
        "Qual o valor do contrato de compra de dirigíveis para o carnaval do Recife em 2031?",
        "assunto plausível, ausente do corpus e fora do período",
    ),
    _item(
        "gs-nan-0002",
        "Existe licitação aberta para fornecimento de trens-bala entre Recife e Caruaru?",
        "obra inexistente; exige recusa",
    ),
    _item(
        "gs-nan-0003",
        "Qual o valor estimado da compra de 500 tratores autônomos para "
        "Fernando de Noronha em 2031?",
        "combinação improvável; ano fora do corpus",
    ),
    _item(
        "gs-nan-0004",
        "Quem ganhou a concorrência para construir o elevador espacial de Pernambuco?",
        "assunto absurdo porém formulado como pergunta real",
    ),
    _item(
        "gs-nan-0005",
        "Qual o prazo de propostas da contratação 00000000000000-1-000000/2031?",
        "ID sintaticamente válido, inexistente no corpus",
    ),
    _item(
        "gs-nan-0006",
        "Há contratação para compra de submarinos nucleares em Olinda?",
        "termo distintivo ausente",
    ),
]


def save_golden(items: list[GoldenItem], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [item.to_dict() for item in items]
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, width=100),
        encoding="utf-8",
    )
    return path


def load_golden(path: str | Path) -> list[GoldenItem]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    return [GoldenItem(**item) for item in data]


def validate_golden(conn, items: list[GoldenItem]) -> list[str]:
    """IDs esperados que não existem no corpus atual (golden set defasado)."""
    known = {row[0] for row in conn.execute("SELECT numero_controle_pncp FROM contratacoes")}
    missing: list[str] = []
    for item in items:
        for fonte in item.fontes_esperadas:
            numero = fonte.get("numero_controle_pncp")
            if numero and numero not in known:
                missing.append(f"{item.id}:{numero}")
    return missing
