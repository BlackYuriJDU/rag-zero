"""Testes dos modelos do PNCP — garantem que os aliases camelCase do payload funcionam."""

from rag_zero.pncp.models import Page

PAYLOAD = {
    "data": [
        {
            "numeroControlePNCP": "12345678000196-1-000123/2026",
            "objetoCompra": "Aquisição de gêneros alimentícios para merenda escolar",
            "valorTotalEstimado": 120000.50,
            "modalidadeNome": "Pregão - Eletrônico",
            "situacaoCompraNome": "Divulgada no PNCP",
            "dataPublicacaoPncp": "2026-10-14T10:00:00",
            "orgaoEntidade": {"cnpj": "12345678000196", "razaoSocial": "Prefeitura Municipal X"},
            "unidadeOrgao": {"ufSigla": "PE", "municipioNome": "Recife"},
        }
    ],
    "totalRegistros": 1,
    "totalPaginas": 1,
    "numeroPagina": 1,
    "paginasRestantes": 0,
    "empty": False,
}


def test_page_parses_envelope() -> None:
    page = Page.model_validate(PAYLOAD)
    assert page.total_registros == 1
    assert page.paginas_restantes == 0
    assert page.empty is False
    assert len(page.data) == 1


def test_contratacao_aliases_and_property() -> None:
    item = Page.model_validate(PAYLOAD).data[0]
    assert item.objeto_compra is not None and "merenda" in item.objeto_compra
    assert item.valor_total_estimado == 120000.50
    assert item.uf == "PE"
    assert item.orgao_entidade is not None
    assert item.orgao_entidade.razao_social == "Prefeitura Municipal X"


def test_empty_page_defaults() -> None:
    page = Page()
    assert page.data == []
    assert page.total_registros == 0
    assert page.empty is True
