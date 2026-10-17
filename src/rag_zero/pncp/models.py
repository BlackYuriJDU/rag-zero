"""Modelos do payload da API do PNCP.

Os nomes de campo na API são camelCase; usamos ``alias`` para mapear.
``extra="allow"`` mantém o payload completo por perto — o PNCP evolui o schema
sem aviso, e não queremos perder dados nem quebrar o pipeline por isso.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class OrgaoEntidade(BaseModel):
    model_config = ConfigDict(extra="allow")

    cnpj: str | None = None
    razao_social: str | None = Field(default=None, alias="razaoSocial")


class UnidadeOrgao(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    uf_sigla: str | None = Field(default=None, alias="ufSigla")
    municipio_nome: str | None = Field(default=None, alias="municipioNome")
    nome_unidade: str | None = Field(default=None, alias="nomeUnidade")


class Contratacao(BaseModel):
    """Uma contratação publicada no PNCP."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    numero_controle_pncp: str | None = Field(default=None, alias="numeroControlePNCP")
    ano_compra: int | None = Field(default=None, alias="anoCompra")
    sequencial_compra: int | None = Field(default=None, alias="sequencialCompra")
    numero_compra: str | None = Field(default=None, alias="numeroCompra")
    processo: str | None = Field(default=None, alias="processo")
    srp: bool | None = Field(default=None, alias="srp")
    objeto_compra: str | None = Field(default=None, alias="objetoCompra")
    valor_total_estimado: float | None = Field(default=None, alias="valorTotalEstimado")
    valor_total_homologado: float | None = Field(default=None, alias="valorTotalHomologado")
    modalidade_nome: str | None = Field(default=None, alias="modalidadeNome")
    modo_disputa_nome: str | None = Field(default=None, alias="modoDisputaNome")
    situacao_compra_nome: str | None = Field(default=None, alias="situacaoCompraNome")
    data_publicacao_pncp: datetime | None = Field(default=None, alias="dataPublicacaoPncp")
    data_abertura_proposta: datetime | None = Field(default=None, alias="dataAberturaProposta")
    data_encerramento_proposta: datetime | None = Field(
        default=None, alias="dataEncerramentoProposta"
    )
    link_sistema_origem: str | None = Field(default=None, alias="linkSistemaOrigem")
    orgao_entidade: OrgaoEntidade | None = Field(default=None, alias="orgaoEntidade")
    unidade_orgao: UnidadeOrgao | None = Field(default=None, alias="unidadeOrgao")

    @property
    def uf(self) -> str | None:
        return self.unidade_orgao.uf_sigla if self.unidade_orgao else None


class Page(BaseModel):
    """Envelope paginado retornado pelos endpoints de consulta."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    data: list[Contratacao] = Field(default_factory=list)
    total_registros: int = Field(default=0, alias="totalRegistros")
    total_paginas: int = Field(default=0, alias="totalPaginas")
    numero_pagina: int = Field(default=0, alias="numeroPagina")
    paginas_restantes: int = Field(default=0, alias="paginasRestantes")
    empty: bool = True
