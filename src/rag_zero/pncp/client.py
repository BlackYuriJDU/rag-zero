"""Cliente mínimo e educado da API pública de consulta do PNCP.

Quirks documentados da API (aprendidos na documentação oficial e respeitados aqui):

- Datas no formato ``yyyyMMdd`` (sem hífens).
- ``pagina`` começa em 1 e é obrigatória nas listagens (omitir = 400).
- ``tamanhoPagina``: 10 a 50 em contratações (até 500 em atas/contratos).
- ``204 No Content`` = nenhum resultado (não é erro).
- ``400/422`` = erro do cliente (sem retry); ``5xx/429`` = retry com backoff.
- Respostas paginadas vêm com ``totalRegistros``, ``totalPaginas``,
  ``numeroPagina`` e ``paginasRestantes``.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import date

import httpx

from rag_zero.config import settings
from rag_zero.pncp.models import Contratacao, Page

BASE_URL = "https://pncp.gov.br/api/consulta"

# Códigos oficiais de modalidade de contratação (Lei 14.133/2021).
MODALIDADES: dict[int, str] = {
    1: "Leilão - Eletrônico",
    2: "Diálogo Competitivo",
    3: "Concurso",
    4: "Concorrência - Eletrônica",
    5: "Concorrência - Presencial",
    6: "Pregão - Eletrônico",
    7: "Pregão - Presencial",
    8: "Dispensa",
    9: "Inexigibilidade",
    10: "Manifestação de Interesse",
    11: "Pré-qualificação",
    12: "Credenciamento",
    13: "Leilão - Presencial",
}


class PNCPError(RuntimeError):
    """Erro ao conversar com a API do PNCP."""


class PNCPClient:
    def __init__(
        self,
        base_url: str = BASE_URL,
        timeout: float = 30.0,
        min_interval: float | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            headers={"User-Agent": settings.user_agent, "Accept": "application/json"},
            follow_redirects=True,
        )
        self._min_interval = (
            min_interval if min_interval is not None else settings.min_request_interval
        )
        self._last_request_at = 0.0

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> PNCPClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # --- internals -----------------------------------------------------------

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

    def _request(
        self, path: str, params: dict[str, object], *, retries: int = 3
    ) -> httpx.Response | None:
        """GET com retry em 5xx/429. Retorna ``None`` para 204 (sem resultados)."""
        delay = 2.0
        for attempt in range(1, retries + 1):
            self._throttle()
            try:
                response = self._client.get(path, params=params)
            except httpx.TransportError as exc:
                if attempt == retries:
                    raise PNCPError(f"falha de rede após {retries} tentativas: {exc}") from exc
                time.sleep(delay)
                delay *= 2
                continue

            self._last_request_at = time.monotonic()

            if response.status_code == 204:
                return None
            if response.status_code in (400, 422):
                raise PNCPError(
                    f"PNCP rejeitou os parâmetros ({response.status_code}): {response.text[:300]}"
                )
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == retries:
                    raise PNCPError(
                        f"PNCP indisponível ({response.status_code}) após {retries} tentativas"
                    )
                time.sleep(delay)
                delay *= 2
                continue
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise PNCPError(f"resposta inesperada do PNCP: {exc}") from exc
            return response

        raise PNCPError("fluxo de retry inconsistente")  # inalcançável

    # --- public API ----------------------------------------------------------

    def contratacoes_por_publicacao(
        self,
        data_inicial: date,
        data_final: date,
        modalidade: int,
        pagina: int = 1,
        tamanho_pagina: int = 50,
        uf: str | None = None,
    ) -> Page:
        """Contratações publicadas no período, por modalidade e UF opcional."""
        params: dict[str, object] = {
            "dataInicial": data_inicial.strftime("%Y%m%d"),
            "dataFinal": data_final.strftime("%Y%m%d"),
            "codigoModalidadeContratacao": modalidade,
            "pagina": pagina,
            "tamanhoPagina": tamanho_pagina,
        }
        if uf:
            params["uf"] = uf.upper()
        response = self._request("/v1/contratacoes/publicacao", params)
        if response is None:
            return Page()
        return Page.model_validate(response.json())

    def iter_contratacoes_por_publicacao(
        self,
        data_inicial: date,
        data_final: date,
        modalidade: int,
        uf: str | None = None,
        tamanho_pagina: int = 50,
        max_paginas: int = 100,
    ) -> Iterator[Contratacao]:
        """Itera por todas as páginas do período, respeitando o throttle."""
        pagina = 1
        while pagina <= max_paginas:
            page = self.contratacoes_por_publicacao(
                data_inicial,
                data_final,
                modalidade,
                pagina=pagina,
                tamanho_pagina=tamanho_pagina,
                uf=uf,
            )
            yield from page.data
            if page.paginas_restantes <= 0 or not page.data:
                break
            pagina += 1
