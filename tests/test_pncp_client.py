from datetime import date
from types import SimpleNamespace

import httpx
import pytest

from rag_zero.pncp import client as pncp
from rag_zero.pncp.client import PNCPClient, PNCPError

START = date(2026, 1, 2)
END = date(2026, 1, 31)


@pytest.fixture
def stub_api(monkeypatch):
    sleeps = []
    client_class = httpx.Client
    monkeypatch.setattr(pncp, "time", SimpleNamespace(monotonic=lambda: 100.0, sleep=sleeps.append))

    def connect(handler):
        def mock_client(**kwargs):
            return client_class(transport=httpx.MockTransport(handler), trust_env=False, **kwargs)

        monkeypatch.setattr(pncp.httpx, "Client", mock_client)
        return PNCPClient(base_url="https://pncp.invalid/api/consulta", min_interval=0)

    return connect, sleeps


def test_iterates_pages_until_last(stub_api):
    connect, sleeps = stub_api
    seen = []

    def respond(request):
        assert request.url.path.endswith("/v1/contratacoes/publicacao")
        params = request.url.params
        assert params["dataInicial"] == "20260102"
        assert params["dataFinal"] == "20260131"
        assert params["codigoModalidadeContratacao"] == "6"
        assert params["tamanhoPagina"] == "10"
        assert params["uf"] == "SP"
        page = int(params["pagina"])
        seen.append(page)
        return httpx.Response(
            200,
            json={
                "data": [{"numeroControlePNCP": f"synthetic-{page}"}],
                "paginasRestantes": 3 - page,
            },
        )

    with connect(respond) as client:
        rows = list(client.iter_contratacoes_por_publicacao(START, END, 6, "sp", tamanho_pagina=10))
    assert seen == [1, 2, 3]
    assert [row.numero_controle_pncp for row in rows] == [
        "synthetic-1",
        "synthetic-2",
        "synthetic-3",
    ]
    assert sleeps == []


@pytest.mark.parametrize("status", [429, 503])
def test_retries_transient_status_then_returns_result(stub_api, status):
    connect, sleeps = stub_api
    responses = iter([httpx.Response(status), httpx.Response(200, json={"data": []})])

    with connect(lambda request: next(responses)) as client:
        assert client.contratacoes_por_publicacao(START, END, 6).data == []
    assert sleeps == [2.0]


def test_stops_retrying_at_limit(stub_api):
    connect, sleeps = stub_api
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(503)

    with connect(respond) as client, pytest.raises(PNCPError, match=r"503.*3"):
        client.contratacoes_por_publicacao(START, END, 6)
    assert len(requests) == 3
    assert sleeps == [2.0, 4.0]


@pytest.mark.parametrize("status", [400, 401, 422])
def test_permanent_error_is_not_retried(stub_api, status):
    connect, sleeps = stub_api
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(status, text="synthetic error")

    with connect(respond) as client, pytest.raises(PNCPError, match=str(status)):
        client.contratacoes_por_publicacao(START, END, 6)
    assert len(requests) == 1
    assert sleeps == []


def test_no_content_is_empty_and_ends_iteration(stub_api):
    connect, sleeps = stub_api
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(204)

    with connect(respond) as client:
        assert list(client.iter_contratacoes_por_publicacao(START, END, 6)) == []
    assert len(requests) == 1
    assert sleeps == []
