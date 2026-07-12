"""Teste de fumaça da API (multi-modelo).

Roda mesmo sem modelos treinados: valida que a aplicação sobe e que os
endpoints de previsão exigem a escolha de um ticker.
"""
from fastapi.testclient import TestClient

from src.api.main import app


def test_health_endpoint():
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "available_symbols" in body
        assert isinstance(body["available_symbols"], list)


def test_predict_requires_valid_symbol():
    with TestClient(app) as client:
        # ticker inexistente -> 404 (sem modelo) ou 503 (nenhum carregado)
        resp = client.post(
            "/predict", json={"symbol": "ZZZZ", "prices": [1, 2, 3], "horizon": 1}
        )
        assert resp.status_code in (404, 503, 422)


def test_predict_missing_symbol_is_rejected():
    with TestClient(app) as client:
        # sem o campo obrigatório `symbol` -> 422 de validação
        resp = client.post("/predict", json={"prices": [1, 2, 3]})
        assert resp.status_code == 422
