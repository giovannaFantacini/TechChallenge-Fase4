"""Teste de fumaça da API (health check).

Roda mesmo sem modelo treinado: valida que a aplicação sobe e responde.
"""
from fastapi.testclient import TestClient

from src.api.main import app


def test_health_endpoint():
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "model_loaded" in body


def test_predict_without_model_or_short_input():
    with TestClient(app) as client:
        resp = client.post("/predict", json={"prices": [1, 2, 3], "horizon": 1})
        # 503 se não há modelo carregado, 422 se a janela é insuficiente
        assert resp.status_code in (422, 503)
