"""Instrumentação de monitoramento.

Etapa 5 do Tech Challenge: expõe métricas no formato Prometheus (tempo de
resposta, contagem de requisições, latência de inferência e uso de recursos)
para acompanhar a performance do modelo em produção.
"""
from __future__ import annotations

import time

import psutil
from prometheus_client import Counter, Gauge, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# --- Métricas HTTP -----------------------------------------------------------
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total de requisições HTTP recebidas",
    ["method", "endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Tempo de resposta das requisições HTTP (segundos)",
    ["endpoint"],
)

# --- Métricas de inferência --------------------------------------------------
PREDICTION_COUNT = Counter(
    "model_predictions_total", "Total de previsões geradas pelo modelo"
)
INFERENCE_LATENCY = Histogram(
    "model_inference_duration_seconds",
    "Tempo de inferência do modelo (segundos)",
)

# --- Uso de recursos ---------------------------------------------------------
CPU_USAGE = Gauge("process_cpu_percent", "Uso de CPU do processo (%)")
MEMORY_USAGE = Gauge(
    "process_memory_mb", "Uso de memória residente do processo (MB)"
)


def refresh_resource_metrics() -> None:
    """Atualiza os gauges de CPU/memória no momento do scrape."""
    process = psutil.Process()
    CPU_USAGE.set(process.cpu_percent(interval=None))
    MEMORY_USAGE.set(process.memory_info().rss / (1024 * 1024))


class MonitoringMiddleware(BaseHTTPMiddleware):
    """Mede tempo de resposta e contabiliza cada requisição."""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start

        endpoint = request.url.path
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=response.status_code,
        ).inc()
        response.headers["X-Process-Time-ms"] = f"{elapsed * 1000:.2f}"
        return response
