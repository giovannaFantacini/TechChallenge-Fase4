"""Instrumentação de monitoramento.

Etapa 5 do Tech Challenge: expõe métricas no formato Prometheus (tempo de
resposta, contagem de requisições, latência de inferência e uso de recursos)
para acompanhar a performance do modelo em produção.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import psutil
from prometheus_client import REGISTRY, Counter, Gauge, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# Momento em que o processo subiu (para calcular o uptime no painel)
START_TIME = time.time()

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
    "model_predictions_total",
    "Total de previsões geradas pelo modelo",
    ["symbol"],
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


# --- Agregação para o painel /monitor ----------------------------------------
def _quantile_from_buckets(
    buckets: List[Tuple[float, float]], q: float
) -> Optional[float]:
    """Estima um quantil a partir dos buckets cumulativos de um histograma.

    ``buckets`` são pares ``(limite_superior, contagem_acumulada)`` ordenados.
    Faz interpolação linear dentro do bucket onde o alvo cai.
    """
    if not buckets:
        return None
    buckets = sorted(buckets, key=lambda b: b[0])
    total = buckets[-1][1]
    if total <= 0:
        return None

    target = q * total
    prev_le, prev_count = 0.0, 0.0
    for le, count in buckets:
        if count >= target:
            if le == float("inf"):
                return prev_le
            if count == prev_count:
                return le
            frac = (target - prev_count) / (count - prev_count)
            return prev_le + frac * (le - prev_le)
        if le != float("inf"):
            prev_le = le
        prev_count = count
    return None


def snapshot() -> dict:
    """Lê o registry do Prometheus e devolve um resumo pronto para a UI.

    Percorre as amostras coletadas em vez de parsear o texto exposto em
    ``/metrics`` — é mais robusto e não depende do formato de saída.
    """
    refresh_resource_metrics()

    endpoints: Dict[str, dict] = {}
    predictions: Dict[str, float] = {}
    infer = {"count": 0.0, "sum": 0.0, "buckets": []}
    resources = {"cpu_percent": 0.0, "memory_mb": 0.0}

    def ep(name: str) -> dict:
        return endpoints.setdefault(
            name,
            {"endpoint": name, "requests": 0.0, "by_status": {},
             "latency_count": 0.0, "latency_sum": 0.0, "buckets": []},
        )

    for family in REGISTRY.collect():
        for s in family.samples:
            n, lbl, val = s.name, s.labels, s.value

            if n == "http_requests_total":
                e = ep(lbl.get("endpoint", "?"))
                e["requests"] += val
                code = str(lbl.get("status_code", "?"))
                e["by_status"][code] = e["by_status"].get(code, 0.0) + val
            elif n == "http_request_duration_seconds_count":
                ep(lbl.get("endpoint", "?"))["latency_count"] += val
            elif n == "http_request_duration_seconds_sum":
                ep(lbl.get("endpoint", "?"))["latency_sum"] += val
            elif n == "http_request_duration_seconds_bucket":
                ep(lbl.get("endpoint", "?"))["buckets"].append(
                    (float(lbl.get("le", "inf")), val)
                )
            elif n == "model_predictions_total":
                sym = lbl.get("symbol", "?")
                predictions[sym] = predictions.get(sym, 0.0) + val
            elif n == "model_inference_duration_seconds_count":
                infer["count"] += val
            elif n == "model_inference_duration_seconds_sum":
                infer["sum"] += val
            elif n == "model_inference_duration_seconds_bucket":
                infer["buckets"].append((float(lbl.get("le", "inf")), val))
            elif n == "process_cpu_percent":
                resources["cpu_percent"] = val
            elif n == "process_memory_mb":
                resources["memory_mb"] = val

    rows = []
    total_requests = 0.0
    total_latency_sum = 0.0
    total_latency_count = 0.0
    for e in endpoints.values():
        cnt = e["latency_count"]
        avg_ms = (e["latency_sum"] / cnt * 1000) if cnt else None
        p95 = _quantile_from_buckets(e["buckets"], 0.95)
        rows.append(
            {
                "endpoint": e["endpoint"],
                "requests": int(e["requests"]),
                "by_status": {k: int(v) for k, v in e["by_status"].items()},
                "avg_ms": round(avg_ms, 2) if avg_ms is not None else None,
                "p95_ms": round(p95 * 1000, 2) if p95 is not None else None,
            }
        )
        total_requests += e["requests"]
        total_latency_sum += e["latency_sum"]
        total_latency_count += cnt

    rows.sort(key=lambda r: r["requests"], reverse=True)

    infer_avg = (infer["sum"] / infer["count"] * 1000) if infer["count"] else None
    infer_p95 = _quantile_from_buckets(infer["buckets"], 0.95)

    return {
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "totals": {
            "requests": int(total_requests),
            "avg_response_ms": round(
                total_latency_sum / total_latency_count * 1000, 2
            ) if total_latency_count else None,
            "predictions": int(sum(predictions.values())),
        },
        "endpoints": rows,
        "predictions_by_symbol": {
            k: int(v) for k, v in sorted(
                predictions.items(), key=lambda kv: kv[1], reverse=True
            )
        },
        "inference": {
            "count": int(infer["count"]),
            "avg_ms": round(infer_avg, 2) if infer_avg is not None else None,
            "p95_ms": round(infer_p95 * 1000, 2) if infer_p95 is not None else None,
        },
        "resources": {
            "cpu_percent": round(resources["cpu_percent"], 1),
            "memory_mb": round(resources["memory_mb"], 1),
        },
    }
