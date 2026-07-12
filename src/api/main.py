"""API RESTful (FastAPI) que serve o modelo LSTM.

Etapa 4 do Tech Challenge. Endpoints principais:
  GET  /health          -> checagem de saúde
  GET  /model/info      -> métricas e metadados do modelo treinado
  POST /predict         -> previsão a partir de preços informados
  POST /predict/latest  -> busca dados atuais no Yahoo Finance e prevê
  GET  /metrics         -> métricas Prometheus (monitoramento)
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api import monitoring
from src.api.monitoring import (
    INFERENCE_LATENCY,
    PREDICTION_COUNT,
    MonitoringMiddleware,
    refresh_resource_metrics,
)
from src.api.predictor import ModelNotLoadedError, predictor
from src.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictLatestRequest,
    PredictRequest,
    PredictResponse,
)
from src.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega o modelo na subida da aplicação."""
    logger.info("Carregando artefatos do modelo...")
    predictor.load()
    yield
    logger.info("Encerrando aplicação.")


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=(
        "API de previsão de preços de fechamento de ações usando uma rede "
        "neural LSTM. Tech Challenge - Fase 4 (Pós Tech MLET)."
    ),
    lifespan=lifespan,
)

app.add_middleware(MonitoringMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["infra"])
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=predictor.is_loaded,
        symbol=predictor.symbol if predictor.is_loaded else None,
        version=settings.api_version,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["model"])
async def model_info():
    if not predictor.is_loaded or not predictor.metadata:
        raise HTTPException(
            status_code=503,
            detail="Modelo/metadados indisponíveis. Treine o modelo primeiro.",
        )
    md = predictor.metadata
    return ModelInfoResponse(
        symbol=md["symbol"],
        sequence_length=md["sequence_length"],
        metrics=md["metrics"],
        trained_at=md["trained_at"],
        hyperparameters=md["hyperparameters"],
    )


@app.post("/predict", response_model=PredictResponse, tags=["model"])
async def predict(req: PredictRequest):
    try:
        with INFERENCE_LATENCY.time():
            result = predictor.predict(req.prices, horizon=req.horizon)
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    PREDICTION_COUNT.inc()
    return PredictResponse(
        symbol=predictor.symbol,
        horizon=req.horizon,
        last_input_price=req.prices[-1],
        predictions=result["predictions"],
        inference_ms=result["inference_ms"],
    )


@app.post("/predict/latest", response_model=PredictResponse, tags=["model"])
async def predict_latest(req: PredictLatestRequest):
    """Busca os preços mais recentes no Yahoo Finance e prevê o futuro."""
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=503, detail="Nenhum modelo carregado."
        )

    import yfinance as yf

    symbol = req.symbol or predictor.symbol
    # baixa período suficiente para preencher a janela (com folga p/ feriados)
    period_days = predictor.sequence_length * 2 + 30
    df = yf.download(
        symbol, period=f"{period_days}d", progress=False, auto_adjust=True
    )
    if df.empty:
        raise HTTPException(
            status_code=404, detail=f"Sem dados para o ticker '{symbol}'."
        )
    if hasattr(df.columns, "get_level_values"):
        try:
            df.columns = df.columns.get_level_values(0)
        except Exception:
            pass

    prices = df["Close"].dropna().tolist()
    if len(prices) < predictor.sequence_length:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Histórico insuficiente ({len(prices)} pts) para a janela de "
                f"{predictor.sequence_length}."
            ),
        )

    try:
        with INFERENCE_LATENCY.time():
            result = predictor.predict(prices, horizon=req.horizon)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    PREDICTION_COUNT.inc()
    return PredictResponse(
        symbol=symbol,
        horizon=req.horizon,
        last_input_price=round(prices[-1], 4),
        predictions=result["predictions"],
        inference_ms=result["inference_ms"],
    )


@app.get("/metrics", tags=["infra"])
async def metrics():
    """Endpoint scrapeado pelo Prometheus."""
    refresh_resource_metrics()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
