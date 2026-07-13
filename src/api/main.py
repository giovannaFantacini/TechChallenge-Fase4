"""API RESTful (FastAPI) que serve **um modelo LSTM por ticker**.

Etapa 4 do Tech Challenge. A API foi treinada para múltiplos tickers e a
pessoa deve **escolher** qual modelo usar antes de prever.

Endpoints:
  GET  /                -> página inicial (escolha do ticker)
  GET  /health          -> saúde + tickers disponíveis
  GET  /models          -> catálogo dos modelos treinados (para escolha)
  POST /predict         -> previsão a partir de preços informados (exige symbol)
  POST /predict/latest  -> baixa dados atuais e prevê (exige symbol)
  GET  /metrics         -> métricas Prometheus (monitoramento)
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api.monitoring import (
    INFERENCE_LATENCY,
    PREDICTION_COUNT,
    MonitoringMiddleware,
    refresh_resource_metrics,
)
from src.api.predictor import ModelNotLoadedError, registry
from src.api.schemas import (
    HealthResponse,
    ModelsResponse,
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

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _build_description() -> str:
    symbols = registry.available_symbols()
    lista = ", ".join(f"**{s}**" for s in symbols) if symbols else "(nenhum ainda)"
    return (
        "API de previsão de preços de fechamento de ações usando redes neurais "
        "**LSTM**. Tech Challenge - Fase 4 (Pós Tech MLET).\n\n"
        f"⚠️ Esta API serve **{len(symbols)} modelos distintos**, um por ticker: "
        f"{lista}.\n\n"
        "**Escolha o ticker** (campo `symbol`) antes de chamar `/predict` ou "
        "`/predict/latest`. Consulte `GET /models` para ver as opções e suas "
        "métricas de avaliação (MAE, RMSE, MAPE)."
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega todos os modelos treinados na subida da aplicação."""
    logger.info("Carregando modelos treinados...")
    registry.load_all()
    logger.info("Modelos disponíveis: %s", registry.available_symbols())
    # Atualiza a descrição do Swagger com os tickers efetivamente carregados
    app.description = _build_description()
    app.openapi_schema = None  # força regenerar o schema com a nova descrição
    yield
    logger.info("Encerrando aplicação.")


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=_build_description(),
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
async def home():
    """Página inicial: permite escolher o ticker e testar as previsões."""
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return Response("<h1>Stock LSTM API</h1><p>Veja /docs</p>", media_type="text/html")


@app.get("/health", response_model=HealthResponse, tags=["infra"])
async def health():
    return HealthResponse(
        status="ok",
        models_loaded=len(registry.available_symbols()),
        available_symbols=registry.available_symbols(),
        version=settings.api_version,
    )


@app.get("/models", response_model=ModelsResponse, tags=["model"])
async def list_models():
    """Catálogo dos modelos treinados — use para escolher o `symbol`."""
    if not registry.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Nenhum modelo carregado. Treine com `python scripts/train.py`.",
        )
    catalog = registry.catalog()
    return ModelsResponse(
        count=len(catalog),
        available_symbols=registry.available_symbols(),
        models=catalog,
    )


@app.post("/predict", response_model=PredictResponse, tags=["model"])
async def predict(req: PredictRequest):
    try:
        with INFERENCE_LATENCY.time():
            result = registry.predict(req.symbol, req.prices, horizon=req.horizon)
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    PREDICTION_COUNT.labels(symbol=result["symbol"]).inc()
    # Devolve o histórico informado (limitado à cauda) para plotagem.
    history = [round(float(p), 4) for p in req.prices[-250:]]
    return PredictResponse(
        symbol=result["symbol"],
        horizon=req.horizon,
        last_input_price=req.prices[-1],
        predictions=result["predictions"],
        inference_ms=result["inference_ms"],
        history=history,
    )


@app.post("/predict/latest", response_model=PredictResponse, tags=["model"])
async def predict_latest(req: PredictLatestRequest):
    """Baixa os preços mais recentes do ticker no Yahoo Finance e prevê."""
    try:
        loaded = registry.get(req.symbol)
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    import yfinance as yf

    seq_len = loaded.sequence_length
    period_days = seq_len * 2 + 30  # folga para feriados/fins de semana
    df = yf.download(
        req.symbol, period=f"{period_days}d", progress=False, auto_adjust=True
    )
    if df.empty:
        raise HTTPException(
            status_code=404, detail=f"Sem dados para o ticker '{req.symbol}'."
        )
    if hasattr(df.columns, "get_level_values"):
        try:
            df.columns = df.columns.get_level_values(0)
        except Exception:
            pass

    prices = df["Close"].dropna().tolist()
    if len(prices) < seq_len:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Histórico insuficiente ({len(prices)} pts) para a janela de "
                f"{seq_len}."
            ),
        )

    try:
        with INFERENCE_LATENCY.time():
            result = registry.predict(req.symbol, prices, horizon=req.horizon)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Datas do histórico e projeção das datas futuras (dias úteis) para o gráfico.
    import pandas as pd

    hist_dates = [d.strftime("%Y-%m-%d") for d in df.index]
    last_date = df.index[-1]
    future = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1), periods=req.horizon
    )
    pred_dates = [d.strftime("%Y-%m-%d") for d in future]

    PREDICTION_COUNT.labels(symbol=result["symbol"]).inc()
    return PredictResponse(
        symbol=result["symbol"],
        horizon=req.horizon,
        last_input_price=round(prices[-1], 4),
        predictions=result["predictions"],
        inference_ms=result["inference_ms"],
        history=[round(float(p), 4) for p in prices],
        history_dates=hist_dates,
        prediction_dates=pred_dates,
    )


@app.get("/metrics", tags=["infra"])
async def metrics():
    """Endpoint scrapeado pelo Prometheus."""
    refresh_resource_metrics()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
