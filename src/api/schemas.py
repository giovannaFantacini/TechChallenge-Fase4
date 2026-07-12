"""Modelos de entrada e saída (Pydantic) da API."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Previsão a partir de uma sequência de preços fornecida pelo usuário."""

    prices: List[float] = Field(
        ...,
        description=(
            "Lista de preços de fechamento históricos, em ordem cronológica. "
            "Deve conter no mínimo `sequence_length` valores."
        ),
        examples=[[100.1, 101.3, 99.8, 102.5, 103.0]],
    )
    horizon: int = Field(
        1,
        ge=1,
        le=30,
        description="Quantos dias no futuro prever (previsão recursiva).",
    )


class PredictResponse(BaseModel):
    symbol: str
    horizon: int
    last_input_price: float
    predictions: List[float]
    inference_ms: float


class PredictLatestRequest(BaseModel):
    """Busca os dados mais recentes no Yahoo Finance e prevê o futuro."""

    symbol: Optional[str] = Field(
        None, description="Ticker. Se omitido, usa o símbolo do modelo treinado."
    )
    horizon: int = Field(1, ge=1, le=30)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    symbol: Optional[str] = None
    version: str


class ModelInfoResponse(BaseModel):
    symbol: str
    sequence_length: int
    metrics: dict
    trained_at: str
    hyperparameters: dict
