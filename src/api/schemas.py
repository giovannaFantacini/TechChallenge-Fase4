"""Modelos de entrada e saída (Pydantic) da API.

A API serve **vários modelos, um por ticker**. Por isso os endpoints de
previsão exigem o campo ``symbol`` — a pessoa escolhe qual modelo usar.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Previsão a partir de uma sequência de preços fornecida pelo usuário."""

    symbol: str = Field(
        ...,
        description=(
            "Ticker do modelo a ser usado. Deve ser um dos modelos treinados "
            "(consulte GET /models). Ex.: DIS, AAPL, MSFT."
        ),
        examples=["AAPL"],
    )
    prices: List[float] = Field(
        ...,
        description=(
            "Lista de preços de fechamento históricos, em ordem cronológica. "
            "Deve conter no mínimo `sequence_length` valores do modelo escolhido."
        ),
        examples=[[100.1, 101.3, 99.8, 102.5, 103.0]],
    )
    horizon: int = Field(
        1,
        ge=1,
        le=30,
        description="Quantos dias no futuro prever (previsão recursiva).",
    )


class PredictLatestRequest(BaseModel):
    """Busca os dados mais recentes no Yahoo Finance e prevê o futuro."""

    symbol: str = Field(
        ...,
        description=(
            "Ticker do modelo treinado a ser usado (GET /models). "
            "A API baixa o histórico recente desse ticker automaticamente."
        ),
        examples=["DIS"],
    )
    horizon: int = Field(1, ge=1, le=30)


class PredictResponse(BaseModel):
    symbol: str
    horizon: int
    last_input_price: float
    predictions: List[float]
    inference_ms: float
    # Série histórica usada como contexto (para plotar o gráfico na interface).
    history: List[float] = []
    history_dates: Optional[List[str]] = None      # datas do histórico (se houver)
    prediction_dates: Optional[List[str]] = None    # datas futuras previstas (se houver)


class ModelCatalogItem(BaseModel):
    symbol: str
    sequence_length: int
    metrics: dict
    trained_at: Optional[str] = None
    period: dict


class ModelsResponse(BaseModel):
    """Catálogo dos modelos disponíveis para escolha antes do predict."""

    count: int
    available_symbols: List[str]
    models: List[ModelCatalogItem]


class HealthResponse(BaseModel):
    status: str
    models_loaded: int
    available_symbols: List[str]
    version: str
