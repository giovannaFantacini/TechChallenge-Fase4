"""Métricas de avaliação do modelo.

Etapa 2 do Tech Challenge: MAE, RMSE e MAPE para medir a precisão das
previsões em unidades reais de preço (após inverter a escala).
"""

from __future__ import annotations

import numpy as np


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_absolute_percentage_error(
    y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    """MAPE em porcentagem. Ignora pontos com valor real ~0 para evitar
    divisão por zero."""
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    mask = np.abs(y_true) > 1e-8
    return float(
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Retorna as três métricas exigidas pelo desafio em um dicionário."""
    return {
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
        "rmse": round(root_mean_squared_error(y_true, y_pred), 4),
        "mape": round(mean_absolute_percentage_error(y_true, y_pred), 4),
    }
