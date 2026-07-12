"""Serviço de inferência: carrega os artefatos e gera previsões.

Encapsula o modelo LSTM, o scaler e os metadados para que a camada da API
apenas orquestre requisições HTTP.
"""
from __future__ import annotations

import json
import logging
import time
from typing import List

import joblib
import numpy as np

from src.config import Settings, settings

logger = logging.getLogger(__name__)


class ModelNotLoadedError(RuntimeError):
    """Levantada quando se tenta prever sem um modelo carregado."""


class Predictor:
    """Carrega os artefatos salvos e expõe métodos de previsão."""

    def __init__(self, cfg: Settings = settings):
        self.cfg = cfg
        self.model = None
        self.scaler = None
        self.metadata: dict = {}
        self.sequence_length = cfg.sequence_length
        self.symbol = cfg.symbol

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.scaler is not None

    def load(self) -> None:
        """Carrega modelo, scaler e metadados do diretório ``models/``.

        Import do TensorFlow é adiado para acelerar o import do módulo e não
        exigir TF em contextos que só leem metadados.
        """
        if not self.cfg.model_path.exists():
            logger.warning(
                "Modelo não encontrado em %s. Treine antes de servir "
                "(python -m src.model.train).",
                self.cfg.model_path,
            )
            return

        from tensorflow.keras.models import load_model

        self.model = load_model(self.cfg.model_path)
        self.scaler = joblib.load(self.cfg.scaler_path)

        if self.cfg.metadata_path.exists():
            with open(self.cfg.metadata_path, encoding="utf-8") as fh:
                self.metadata = json.load(fh)
            self.sequence_length = self.metadata.get(
                "sequence_length", self.cfg.sequence_length
            )
            self.symbol = self.metadata.get("symbol", self.cfg.symbol)

        logger.info("Modelo carregado (%s).", self.symbol)

    def predict(self, prices: List[float], horizon: int = 1) -> dict:
        """Previsão recursiva de ``horizon`` passos a partir de ``prices``.

        Usa a última janela de ``sequence_length`` preços; cada previsão é
        realimentada como entrada para prever o passo seguinte.
        """
        if not self.is_loaded:
            raise ModelNotLoadedError(
                "Nenhum modelo carregado. Treine o modelo primeiro."
            )

        if len(prices) < self.sequence_length:
            raise ValueError(
                f"São necessários pelo menos {self.sequence_length} preços; "
                f"recebidos {len(prices)}."
            )

        start = time.perf_counter()

        window = np.array(prices[-self.sequence_length :], dtype="float32")
        scaled = self.scaler.transform(window.reshape(-1, 1)).ravel()

        predictions_scaled: List[float] = []
        current = scaled.copy()
        for _ in range(horizon):
            x = current[-self.sequence_length :].reshape(
                1, self.sequence_length, 1
            )
            next_scaled = float(self.model.predict(x, verbose=0)[0, 0])
            predictions_scaled.append(next_scaled)
            current = np.append(current, next_scaled)

        predictions = (
            self.scaler.inverse_transform(
                np.array(predictions_scaled).reshape(-1, 1)
            )
            .ravel()
            .tolist()
        )

        inference_ms = (time.perf_counter() - start) * 1000
        return {
            "predictions": [round(p, 4) for p in predictions],
            "inference_ms": round(inference_ms, 2),
        }


# Instância única compartilhada pela API
predictor = Predictor()
