"""Registro de inferência: carrega **um modelo por ticker** e prevê.

Cada ticker treinado tem seu próprio modelo LSTM + scaler + metadados em
``models/<TICKER>/``. Este módulo descobre todos os modelos disponíveis e
expõe métodos de previsão que exigem a escolha explícita do ticker.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import joblib
import numpy as np

from src.config import Settings, settings

logger = logging.getLogger(__name__)


class ModelNotLoadedError(RuntimeError):
    """Nenhum modelo disponível para o ticker solicitado."""


@dataclass
class LoadedModel:
    symbol: str
    model: object
    scaler: object
    metadata: dict
    sequence_length: int


class ModelRegistry:
    """Carrega e mantém em memória todos os modelos treinados por ticker."""

    def __init__(self, cfg: Settings = settings):
        self.cfg = cfg
        self._models: Dict[str, LoadedModel] = {}

    # ----- Descoberta / carregamento -------------------------------------
    def load_all(self) -> None:
        """Carrega o modelo de cada ticker configurado que já foi treinado.

        Import do TensorFlow é adiado para acelerar o import do módulo.
        """
        from tensorflow.keras.models import load_model

        self._models.clear()
        for symbol in self.cfg.symbols:
            model_path = self.cfg.model_path(symbol)
            if not model_path.exists():
                logger.warning(
                    "[%s] Modelo não encontrado (%s). Treine com "
                    "`python scripts/train.py`.",
                    symbol,
                    model_path,
                )
                continue

            metadata: dict = {}
            meta_path = self.cfg.metadata_path(symbol)
            if meta_path.exists():
                with open(meta_path, encoding="utf-8") as fh:
                    metadata = json.load(fh)

            self._models[symbol.upper()] = LoadedModel(
                symbol=symbol.upper(),
                model=load_model(model_path),
                scaler=joblib.load(self.cfg.scaler_path(symbol)),
                metadata=metadata,
                sequence_length=metadata.get(
                    "sequence_length", self.cfg.sequence_length
                ),
            )
            logger.info("[%s] Modelo carregado.", symbol.upper())

        if not self._models:
            logger.warning("Nenhum modelo carregado. Treine antes de servir.")

    # ----- Consultas ------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return len(self._models) > 0

    def available_symbols(self) -> List[str]:
        return sorted(self._models.keys())

    def get(self, symbol: Optional[str]) -> LoadedModel:
        if symbol is None:
            raise ModelNotLoadedError(
                "Informe o ticker (symbol). Disponíveis: "
                f"{self.available_symbols()}."
            )
        key = symbol.upper()
        if key not in self._models:
            raise ModelNotLoadedError(
                f"Nenhum modelo treinado para '{symbol}'. "
                f"Escolha um dos disponíveis: {self.available_symbols()}."
            )
        return self._models[key]

    def catalog(self) -> List[dict]:
        """Resumo de cada modelo treinado — usado pela interface para escolha."""
        items = []
        for sym in self.available_symbols():
            lm = self._models[sym]
            md = lm.metadata
            items.append(
                {
                    "symbol": sym,
                    "sequence_length": lm.sequence_length,
                    "metrics": md.get("metrics", {}),
                    "trained_at": md.get("trained_at"),
                    "period": {
                        "start": md.get("start_date"),
                        "end": md.get("end_date"),
                    },
                }
            )
        return items

    # ----- Inferência -----------------------------------------------------
    def predict(
        self, symbol: str, prices: List[float], horizon: int = 1
    ) -> dict:
        """Previsão recursiva de ``horizon`` passos para o ticker escolhido."""
        loaded = self.get(symbol)
        seq_len = loaded.sequence_length

        if len(prices) < seq_len:
            raise ValueError(
                f"São necessários pelo menos {seq_len} preços para '{symbol}'; "
                f"recebidos {len(prices)}."
            )

        start = time.perf_counter()

        window = np.array(prices[-seq_len:], dtype="float32")
        scaled = loaded.scaler.transform(window.reshape(-1, 1)).ravel()

        predictions_scaled: List[float] = []
        current = scaled.copy()
        for _ in range(horizon):
            x = current[-seq_len:].reshape(1, seq_len, 1)
            next_scaled = float(loaded.model.predict(x, verbose=0)[0, 0])
            predictions_scaled.append(next_scaled)
            current = np.append(current, next_scaled)

        predictions = (
            loaded.scaler.inverse_transform(
                np.array(predictions_scaled).reshape(-1, 1)
            )
            .ravel()
            .tolist()
        )

        inference_ms = (time.perf_counter() - start) * 1000
        return {
            "symbol": loaded.symbol,
            "predictions": [round(p, 4) for p in predictions],
            "inference_ms": round(inference_ms, 2),
        }


# Instância única compartilhada pela API
registry = ModelRegistry()
