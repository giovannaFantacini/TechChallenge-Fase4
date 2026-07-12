"""Pipeline de treino ponta a ponta.

Orquestra as etapas 1 a 3 do Tech Challenge:
  coleta -> pré-processamento -> treino -> avaliação -> salvamento.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import joblib
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.config import MODELS_DIR, Settings, settings
from src.data.collector import download_data, prepare_datasets
from src.model.evaluate import compute_metrics
from src.model.lstm_model import build_model

logger = logging.getLogger(__name__)


def train_pipeline(cfg: Settings = settings) -> dict:
    """Executa o pipeline completo e salva os artefatos em ``models/``.

    Retorna um dicionário com as métricas de teste e o histórico resumido.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Coleta ------------------------------------------------------------
    df = download_data(cfg.symbol, cfg.start_date, cfg.end_date)

    # 2. Pré-processamento -------------------------------------------------
    data = prepare_datasets(
        df,
        target_column=cfg.target_column,
        sequence_length=cfg.sequence_length,
        train_split=cfg.train_split,
    )
    scaler = data["scaler"]

    # 3. Modelo ------------------------------------------------------------
    model = build_model(
        sequence_length=cfg.sequence_length,
        lstm_units=cfg.lstm_units,
        lstm_layers=cfg.lstm_layers,
        dropout=cfg.dropout,
        learning_rate=cfg.learning_rate,
    )
    model.summary(print_fn=logger.info)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=cfg.patience,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=cfg.patience // 2, min_lr=1e-5
        ),
    ]

    logger.info("Iniciando treino...")
    history = model.fit(
        data["X_train"],
        data["y_train"],
        validation_split=0.1,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    # 4. Avaliação (em escala real de preço) -------------------------------
    y_pred_scaled = model.predict(data["X_test"], verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled).ravel()
    y_true = scaler.inverse_transform(
        data["y_test"].reshape(-1, 1)
    ).ravel()

    metrics = compute_metrics(y_true, y_pred)
    logger.info("Métricas de teste: %s", metrics)

    # 5. Salvamento --------------------------------------------------------
    model.save(cfg.model_path)
    joblib.dump(scaler, cfg.scaler_path)

    metadata = {
        "symbol": cfg.symbol,
        "start_date": cfg.start_date,
        "end_date": cfg.end_date,
        "target_column": cfg.target_column,
        "sequence_length": cfg.sequence_length,
        "train_split": cfg.train_split,
        "hyperparameters": {
            "lstm_units": cfg.lstm_units,
            "lstm_layers": cfg.lstm_layers,
            "dropout": cfg.dropout,
            "learning_rate": cfg.learning_rate,
            "epochs_run": len(history.history["loss"]),
            "batch_size": cfg.batch_size,
        },
        "metrics": metrics,
        "n_train": int(len(data["X_train"])),
        "n_test": int(len(data["X_test"])),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(cfg.metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)

    logger.info("Artefatos salvos em %s", MODELS_DIR)
    return {"metrics": metrics, "metadata": metadata}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    result = train_pipeline()
    print(json.dumps(result["metrics"], indent=2))
