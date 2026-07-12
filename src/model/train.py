"""Pipeline de treino ponta a ponta (um modelo por ticker).

Orquestra as etapas 1 a 3 do Tech Challenge:
  coleta -> pré-processamento -> treino -> avaliação -> salvamento.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import joblib
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.config import Settings, settings
from src.data.collector import download_data, prepare_datasets
from src.model.evaluate import compute_metrics
from src.model.lstm_model import build_model

logger = logging.getLogger(__name__)


def train_symbol(symbol: str, cfg: Settings = settings) -> dict:
    """Treina, avalia e salva o modelo de **um** ticker em ``models/<TICKER>/``.

    Retorna um dicionário com as métricas de teste e os metadados salvos.
    """
    symbol = symbol.upper()
    out_dir = cfg.symbol_dir(symbol)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Coleta ------------------------------------------------------------
    df = download_data(symbol, cfg.start_date, cfg.end_date)

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

    logger.info("[%s] Iniciando treino...", symbol)
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
    y_true = scaler.inverse_transform(data["y_test"].reshape(-1, 1)).ravel()

    metrics = compute_metrics(y_true, y_pred)
    logger.info("[%s] Métricas de teste: %s", symbol, metrics)

    # 5. Salvamento --------------------------------------------------------
    model.save(cfg.model_path(symbol))
    joblib.dump(scaler, cfg.scaler_path(symbol))

    metadata = {
        "symbol": symbol,
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
    with open(cfg.metadata_path(symbol), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)

    logger.info("[%s] Artefatos salvos em %s", symbol, out_dir)
    return {"metrics": metrics, "metadata": metadata}


def train_all(cfg: Settings = settings) -> dict:
    """Treina um modelo para cada ticker em ``cfg.symbols``.

    Retorna ``{ticker: resultado}``. Falhas em um ticker não interrompem os
    demais — o erro é registrado e a execução segue.
    """
    results: dict = {}
    for symbol in cfg.symbols:
        try:
            results[symbol] = train_symbol(symbol, cfg)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[%s] Falha no treino: %s", symbol, exc)
            results[symbol] = {"error": str(exc)}
    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    all_results = train_all()
    summary = {
        sym: r.get("metrics", r.get("error")) for sym, r in all_results.items()
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
