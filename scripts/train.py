#!/usr/bin/env python
"""CLI de treino do modelo LSTM.

Exemplos:
    python scripts/train.py
    python scripts/train.py --symbol AAPL --start 2019-01-01 --end 2024-12-31
    python scripts/train.py --epochs 50 --sequence-length 90
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Permite `python scripts/train.py` a partir da raiz do projeto
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Settings  # noqa: E402
from src.model.train import train_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Treina o modelo LSTM de ações.")
    p.add_argument("--symbol", help="Ticker da empresa (ex.: DIS, AAPL).")
    p.add_argument("--start", dest="start_date", help="Data inicial YYYY-MM-DD.")
    p.add_argument("--end", dest="end_date", help="Data final YYYY-MM-DD.")
    p.add_argument("--sequence-length", type=int, help="Tamanho da janela.")
    p.add_argument("--epochs", type=int, help="Número máximo de épocas.")
    p.add_argument("--batch-size", type=int, help="Tamanho do batch.")
    p.add_argument("--lstm-units", type=int, help="Unidades por camada LSTM.")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    overrides = {k: v for k, v in vars(args).items() if v is not None}
    cfg = Settings(**overrides)

    logging.getLogger(__name__).info(
        "Config: symbol=%s período=%s→%s janela=%d épocas=%d",
        cfg.symbol, cfg.start_date, cfg.end_date, cfg.sequence_length, cfg.epochs,
    )
    result = train_pipeline(cfg)
    print("\n=== Métricas de teste ===")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
