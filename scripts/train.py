#!/usr/bin/env python
"""CLI de treino dos modelos LSTM (um por ticker).

Exemplos:
    python scripts/train.py                       # treina todos os SYMBOLS
    python scripts/train.py --symbols DIS,AAPL    # treina só esses
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
from src.model.train import train_all  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Treina os modelos LSTM de ações.")
    p.add_argument(
        "--symbols",
        help="Tickers separados por vírgula (ex.: DIS,AAPL,MSFT).",
    )
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
        "Treinando %s | período=%s→%s janela=%d épocas=%d",
        cfg.symbols, cfg.start_date, cfg.end_date, cfg.sequence_length, cfg.epochs,
    )
    results = train_all(cfg)

    print("\n=== Resumo do treino ===")
    summary = {s: r.get("metrics", r.get("error")) for s, r in results.items()}
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
