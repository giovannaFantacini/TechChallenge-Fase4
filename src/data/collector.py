"""Coleta e pré-processamento dos dados de preços de ações.

Etapa 1 do Tech Challenge: coleta via ``yfinance`` e transformação da série
temporal em janelas (sequências) prontas para alimentar a LSTM.
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


def download_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Baixa os preços históricos de uma ação do Yahoo Finance.

    Retorna um ``DataFrame`` com colunas simples (Open, High, Low, Close,
    Volume). Versões recentes do yfinance retornam colunas em ``MultiIndex``
    quando há vários tickers — normalizamos para o caso de ticker único.
    """
    logger.info("Baixando %s de %s a %s", symbol, start_date, end_date)
    df = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=True,
    )

    if df.empty:
        raise ValueError(
            f"Nenhum dado retornado para '{symbol}'. "
            "Verifique o ticker e o intervalo de datas."
        )

    # Achata colunas MultiIndex ('Close', 'DIS') -> 'Close'
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna()
    logger.info("Coletados %d registros", len(df))
    return df


def create_sequences(
    series: np.ndarray, sequence_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Transforma uma série 1D escalada em pares (janela, próximo valor).

    Para cada posição ``i``, ``X[i]`` são ``sequence_length`` valores
    consecutivos e ``y[i]`` é o valor imediatamente seguinte — exatamente o
    que a LSTM aprende a prever.
    """
    X, y = [], []
    for i in range(sequence_length, len(series)):
        X.append(series[i - sequence_length : i, 0])
        y.append(series[i, 0])
    X = np.array(X)
    y = np.array(y)
    # LSTM espera formato (amostras, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y


def prepare_datasets(
    df: pd.DataFrame,
    target_column: str,
    sequence_length: int,
    train_split: float,
) -> dict:
    """Escala os preços e monta os conjuntos de treino e teste.

    O ``MinMaxScaler`` é ajustado apenas no trecho de treino para evitar
    vazamento de informação (data leakage) do futuro para o passado.
    """
    prices = df[[target_column]].values.astype("float32")

    split_idx = int(len(prices) * train_split)
    train_prices = prices[:split_idx]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_prices)

    scaled = scaler.transform(prices)

    # O teste inclui os últimos `sequence_length` pontos do treino para que a
    # primeira janela de teste tenha contexto suficiente.
    scaled_train = scaled[:split_idx]
    scaled_test = scaled[split_idx - sequence_length :]

    X_train, y_train = create_sequences(scaled_train, sequence_length)
    X_test, y_test = create_sequences(scaled_test, sequence_length)

    logger.info(
        "Treino: %d amostras | Teste: %d amostras", len(X_train), len(X_test)
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
        "dates": df.index,
    }
