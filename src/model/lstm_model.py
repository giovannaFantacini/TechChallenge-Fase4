"""Arquitetura do modelo LSTM.

Etapa 2 do Tech Challenge: rede neural recorrente empilhada com dropout para
regularização, capaz de capturar padrões temporais na série de preços.
"""

from __future__ import annotations

from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def build_model(
    sequence_length: int,
    lstm_units: int = 64,
    lstm_layers: int = 2,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> Sequential:
    """Constrói e compila a LSTM.

    - ``lstm_layers`` camadas LSTM empilhadas (todas menos a última retornam a
      sequência completa para alimentar a camada seguinte).
    - ``Dropout`` após cada LSTM reduz overfitting.
    - Saída ``Dense(1)`` = previsão do próximo preço de fechamento (escalado).
    """
    model = Sequential(name="lstm_stock_predictor")
    model.add(Input(shape=(sequence_length, 1)))

    for layer_idx in range(lstm_layers):
        is_last = layer_idx == lstm_layers - 1
        model.add(
            LSTM(
                lstm_units,
                return_sequences=not is_last,
                name=f"lstm_{layer_idx + 1}",
            )
        )
        model.add(Dropout(dropout, name=f"dropout_{layer_idx + 1}"))

    model.add(Dense(1, name="output"))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=["mae"],
    )
    return model
