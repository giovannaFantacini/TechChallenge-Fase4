"""Testes unitários das métricas de avaliação (não exigem TensorFlow)."""
import numpy as np

from src.data.collector import create_sequences
from src.model.evaluate import compute_metrics


def test_metrics_zero_error():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = compute_metrics(y, y.copy())
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
    assert m["mape"] == 0.0


def test_metrics_known_values():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    m = compute_metrics(y_true, y_pred)
    # erros absolutos: 2, 2, 3 -> MAE = 7/3
    assert abs(m["mae"] - 7 / 3) < 1e-3
    # rmse = sqrt((4+4+9)/3)
    assert abs(m["rmse"] - np.sqrt(17 / 3)) < 1e-3


def test_create_sequences_shape():
    series = np.arange(10, dtype="float32").reshape(-1, 1)
    X, y = create_sequences(series, sequence_length=3)
    assert X.shape == (7, 3, 1)
    assert y.shape == (7,)
    # primeira janela [0,1,2] -> alvo 3
    assert list(X[0].ravel()) == [0.0, 1.0, 2.0]
    assert y[0] == 3.0
