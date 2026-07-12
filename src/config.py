"""Configuração central do projeto.

Todos os hiperparâmetros e caminhos ficam concentrados aqui e podem ser
sobrescritos por variáveis de ambiente (arquivo .env). Assim o mesmo código
roda em desenvolvimento, em contêiner Docker e em produção sem alterações.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Diretório raiz do projeto (…/TechChallenge-Fase4)
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"


class Settings(BaseSettings):
    """Configurações da aplicação carregadas de variáveis de ambiente/.env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # ----- Coleta de dados -----
    symbol: str = "DIS"                 # símbolo (ticker) da empresa
    start_date: str = "2018-01-01"      # início do histórico
    end_date: str = "2024-07-20"        # fim do histórico
    target_column: str = "Close"        # coluna que será prevista

    # ----- Janela temporal / sequências -----
    sequence_length: int = 60           # nº de dias usados para prever o próximo
    train_split: float = 0.8            # fração dos dados para treino

    # ----- Hiperparâmetros do modelo LSTM -----
    lstm_units: int = 64
    lstm_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 1e-3
    epochs: int = 100
    batch_size: int = 32
    patience: int = 12                  # early stopping

    # ----- Artefatos salvos -----
    model_filename: str = "lstm_model.keras"
    scaler_filename: str = "scaler.pkl"
    metadata_filename: str = "metadata.json"

    # ----- API -----
    api_title: str = "Stock Price Prediction API - LSTM"
    api_version: str = "1.0.0"

    @property
    def model_path(self) -> Path:
        return MODELS_DIR / self.model_filename

    @property
    def scaler_path(self) -> Path:
        return MODELS_DIR / self.scaler_filename

    @property
    def metadata_path(self) -> Path:
        return MODELS_DIR / self.metadata_filename


settings = Settings()
