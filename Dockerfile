# Imagem de produção da API de previsão de ações (LSTM).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=8000

WORKDIR /app

# Dependências de sistema mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instala dependências Python primeiro (aproveita cache de camadas).
# Usa tensorflow-cpu no servidor -> imagem menor e sem CUDA.
COPY requirements-docker.txt .
RUN pip install --upgrade pip && pip install -r requirements-docker.txt

# Copia o código e os modelos já treinados (models/<TICKER>/...)
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY models/ ./models/

# Render/Cloud Run injetam a porta via $PORT; localmente cai em 8000.
EXPOSE 8000

# Healthcheck usa o endpoint /health na porta configurada
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# `exec` garante que o uvicorn seja o PID 1 e receba os sinais (SIGTERM)
CMD ["sh", "-c", "exec uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}"]
