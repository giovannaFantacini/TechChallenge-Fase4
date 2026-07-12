# Imagem de produção da API de previsão de ações (LSTM).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# Dependências de sistema mínimas para numpy/tensorflow
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instala dependências Python primeiro (aproveita cache de camadas)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copia o código e os artefatos do modelo (se já treinados)
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY models/ ./models/

EXPOSE 8000

# Healthcheck usa o endpoint /health da própria API
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
