# Tech Challenge — Fase 4 · Previsão de Ações com LSTM

Pipeline completo de **Deep Learning** para prever o **preço de fechamento** de
uma ação usando uma rede neural **LSTM (Long Short-Term Memory)** — da coleta
dos dados ao **deploy de uma API RESTful** com **monitoramento**.

> Pós Tech MLET · Tech Challenge Fase 4

---

## 🎯 Objetivo

Construir e servir um modelo preditivo de séries temporais que, dado o
histórico recente de preços de uma ação, prevê o(s) próximo(s) preço(s) de
fechamento. O projeto cobre todas as etapas exigidas:

| Etapa | Requisito | Onde está |
|-------|-----------|-----------|
| 1 | Coleta e pré-processamento (yfinance) | [`src/data/collector.py`](src/data/collector.py) |
| 2 | Modelo LSTM + treino + avaliação (MAE, RMSE, MAPE) | [`src/model/`](src/model) |
| 3 | Salvamento/exportação do modelo | [`src/model/train.py`](src/model/train.py) |
| 4 | Deploy via API RESTful (FastAPI) | [`src/api/`](src/api) |
| 5 | Escalabilidade e monitoramento (Prometheus) | [`src/api/monitoring.py`](src/api/monitoring.py) |

---

## 🗂️ Estrutura do projeto

> 🔀 **Multi-modelo:** o projeto treina **um modelo LSTM por ticker**. A API
> serve todos eles e a pessoa **escolhe o ticker** (campo `symbol`) antes de
> prever. Os tickers vêm da variável `SYMBOLS` (padrão: `DIS, AAPL, MSFT`).

```
TechChallenge-Fase4/
├── src/
│   ├── config.py             # Configuração central (SYMBOLS, hiperparâmetros)
│   ├── data/
│   │   └── collector.py      # Download (yfinance) + janelas + scaling
│   ├── model/
│   │   ├── lstm_model.py     # Arquitetura da LSTM (Keras)
│   │   ├── train.py          # Treino por ticker (train_symbol / train_all)
│   │   └── evaluate.py       # MAE, RMSE, MAPE
│   └── api/
│       ├── main.py           # App FastAPI (endpoints)
│       ├── predictor.py      # Registro que carrega 1 modelo por ticker
│       ├── schemas.py        # Contratos Pydantic
│       ├── monitoring.py     # Métricas Prometheus + middleware
│       └── static/index.html # Página inicial p/ escolher o ticker e prever
├── scripts/train.py          # CLI de treino (todos os tickers ou subconjunto)
├── tests/                    # Testes (métricas + smoke da API)
├── models/                   # Artefatos por ticker: models/<TICKER>/...
│   ├── DIS/{lstm_model.keras, scaler.pkl, metadata.json}
│   ├── AAPL/...
│   └── MSFT/...
├── monitoring/prometheus.yml # Config do Prometheus
├── Dockerfile
├── docker-compose.yml        # API + Prometheus
├── requirements.txt
└── README.md
```

---

## 🚀 Como executar

### 1. Ambiente

```bash
cd TechChallenge-Fase4
python -m venv .venv && source .venv/bin/activate   # opcional
pip install -r requirements.txt
```

### 2. Treinar os modelos (um por ticker)

Baixa os dados, treina uma LSTM para cada ticker, avalia e salva os artefatos:

```bash
python scripts/train.py                       # treina todos os SYMBOLS (DIS, AAPL, MSFT)
python scripts/train.py --symbols DIS,AAPL    # treina só esses
python scripts/train.py --epochs 50 --sequence-length 90
```

Para cada ticker é criada a pasta `models/<TICKER>/` com:
- `lstm_model.keras` — modelo treinado
- `scaler.pkl` — `MinMaxScaler` ajustado
- `metadata.json` — hiperparâmetros e métricas (MAE, RMSE, MAPE)

### 3. Subir a API

```bash
uvicorn src.api.main:app --reload
# Página inicial (escolha do ticker): http://localhost:8000/
# Documentação interativa (Swagger):  http://localhost:8000/docs
```

### 4. Com Docker (API + Prometheus)

```bash
docker compose up --build
# API .......... http://localhost:8000/docs
# Prometheus ... http://localhost:9090
```

> Treine o modelo **antes** de buildar (ou monte o volume `./models`) para que
> os artefatos estejam disponíveis dentro do contêiner.

---

## 🔌 Endpoints da API

| Método | Rota | Descrição |
|--------|------|-----------|
| `GET`  | `/` | Página inicial — **escolha do ticker** e previsão |
| `GET`  | `/health` | Saúde do serviço e tickers disponíveis |
| `GET`  | `/models` | **Catálogo dos modelos treinados** (para escolher o `symbol`) |
| `POST` | `/predict` | Previsão a partir de preços informados (**exige `symbol`**) |
| `POST` | `/predict/latest` | Baixa dados atuais do ticker e prevê (**exige `symbol`**) |
| `GET`  | `/metrics` | Métricas Prometheus (monitoramento) |
| `GET`  | `/docs` | Swagger UI |

> ⚠️ **Escolha do modelo:** `/predict` e `/predict/latest` **exigem** o campo
> `symbol`, que deve ser um dos tickers treinados. Consulte `GET /models` para
> ver as opções disponíveis e suas métricas.

### Exemplo — `GET /models`

```json
{
  "count": 3,
  "available_symbols": ["AAPL", "DIS", "MSFT"],
  "models": [
    {"symbol": "AAPL", "sequence_length": 60, "metrics": {"mae": 3.1, "rmse": 4.0, "mape": 1.8}, ...},
    {"symbol": "DIS",  "sequence_length": 60, "metrics": {"mae": 2.4, "rmse": 3.1, "mape": 2.2}, ...},
    {"symbol": "MSFT", "sequence_length": 60, "metrics": {"mae": 5.7, "rmse": 7.2, "mape": 1.5}, ...}
  ]
}
```

### Exemplo — `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "prices": [100.1, 101.3, 99.8, 102.5, ...], "horizon": 5}'
```

Resposta:

```json
{
  "symbol": "AAPL",
  "horizon": 5,
  "last_input_price": 102.5,
  "predictions": [102.7, 103.1, 103.4, 103.2, 103.6],
  "inference_ms": 48.21
}
```

> `prices` deve conter pelo menos `sequence_length` valores (padrão **60**),
> em ordem cronológica. `horizon` faz previsão recursiva de N dias.

### Exemplo — `POST /predict/latest`

```bash
curl -X POST http://localhost:8000/predict/latest \
  -H "Content-Type: application/json" \
  -d '{"symbol": "DIS", "horizon": 3}'
```

A API baixa automaticamente o histórico recente do ticker escolhido e prevê.

---

## 🧠 Modelo LSTM

- **Um modelo independente por ticker** (mesma arquitetura, pesos e scaler
  próprios), salvo em `models/<TICKER>/`.
- Entrada: janela deslizante de `sequence_length` preços de fechamento.
- Arquitetura: `LSTM` empilhadas (`lstm_layers`) com `Dropout`, seguidas de
  `Dense(1)`.
- Pré-processamento: `MinMaxScaler` **ajustado apenas no treino** (evita data
  leakage).
- Otimizador `Adam`, perda `MSE`, `EarlyStopping` + `ReduceLROnPlateau`.
- Previsão multi-passo por realimentação recursiva.

### Avaliação

Métricas calculadas em **escala real de preço** (após inverter o scaling):

- **MAE** — Erro Absoluto Médio
- **RMSE** — Raiz do Erro Quadrático Médio
- **MAPE** — Erro Percentual Absoluto Médio

Cada modelo tem suas próprias métricas, registradas em
`models/<TICKER>/metadata.json` e disponíveis em `GET /models`.

---

## 📈 Escalabilidade e Monitoramento

O endpoint `/metrics` expõe, no formato Prometheus:

- `http_requests_total` — contagem por rota/método/status
- `http_request_duration_seconds` — **tempo de resposta** (histograma)
- `model_inference_duration_seconds` — latência de inferência
- `model_predictions_total{symbol=...}` — total de previsões **por ticker**
- `process_cpu_percent` / `process_memory_mb` — **uso de recursos**

Cada resposta HTTP inclui o cabeçalho `X-Process-Time-ms`.

O `docker-compose.yml` já sobe um **Prometheus** apontando para a API
(`monitoring/prometheus.yml`), permitindo dashboards e alertas.

**Escalabilidade:** a aplicação é *stateless* (o estado vive nos artefatos em
`models/`), então escala horizontalmente atrás de um load balancer. Em
produção, rode múltiplos workers Uvicorn/Gunicorn:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## ✅ Testes

```bash
pytest
```

- `tests/test_metrics.py` — métricas e geração de sequências (sem TensorFlow).
- `tests/test_api.py` — smoke test da API (health + validação).

---

## ⚙️ Configuração

Todos os parâmetros podem ser ajustados por variáveis de ambiente ou `.env`
(veja [`.env.example`](.env.example)). Principais:

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `SYMBOLS` | `DIS,AAPL,MSFT` | Tickers (um modelo por ticker), separados por vírgula |
| `START_DATE` / `END_DATE` | `2018-01-01` / `2024-07-20` | Intervalo do histórico |
| `SEQUENCE_LENGTH` | `60` | Tamanho da janela temporal |
| `LSTM_UNITS` / `LSTM_LAYERS` | `64` / `2` | Capacidade da rede |
| `EPOCHS` / `BATCH_SIZE` | `100` / `32` | Treino |

---

## 📦 Entregáveis do desafio

- ✅ Código-fonte do modelo LSTM + documentação (este repositório)
- ✅ `Dockerfile` e `docker-compose.yml` para deploy da API
- ✅ API RESTful (FastAPI) servindo o modelo
- ✅ Monitoramento (Prometheus) de tempo de resposta e uso de recursos
- 🎥 Vídeo de demonstração — 
- 🌐 Link da API em produção — 
