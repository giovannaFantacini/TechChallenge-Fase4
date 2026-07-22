# Tech Challenge — Fase 4 · Previsão de Ações com LSTM

Este projeto implementa, de ponta a ponta, um pipeline de **Deep Learning** que
prevê o **preço de fechamento** de ações usando redes neurais **LSTM (Long
Short-Term Memory)**. Ele cobre todo o ciclo: da coleta dos dados históricos ao
treino e avaliação do modelo, terminando em uma **API RESTful** que serve as
previsões e é **monitorada** em tempo real.

> Pós Tech MLET · Tech Challenge Fase 4

---

## Objetivo

O objetivo é, dado o histórico recente de preços de uma ação, prever o(s)
próximo(s) preço(s) de fechamento. Cada requisito do desafio foi endereçado em
uma parte específica do código:

| Etapa | Requisito | Onde está |
|-------|-----------|-----------|
| 1 | Coleta e pré-processamento (yfinance) | [`src/data/collector.py`](src/data/collector.py) |
| 2 | Modelo LSTM + treino + avaliação (MAE, RMSE, MAPE) | [`src/model/`](src/model) |
| 3 | Salvamento/exportação do modelo | [`src/model/train.py`](src/model/train.py) |
| 4 | Deploy via API RESTful (FastAPI) | [`src/api/`](src/api) |
| 5 | Escalabilidade e monitoramento (Prometheus) | [`src/api/monitoring.py`](src/api/monitoring.py) |

Uma decisão de projeto importante: em vez de um único modelo, o sistema treina
**um modelo LSTM independente para cada ticker**. A API carrega todos eles e o
usuário **escolhe qual ticker** quer prever. Por padrão, são treinados três:
**DIS**, **AAPL** e **MSFT** (configurável pela variável `SYMBOLS`).

---

## 🔎 Como funciona (a pipeline)

O fluxo completo, do dado bruto à previsão servida pela API:

**1. Coleta.** Os preços históricos são baixados do Yahoo Finance com a
biblioteca `yfinance`. O código normaliza o retorno (que pode vir com colunas
em `MultiIndex`) e trabalha com a coluna de fechamento (`Close`).

**2. Pré-processamento.** A série é escalada para o intervalo [0, 1] com um
`MinMaxScaler` — mas o scaler é **ajustado apenas na parte de treino**, nunca no
conjunto de teste. Isso evita *data leakage* (deixar o modelo "enxergar" o
futuro). Em seguida a série é recortada em **janelas deslizantes**: cada exemplo
são `sequence_length` dias consecutivos (padrão **60**) e o alvo é o preço do
dia seguinte.

**3. Modelo.** Uma rede `Sequential` do Keras com camadas **LSTM empilhadas**
(com `Dropout` para regularização) e uma camada `Dense(1)` na saída, que produz
o próximo preço (ainda em escala normalizada).

**4. Treino e avaliação.** O modelo é treinado com `Adam` e perda `MSE`, usando
`EarlyStopping` (para parar quando a validação para de melhorar) e
`ReduceLROnPlateau` (para reduzir a taxa de aprendizado automaticamente). A
avaliação é feita **em escala real de preço** (desfazendo a normalização) com as
métricas **MAE**, **RMSE** e **MAPE**.

**5. Salvamento.** Para cada ticker são gravados três artefatos em
`models/<TICKER>/`: o modelo (`.keras`), o `scaler` (`.pkl`) e um
`metadata.json` com hiperparâmetros e métricas. É isso que a API carrega — ela
**não treina nada em tempo de execução**, apenas faz inferência.

**6. API.** Uma aplicação FastAPI carrega todos os modelos disponíveis na
subida e expõe endpoints de previsão. Como há vários modelos, todo pedido de
previsão precisa informar qual ticker usar. A previsão de vários dias à frente é
feita de forma **recursiva**: a previsão de um dia é realimentada como entrada
para prever o dia seguinte.

**7. Monitoramento.** Um middleware mede o tempo de cada requisição e o endpoint
`/metrics` expõe, no formato Prometheus, latência, contagem de previsões por
ticker e uso de CPU/memória.

---

## 🗂️ Estrutura do projeto

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
│       ├── schemas.py        # Contratos de entrada/saída (Pydantic)
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

## Como rodar

### 1. Preparar o ambiente

```bash
cd TechChallenge-Fase4
python -m venv .venv && source .venv/bin/activate   # recomendado
pip install -r requirements.txt
```

### 2. Treinar os modelos (um por ticker)

Este passo baixa os dados, treina uma LSTM para cada ticker, avalia e salva os
artefatos:

```bash
python scripts/train.py                       # treina todos os SYMBOLS (DIS, AAPL, MSFT)
python scripts/train.py --symbols DIS,AAPL    # treina só esses
python scripts/train.py --epochs 50 --sequence-length 90
```

Ao final, cada ticker tem sua pasta `models/<TICKER>/` com:
- `lstm_model.keras` — o modelo treinado
- `scaler.pkl` — o `MinMaxScaler` ajustado
- `metadata.json` — hiperparâmetros e métricas (MAE, RMSE, MAPE)

### 3. Subir a API

```bash
uvicorn src.api.main:app --reload
```

- Página inicial (escolha do ticker): <http://localhost:8000/>
- Documentação interativa (Swagger): <http://localhost:8000/docs>

A página inicial lista os modelos disponíveis (com suas métricas), deixa você
**escolher um ticker** e testar as duas formas de previsão sem escrever código.
O resultado é exibido em um **gráfico de linha** que mostra o histórico de
preços (em azul) e a **continuação prevista** (em laranja), conectados no último
ponto real — deixando visível onde termina o passado e começa a previsão.

### 4. (Opcional) Rodar com Docker + Prometheus

Para subir a API junto com um Prometheus já configurado:

```bash
docker compose up --build
# API .......... http://localhost:8000/
# Prometheus ... http://localhost:9090
```

> Treine os modelos **antes** de buildar (os artefatos em `models/` entram na
> imagem) ou use o volume já mapeado no `docker-compose.yml`.

### 5. Rodar os testes

```bash
pytest
```

---

## 🔌 Endpoints da API

| Método | Rota | Descrição |
|--------|------|-----------|
| `GET`  | `/` | Página inicial — **escolha do ticker** e previsão |
| `GET`  | `/health` | Saúde do serviço e tickers disponíveis |
| `GET`  | `/models` | **Catálogo dos modelos treinados** (para escolher o `symbol`) |
| `POST` | `/predict` | Previsão a partir de preços informados (**exige `symbol`**) |
| `POST` | `/predict/latest` | Baixa dados atuais do ticker e prevê (**exige `symbol`**) |
| `GET`  | `/monitor` | **Painel visual de monitoramento** (auto-refresh) |
| `GET`  | `/monitor/data` | Métricas agregadas em JSON (alimenta o painel) |
| `GET`  | `/metrics` | Métricas Prometheus, formato texto (para *scraping*) |
| `GET`  | `/docs` | Swagger UI |

> ⚠️ Como a API serve **vários modelos**, `/predict` e `/predict/latest`
> **exigem** o campo `symbol`, que deve ser um dos tickers treinados. Consulte
> `GET /models` para ver as opções e suas métricas.

### `GET /models` — descobrir os modelos disponíveis

```json
{
  "count": 3,
  "available_symbols": ["AAPL", "DIS", "MSFT"],
  "models": [
    {"symbol": "AAPL", "sequence_length": 60, "metrics": {"mae": 6.49, "rmse": 8.50, "mape": 2.70}, ...},
    {"symbol": "DIS",  "sequence_length": 60, "metrics": {"mae": 2.13, "rmse": 3.01, "mape": 2.06}, ...},
    {"symbol": "MSFT", "sequence_length": 60, "metrics": {"mae": 13.20, "rmse": 16.93, "mape": 3.03}, ...}
  ]
}
```

### `POST /predict` — prever a partir de preços que você fornece

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "prices": [100.1, 101.3, 99.8, 102.5, ...], "horizon": 5}'
```

```json
{
  "symbol": "AAPL",
  "horizon": 5,
  "last_input_price": 102.5,
  "predictions": [102.7, 103.1, 103.4, 103.2, 103.6],
  "inference_ms": 48.21
}
```

> `prices` deve ter pelo menos `sequence_length` valores (padrão **60**), em
> ordem cronológica. `horizon` é quantos dias à frente prever (recursivamente).

### `POST /predict/latest` — prever com os dados mais recentes

```bash
curl -X POST http://localhost:8000/predict/latest \
  -H "Content-Type: application/json" \
  -d '{"symbol": "DIS", "horizon": 3}'
```

Aqui você não precisa mandar os preços: a API baixa sozinha o histórico recente
do ticker escolhido no Yahoo Finance e devolve a previsão.

---

## 🧠 Detalhes do modelo LSTM

- **Um modelo independente por ticker** — mesma arquitetura, mas pesos e scaler
  próprios, salvos em `models/<TICKER>/`.
- **Entrada:** janela deslizante de `sequence_length` preços de fechamento.
- **Arquitetura:** camadas `LSTM` empilhadas (`lstm_layers`) com `Dropout`,
  seguidas de `Dense(1)`.
- **Pré-processamento:** `MinMaxScaler` **ajustado só no treino** (evita data
  leakage).
- **Treino:** otimizador `Adam`, perda `MSE`, com `EarlyStopping` e
  `ReduceLROnPlateau`.
- **Previsão multi-passo:** feita por realimentação recursiva das previsões.

### Avaliação

As métricas são calculadas em **escala real de preço** (após desfazer a
normalização):

- **MAE** — Erro Absoluto Médio
- **RMSE** — Raiz do Erro Quadrático Médio
- **MAPE** — Erro Percentual Absoluto Médio

Resultados obtidos no treino dos três modelos (histórico de 2020-01-01 a
2026-07-01, janela de 60 dias):

| Ticker | MAE | RMSE | MAPE |
|--------|-----|------|------|
| DIS  | 2.13  | 3.01  | **2.06%** |
| AAPL | 6.49  | 8.50  | **2.70%** |
| MSFT | 13.20 | 16.93 | **3.03%** |

Cada modelo guarda suas métricas em `models/<TICKER>/metadata.json`, também
acessíveis via `GET /models`.

> Observação: os modelos são treinados com dados até `END_DATE` (2026-07-01).
> Para manter as previsões "ao vivo" (`/predict/latest`) atualizadas, retreine
> periodicamente com uma data final mais recente, por exemplo:
> `python scripts/train.py --end 2026-12-31`.

---

## 📈 Escalabilidade e Monitoramento

O endpoint `/metrics` expõe, no formato Prometheus:

- `http_requests_total` — contagem por rota/método/status
- `http_request_duration_seconds` — **tempo de resposta** (histograma)
- `model_inference_duration_seconds` — latência de inferência
- `model_predictions_total{symbol=...}` — total de previsões **por ticker**
- `process_cpu_percent` / `process_memory_mb` — **uso de recursos**

Além disso, cada resposta HTTP traz o cabeçalho `X-Process-Time-ms` com o tempo
de processamento.

### Painel `/monitor` — monitoramento sem infraestrutura extra

Como `/metrics` devolve **texto cru** (feito para máquina ler), a API também
serve um **painel visual em `/monitor`**, na mesma origem da página principal —
sem precisar subir o Prometheus nem outro serviço:

- Cartões: total de requisições, tempo médio de resposta, previsões, latência de
  inferência, memória e CPU
- Tabela por endpoint: contagem, **média**, **p95** e quebra por status HTTP
- Barras de **previsões por ticker**
- Uptime e **auto-refresh** a cada 5s

`GET /monitor/data` devolve esses mesmos números em JSON — lidos direto do
registry do Prometheus, com o p95 estimado por interpolação dos buckets do
histograma.

### Prometheus (opcional, para histórico)

O Prometheus é um **serviço separado**, não um endpoint desta API: ele *scrapeia*
o `/metrics` periodicamente e guarda a série temporal. O `docker-compose.yml` já
o sobe apontando para a API (`monitoring/prometheus.yml`):

```bash
docker compose up --build     # API em :8000 · Prometheus em :9090
```

> Publicá-lo na internet exigiria um deploy próprio — e ele **não tem
> autenticação nativa**. Para o ambiente em nuvem, o `/monitor` cobre a
> visualização sem custo nem exposição adicional.

> ⚠️ **Múltiplos workers:** com `--workers > 1` o `prometheus_client` precisa do
> *modo multiprocess*; sem isso cada worker contabiliza apenas as próprias
> métricas. Com 1 worker (padrão do Render) os números ficam consistentes.

**Escalabilidade:** a aplicação é *stateless* (todo o estado vive nos artefatos
em `models/`), então escala horizontalmente. Para mais throughput, basta rodar
múltiplos workers:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## ✅ Testes

```bash
pytest
```

- `tests/test_metrics.py` — valida as métricas e a geração de sequências (roda
  sem TensorFlow).
- `tests/test_api.py` — smoke test da API: sobe a aplicação, checa o `/health` e
  garante que a previsão exige um ticker válido.

---

## ⚙️ Configuração

Todos os parâmetros podem ser ajustados por variáveis de ambiente ou por um
arquivo `.env` (veja [`.env.example`](.env.example)). Os principais:

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `SYMBOLS` | `DIS,AAPL,MSFT` | Tickers (um modelo por ticker), separados por vírgula |
| `START_DATE` / `END_DATE` | `2020-01-01` / `2026-07-01` | Intervalo do histórico |
| `SEQUENCE_LENGTH` | `60` | Tamanho da janela temporal |
| `LSTM_UNITS` / `LSTM_LAYERS` | `64` / `2` | Capacidade da rede |
| `EPOCHS` / `BATCH_SIZE` | `100` / `32` | Treino |

---

## 📦 Entregáveis do desafio

- ✅ Código-fonte do modelo LSTM + documentação (este repositório)
- ✅ `Dockerfile` + `docker-compose.yml` para conteinerizar a API
- ✅ API RESTful (FastAPI) servindo os modelos, com escolha de ticker - https://techchallenge-fase4.onrender.com/
- ✅ Monitoramento (Prometheus) de tempo de resposta e uso de recursos
- 🎥 Vídeo de demonstração —
