.PHONY: install train serve test docker-build docker-up clean

install:
	pip install -r requirements.txt

train:
	python scripts/train.py

serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest

docker-build:
	docker build -t stock-lstm-api .

docker-up:
	docker compose up --build

clean:
	rm -f models/lstm_model.keras models/scaler.pkl models/metadata.json
	find . -type d -name __pycache__ -exec rm -rf {} +
