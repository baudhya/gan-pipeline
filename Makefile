.PHONY: install venv lint format typecheck test train generate download-weights \
        mlflow clean clean-venv clean-all

VENV_DIR ?= .venv
PYTHON   ?= python3

# ── Setup ─────────────────────────────────────────────────────────────────────

venv:
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --quiet --upgrade pip
	@echo "Virtual environment created at $(VENV_DIR)"
	@echo "Activate with: source $(VENV_DIR)/bin/activate"

install:
	pip install -e ".[dev]"

# ── Quality ───────────────────────────────────────────────────────────────────

lint:
	ruff check src tests scripts
	black --check src tests scripts
	isort --check-only src tests scripts

format:
	black src tests scripts
	isort src tests scripts
	ruff check --fix src tests scripts

typecheck:
	mypy src

test:
	pytest

# ── Data ──────────────────────────────────────────────────────────────────────

download-weights:
	python scripts/download_vgg_weights.py

prepare-data:
	python scripts/prepare_data.py \
	  --mode sen12ms \
	  --s1-dir data/raw/s1 \
	  --s2-dir data/raw/s2 \
	  --output-dir data/sar_eo \
	  --sar-already-db \
	  --sar-channels 1

# ── Training ──────────────────────────────────────────────────────────────────

train:
	python scripts/train_pix2pix.py

train-dcgan:
	python scripts/train.py model=dcgan training=default data=celeba experiment_name=dcgan_baseline

generate:
	python scripts/generate.py checkpoint=outputs/dcgan_baseline/checkpoints/epoch_0199.pt

# ── Docker ────────────────────────────────────────────────────────────────────

mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

docker-build:
	docker compose -f docker/docker-compose.yml build

docker-train:
	docker compose -f docker/docker-compose.yml up train

# ── Clean ─────────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -name "*.egg-info" -exec rm -rf {} +

clean-venv:
	rm -rf $(VENV_DIR)
	@echo "Removed $(VENV_DIR)"

clean-all: clean clean-venv
