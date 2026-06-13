.PHONY: install lint format typecheck test train generate clean

install:
	pip install -e ".[dev]"

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

train:
	python scripts/train_pix2pix.py

train-dcgan:
	python scripts/train.py model=dcgan training=default data=celeba experiment_name=dcgan_baseline

generate:
	python scripts/generate.py checkpoint=outputs/dcgan_baseline/checkpoints/epoch_0199.pt

docker-build:
	docker compose -f docker/docker-compose.yml build

docker-train:
	docker compose -f docker/docker-compose.yml up train

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -name "*.egg-info" -exec rm -rf {} +
