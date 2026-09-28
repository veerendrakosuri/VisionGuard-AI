.PHONY: install lint test train evaluate predict

install:
	python -m pip install -e ".[dev]"

lint:
	python -m ruff check .
	python -m mypy src

test:
	python -m pytest

train:
	python scripts/train.py --config configs/patchcore_mvtecad2.yaml

evaluate:
	python scripts/evaluate.py --config configs/patchcore_mvtecad2.yaml

predict:
	python scripts/predict.py --config configs/patchcore_mvtecad2.yaml --image $(IMAGE)
