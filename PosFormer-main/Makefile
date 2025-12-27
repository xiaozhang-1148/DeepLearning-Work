.PHONY: install test train evaluate clean

install:
	pip install -e .

test:
	pytest tests/

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info
