.PHONY: setup validate-feeder generate-data make-splits train-baselines train-gnn evaluate ablations robustness figures test test-fast audit clean

PYTHON ?= python

setup:
	$(PYTHON) -m pip install -r requirements-lock.txt
	$(PYTHON) -m pip install -e .

validate-feeder:
	$(PYTHON) scripts/validate_feeder.py --config configs/ieee123_base.yaml
	$(PYTHON) scripts/validate_feeder.py --config configs/secondary_ieee13.yaml

generate-data:
	$(PYTHON) scripts/generate_data.py --config configs/ieee123_base.yaml
	$(PYTHON) scripts/generate_data.py --config configs/secondary_ieee13.yaml

audit:
	$(PYTHON) scripts/audit_dataset.py --config configs/ieee123_base.yaml

make-splits:
	$(PYTHON) scripts/make_splits.py --config configs/ieee123_base.yaml

train-baselines:
	$(PYTHON) scripts/train_baselines.py --config configs/ieee123_base.yaml

train-gnn:
	$(PYTHON) scripts/train_gnn.py --config configs/ieee123_base.yaml

evaluate:
	$(PYTHON) scripts/evaluate.py --config configs/ieee123_base.yaml

ablations:
	$(PYTHON) scripts/run_ablations.py --config configs/ieee123_base.yaml

robustness:
	$(PYTHON) scripts/run_robustness.py --config configs/ieee123_base.yaml

figures:
	$(PYTHON) scripts/generate_figures.py --config configs/ieee123_base.yaml

test:
	$(PYTHON) -m pytest tests/ -v

test-fast:
	$(PYTHON) -m pytest tests/ -v -m "not simulation"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
