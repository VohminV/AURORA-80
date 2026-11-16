PYTHON ?= python
CUDA_PYTHON ?= .venv-cuda/bin/python
PIP ?= $(PYTHON) -m pip
CONFIG ?= configs/aurora_100m.yaml

.PHONY: install install-dev test lint format format-check check smoke-data smoke-train \
	smoke-generate smoke-perplexity eval quant-compare state-check ablate profile benchmark \
	benchmark-cuda clean

install:
	$(PIP) install -e ".[tokenizer]"

install-dev:
	$(PIP) install -e ".[tokenizer,dev]"

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check aurora tests scripts
	$(PYTHON) -m mypy aurora

format:
	$(PYTHON) -m ruff format aurora tests scripts

format-check:
	$(PYTHON) -m ruff format --check aurora tests scripts

check: lint format-check test

smoke-data:
	$(PYTHON) scripts/make_demo_corpus.py --output data/demo.txt --lines 2000
	$(PYTHON) -m aurora.data.prepare \
		--input data/demo.txt \
		--output-dir data/smoke \
		--tokenizer byte \
		--val-fraction 0.05

smoke-train:
	$(PYTHON) -m aurora.training.pretrain --config configs/smoke.yaml

smoke-generate:
	$(PYTHON) -m aurora.generation \
		--checkpoint runs/smoke/best.pt \
		--prompt "AURORA" \
		--tokenizer byte \
		--max-new-tokens 64 \
		--temperature 0.8 \
		--top-k 20 \
		--top-p 0.95 \
		--device cpu

smoke-perplexity:
	$(PYTHON) -m aurora.benchmarks.perplexity \
		--checkpoint runs/smoke/best.pt \
		--data data/smoke/val.bin \
		--batch-size 4 \
		--batches 20 \
		--device cpu

eval:
	mkdir -p reports
	$(PYTHON) -m aurora.benchmarks.evaluate \
		--checkpoint runs/smoke/best.pt \
		--data val=data/smoke/val.bin \
		--output reports/smoke_eval.json

quant-compare:
	mkdir -p reports
	$(PYTHON) -m aurora.benchmarks.quantization \
		--checkpoint runs/smoke/best.pt \
		--data data/smoke/val.bin \
		--group-size 16 \
		--min-features 16 \
		--output reports/smoke_quantization.json

state-check:
	mkdir -p reports
	$(PYTHON) -m aurora.benchmarks.state_stability \
		--checkpoint runs/smoke/best.pt \
		--lengths 4096 8192 16384 32768 \
		--chunk-size 256 \
		--output reports/state_stability.json

ablate:
	mkdir -p reports
	$(PYTHON) -m aurora.benchmarks.ablation \
		--config configs/smoke.yaml \
		--variants core core_memory full \
		--output reports/smoke_ablation.json

profile:
	mkdir -p reports
	$(PYTHON) scripts/profile_components.py --config $(CONFIG) \
		--output reports/profile_components.json

benchmark:
	mkdir -p reports
	$(PYTHON) scripts/benchmark.py --config $(CONFIG) --output reports/benchmark.json

benchmark-cuda:
	mkdir -p reports
	$(CUDA_PYTHON) scripts/benchmark.py \
		--config $(CONFIG) \
		--device cuda \
		--seq-len 512 \
		--batch-size 1 \
		--train-steps 3 \
		--inference-steps 3 \
		--generation-tokens 4 \
		--output reports/benchmark_cuda.json

clean:
	rm -rf runs .pytest_cache .mypy_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
