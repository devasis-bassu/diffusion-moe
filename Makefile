.PHONY: install test lint train train-distributed eval

NUM_GPUS ?= $(shell python -c "import torch; print(torch.cuda.device_count() or 1)" 2>/dev/null || echo 1)

install:
	uv pip install -r requirements.txt
	uv pip install -e .

test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ scripts/ tests/

# Local: single device (CUDA > MPS > CPU, auto-detected by Trainer).
train:
	python scripts/train.py

# Remote (vast.ai): multiple CUDA GPUs via torch.distributed + DDP.
# Override GPU count with e.g. `make train-distributed NUM_GPUS=8`.
train-distributed:
	torchrun --nproc_per_node=$(NUM_GPUS) scripts/train.py

eval:
	python scripts/evaluate.py --checkpoint $(CHECKPOINT)
