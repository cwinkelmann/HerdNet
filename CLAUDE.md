# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HerdNet is a PyTorch-based deep learning framework for animal localization and counting in aerial drone imagery. The installable package is named **`animaloc`**. It is a fork of [Alexandre Delplanque's HerdNet](https://github.com/Alexandre-Delplanque/HerdNet), extended for Marine Iguana detection on the Galapagos Islands.

## Build & Install

```bash
pip install -e .              # Editable install (recommended for development)
pip install -e ".[dev]"       # Includes pytest
pip install -e ".[all]"      # Everything: dev + tracking + notebook + viz
```

Optional extras: `tracking` (wandb), `notebook` (jupyterlab, ipywidgets), `viz` (fiftyone).

Requires Python >= 3.11. PyTorch must be installed separately (not in pyproject.toml).

## Running Tests

```bash
pytest tests/ -v                          # All tests
pytest tests/test_train.py -v             # Training pipeline tests
pytest tests/test_inference.py -v         # Inference tests
pytest tests/test_inference.py::TestInference::test_inference_runs -v  # Single test
```

Tests use Hydra configs from `configs/demo/` and require clearing Hydra global state between runs (handled by the `clear_hydra` autouse fixture).

## Training & Inference CLI

```bash
python tools/train.py                     # Training (Hydra, default config in tools/train.py)
python tools/train.py --config-path <dir> --config-name <name> key=value  # With overrides
python tools/infer.py --config-dir <dir> --images <path> --model <path>   # Inference
python tools/patcher.py <root> <h> <w> <overlap> <dest>                   # Create image patches
python tools/view.py <root> <gt_csv> [-dets <det_csv>]                    # FiftyOne visualization
```

## Architecture

When working on model architecture, dataset pipeline, or training flow, read `.claude/ARCHITECTURE.md` for detailed design patterns and package structure.

**Quick reference:**
- Entry point: `animaloc.utils.train.main(cfg)` called by `tools/train.py`
- Models: `animaloc/models/` — registered via `@MODELS.register()`
- Datasets: `animaloc/datasets/csv.py` — `CSVDataset` with format `images,x,y,labels`
- Training: `animaloc/train/trainers.py` — `Trainer` class
- Evaluation: `animaloc/eval/` — `HerdNetEvaluator` + `HerdNetStitcher` for tiled inference
- Configs: `configs/demo/` — modular Hydra YAML (datasets, model, losses, training_settings)

## Sample Data

Models and datasets are on HuggingFace:
- **Models**: `karisu/HerdNet`
- **Dataset**: `karisu/General_Dataset`

```python
from huggingface_hub import hf_hub_download, snapshot_download
model_path = hf_hub_download(repo_id="karisu/HerdNet", filename="general_2022/20220413_HerdNet_General_dataset_2022.pth", local_dir="./models")
sample_data = snapshot_download(repo_id="karisu/General_Dataset", repo_type="dataset", local_dir="./data", allow_patterns=["test_sample/*"])
```

## Best Practices

### Adding a new model
1. Create `animaloc/models/herdnet_<name>.py`
2. Decorate with `@MODELS.register()`
3. Implement `reshape_classes(num_classes)` and `check_trainable_parameters()`
4. Create a Hydra config under `configs/demo/model/<Name>.yaml`
5. Add a test case in `tests/test_train.py`

### Adding a new loss
1. Add to `animaloc/train/losses/` and register with `@LOSSES.register()`
2. Create a loss config under `configs/demo/losses/`
3. Reference in model config's `defaults` list

### Config conventions
- Use `${model.kwargs.down_ratio}` style interpolation to keep values DRY
- Override anything from CLI: `python tools/train.py model.kwargs.pretrained=False`
- Always set `wandb_flag=False` in tests

### Testing
- Tests auto-detect device (CUDA > MPS > CPU) — never hardcode a device
- Use `configs/demo/` configs with `_common_overrides()` for minimal test runs
- Clear Hydra global state between tests (`clear_hydra` fixture)
- Keep `epochs=2`, `batch_size=2`, `warmup_iters=1` for fast test iterations

### Code style
- All models must implement `reshape_classes()` for runtime class count changes
- Use `loguru.logger` (not `print`) for all logging
- Guard optional imports: `try: import wandb except ImportError: wandb = None`
- Use `weights_only=False` explicitly on `torch.load()` calls (until model files are converted to safe tensors)

## Logging

Training writes logs to **both stdout and a loguru file sink** (`YYYYMMDD_training.log` in the working directory). The file log includes all loguru messages and training/validation progress from the `CustomLogger`.

## Skills

Custom skills are in `.claude/skills/`. Use them via slash commands:

- `/train [config] [overrides...]` — Start training in background, auto-check logs after launch
- `/train-status [log path]` — Check if training is running, report epoch/loss/metrics
- `/read-logs [path]` — Display training or validation log contents
- `/analyse-logs [path]` — Deep analysis: loss trends, convergence, anomalies, recommendations

### General behavior for long-running tasks
- Always use `run_in_background: true` for training, inference, evaluation
- Proactively check logs after launching to confirm the job started
- Report errors immediately without waiting for the user to ask
