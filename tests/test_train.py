"""
Training pipeline tests.

Uses HuggingFace sample data (downloaded + patched in conftest.py).
Runs 2-epoch training with each model architecture to verify the full pipeline.
"""
import torch
from pathlib import Path

import pytest
from animaloc.utils.train import main


def _detect_device():
    """Pick the best available device for testing."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _common_overrides(training_data, tmp_output_dir):
    """Return overrides shared by all training tests."""
    return [
        f"datasets.train.csv_file={training_data['train_csv']}",
        f"datasets.train.root_dir={training_data['train_root']}",
        f"datasets.validate.csv_file={training_data['val_csv']}",
        f"datasets.validate.root_dir={training_data['val_root']}",
        "training_settings.epochs=2",
        "training_settings.batch_size=2",
        "training_settings.num_workers=0",
        "training_settings.warmup_iters=1",
        "wandb_flag=False",
        "model.load_from=null",
        f"hydra.run.dir={tmp_output_dir}",
        f"device_name={_detect_device()}",
    ]


@pytest.fixture
def tmp_output_dir(tmp_path):
    return str(tmp_path / "output")


def test_train_dla34(load_config, training_data, tmp_output_dir):
    overrides = _common_overrides(training_data, tmp_output_dir) + [
        "datasets.num_classes=7",
        "++datasets.class_def={1: buffalo, 2: elephant, 3: kob, 4: topi, 5: warthog, 6: waterbuck}",
        "losses.CrossEntropyLoss.kwargs.weight=[0.1,1.0,2.0,1.0,6.0,12.0,1.0]",
    ]
    cfg = load_config("dla34_delplanque", overrides=overrides)
    results = main(cfg)
    assert results is not None


def test_train_timm_dla34(load_config, training_data, tmp_output_dir):
    overrides = _common_overrides(training_data, tmp_output_dir) + [
        "datasets.num_classes=7",
        "++datasets.class_def={1: buffalo, 2: elephant, 3: kob, 4: topi, 5: warthog, 6: waterbuck}",
        "losses.CrossEntropyLoss.kwargs.weight=[0.1,1.0,2.0,1.0,6.0,12.0,1.0]",
    ]
    cfg = load_config("dla34_timm", overrides=overrides)
    results = main(cfg)
    assert results is not None


@pytest.mark.slow
def test_train_convnext_camouflaged(load_config, training_data, tmp_output_dir):
    overrides = _common_overrides(training_data, tmp_output_dir) + [
        "datasets.num_classes=7",
        "++datasets.class_def={1: buffalo, 2: elephant, 3: kob, 4: topi, 5: warthog, 6: waterbuck}",
        "losses.CrossEntropyLoss.kwargs.weight=[0.1,1.0,2.0,1.0,6.0,12.0,1.0]",
    ]
    cfg = load_config("convnext_camouflaged", overrides=overrides)
    results = main(cfg)
    assert results is not None
