import os
from pathlib import Path

import hydra
from omegaconf import DictConfig

from tools.train import main

import pytest

import pytest
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from pathlib import Path


@pytest.fixture(autouse=True)
def clear_hydra():
    """Clear Hydra state before each test."""
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


@pytest.fixture
def load_config():
    config_dir = str(Path(__file__).parent.parent / "configs" / "demo")

    def _load(config_name: str, overrides: list = None):
        with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
            return compose(config_name=config_name, overrides=overrides or [])

    return _load

def test_train_dla34(load_config):

    cfg = load_config("dla34_delplanque")
    result = main(cfg)


def test_train_timm_dla34(load_config):

    cfg = load_config("dla34_timm")
    result = main(cfg)





def test_train_convnext_camouflaged(load_config):
    cfg = load_config("convnext_camouflaged")
    result = main(cfg)