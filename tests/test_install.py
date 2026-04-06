"""
Installation and import tests.

Fast smoke tests that verify all dependencies are importable
and key animaloc modules load correctly. These run on every
environment; training tests are marked slow and skipped by default.
"""
import importlib
import sys

import pytest


# --- Dependency imports ---

REQUIRED_PACKAGES = [
    "torch",
    "timm",
    "albumentations",
    "hydra",
    "cv2",
    "PIL",
    "sklearn",
    "skimage",
    "matplotlib",
    "tqdm",
    "loguru",
    "seaborn",
    "numpy",
    "scipy",
    "pandas",
    "huggingface_hub",
]


@pytest.mark.parametrize("package", REQUIRED_PACKAGES)
def test_import_dependency(package):
    """Each core dependency should be importable."""
    importlib.import_module(package)


# --- Version checks ---

def test_pillow_version():
    from PIL import Image
    major = int(Image.__version__.split(".")[0])
    assert major >= 10, f"Pillow too old: {Image.__version__}"


def test_opencv_version():
    import cv2
    major, minor = (int(x) for x in cv2.__version__.split(".")[:2])
    assert (major, minor) >= (4, 8), f"OpenCV too old: {cv2.__version__}"


def test_numpy_version():
    import numpy as np
    major = int(np.__version__.split(".")[0])
    assert major >= 2, f"NumPy too old: {np.__version__}"


# --- Animaloc module imports ---

ANIMALOC_MODULES = [
    "animaloc",
    "animaloc.models",
    "animaloc.models.herdnet",
    "animaloc.models.herdnet_timm_dla",
    "animaloc.models.herdnet_timm_convnext_camouflaged",
    "animaloc.models.utils",
    "animaloc.datasets",
    "animaloc.datasets.csv",
    "animaloc.data",
    "animaloc.utils.train",
    "animaloc.utils.augmentations",
]


@pytest.mark.parametrize("module", ANIMALOC_MODULES)
def test_import_animaloc_module(module):
    """Each animaloc sub-module should be importable."""
    importlib.import_module(module)