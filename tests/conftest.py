"""
Shared test fixtures.

Downloads sample data from HuggingFace (karisu/General_Dataset).
Cached in tests/.cache/ so downloads only happen once per machine.
"""
import pytest
from pathlib import Path

from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra

CACHE_DIR = Path(__file__).parent / ".cache"


@pytest.fixture(autouse=True)
def clear_hydra():
    """Clear Hydra state before each test."""
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


@pytest.fixture(scope="session")
def training_data():
    """Download the General_Dataset test sample from HuggingFace.

    Returns dict with train/val csv_file and root_dir paths.
    Same pattern as the demo notebook.
    """
    from huggingface_hub import snapshot_download, hf_hub_download

    data_dir = CACHE_DIR / "data"
    images_dir = data_dir / "test_sample"
    csv_path = data_dir / "test_sample.csv"

    if not images_dir.exists() or not csv_path.exists():
        snapshot_download(
            repo_id="karisu/General_Dataset",
            repo_type="dataset",
            local_dir=str(data_dir),
            allow_patterns=["test_sample/*"],
            revision="main",
        )
        hf_hub_download(
            repo_id="karisu/General_Dataset",
            repo_type="dataset",
            filename="test_sample.csv",
            local_dir=str(data_dir),
        )

    assert images_dir.exists(), f"Sample images not found at {images_dir}"
    assert csv_path.exists(), f"Sample CSV not found at {csv_path}"

    return {
        "train_csv": str(csv_path),
        "train_root": str(images_dir),
        "val_csv": str(csv_path),
        "val_root": str(images_dir),
    }


@pytest.fixture
def load_config():
    """Load a Hydra config from configs/demo/."""
    config_dir = str(Path(__file__).parent.parent / "configs" / "demo")

    def _load(config_name: str, overrides: list = None):
        with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
            return compose(config_name=config_name, overrides=overrides or [])

    return _load
