"""
Shared test fixtures.

Downloads sample data from HuggingFace (karisu/General_Dataset)
and creates 512x512 training patches using HerdNet's PatchesBuffer.
Cached in tests/.cache/ so downloads only happen once per machine.
"""
import pytest
from pathlib import Path

from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra

CACHE_DIR = Path(__file__).parent / ".cache"
PATCH_SIZE = 512


@pytest.fixture(autouse=True)
def clear_hydra():
    """Clear Hydra state before each test."""
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


@pytest.fixture(scope="session")
def training_data():
    """Download sample data from HuggingFace and create 512x512 patches.

    Uses the same HuggingFace download as the demo notebook,
    then HerdNet's PatchesBuffer to create training crops.
    """
    from huggingface_hub import snapshot_download, hf_hub_download
    from PIL import Image
    import numpy as np
    from animaloc.data import PatchesBuffer

    # --- Download from HuggingFace (same as notebook) ---
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

    # --- Create 512x512 patches using HerdNet's PatchesBuffer ---
    patches_dir = CACHE_DIR / "patches"
    gt_csv = patches_dir / "gt.csv"

    if not gt_csv.exists():
        patches_dir.mkdir(parents=True, exist_ok=True)

        buffer = PatchesBuffer(
            csv_file=str(csv_path),
            root_dir=str(images_dir),
            patch_size=(PATCH_SIZE, PATCH_SIZE),
            overlap=0,
            min_visibility=0.0,
        )

        buffer.buffer.drop(columns="limits").to_csv(str(gt_csv), index=False)

        for img_name in buffer.buffer["base_images"].unique():
            pil_img = Image.open(str(images_dir / img_name)).convert("RGB")

            img_patches = buffer.buffer[buffer.buffer["base_images"] == img_name]
            for row in img_patches[["images", "limits"]].to_numpy().tolist():
                patch_name, limits = row[0], row[1]
                cropped = pil_img.crop(limits.get_tuple)
                # Pad to PATCH_SIZE x PATCH_SIZE with black if needed
                if cropped.size != (PATCH_SIZE, PATCH_SIZE):
                    padded = Image.new("RGB", (PATCH_SIZE, PATCH_SIZE), (0, 0, 0))
                    padded.paste(cropped, (0, 0))
                    cropped = padded
                cropped.save(str(patches_dir / patch_name))

    assert gt_csv.exists(), f"Patching did not create {gt_csv}"

    return {
        "train_csv": str(gt_csv),
        "train_root": str(patches_dir),
        "val_csv": str(gt_csv),
        "val_root": str(patches_dir),
    }


@pytest.fixture
def load_config():
    """Load a Hydra config from configs/demo/."""
    config_dir = str(Path(__file__).parent.parent / "configs" / "demo")

    def _load(config_name: str, overrides: list = None):
        with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
            return compose(config_name=config_name, overrides=overrides or [])

    return _load
