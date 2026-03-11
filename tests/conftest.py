"""
Shared test fixtures.

Downloads sample data from HuggingFace and creates training patches.
Cached in tests/.cache/ so downloads + patching only happen once per machine.
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
def hf_data_dir():
    """Download the General_Dataset test sample from HuggingFace.

    Returns path to directory containing test_sample/ images and test_sample.csv.
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

    return data_dir


@pytest.fixture(scope="session")
def training_data(hf_data_dir):
    """Create 512x512 training patches from the HuggingFace sample data.

    Uses animaloc.data.patches API to split full-size images into crops.
    Returns dict with train/val csv_file and root_dir paths.
    """
    import PIL
    import numpy
    import cv2
    import torchvision
    from albumentations import PadIfNeeded
    from animaloc.data import PatchesBuffer

    source_images = hf_data_dir / "test_sample"
    source_csv = hf_data_dir / "test_sample.csv"

    patches_dir = CACHE_DIR / "patches"
    gt_csv = patches_dir / "gt.csv"

    if not patches_dir.exists() or not gt_csv.exists():
        patches_dir.mkdir(parents=True, exist_ok=True)

        # Build patch buffer (annotations + limits)
        buffer = PatchesBuffer(
            csv_file=str(source_csv),
            root_dir=str(source_images),
            patch_size=(PATCH_SIZE, PATCH_SIZE),
            overlap=0,
            min_visibility=0.0,
        )

        # Save annotations CSV
        buffer.buffer.drop(columns="limits").to_csv(str(gt_csv), index=False)

        # Export annotated patch images
        for img_name in buffer.buffer["base_images"].unique():
            img_path = source_images / img_name
            pil_img = PIL.Image.open(str(img_path)).convert("RGB")

            padder = PadIfNeeded(
                PATCH_SIZE, PATCH_SIZE,
                position=PadIfNeeded.PositionType.TOP_LEFT,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
            )

            img_patches = buffer.buffer[buffer.buffer["base_images"] == img_name]
            for row in img_patches[["images", "limits"]].to_numpy().tolist():
                patch_name, limits = row[0], row[1]
                cropped = numpy.array(pil_img.crop(limits.get_tuple))
                padded = PIL.Image.fromarray(padder(image=cropped)["image"])
                padded.save(str(patches_dir / patch_name))

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
