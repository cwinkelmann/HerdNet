"""
Pretrained model tests — inference and training with HuggingFace checkpoints.

Downloads pretrained models from karisu/HerdNet and verifies:
1. Model loads and produces correct output shapes (inference)
2. Model loads pretrained weights, reshapes classes, and trains (fine-tuning)

Models:
- general_2022: HerdNet DLA34, 7 classes (General Dataset 2022)
- timm_dla34: HerdNetTimmDLA, 3 classes (iguana) → reshaped to 7
- convnext_camouflaged: CamouflageHerdNetConvNeXt, 3 classes (iguana) → reshaped to 7
"""
import torch
import pytest
from pathlib import Path
from huggingface_hub import hf_hub_download

from animaloc.utils.train import main


CACHE_DIR = Path(__file__).parent / ".cache" / "models"


def _detect_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _common_overrides(training_data, tmp_output_dir):
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
        f"hydra.run.dir={tmp_output_dir}",
        f"device_name={_detect_device()}",
    ]


# 7-class overrides matching the General Dataset test data
_CLASS_OVERRIDES = [
    "datasets.num_classes=7",
    "++datasets.class_def={1: buffalo, 2: elephant, 3: kob, 4: topi, 5: warthog, 6: waterbuck}",
    "losses.CrossEntropyLoss.kwargs.weight=[0.1,1.0,2.0,1.0,6.0,12.0,1.0]",
]


@pytest.fixture(scope="session")
def model_general_2022():
    """Download HerdNet DLA34 pretrained on General Dataset 2022 (7 classes)."""
    return hf_hub_download(
        repo_id="karisu/HerdNet",
        filename="general_2022/model.pth",
        local_dir=str(CACHE_DIR),
    )


@pytest.fixture(scope="session")
def model_timm_dla34():
    """Download HerdNetTimmDLA pretrained on iguana (3 classes)."""
    return hf_hub_download(
        repo_id="karisu/HerdNet",
        filename="timm_dla34/model.pth",
        local_dir=str(CACHE_DIR),
    )


@pytest.fixture(scope="session")
def model_convnext():
    """Download CamouflageHerdNetConvNeXt pretrained on iguana (3 classes)."""
    return hf_hub_download(
        repo_id="karisu/HerdNet",
        filename="convnext_camouflaged/model.pth",
        local_dir=str(CACHE_DIR),
    )


# --- Inference tests ---

class TestInferenceGeneral2022:
    def test_inference(self, model_general_2022):
        from animaloc.models import HerdNet
        from animaloc.models.utils import LossWrapper, load_model

        model = HerdNet(num_classes=7, down_ratio=2, head_conv=64)
        wrapped = LossWrapper(model, [])
        load_model(wrapped, model_general_2022, device="cpu")
        wrapped.eval()

        x = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            out = wrapped.model(x)

        assert len(out) == 2, f"Expected 2 outputs, got {len(out)}"
        assert out[0].shape == (1, 1, 256, 256), f"Heatmap shape: {out[0].shape}"
        assert out[1].shape[0] == 1 and out[1].shape[1] == 7, f"Cls shape: {out[1].shape}"


class TestInferenceTimmDla34:
    def test_inference(self, model_timm_dla34):
        from animaloc.models.herdnet_timm_dla import HerdNetTimmDLA
        from animaloc.models.utils import LossWrapper, load_model

        model = HerdNetTimmDLA(num_classes=3, backbone="timm/dla34", down_ratio=4, head_conv=128)
        wrapped = LossWrapper(model, [])
        load_model(wrapped, model_timm_dla34, device="cpu")
        wrapped.eval()

        x = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            out = wrapped.model(x)

        assert len(out) == 2, f"Expected 2 outputs, got {len(out)}"
        assert out[0].shape == (1, 1, 128, 128), f"Heatmap shape: {out[0].shape}"
        assert out[1].shape[0] == 1 and out[1].shape[1] == 3, f"Cls shape: {out[1].shape}"


@pytest.mark.slow
class TestInferenceConvNext:
    def test_inference(self, model_convnext):
        from animaloc.models.herdnet_timm_convnext_camouflaged import CamouflageHerdNetConvNeXt
        from animaloc.models.utils import LossWrapper, load_model

        model = CamouflageHerdNetConvNeXt(num_classes=3, down_ratio=4, backbone_size="base")
        wrapped = LossWrapper(model, [])
        load_model(wrapped, model_convnext, device="cpu")
        wrapped.eval()

        x = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            out = wrapped.model(x)

        assert len(out) == 2, f"Expected 2 outputs, got {len(out)}"
        assert out[0].shape == (1, 1, 128, 128), f"Heatmap shape: {out[0].shape}"
        assert out[1].shape[0] == 1 and out[1].shape[1] == 3, f"Cls shape: {out[1].shape}"


# --- Training tests (load pretrained → reshape_classes → fine-tune 2 epochs) ---

@pytest.fixture
def tmp_output_dir(tmp_path):
    return str(tmp_path / "output")


@pytest.mark.slow
class TestTrainGeneral2022:
    """Fine-tune 7-class DLA34 on 7-class General Dataset (no reshape needed)."""
    def test_finetune(self, load_config, training_data, tmp_output_dir, model_general_2022):
        overrides = _common_overrides(training_data, tmp_output_dir) + _CLASS_OVERRIDES + [
            f"model.load_from={model_general_2022}",
        ]
        cfg = load_config("dla34_delplanque", overrides=overrides)
        results = main(cfg)
        assert results is not None


@pytest.mark.slow
class TestTrainTimmDla34:
    """Load 3-class timm DLA34, reshape to 7 classes, fine-tune on General Dataset."""
    def test_finetune(self, load_config, training_data, tmp_output_dir, model_timm_dla34):
        overrides = _common_overrides(training_data, tmp_output_dir) + _CLASS_OVERRIDES + [
            "model.kwargs.head_conv=128",
            "model.kwargs.down_ratio=4",
            f"model.load_from={model_timm_dla34}",
        ]
        cfg = load_config("dla34_timm", overrides=overrides)
        results = main(cfg)
        assert results is not None


@pytest.mark.slow
class TestTrainConvNext:
    """Load 3-class ConvNeXt, reshape to 7 classes, fine-tune on General Dataset."""
    def test_finetune(self, load_config, training_data, tmp_output_dir, model_convnext):
        overrides = _common_overrides(training_data, tmp_output_dir) + _CLASS_OVERRIDES + [
            f"model.load_from={model_convnext}",
        ]
        cfg = load_config("convnext_camouflaged", overrides=overrides)
        results = main(cfg)
        assert results is not None
