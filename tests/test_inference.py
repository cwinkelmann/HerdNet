"""
Test inference functionality
============================

Run with: pytest tests/test_inference.py -v
"""

import pytest
import pandas as pd
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Paths for testing
TEST_MODEL_PATH =  PROJECT_ROOT / "best_models/17-02-44_convnext_camouflaged_zr7ljum7/best_model.pth"
TEST_DATA_DIR = Path("data/single_images/ISWF01_22012023_subset")


@pytest.fixture
def inference_config() -> DictConfig:
    """Create a config for inference testing based on actual training config."""
    cfg = OmegaConf.create({
        "losses": {
            "FocalLoss": {
                "print_name": "focal_loss",
                "from_torch": False,
                "output_idx": 0,
                "target_idx": 0,
                "lambda_const": 1.0,
                "kwargs": {
                    "alpha": 2,
                    "beta": 4,
                    "reduction": "mean",
                    "normalize": False,
                }
            },
            "CrossEntropyLoss": {
                "print_name": "ce_loss",
                "from_torch": True,
                "output_idx": 1,
                "target_idx": 1,
                "lambda_const": 1.0,
                "background_class_weight": 0.1,
                "kwargs": {
                    "reduction": "mean",
                    "weight": [0.1, 5, 0.1],
                }
            }
        },
        "datasets": {
            "img_size": [512, 512],
            "anno_type": "point",
            "num_classes": 3,
            "collate_fn": None,
            "class_def": {
                1: "iguana_point",
                2: "hard_negative",
            },
            "train": {
                "name": "CSVDataset",
                "csv_file": None,
                "root_dir": None,
                "sampler": None,
                "augmentation_multiplier": 1,
                "albu_transforms": None,
                "end_transforms": None,
            },
            "validate": {
                "name": "CSVDataset",
                "csv_file": None,
                "root_dir": None,
                "albu_transforms": {
                    "Normalize": {"p": 1.0}
                },
                "end_transforms": {
                    "DownSample": {
                        "down_ratio": 4,
                        "anno_type": "point",
                    }
                },
            },
            "test": {
                "name": "CSVDataset",
                "csv_file": None,  # Will be auto-generated for plain_inference
                "root_dir": str(TEST_DATA_DIR),
                "albu_transforms": {
                    "Normalize": {"p": 1.0}
                },
                "end_transforms": {
                    "DownSample": {
                        "down_ratio": 4,
                        "anno_type": "point",
                    }
                },
            },
        },
        "training_settings": {
            "trainer": "Trainer",
            "epochs": 50,
            "valid_freq": 1,
            "print_freq": 225,
            "batch_size": 12,
            "optimizer": "adamW",
            "lr": 0.0001,
            "backbone_lr": 1e-05,
            "head_lr": 0.001,
            "weight_decay": 0.00016,
            "num_workers": 2,
            "auto_lr": {
                "mode": "max",
                "patience": 35,
                "threshold": 0.0001,
                "threshold_mode": "rel",
                "cooldown": 10,
                "min_lr": 1e-07,
                "verbose": True,
            },
            "warmup_iters": 500,
            "vizual_fn": "visualize_sample",
            "visualiser": {
                "name": "HeatMapVisualizer",
                "output_dir": "./visualizations",
                "down_ratio": 4,
            },
            "loss_evaluation": None,
            "evaluator": {
                "name": "HerdNetEvaluator",
                "threshold": 100,
                "select_mode": "max",
                "validate_on": "f1_score",
                "kwargs": {
                    "print_freq": 125,
                    "lmds_kwargs": {
                        "kernel_size": [9, 9],
                        "adapt_ts": 0.3,
                        "scale_factor": 1,
                        "up": True,
                    }
                }
            },
            "stitcher": {
                "name": "HerdNetStitcher",
                "kwargs": {
                    "overlap": 120,
                    "down_ratio": 4,
                    "up": False,
                    "reduction": "mean",
                }
            },
            "debug_visualiser": None,
        },
        "model": {
            "name": "CamouflageHerdNetConvNeXt",
            "from_torchvision": False,
            "load_from": str(TEST_MODEL_PATH),
            "resume_from": None,
            "kwargs": {
                "pretrained": True,
                "down_ratio": 4,
                "backbone_size": "base",
            },
            "freeze": None,
        },
        "wandb_notes": "test inference",
        "wandb_flag": False,
        "wandb_project": "test",
        "wandb_entity": "karisu",
        "wandb_run": "test_inference",
        "wandb_tags": ["test"],
        "seed": 1,
        "device_name": None,  # Will auto-detect
    })
    return cfg


class TestInference:
    """Test suite for inference functionality."""

    def test_model_path_exists(self):
        """Test that the model file exists."""
        assert Path(TEST_MODEL_PATH).exists(), f"Model not found: {TEST_MODEL_PATH}"

    def test_data_dir_exists(self):
        """Test that the test data directory exists."""
        assert Path(TEST_DATA_DIR).exists(), f"Data dir not found: {TEST_DATA_DIR}"

    def test_data_dir_has_images(self):
        """Test that the data directory contains images."""
        data_dir = Path(TEST_DATA_DIR)
        images = list(data_dir.glob("*.JPG")) + list(data_dir.glob("*.jpg"))
        assert len(images) > 0, f"No images found in {TEST_DATA_DIR}"
        print(f"Found {len(images)} images")

    def test_inference_runs(self, inference_config):
        """Test that inference runs without errors."""
        from tools.inference_test import inference

        # Run inference
        detections = inference(inference_config, plain_inference=True, vis_detections=False)

        # Basic assertions
        assert detections is not None
        assert isinstance(detections, pd.DataFrame)
        print(f"Got {len(detections)} detections")

    def test_inference_output_format(self, inference_config):
        """Test that inference output has expected columns."""
        from tools.inference_test import inference

        detections = inference(inference_config, plain_inference=True, vis_detections=False)

        # Check required columns exist
        required_columns = ['images', 'x', 'y', 'scores', 'labels']
        for col in required_columns:
            assert col in detections.columns, f"Missing column: {col}"

        assert len(detections) == 16
    #
    # def test_inference_coordinates_valid(self, inference_config):
    #     """Test that detection coordinates are valid."""
    #     from tools.inference_test import inference

        detections = inference(inference_config, plain_inference=True, vis_detections=False)

        if len(detections) > 0:
            # Coordinates should be positive
            assert (detections['x'] >= 0).all(), "Negative x coordinates found"
            assert (detections['y'] >= 0).all(), "Negative y coordinates found"

            # Scores should be between 0 and 1
            assert (detections['scores'] >= 0).all(), "Scores below 0 found"
            assert (detections['scores'] <= 1).all(), "Scores above 1 found"

    def test_inference_detects_iguanas(self, inference_config):
        """Test that inference detects at least some iguanas."""
        from tools.inference_test import inference

        detections = inference(inference_config, plain_inference=True, vis_detections=False)

        # Filter by confidence threshold
        confident_detections = detections[detections['scores'] > 0.5]

        # We expect at least some detections in the test images
        assert len(confident_detections) > 0, "No confident detections found"
        print(f"Found {len(confident_detections)} detections with score > 0.5")


class TestInferencePerformance:
    """Performance-related tests."""

    @pytest.mark.slow
    def test_inference_speed(self, inference_config):
        """Test inference completes within reasonable time."""
        import time
        from tools.inference_test import inference

        start = time.time()
        detections = inference(inference_config, plain_inference=True, vis_detections=False)
        elapsed = time.time() - start

        n_images = len(pd.unique(detections['images'])) if len(detections) > 0 else 0

        print(f"Inference took {elapsed:.2f}s for {n_images} images")
        print(f"Average: {elapsed/max(n_images, 1):.2f}s per image")

        # Adjust threshold based on your hardware
        assert elapsed < 300, f"Inference too slow: {elapsed:.2f}s"


# Optional: Parametrized test for different confidence thresholds
@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7, 0.9])
def test_detection_counts_at_thresholds(inference_config, threshold):
    """Test detection counts at various confidence thresholds."""
    from tools.inference_test import inference

    detections = inference(inference_config, plain_inference=True, vis_detections=False)

    n_above = len(detections[detections['scores'] >= threshold])
    n_total = len(detections)

    print(f"Threshold {threshold}: {n_above}/{n_total} detections")

    # Higher thresholds should have fewer detections
    assert n_above <= n_total