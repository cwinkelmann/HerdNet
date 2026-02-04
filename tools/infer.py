#!/usr/bin/env python
"""
Inference script for HerdNet using Hydra config.

Usage:
    python infer.py --config-dir ./models/general_2022 --images ./data/test_sample/

    # With additional overrides:
    python infer.py --config-dir ./models/general_2022 --images ./data/test_sample/ \
        --model ./models/general_2022/model.pth \
        --overrides "inference.patch_size=512" "inference.overlap=160"
"""

__author__ = "Alexandre Delplanque, Christian Winkelmann (refactored)"
__license__ = "MIT License"
__version__ = "0.3.0"

import argparse
from pathlib import Path

from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from animaloc.utils.inference import inference


def parse_args():
    parser = argparse.ArgumentParser(
        prog='infer',
        description='Run HerdNet inference using Hydra config'
    )

    parser.add_argument('--config-dir', type=str, required=True,
        help='Path to config directory containing config.yaml')
    parser.add_argument('--config-name', type=str, default='config',
        help='Name of config file (without .yaml). Defaults to "config"')
    parser.add_argument('--images', type=str, required=True,
        help='Path to images directory')
    parser.add_argument('--model', type=str, default=None,
        help='Path to model .pth file (overrides config)')
    parser.add_argument('--output', type=str, default=None,
        help='Output directory for results (overrides config)')
    parser.add_argument('--vis', action='store_true',
        help='Visualize detections')
    parser.add_argument('--overrides', nargs='*', default=[],
        help='Additional Hydra-style overrides, e.g., "inference.patch_size=512"')

    return parser.parse_args()


def main():
    args = parse_args()

    # Clear any previous Hydra state
    GlobalHydra.instance().clear()

    # Convert to absolute path (required by Hydra)
    config_dir = str(Path(args.config_dir).resolve())

    # Build overrides list
    overrides = list(args.overrides)

    # Add image directory override
    overrides.append(f"datasets.test.root_dir={args.images}")

    # Add model path override if provided
    if args.model:
        overrides.append(f"model.load_from={args.model}")

    # Add output directory override if provided
    if args.output:
        overrides.append(f"work_dir={args.output}")

    # Load config
    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg = compose(config_name=args.config_name, overrides=overrides)

    # Print config
    print("=" * 50)
    print("Configuration:")
    print("=" * 50)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 50)

    # Run inference
    detections = inference(cfg, plain_inference=True, vis_detections=args.vis)

    print(f"\nDetections: {len(detections)} total")
    print(detections.head())

    return detections


if __name__ == '__main__':
    main()