"""
HerdNet Inference Script Template
===================================
Full inference pipeline for running a trained HerdNet model on orthomosaics.
Handles: checkpoint loading, overlapping tile inference (stitcher), peak
detection (LMDS), coordinate conversion, and CSV export.

Usage:
    python inference_script_template.py

Customize:
    1. Set CHECKPOINT_PATH to your trained model
    2. Set IMAGE_PATH to your orthomosaic
    3. Adjust model config to match training config exactly
    4. Adjust LMDS parameters for your detection sensitivity needs
"""

import os
import torch
import torchvision
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# animaloc imports
from animaloc.models import HerdNetTimmDLA
from animaloc.models.utils import LossWrapper
from animaloc.eval.stitchers import HerdNetStitcher
from animaloc.eval.lmds import HerdNetLMDS


# =============================================================================
# CONFIGURATION -- MUST MATCH TRAINING CONFIG
# =============================================================================

# Paths
CHECKPOINT_PATH = 'best_model.pth'                   # TODO: Update
IMAGE_PATH = 'orthomosaic.tif'                        # TODO: Update
OUTPUT_CSV = 'detections.csv'
OUTPUT_IMAGE = 'detections_overlay.png'

# Model config -- MUST match the training configuration exactly
NUM_CLASSES = 3          # background(0) + iguana(1) + hard_negative(2)
DOWN_RATIO = 4           # Must match training down_ratio
HEAD_CONV = 64           # Must match training head_conv
BACKBONE = 'timm/dla34'  # Must match training backbone

# Stitcher config
PATCH_SIZE = (512, 512)  # Must match training patch size
OVERLAP = 120            # Overlap between tiles in pixels
REDUCTION = 'mean'       # Averaging for smooth blending

# LMDS config (peak detection)
LMDS_KERNEL = (5, 5)    # Optimal for iguana spacing
LMDS_ADAPT_TS = 0.5     # Adaptive threshold (0.5 = optimal)
LMDS_NEG_TS = 0.1       # Negative sample threshold

# Class names for output
CLASS_NAMES = {
    1: 'iguana',
    2: 'hard_negative',
}
CLASS_COLORS = {
    1: 'lime',
    2: 'red',
}


# =============================================================================
# DEVICE SELECTION
# =============================================================================

def get_device():
    """Select best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


# =============================================================================
# LOAD MODEL
# =============================================================================

def load_model(checkpoint_path, device):
    """Load trained HerdNet model from checkpoint."""

    # Create model with same architecture as training
    model = HerdNetTimmDLA(
        backbone=BACKBONE,
        num_classes=NUM_CLASSES,
        down_ratio=DOWN_RATIO,
        head_conv=HEAD_CONV,
        pretrained=False,    # We have our own weights
        debug=False,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle state dict -- may have LossWrapper prefix
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Strip 'model.' prefix if checkpoint was saved from LossWrapper
    clean_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('model.', '', 1) if key.startswith('model.') else key
        clean_state_dict[new_key] = value

    missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
    if missing:
        print(f"Warning: Missing keys: {missing[:5]}...")
    if unexpected:
        print(f"Warning: Unexpected keys: {unexpected[:5]}...")

    model.to(device)
    model.eval()

    # Wrap for stitcher (requires LossWrapper interface)
    wrapped = LossWrapper(model, losses=[], mode='preds_only')

    return wrapped


# =============================================================================
# RUN INFERENCE
# =============================================================================

def run_inference(model, image_path, device):
    """Run inference on an orthomosaic using tiled stitching."""

    # Load image
    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    print(f"Image size: {width} x {height} pixels")

    # Convert to tensor and normalize (ImageNet stats)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    image_tensor = transform(image)  # [3, H, W]

    # Create stitcher
    stitcher = HerdNetStitcher(
        model=model,
        size=PATCH_SIZE,
        overlap=OVERLAP,
        batch_size=1,
        down_ratio=DOWN_RATIO,
        up=False,            # Don't upsample -- handle coords manually
        reduction=REDUCTION,
        device_name=str(device),
    )

    # Run stitcher
    print("Running tiled inference...")
    with torch.no_grad():
        output = stitcher(image_tensor)
    # output shape: [1, 1+C, H/DR, W/DR]
    print(f"Stitched output shape: {output.shape}")

    # Split into heatmap and class map
    heatmap = output[:, :1, :, :]        # [1, 1, H/DR, W/DR]
    clsmap = output[:, 1:, :, :]         # [1, C-1, H/DR, W/DR]

    # Apply LMDS (peak detection)
    lmds = HerdNetLMDS(
        up=False,                         # False because stitcher already handled scaling
        kernel_size=LMDS_KERNEL,
        adapt_ts=LMDS_ADAPT_TS,
        neg_ts=LMDS_NEG_TS,
        scale_factor=1,                   # 1 because stitcher already matched resolutions
    )

    counts, locs, labels, cls_scores, det_scores = lmds([heatmap, clsmap])

    print(f"Detected {sum(counts[0])} animals:")
    for i, count in enumerate(counts[0]):
        class_name = CLASS_NAMES.get(i + 1, f'class_{i + 1}')
        print(f"  {class_name}: {count}")

    return locs[0], labels[0], cls_scores[0], det_scores[0], image


# =============================================================================
# EXPORT RESULTS
# =============================================================================

def export_csv(locs, labels, cls_scores, det_scores, output_path):
    """Export detections to CSV with image coordinates."""

    results = []
    for (row, col), label, cls_score, det_score in zip(
        locs, labels, cls_scores, det_scores
    ):
        results.append({
            'x': int(col * DOWN_RATIO),
            'y': int(row * DOWN_RATIO),
            'label': label,
            'class_name': CLASS_NAMES.get(label, f'class_{label}'),
            'class_score': round(cls_score, 4),
            'detection_score': round(det_score, 4),
        })

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(results)} detections to {output_path}")

    return df


def overlay_detections(image, locs, labels, det_scores, output_path):
    """Overlay detections on the original image and save."""

    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    ax.imshow(image)

    for (row, col), label, score in zip(locs, labels, det_scores):
        x = col * DOWN_RATIO
        y = row * DOWN_RATIO
        color = CLASS_COLORS.get(label, 'blue')

        circle = plt.Circle(
            (x, y), radius=12,
            color=color, fill=False, linewidth=2,
        )
        ax.add_patch(circle)

        # Add score label
        ax.text(
            x + 15, y, f'{score:.2f}',
            color=color, fontsize=6,
            bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.5),
        )

    # Legend
    for label, name in CLASS_NAMES.items():
        color = CLASS_COLORS.get(label, 'blue')
        ax.plot([], [], 'o', color=color, label=name)
    ax.legend(loc='upper right', fontsize=14)

    total = len(locs)
    ax.set_title(f'Detections: {total} total', fontsize=16)
    ax.axis('off')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved overlay to {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    device = get_device()
    print(f"Using device: {device}")
    print(f"Model: {BACKBONE}, DR={DOWN_RATIO}, classes={NUM_CLASSES}")
    print()

    # Load model
    model = load_model(CHECKPOINT_PATH, device)

    # Run inference
    locs, labels, cls_scores, det_scores, image = run_inference(
        model, IMAGE_PATH, device
    )

    # Export results
    df = export_csv(locs, labels, cls_scores, det_scores, OUTPUT_CSV)

    # Visualize
    overlay_detections(image, locs, labels, det_scores, OUTPUT_IMAGE)

    # Summary
    print("\n" + "=" * 50)
    print("Inference complete!")
    print(f"  Total detections: {len(locs)}")
    print(f"  CSV output: {OUTPUT_CSV}")
    print(f"  Image output: {OUTPUT_IMAGE}")
    print("=" * 50)


if __name__ == '__main__':
    main()
