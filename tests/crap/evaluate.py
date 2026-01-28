#!/usr/bin/env python3
"""
Evaluate Iguana Classifier and Find Optimal Threshold

This script:
1. Loads a trained model checkpoint
2. Runs inference on validation data
3. Sweeps through thresholds to find optimal F1/F3
4. Outputs detailed metrics and plots

Usage:
    python evaluate_classifier.py \
        --checkpoint outputs_classifier/best.pth \
        --val_csv val.csv \
        --val_image_dir ./val_images \
        --tiled \
        --output_dir ./eval_results
"""

import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import from classifier module
from classifier_v2 import (
    IguanaClassifier,
    IguanaTiledDataset,
    IguanaPresenceDataset,
    tiled_collate_fn,
)


def compute_metrics_at_threshold(
        probs: np.ndarray,
        labels: np.ndarray,
        threshold: float
) -> Dict[str, float]:
    """Compute classification metrics at a given threshold."""
    preds = (probs > threshold).astype(float)

    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    accuracy = (tp + tn) / len(labels)

    # F-scores
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    f2 = 5 * precision * recall / max(4 * precision + recall, 1e-6)
    f3 = 10 * precision * recall / max(9 * precision + recall, 1e-6)
    f05 = 1.25 * precision * recall / max(0.25 * precision + recall, 1e-6)

    return {
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'accuracy': accuracy,
        'f1': f1,
        'f2': f2,
        'f3': f3,
        'f0.5': f05,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
    }


def sweep_thresholds(
        probs: np.ndarray,
        labels: np.ndarray,
        thresholds: np.ndarray = None,
) -> pd.DataFrame:
    """Compute metrics across a range of thresholds."""
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)

    results = []
    for thresh in thresholds:
        metrics = compute_metrics_at_threshold(probs, labels, thresh)
        results.append(metrics)

    return pd.DataFrame(results)


def find_optimal_thresholds(df: pd.DataFrame) -> Dict[str, Dict]:
    """Find optimal thresholds for different metrics."""
    optimals = {}

    for metric in ['f1', 'f2', 'f3', 'f0.5', 'accuracy']:
        idx = df[metric].idxmax()
        row = df.iloc[idx]
        optimals[metric] = {
            'threshold': row['threshold'],
            'value': row[metric],
            'precision': row['precision'],
            'recall': row['recall'],
            'tp': row['tp'],
            'fp': row['fp'],
            'fn': row['fn'],
            'tn': row['tn'],
        }

    # Also find threshold for specific recall targets
    for target_recall in [0.90, 0.95, 0.99]:
        mask = df['recall'] >= target_recall
        if mask.any():
            # Among thresholds achieving target recall, pick highest precision
            subset = df[mask]
            idx = subset['precision'].idxmax()
            row = subset.loc[idx]
            optimals[f'recall_{int(target_recall * 100)}'] = {
                'threshold': row['threshold'],
                'recall': row['recall'],
                'precision': row['precision'],
                'f1': row['f1'],
                'f3': row['f3'],
            }

    return optimals


@torch.no_grad()
def run_inference(
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device,
        is_tiled: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Run inference and collect predictions with metadata."""
    model.eval()

    all_probs = []
    all_labels = []
    all_metadata = []

    for images, targets in tqdm(loader, desc="Running inference"):
        images = images.to(device)

        if is_tiled:
            labels = targets['label']
            names = targets['name']
            crop_xs = targets['crop_x']
            crop_ys = targets['crop_y']
        else:
            labels = targets
            names = [None] * len(labels)
            crop_xs = [None] * len(labels)
            crop_ys = [None] * len(labels)

        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()

        all_probs.extend(probs)
        all_labels.extend(labels.numpy())

        for i in range(len(labels)):
            all_metadata.append({
                'name': names[i] if is_tiled else None,
                'crop_x': crop_xs[i] if is_tiled else None,
                'crop_y': crop_ys[i] if is_tiled else None,
                'prob': float(probs[i]),
                'label': int(labels[i]),
            })

    return np.array(all_probs), np.array(all_labels), all_metadata


def plot_threshold_curves(df: pd.DataFrame, output_dir: Path):
    """Generate visualization plots for threshold analysis."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Precision-Recall curve
    ax = axes[0, 0]
    ax.plot(df['recall'], df['precision'], 'b-', linewidth=2)
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Mark optimal F1 and F3 points
    f1_idx = df['f1'].idxmax()
    f3_idx = df['f3'].idxmax()
    ax.scatter(df.loc[f1_idx, 'recall'], df.loc[f1_idx, 'precision'],
               c='red', s=100, zorder=5, label=f"Best F1 (t={df.loc[f1_idx, 'threshold']:.2f})")
    ax.scatter(df.loc[f3_idx, 'recall'], df.loc[f3_idx, 'precision'],
               c='green', s=100, zorder=5, label=f"Best F3 (t={df.loc[f3_idx, 'threshold']:.2f})")
    ax.legend()

    # 2. F-scores vs threshold
    ax = axes[0, 1]
    ax.plot(df['threshold'], df['f1'], 'r-', linewidth=2, label='F1')
    ax.plot(df['threshold'], df['f2'], 'orange', linewidth=2, label='F2')
    ax.plot(df['threshold'], df['f3'], 'g-', linewidth=2, label='F3')
    ax.plot(df['threshold'], df['f0.5'], 'purple', linewidth=2, label='F0.5')
    ax.axvline(df.loc[f1_idx, 'threshold'], c='red', ls='--', alpha=0.5)
    ax.axvline(df.loc[f3_idx, 'threshold'], c='green', ls='--', alpha=0.5)
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('F-score', fontsize=12)
    ax.set_title('F-scores vs Threshold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])

    # 3. Precision and Recall vs threshold
    ax = axes[1, 0]
    ax.plot(df['threshold'], df['precision'], 'b-', linewidth=2, label='Precision')
    ax.plot(df['threshold'], df['recall'], 'r-', linewidth=2, label='Recall')
    ax.axvline(df.loc[f1_idx, 'threshold'], c='gray', ls='--', alpha=0.5,
               label=f"Best F1 (t={df.loc[f1_idx, 'threshold']:.2f})")
    ax.axvline(df.loc[f3_idx, 'threshold'], c='green', ls='--', alpha=0.5,
               label=f"Best F3 (t={df.loc[f3_idx, 'threshold']:.2f})")
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Precision & Recall vs Threshold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # 4. Confusion matrix counts vs threshold
    ax = axes[1, 1]
    ax.plot(df['threshold'], df['tp'], 'g-', linewidth=2, label='TP')
    ax.plot(df['threshold'], df['fp'], 'r-', linewidth=2, label='FP')
    ax.plot(df['threshold'], df['fn'], 'orange', linewidth=2, label='FN')
    ax.plot(df['threshold'], df['tn'], 'b-', linewidth=2, label='TN')
    ax.axvline(df.loc[f3_idx, 'threshold'], c='green', ls='--', alpha=0.5)
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Confusion Matrix Counts vs Threshold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])

    plt.tight_layout()
    plt.savefig(output_dir / 'threshold_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved threshold analysis plot to {output_dir / 'threshold_analysis.png'}")

    # Additional plot: probability distribution
    fig, ax = plt.subplots(figsize=(10, 6))

    # We'll need to reload probs/labels for histogram - skip if not available
    print("Threshold curves plotted successfully")


def plot_probability_histogram(
        probs: np.ndarray,
        labels: np.ndarray,
        optimal_thresholds: Dict,
        output_dir: Path
):
    """Plot histogram of predicted probabilities by class."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    pos_probs = probs[labels == 1]
    neg_probs = probs[labels == 0]

    bins = np.linspace(0, 1, 51)

    ax.hist(neg_probs, bins=bins, alpha=0.6, label=f'Negative (n={len(neg_probs)})', color='blue')
    ax.hist(pos_probs, bins=bins, alpha=0.6, label=f'Positive (n={len(pos_probs)})', color='red')

    # Mark optimal thresholds
    colors = {'f1': 'red', 'f3': 'green', 'recall_95': 'orange'}
    for metric, color in colors.items():
        if metric in optimal_thresholds:
            thresh = optimal_thresholds[metric]['threshold']
            ax.axvline(thresh, c=color, ls='--', linewidth=2,
                       label=f'{metric} threshold: {thresh:.3f}')

    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Predicted Probabilities by Class', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'probability_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved probability distribution to {output_dir / 'probability_distribution.png'}")


def plot_false_negatives(
        model: torch.nn.Module,
        dataset,
        metadata: List[Dict],
        probs: np.ndarray,
        labels: np.ndarray,
        threshold: float,
        device: torch.device,
        output_dir: Path,
        max_samples: int = None,
):
    """
    Plot all false negatives (missed detections).

    Args:
        model: Trained model for generating heatmaps
        dataset: IguanaTiledDataset instance
        metadata: List of prediction metadata dicts
        probs: Array of predicted probabilities
        labels: Array of ground truth labels
        threshold: Classification threshold
        device: torch device
        output_dir: Directory to save plots
        max_samples: Maximum number of samples to plot (None = all)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping false negative plots")
        return

    # Find false negatives
    preds = (probs > threshold).astype(int)
    fn_mask = (labels == 1) & (preds == 0)
    fn_indices = np.where(fn_mask)[0]

    if len(fn_indices) == 0:
        print("No false negatives found!")
        return

    print(f"\nPlotting {len(fn_indices)} false negatives (threshold={threshold:.3f})...")

    fn_dir = output_dir / 'false_negatives'
    fn_dir.mkdir(parents=True, exist_ok=True)

    # Sort by probability (lowest first - most confident misses)
    fn_probs = probs[fn_indices]
    sorted_order = np.argsort(fn_probs)
    fn_indices = fn_indices[sorted_order]

    if max_samples is not None:
        fn_indices = fn_indices[:max_samples]

    model.eval()
    crop_size = dataset.crop_size

    for i, idx in enumerate(tqdm(fn_indices, desc="Plotting false negatives")):
        # Get the sample
        img_tensor, targets = dataset[idx]

        meta = metadata[idx]
        prob = meta['prob']
        name = meta['name']
        crop_x = meta['crop_x']
        crop_y = meta['crop_y']

        # Get points in this tile
        if hasattr(targets, 'items'):  # Dict from tiled dataset
            points_in_crop = targets['points'].numpy()
        else:
            points_in_crop = np.array([])

        # Unnormalize image for visualization
        img = img_tensor.permute(1, 2, 0).numpy()
        img = img * np.array(dataset.std) + np.array(dataset.mean)
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

        # Get heatmap from model
        with torch.no_grad():
            logit, patch_logits = model(
                img_tensor.unsqueeze(0).to(device),
                return_patches=True,
                upsample_patches=False
            )

            patch_probs = torch.sigmoid(patch_logits[0]).cpu().numpy()

            # Upsample heatmap
            heatmap = torch.nn.functional.interpolate(
                torch.sigmoid(patch_logits).unsqueeze(1),
                size=(crop_size, crop_size),
                mode='bilinear',
                align_corners=False
            )[0, 0].cpu().numpy()

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 1. Original image with GT points
        axes[0].imshow(img)
        for pt in points_in_crop:
            px, py = pt[0], pt[1]
            circle = plt.Circle((px, py), 15, color='lime', fill=False, linewidth=2)
            axes[0].add_patch(circle)
            axes[0].plot(px, py, 'g+', markersize=12, markeredgewidth=2)
        axes[0].set_title(f"Original (GT: {len(points_in_crop)} pts)", fontsize=10)
        axes[0].axis('off')

        # 2. Heatmap overlay
        axes[1].imshow(img)
        heatmap_overlay = axes[1].imshow(heatmap, cmap='hot', alpha=0.5, vmin=0, vmax=1)
        for pt in points_in_crop:
            px, py = pt[0], pt[1]
            axes[1].plot(px, py, 'g+', markersize=12, markeredgewidth=2)
        axes[1].set_title(f"Heatmap (max={heatmap.max():.2f})", fontsize=10)
        axes[1].axis('off')
        plt.colorbar(heatmap_overlay, ax=axes[1], fraction=0.046)

        # 3. Patch grid
        im = axes[2].imshow(patch_probs, cmap='hot', vmin=0, vmax=1)
        grid_h, grid_w = patch_probs.shape
        patch_size = crop_size / grid_h
        for pt in points_in_crop:
            px, py = pt[0], pt[1]
            axes[2].plot(px / patch_size, py / patch_size, 'g+', markersize=15, markeredgewidth=2)
        axes[2].set_title(f"Patch Grid ({grid_h}x{grid_w})", fontsize=10)
        plt.colorbar(im, ax=axes[2], fraction=0.046)

        # Main title
        fig.suptitle(
            f"FALSE NEGATIVE #{i + 1}\n"
            f"Prob: {prob:.4f} (threshold: {threshold:.3f}) | Points: {len(points_in_crop)}\n"
            f"Source: {name} @ ({crop_x}, {crop_y})",
            fontsize=11
        )

        plt.tight_layout()
        plt.savefig(fn_dir / f'{i:04d}_prob{prob:.4f}.png', dpi=120, bbox_inches='tight')
        plt.close()

    print(f"Saved {len(fn_indices)} false negative plots to {fn_dir}")

    # Create summary grid of worst false negatives
    n_summary = min(16, len(fn_indices))
    if n_summary > 0:
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.flatten()

        for i, idx in enumerate(fn_indices[:n_summary]):
            img_tensor, targets = dataset[idx]

            img = img_tensor.permute(1, 2, 0).numpy()
            img = img * np.array(dataset.std) + np.array(dataset.mean)
            img = np.clip(img * 255, 0, 255).astype(np.uint8)

            if hasattr(targets, 'items'):
                points_in_crop = targets['points'].numpy()
            else:
                points_in_crop = np.array([])

            axes[i].imshow(img)
            for pt in points_in_crop:
                circle = plt.Circle((pt[0], pt[1]), 12, color='lime', fill=False, linewidth=2)
                axes[i].add_patch(circle)
            axes[i].set_title(f"p={probs[idx]:.3f}, pts={len(points_in_crop)}", fontsize=9)
            axes[i].axis('off')

        # Hide unused axes
        for i in range(n_summary, len(axes)):
            axes[i].axis('off')

        plt.suptitle(f"Worst False Negatives (threshold={threshold:.3f})", fontsize=14)
        plt.tight_layout()
        plt.savefig(fn_dir / 'summary_grid.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved false negatives summary grid to {fn_dir / 'summary_grid.png'}")

    return len(fn_indices)


def plot_false_positives(
        model: torch.nn.Module,
        dataset,
        metadata: List[Dict],
        probs: np.ndarray,
        labels: np.ndarray,
        threshold: float,
        device: torch.device,
        output_dir: Path,
        max_samples: int = None,
):
    """
    Plot all false positives (false alarms).

    Similar to plot_false_negatives but for FP cases.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping false positive plots")
        return

    # Find false positives
    preds = (probs > threshold).astype(int)
    fp_mask = (labels == 0) & (preds == 1)
    fp_indices = np.where(fp_mask)[0]

    if len(fp_indices) == 0:
        print("No false positives found!")
        return

    print(f"\nPlotting {len(fp_indices)} false positives (threshold={threshold:.3f})...")

    fp_dir = output_dir / 'false_positives'
    fp_dir.mkdir(parents=True, exist_ok=True)

    # Sort by probability (highest first - most confident false alarms)
    fp_probs = probs[fp_indices]
    sorted_order = np.argsort(-fp_probs)  # Descending
    fp_indices = fp_indices[sorted_order]

    if max_samples is not None:
        fp_indices = fp_indices[:max_samples]

    model.eval()
    crop_size = dataset.crop_size

    for i, idx in enumerate(tqdm(fp_indices, desc="Plotting false positives")):
        img_tensor, targets = dataset[idx]

        meta = metadata[idx]
        prob = meta['prob']
        name = meta['name']
        crop_x = meta['crop_x']
        crop_y = meta['crop_y']

        # Unnormalize image
        img = img_tensor.permute(1, 2, 0).numpy()
        img = img * np.array(dataset.std) + np.array(dataset.mean)
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

        # Get heatmap
        with torch.no_grad():
            logit, patch_logits = model(
                img_tensor.unsqueeze(0).to(device),
                return_patches=True,
                upsample_patches=False
            )

            patch_probs = torch.sigmoid(patch_logits[0]).cpu().numpy()

            heatmap = torch.nn.functional.interpolate(
                torch.sigmoid(patch_logits).unsqueeze(1),
                size=(crop_size, crop_size),
                mode='bilinear',
                align_corners=False
            )[0, 0].cpu().numpy()

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 1. Original image
        axes[0].imshow(img)
        axes[0].set_title("Original (no GT points)", fontsize=10)
        axes[0].axis('off')

        # 2. Heatmap overlay
        axes[1].imshow(img)
        heatmap_overlay = axes[1].imshow(heatmap, cmap='hot', alpha=0.5, vmin=0, vmax=1)
        axes[1].set_title(f"Heatmap (max={heatmap.max():.2f})", fontsize=10)
        axes[1].axis('off')
        plt.colorbar(heatmap_overlay, ax=axes[1], fraction=0.046)

        # 3. Patch grid
        im = axes[2].imshow(patch_probs, cmap='hot', vmin=0, vmax=1)
        axes[2].set_title(f"Patch Grid", fontsize=10)
        plt.colorbar(im, ax=axes[2], fraction=0.046)

        fig.suptitle(
            f"FALSE POSITIVE #{i + 1}\n"
            f"Prob: {prob:.4f} (threshold: {threshold:.3f})\n"
            f"Source: {name} @ ({crop_x}, {crop_y})",
            fontsize=11
        )

        plt.tight_layout()
        plt.savefig(fp_dir / f'{i:04d}_prob{prob:.4f}.png', dpi=120, bbox_inches='tight')
        plt.close()

    print(f"Saved {len(fp_indices)} false positive plots to {fp_dir}")

    # Summary grid
    n_summary = min(16, len(fp_indices))
    if n_summary > 0:
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.flatten()

        for i, idx in enumerate(fp_indices[:n_summary]):
            img_tensor, _ = dataset[idx]

            img = img_tensor.permute(1, 2, 0).numpy()
            img = img * np.array(dataset.std) + np.array(dataset.mean)
            img = np.clip(img * 255, 0, 255).astype(np.uint8)

            axes[i].imshow(img)
            axes[i].set_title(f"p={probs[idx]:.3f}", fontsize=9)
            axes[i].axis('off')

        for i in range(n_summary, len(axes)):
            axes[i].axis('off')

        plt.suptitle(f"Worst False Positives (threshold={threshold:.3f})", fontsize=14)
        plt.tight_layout()
        plt.savefig(fp_dir / 'summary_grid.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved false positives summary grid to {fp_dir / 'summary_grid.png'}")

    return len(fp_indices)


def main():
    parser = argparse.ArgumentParser(description="Evaluate classifier and find optimal threshold")

    # Required
    parser.add_argument('--checkpoint', required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--val_csv', required=True,
                        help='Path to validation CSV')
    parser.add_argument('--val_image_dir', required=True,
                        help='Path to validation images')

    # Model
    parser.add_argument('--backbone', default='vit_large_patch14_reg4_dinov2.lvd142m')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.3)

    # Dataset
    parser.add_argument('--crop_size', type=int, default=518)
    parser.add_argument('--tiled', action='store_true',
                        help='Use tiled dataset (deterministic)')
    parser.add_argument('--tile_overlap', type=int, default=0)
    parser.add_argument('--crops_per_image', type=int, default=8,
                        help='For non-tiled: crops per image')

    # Evaluation
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', default='./eval_results')

    # Threshold sweep
    parser.add_argument('--threshold_min', type=float, default=0.01)
    parser.add_argument('--threshold_max', type=float, default=0.99)
    parser.add_argument('--threshold_step', type=float, default=0.01)

    # Error visualization
    parser.add_argument('--plot_errors', action='store_true',
                        help='Plot all false negatives and false positives')
    parser.add_argument('--plot_threshold', type=float, default=None,
                        help='Threshold for error plotting (default: optimal F3 threshold)')
    parser.add_argument('--max_fn_plots', type=int, default=None,
                        help='Max false negatives to plot (default: all)')
    parser.add_argument('--max_fp_plots', type=int, default=None,
                        help='Max false positives to plot (default: all)')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading model from {args.checkpoint}")

    # Load checkpoint first to get model config
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Get model params from checkpoint if available, otherwise use args
    backbone_name = ckpt.get('backbone', args.backbone)
    hidden_dim = ckpt.get('hidden_dim', args.hidden_dim)
    dropout = ckpt.get('dropout', args.dropout)

    # Validate backbone name - common mistake is missing 'm' suffix for DINOv2
    if 'dinov2' in backbone_name.lower():
        if backbone_name.endswith('.lvd142'):
            corrected = backbone_name + 'm'
            print(f"\n⚠️  WARNING: DINOv2 backbone names end with 'm' (for LVD-142M dataset)")
            print(f"   You specified: {backbone_name}")
            print(f"   Auto-correcting to: {corrected}")
            backbone_name = corrected
        elif not backbone_name.endswith('m'):
            print(f"\n⚠️  NOTE: DINOv2 backbone '{backbone_name}' may be invalid.")
            print(f"   Common valid names:")
            print(f"     vit_small_patch14_dinov2.lvd142m")
            print(f"     vit_base_patch14_dinov2.lvd142m")
            print(f"     vit_large_patch14_dinov2.lvd142m")
            print(f"     vit_base_patch14_reg4_dinov2.lvd142m  (with register tokens)")
            print(f"     vit_large_patch14_reg4_dinov2.lvd142m (with register tokens)")

    print(f"\n  Model config:")
    print(f"    backbone: {backbone_name}")
    print(f"    hidden_dim: {hidden_dim}")
    print(f"    dropout: {dropout}")

    # Create model - use pretrained=False since we load weights from checkpoint
    try:
        model = IguanaClassifier(
            backbone=backbone_name,
            freeze_backbone=True,
            hidden_dim=hidden_dim,
            dropout=dropout,
        ).to(device)
    except RuntimeError as e:
        if 'Invalid pretrained tag' in str(e):
            print(f"\n❌ ERROR: Invalid backbone name '{backbone_name}'")
            print(f"\nTry one of these DINOv2 models:")
            try:
                import timm
                dinov2_models = timm.list_models('*dinov2*')
                for m in sorted(set(dinov2_models))[:15]:
                    print(f"  - {m}")
            except:
                print("  vit_base_patch14_dinov2.lvd142m")
                print("  vit_large_patch14_dinov2.lvd142m")
                print("  vit_large_patch14_reg4_dinov2.lvd142m")
            raise SystemExit(1)
        raise

    model.load_state_dict(ckpt['model_state_dict'])

    # Print checkpoint info
    print(f"\n  Checkpoint info:")
    if 'epoch' in ckpt:
        print(f"    Epoch: {ckpt['epoch']}")
    if 'best_f3' in ckpt:
        print(f"    Best F3: {ckpt['best_f3']:.4f}")
    if 'threshold' in ckpt:
        print(f"    Saved threshold: {ckpt['threshold']}")

    # Load dataset
    print(f"\nLoading validation data...")
    if args.tiled:
        val_ds = IguanaTiledDataset(
            args.val_csv, args.val_image_dir,
            crop_size=args.crop_size,
            overlap=args.tile_overlap,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
            collate_fn=tiled_collate_fn,
        )
    else:
        val_ds = IguanaPresenceDataset(
            args.val_csv, args.val_image_dir,
            crop_size=args.crop_size,
            crops_per_image=args.crops_per_image,
            positive_ratio=0.5,
            augment=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )

    # Run inference
    print(f"\nRunning inference on {len(val_ds)} samples...")
    probs, labels, metadata = run_inference(model, val_loader, device, is_tiled=args.tiled)

    print(f"  Total samples: {len(labels)}")
    print(f"  Positive samples: {labels.sum()} ({100 * labels.mean():.1f}%)")
    print(f"  Negative samples: {len(labels) - labels.sum()} ({100 * (1 - labels.mean()):.1f}%)")
    print(f"  Prob range: [{probs.min():.4f}, {probs.max():.4f}]")
    print(f"  Prob mean: {probs.mean():.4f}")

    # Threshold sweep
    print(f"\nSweeping thresholds from {args.threshold_min} to {args.threshold_max}...")
    thresholds = np.arange(args.threshold_min, args.threshold_max + args.threshold_step, args.threshold_step)
    df = sweep_thresholds(probs, labels, thresholds)

    # Find optimal thresholds
    optimal = find_optimal_thresholds(df)

    # Print results
    print("\n" + "=" * 70)
    print("OPTIMAL THRESHOLDS")
    print("=" * 70)

    print("\n### By F-score:")
    for metric in ['f0.5', 'f1', 'f2', 'f3']:
        opt = optimal[metric]
        print(f"  {metric.upper():5s}: threshold={opt['threshold']:.3f} | "
              f"P={opt['precision']:.4f} R={opt['recall']:.4f} {metric}={opt['value']:.4f} | "
              f"TP={opt['tp']} FP={opt['fp']} FN={opt['fn']} TN={opt['tn']}")

    print("\n### By recall target:")
    for key in ['recall_90', 'recall_95', 'recall_99']:
        if key in optimal:
            opt = optimal[key]
            print(f"  {key}: threshold={opt['threshold']:.3f} | "
                  f"P={opt['precision']:.4f} R={opt['recall']:.4f} | "
                  f"F1={opt['f1']:.4f} F3={opt['f3']:.4f}")

    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # Save threshold sweep results
    df.to_csv(output_dir / 'threshold_sweep.csv', index=False)
    print(f"  Saved threshold sweep to {output_dir / 'threshold_sweep.csv'}")

    # Save optimal thresholds
    with open(output_dir / 'optimal_thresholds.json', 'w') as f:
        json.dump(optimal, f, indent=2)
    print(f"  Saved optimal thresholds to {output_dir / 'optimal_thresholds.json'}")

    # Save predictions with metadata
    pred_df = pd.DataFrame(metadata)
    pred_df.to_csv(output_dir / 'predictions.csv', index=False)
    print(f"  Saved predictions to {output_dir / 'predictions.csv'}")

    # Generate plots
    print("\nGenerating plots...")
    plot_threshold_curves(df, output_dir)
    plot_probability_histogram(probs, labels, optimal, output_dir)

    # Summary for easy reference
    print("\n" + "=" * 70)
    print("SUMMARY - RECOMMENDED THRESHOLDS")
    print("=" * 70)

    f3_opt = optimal['f3']
    f1_opt = optimal['f1']

    print(f"""
For wildlife counting (minimize missed iguanas):
  → Use threshold = {f3_opt['threshold']:.3f}
  → Expected: {f3_opt['recall'] * 100:.1f}% recall, {f3_opt['precision'] * 100:.1f}% precision
  → F3 = {f3_opt['value']:.4f}

For balanced performance:
  → Use threshold = {f1_opt['threshold']:.3f}
  → Expected: {f1_opt['recall'] * 100:.1f}% recall, {f1_opt['precision'] * 100:.1f}% precision
  → F1 = {f1_opt['value']:.4f}
""")

    if 'recall_95' in optimal:
        r95 = optimal['recall_95']
        print(f"""For 95% recall target:
  → Use threshold = {r95['threshold']:.3f}
  → Expected: {r95['recall'] * 100:.1f}% recall, {r95['precision'] * 100:.1f}% precision
""")

    # Plot errors if requested
    if args.plot_errors and args.tiled:
        print("\n" + "=" * 70)
        print("PLOTTING ERRORS")
        print("=" * 70)

        # Use specified threshold or optimal F3 threshold
        plot_thresh = args.plot_threshold if args.plot_threshold is not None else f3_opt['threshold']

        # Plot false negatives
        n_fn = plot_false_negatives(
            model=model,
            dataset=val_ds,
            metadata=metadata,
            probs=probs,
            labels=labels,
            threshold=plot_thresh,
            device=device,
            output_dir=output_dir,
            max_samples=args.max_fn_plots,
        )

        # Plot false positives
        n_fp = plot_false_positives(
            model=model,
            dataset=val_ds,
            metadata=metadata,
            probs=probs,
            labels=labels,
            threshold=plot_thresh,
            device=device,
            output_dir=output_dir,
            max_samples=args.max_fp_plots,
        )

        print(f"\nError summary at threshold={plot_thresh:.3f}:")
        print(f"  False negatives: {n_fn}")
        print(f"  False positives: {n_fp}")
    elif args.plot_errors and not args.tiled:
        print("\nWARNING: --plot_errors requires --tiled dataset")


if __name__ == '__main__':
    main()