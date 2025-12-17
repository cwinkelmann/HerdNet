"""
Visualization script for debugging point detection models.

Plots:
- True Positives (green circles)
- False Positives (red X marks) 
- False Negatives (blue squares)
- Heatmap overlay

Usage:
    python visualize_predictions.py \
        --checkpoint outputs_dinov3/best.pth \
        --val_csv /path/to/val.csv \
        --val_image_dir /path/to/images \
        --output_dir ./visualizations \
        --num_images 20
"""

import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# Import the model (assuming it's in the same directory)
from improved_point_detector_dinov3 import (
    ImprovedPointDetectorDINOv3,
    HeatmapPointDataset,
    collate_fn,
    extract_points_from_heatmap
)


def match_predictions_to_gt(pred_pts: torch.Tensor, 
                            gt_pts: torch.Tensor, 
                            match_radius: float = 25.0) -> Tuple[List, List, List]:
    """
    Match predictions to ground truth points.
    
    Returns:
        tp_pairs: List of (pred_idx, gt_idx, distance) for true positives
        fp_indices: List of prediction indices that are false positives
        fn_indices: List of ground truth indices that are false negatives
    """
    n_pred, n_gt = len(pred_pts), len(gt_pts)
    
    if n_pred == 0:
        return [], [], list(range(n_gt))
    if n_gt == 0:
        return [], list(range(n_pred)), []
    
    # Compute pairwise distances
    dists = torch.cdist(pred_pts, gt_pts)
    
    matched_pred = set()
    matched_gt = set()
    tp_pairs = []
    
    # Greedy matching - assign closest pairs first
    flat = dists.flatten()
    for idx in flat.argsort():
        if flat[idx] > match_radius:
            break
        pi, gi = (idx // n_gt).item(), (idx % n_gt).item()
        if pi not in matched_pred and gi not in matched_gt:
            matched_pred.add(pi)
            matched_gt.add(gi)
            tp_pairs.append((pi, gi, flat[idx].item()))
    
    fp_indices = [i for i in range(n_pred) if i not in matched_pred]
    fn_indices = [i for i in range(n_gt) if i not in matched_gt]
    
    return tp_pairs, fp_indices, fn_indices


def visualize_predictions(
    image: np.ndarray,
    pred_pts: np.ndarray,
    pred_scores: np.ndarray,
    gt_pts: np.ndarray,
    heatmap: np.ndarray,
    match_radius: float = 25.0,
    image_name: str = "",
    save_path: Optional[str] = None,
    show: bool = False
):
    """
    Visualize predictions with TP/FP/FN annotations.
    """
    # Match predictions to ground truth
    pred_tensor = torch.from_numpy(pred_pts).float()
    gt_tensor = torch.from_numpy(gt_pts).float()
    tp_pairs, fp_indices, fn_indices = match_predictions_to_gt(
        pred_tensor, gt_tensor, match_radius
    )
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # --- Plot 1: Original image with all annotations ---
    ax1 = axes[0]
    ax1.imshow(image)
    
    # Plot True Positives (green circles)
    for pi, gi, dist in tp_pairs:
        ax1.plot(pred_pts[pi, 0], pred_pts[pi, 1], 'go', markersize=12, 
                markerfacecolor='none', markeredgewidth=2, label='TP' if pi == tp_pairs[0][0] else '')
    
    # Plot False Positives (red X)
    for pi in fp_indices:
        ax1.plot(pred_pts[pi, 0], pred_pts[pi, 1], 'rx', markersize=12, 
                markeredgewidth=2, label='FP' if pi == fp_indices[0] else '')
    
    # Plot False Negatives (blue squares)
    for gi in fn_indices:
        ax1.plot(gt_pts[gi, 0], gt_pts[gi, 1], 'bs', markersize=12, 
                markerfacecolor='none', markeredgewidth=2, label='FN' if gi == fn_indices[0] else '')
    
    # Add legend
    handles = []
    if tp_pairs:
        handles.append(mpatches.Patch(color='green', label=f'TP: {len(tp_pairs)}'))
    if fp_indices:
        handles.append(mpatches.Patch(color='red', label=f'FP: {len(fp_indices)}'))
    if fn_indices:
        handles.append(mpatches.Patch(color='blue', label=f'FN: {len(fn_indices)}'))
    ax1.legend(handles=handles, loc='upper right', fontsize=10)
    
    # Calculate metrics
    precision = len(tp_pairs) / max(len(tp_pairs) + len(fp_indices), 1)
    recall = len(tp_pairs) / max(len(tp_pairs) + len(fn_indices), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    
    ax1.set_title(f'{image_name}\nP={precision:.2f} R={recall:.2f} F1={f1:.2f}', fontsize=10)
    ax1.axis('off')
    
    # --- Plot 2: Heatmap overlay ---
    ax2 = axes[1]
    ax2.imshow(image)
    
    # Resize heatmap to image size
    hm_resized = np.array(Image.fromarray(heatmap).resize(
        (image.shape[1], image.shape[0]), Image.BILINEAR
    ))
    
    # Create custom colormap (transparent to red)
    colors = [(1, 0, 0, 0), (1, 0, 0, 0.3), (1, 0.5, 0, 0.6), (1, 1, 0, 0.8)]
    cmap = LinearSegmentedColormap.from_list('heatmap', colors)
    
    ax2.imshow(hm_resized, cmap=cmap, vmin=0, vmax=1)
    
    # Mark GT points
    if len(gt_pts) > 0:
        ax2.scatter(gt_pts[:, 0], gt_pts[:, 1], c='cyan', s=30, marker='+', linewidths=1)
    
    ax2.set_title(f'Heatmap (max={heatmap.max():.3f})\nGT points: {len(gt_pts)}', fontsize=10)
    ax2.axis('off')
    
    # --- Plot 3: Detailed FP/FN analysis ---
    ax3 = axes[2]
    ax3.imshow(image)
    
    # Draw match radius circles for FPs
    for pi in fp_indices:
        circle = plt.Circle((pred_pts[pi, 0], pred_pts[pi, 1]), match_radius, 
                            fill=False, color='red', linewidth=1, linestyle='--')
        ax3.add_patch(circle)
        ax3.plot(pred_pts[pi, 0], pred_pts[pi, 1], 'rx', markersize=8, markeredgewidth=2)
        # Show confidence score
        ax3.annotate(f'{pred_scores[pi]:.2f}', 
                    (pred_pts[pi, 0], pred_pts[pi, 1] - 15),
                    color='red', fontsize=8, ha='center')
    
    # Draw FNs with their nearest prediction distance
    for gi in fn_indices:
        ax3.plot(gt_pts[gi, 0], gt_pts[gi, 1], 'bs', markersize=10, 
                markerfacecolor='none', markeredgewidth=2)
        
        # Find distance to nearest prediction
        if len(pred_pts) > 0:
            dists = np.sqrt(((pred_pts - gt_pts[gi]) ** 2).sum(axis=1))
            min_dist = dists.min()
            nearest_pred_idx = dists.argmin()
            
            # Draw line to nearest prediction
            ax3.plot([gt_pts[gi, 0], pred_pts[nearest_pred_idx, 0]],
                    [gt_pts[gi, 1], pred_pts[nearest_pred_idx, 1]],
                    'b--', linewidth=1, alpha=0.5)
            ax3.annotate(f'd={min_dist:.0f}', 
                        (gt_pts[gi, 0], gt_pts[gi, 1] + 15),
                        color='blue', fontsize=8, ha='center')
    
    ax3.set_title(f'FP/FN Analysis\nMatch radius: {match_radius}px', fontsize=10)
    ax3.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    plt.close()
    
    return {
        'tp': len(tp_pairs),
        'fp': len(fp_indices),
        'fn': len(fn_indices),
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def visualize_heatmap_distribution(
    all_heatmaps: List[np.ndarray],
    all_gt_counts: List[int],
    save_path: Optional[str] = None
):
    """
    Visualize heatmap statistics to understand model behavior.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Flatten all heatmap values
    all_values = np.concatenate([h.flatten() for h in all_heatmaps])
    
    # Plot 1: Histogram of all heatmap values
    ax1 = axes[0, 0]
    ax1.hist(all_values, bins=100, edgecolor='black', alpha=0.7)
    ax1.axvline(0.3, color='r', linestyle='--', label='Threshold 0.3')
    ax1.axvline(0.5, color='orange', linestyle='--', label='Threshold 0.5')
    ax1.set_xlabel('Heatmap Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of All Heatmap Values')
    ax1.legend()
    ax1.set_yscale('log')
    
    # Plot 2: Max heatmap value per image
    ax2 = axes[0, 1]
    max_values = [h.max() for h in all_heatmaps]
    ax2.hist(max_values, bins=30, edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(max_values), color='r', linestyle='--', 
                label=f'Mean: {np.mean(max_values):.3f}')
    ax2.set_xlabel('Max Heatmap Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Max Heatmap Value per Image')
    ax2.legend()
    
    # Plot 3: Number of peaks above threshold vs GT count
    ax3 = axes[1, 0]
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    for thresh in thresholds:
        peaks = []
        for h in all_heatmaps:
            # Simple peak counting (values above threshold)
            peaks.append((h > thresh).sum())
        ax3.scatter(all_gt_counts, peaks, alpha=0.5, s=20, label=f'thresh={thresh}')
    
    # Add diagonal line
    max_count = max(max(all_gt_counts), max(peaks))
    ax3.plot([0, max_count], [0, max_count], 'k--', alpha=0.5)
    ax3.set_xlabel('Ground Truth Count')
    ax3.set_ylabel('Predicted Peaks (above threshold)')
    ax3.set_title('Peak Count vs GT Count')
    ax3.legend()
    
    # Plot 4: Heatmap sparsity
    ax4 = axes[1, 1]
    sparsity = [(h > 0.1).sum() / h.size * 100 for h in all_heatmaps]
    ax4.hist(sparsity, bins=30, edgecolor='black', alpha=0.7)
    ax4.set_xlabel('% of heatmap > 0.1')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Heatmap Activation Sparsity')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def run_visualization(
    checkpoint_path: str,
    val_csv: str,
    val_image_dir: str,
    output_dir: str,
    num_images: int = 20,
    threshold: float = 0.3,
    match_radius: float = 25.0,
    image_size: int = 512,
    device: str = 'cuda'
):
    """
    Run visualization on validation set.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    
    # Build model
    backbone = config.get('backbone', 'vit_large_patch16_dinov3.sat493m')
    hidden_dim = config.get('hidden_dim', 256)
    extract_layers = config.get('extract_layers', [8, 16, 24])
    heatmap_stride = config.get('heatmap_stride', 4)
    
    print(f"Building model with backbone: {backbone}")
    model = ImprovedPointDetectorDINOv3(
        backbone=backbone,
        hidden_dim=hidden_dim,
        extract_layers=extract_layers,
        freeze_backbone=True,
        image_size=image_size,
        heatmap_stride=heatmap_stride
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from epoch {checkpoint.get('epoch', '?')}")
    
    # Load dataset
    print(f"Loading validation data...")
    val_dataset = HeatmapPointDataset(
        val_csv, val_image_dir, image_size,
        heatmap_stride=heatmap_stride,
        augment=False
    )
    
    heatmap_size = image_size // heatmap_stride
    scale = image_size / heatmap_size
    
    # Process images
    all_metrics = []
    all_heatmaps = []
    all_gt_counts = []
    
    # Also load original images for visualization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    print(f"\nProcessing {min(num_images, len(val_dataset))} images...")
    
    with torch.no_grad():
        for idx in range(min(num_images, len(val_dataset))):
            # Get data
            img_tensor, target = val_dataset[idx]
            img_name = target['image_name']
            gt_pts = target['points'].numpy()
            
            # Load original image for visualization
            orig_image = np.array(Image.open(
                os.path.join(val_image_dir, img_name)
            ).convert('RGB').resize((image_size, image_size)))
            
            # Run inference
            img_batch = img_tensor.unsqueeze(0).to(device)
            outputs = model(img_batch)
            
            pred_heatmap = torch.sigmoid(outputs['heatmap'][0]).cpu()
            pred_count = outputs['count'][0].cpu().item()
            
            # Extract points
            pred_pts, pred_scores = extract_points_from_heatmap(
                pred_heatmap, threshold=threshold
            )
            pred_pts = (pred_pts * scale).numpy()
            pred_scores = pred_scores.numpy()
            
            # Store for statistics
            all_heatmaps.append(pred_heatmap.numpy())
            all_gt_counts.append(len(gt_pts))
            
            # Visualize
            save_path = output_dir / f'{idx:03d}_{Path(img_name).stem}.png'
            metrics = visualize_predictions(
                image=orig_image,
                pred_pts=pred_pts,
                pred_scores=pred_scores,
                gt_pts=gt_pts,
                heatmap=pred_heatmap.numpy(),
                match_radius=match_radius,
                image_name=img_name,
                save_path=str(save_path)
            )
            metrics['image_name'] = img_name
            metrics['pred_count'] = pred_count
            metrics['gt_count'] = len(gt_pts)
            all_metrics.append(metrics)
            
            print(f"  [{idx+1}/{num_images}] {img_name}: "
                  f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} "
                  f"F1={metrics['f1']:.3f}")
    
    # Visualize heatmap distribution
    visualize_heatmap_distribution(
        all_heatmaps, all_gt_counts,
        save_path=str(output_dir / 'heatmap_distribution.png')
    )
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_tp = sum(m['tp'] for m in all_metrics)
    total_fp = sum(m['fp'] for m in all_metrics)
    total_fn = sum(m['fn'] for m in all_metrics)
    
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    
    print(f"Total TP: {total_tp}")
    print(f"Total FP: {total_fp}")
    print(f"Total FN: {total_fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Find worst images
    print("\nWorst images by F1:")
    sorted_metrics = sorted(all_metrics, key=lambda x: x['f1'])
    for m in sorted_metrics[:5]:
        print(f"  {m['image_name']}: F1={m['f1']:.3f} TP={m['tp']} FP={m['fp']} FN={m['fn']}")
    
    print("\nImages with most FPs:")
    sorted_by_fp = sorted(all_metrics, key=lambda x: x['fp'], reverse=True)
    for m in sorted_by_fp[:5]:
        print(f"  {m['image_name']}: FP={m['fp']} (TP={m['tp']}, FN={m['fn']})")
    
    print("\nImages with most FNs:")
    sorted_by_fn = sorted(all_metrics, key=lambda x: x['fn'], reverse=True)
    for m in sorted_by_fn[:5]:
        print(f"  {m['image_name']}: FN={m['fn']} (TP={m['tp']}, FP={m['fp']})")
    
    # Save metrics to JSON
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump({
            'threshold': threshold,
            'match_radius': match_radius,
            'total': {'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
                     'precision': precision, 'recall': recall, 'f1': f1},
            'per_image': all_metrics
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize point detection predictions')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--val_csv', type=str, required=True,
                        help='Path to validation CSV')
    parser.add_argument('--val_image_dir', type=str, required=True,
                        help='Path to validation images')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                        help='Output directory for visualizations')
    parser.add_argument('--num_images', type=int, default=20,
                        help='Number of images to visualize')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Detection threshold')
    parser.add_argument('--match_radius', type=float, default=25.0,
                        help='Match radius in pixels')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Image size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    run_visualization(
        checkpoint_path=args.checkpoint,
        val_csv=args.val_csv,
        val_image_dir=args.val_image_dir,
        output_dir=args.output_dir,
        num_images=args.num_images,
        threshold=args.threshold,
        match_radius=args.match_radius,
        image_size=args.image_size,
        device=args.device
    )


if __name__ == '__main__':
    main()
