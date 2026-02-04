"""
Visualization functions for point detection evaluation.

Add these to your training script or import as a module.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image


# =============================================================================
# CORE VISUALIZATION FUNCTIONS
# =============================================================================

def denormalize_image(img_tensor: torch.Tensor,
                      mean: List[float] = [0.485, 0.456, 0.406],
                      std: List[float] = [0.229, 0.224, 0.225]) -> np.ndarray:
    """Convert normalized tensor back to displayable image."""
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(std) + np.array(mean)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def visualize_single_prediction(
        image: np.ndarray,
        gt_points: np.ndarray,
        pred_points: np.ndarray,
        pred_scores: np.ndarray,
        conf_map: np.ndarray,
        match_radius: float = 25.0,
        image_name: str = "",
        save_path: Optional[str] = None,
        show: bool = False
) -> Dict:
    """
    Visualize predictions for a single image.

    Returns:
        Dict with TP, FP, FN counts and indices
    """
    # Match predictions to GT
    tp_pairs, fp_indices, fn_indices = match_predictions(
        pred_points, gt_points, match_radius
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # === Plot 1: Detections ===
    ax1 = axes[0]
    ax1.imshow(image)

    # True Positives (green circles)
    for pi, gi, dist in tp_pairs:
        ax1.plot(pred_points[pi, 0], pred_points[pi, 1], 'o',
                 color='lime', markersize=14, markerfacecolor='none',
                 markeredgewidth=2)
        ax1.plot(gt_points[gi, 0], gt_points[gi, 1], '+',
                 color='lime', markersize=10, markeredgewidth=2)

    # False Positives (red X)
    for pi in fp_indices:
        ax1.plot(pred_points[pi, 0], pred_points[pi, 1], 'x',
                 color='red', markersize=12, markeredgewidth=2)

    # False Negatives (blue squares)
    for gi in fn_indices:
        ax1.plot(gt_points[gi, 0], gt_points[gi, 1], 's',
                 color='blue', markersize=12, markerfacecolor='none',
                 markeredgewidth=2)

    # Legend
    handles = []
    if tp_pairs:
        handles.append(mpatches.Patch(color='lime', label=f'TP: {len(tp_pairs)}'))
    if fp_indices:
        handles.append(mpatches.Patch(color='red', label=f'FP: {len(fp_indices)}'))
    if fn_indices:
        handles.append(mpatches.Patch(color='blue', label=f'FN: {len(fn_indices)}'))
    ax1.legend(handles=handles, loc='upper right')

    # Metrics
    p = len(tp_pairs) / max(len(tp_pairs) + len(fp_indices), 1)
    r = len(tp_pairs) / max(len(tp_pairs) + len(fn_indices), 1)
    f1 = 2 * p * r / max(p + r, 1e-6)
    ax1.set_title(f'{image_name}\nP={p:.2f} R={r:.2f} F1={f1:.2f}')
    ax1.axis('off')

    # === Plot 2: Confidence Heatmap ===
    ax2 = axes[1]
    ax2.imshow(image)

    # Upsample confidence map to image size
    conf_up = np.array(Image.fromarray(conf_map).resize(
        (image.shape[1], image.shape[0]), Image.BILINEAR
    ))

    # Overlay heatmap
    cmap = plt.cm.hot
    cmap.set_under(alpha=0)
    im = ax2.imshow(conf_up, cmap=cmap, vmin=0.1, vmax=1.0, alpha=0.6)
    plt.colorbar(im, ax=ax2, fraction=0.046)

    # Mark GT points
    if len(gt_points) > 0:
        ax2.scatter(gt_points[:, 0], gt_points[:, 1],
                    c='cyan', s=40, marker='+', linewidths=2)

    ax2.set_title(f'Confidence Map\nmax={conf_map.max():.3f} mean={conf_map.mean():.3f}')
    ax2.axis('off')

    # === Plot 3: Score Distribution ===
    ax3 = axes[2]

    if len(pred_scores) > 0:
        # Separate TP and FP scores
        tp_scores = [pred_scores[pi] for pi, _, _ in tp_pairs]
        fp_scores = [pred_scores[pi] for pi in fp_indices]

        bins = np.linspace(0, 1, 21)
        if tp_scores:
            ax3.hist(tp_scores, bins=bins, alpha=0.7, color='lime', label=f'TP (n={len(tp_scores)})')
        if fp_scores:
            ax3.hist(fp_scores, bins=bins, alpha=0.7, color='red', label=f'FP (n={len(fp_scores)})')

        ax3.axvline(0.3, color='black', linestyle='--', label='Threshold')
        ax3.legend()
        ax3.set_xlabel('Confidence Score')
        ax3.set_ylabel('Count')
        ax3.set_title('Score Distribution')
    else:
        ax3.text(0.5, 0.5, 'No predictions', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Score Distribution')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

    return {
        'tp': len(tp_pairs),
        'fp': len(fp_indices),
        'fn': len(fn_indices),
        'precision': p,
        'recall': r,
        'f1': f1,
        'tp_pairs': tp_pairs,
        'fp_indices': fp_indices,
        'fn_indices': fn_indices
    }


def match_predictions(pred_pts: np.ndarray, gt_pts: np.ndarray,
                      match_radius: float) -> Tuple[List, List, List]:
    """
    Match predictions to ground truth using greedy nearest neighbor.

    Returns:
        tp_pairs: List of (pred_idx, gt_idx, distance)
        fp_indices: List of unmatched prediction indices
        fn_indices: List of unmatched ground truth indices
    """
    n_pred, n_gt = len(pred_pts), len(gt_pts)

    if n_pred == 0:
        return [], [], list(range(n_gt))
    if n_gt == 0:
        return [], list(range(n_pred)), []

    # Pairwise distances
    pred_t = torch.from_numpy(pred_pts).float()
    gt_t = torch.from_numpy(gt_pts).float()
    dists = torch.cdist(pred_t, gt_t).numpy()

    matched_pred, matched_gt = set(), set()
    tp_pairs = []

    # Greedy matching
    flat_indices = np.argsort(dists.flatten())
    for idx in flat_indices:
        pi, gi = idx // n_gt, idx % n_gt
        if dists[pi, gi] > match_radius:
            break
        if pi not in matched_pred and gi not in matched_gt:
            matched_pred.add(pi)
            matched_gt.add(gi)
            tp_pairs.append((pi, gi, dists[pi, gi]))

    fp_indices = [i for i in range(n_pred) if i not in matched_pred]
    fn_indices = [i for i in range(n_gt) if i not in matched_gt]

    return tp_pairs, fp_indices, fn_indices


# =============================================================================
# BATCH VISUALIZATION
# =============================================================================

def visualize_evaluation(
        model,
        dataloader,
        device,
        stride: int,
        threshold: float = 0.3,
        match_radius: float = 25.0,
        output_dir: str = './visualizations',
        num_images: int = 20,
        extract_points_fn=None,  # Pass your extract_points function
        show_worst: bool = True
):
    """
    Run evaluation with visualization on a subset of images.

    Args:
        model: Trained model
        dataloader: Validation dataloader
        device: torch device
        stride: Model output stride
        threshold: Detection threshold
        match_radius: Matching radius in pixels
        output_dir: Where to save visualizations
        num_images: Number of images to visualize
        extract_points_fn: Function to extract points from model output
        show_worst: If True, prioritize worst performing images
    """
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    all_results = []
    all_data = []

    print(f"Running evaluation with visualization...")

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            outputs = model(images)

            for b in range(len(targets)):
                # Get data
                img_tensor = images[b].cpu()
                img_np = denormalize_image(img_tensor)

                gt_pts = targets[b]['points'].numpy() if 'points' in targets[b] else np.zeros((0, 2))
                if 'n_points' in targets[b]:
                    n_gt = targets[b]['n_points'].item()
                    gt_pts = gt_pts[:n_gt]

                img_name = targets[b].get('name', targets[b].get('image_name', f'img_{batch_idx}_{b}'))

                # Get predictions
                conf = outputs['conf'][b]
                offset = outputs['offset'][b]
                conf_sigmoid = torch.sigmoid(conf).cpu().numpy()

                if extract_points_fn:
                    pred_pts, pred_scores = extract_points_fn(conf, offset, threshold, stride)
                    pred_pts = pred_pts.cpu().numpy()
                    pred_scores = pred_scores.cpu().numpy()
                else:
                    pred_pts = np.zeros((0, 2))
                    pred_scores = np.zeros((0,))

                # Compute metrics
                tp_pairs, fp_idx, fn_idx = match_predictions(pred_pts, gt_pts, match_radius)

                p = len(tp_pairs) / max(len(tp_pairs) + len(fp_idx), 1)
                r = len(tp_pairs) / max(len(tp_pairs) + len(fn_idx), 1)
                f1 = 2 * p * r / max(p + r, 1e-6)

                all_results.append({
                    'idx': len(all_results),
                    'name': img_name,
                    'f1': f1,
                    'precision': p,
                    'recall': r,
                    'tp': len(tp_pairs),
                    'fp': len(fp_idx),
                    'fn': len(fn_idx),
                    'n_gt': len(gt_pts),
                    'n_pred': len(pred_pts),
                    'max_conf': conf_sigmoid.max(),
                })

                all_data.append({
                    'image': img_np,
                    'gt_points': gt_pts,
                    'pred_points': pred_pts,
                    'pred_scores': pred_scores,
                    'conf_map': conf_sigmoid,
                    'name': img_name,
                })

    # Select which images to visualize
    if show_worst:
        # Sort by F1 (worst first)
        sorted_indices = sorted(range(len(all_results)), key=lambda i: all_results[i]['f1'])
    else:
        sorted_indices = list(range(len(all_results)))

    selected = sorted_indices[:num_images]

    print(f"\nVisualizing {len(selected)} images...")

    for rank, idx in enumerate(selected):
        data = all_data[idx]
        result = all_results[idx]

        save_path = output_dir / f'{rank:03d}_{Path(data["name"]).stem}_f1={result["f1"]:.2f}.png'

        visualize_single_prediction(
            image=data['image'],
            gt_points=data['gt_points'],
            pred_points=data['pred_points'],
            pred_scores=data['pred_scores'],
            conf_map=data['conf_map'],
            match_radius=match_radius,
            image_name=data['name'],
            save_path=str(save_path)
        )

        print(f"  [{rank + 1}/{len(selected)}] {data['name']}: F1={result['f1']:.3f} "
              f"(TP={result['tp']} FP={result['fp']} FN={result['fn']})")

    # Summary statistics
    print_evaluation_summary(all_results, output_dir)

    return all_results


def print_evaluation_summary(results: List[Dict], output_dir: Path):
    """Print and save evaluation summary statistics."""

    total_tp = sum(r['tp'] for r in results)
    total_fp = sum(r['fp'] for r in results)
    total_fn = sum(r['fn'] for r in results)

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    print(f"\n{'=' * 60}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total images: {len(results)}")
    print(f"Total TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Per-image stats
    f1_scores = [r['f1'] for r in results]
    print(f"\nPer-image F1: mean={np.mean(f1_scores):.3f} std={np.std(f1_scores):.3f}")
    print(f"  min={np.min(f1_scores):.3f} max={np.max(f1_scores):.3f}")

    # Worst images
    print(f"\nWorst 5 images by F1:")
    sorted_by_f1 = sorted(results, key=lambda x: x['f1'])
    for r in sorted_by_f1[:5]:
        print(f"  {r['name']}: F1={r['f1']:.3f} (TP={r['tp']} FP={r['fp']} FN={r['fn']})")

    # Most FPs
    print(f"\nTop 5 images by FP count:")
    sorted_by_fp = sorted(results, key=lambda x: x['fp'], reverse=True)
    for r in sorted_by_fp[:5]:
        print(f"  {r['name']}: FP={r['fp']} (TP={r['tp']} FN={r['fn']})")

    # Most FNs
    print(f"\nTop 5 images by FN count:")
    sorted_by_fn = sorted(results, key=lambda x: x['fn'], reverse=True)
    for r in sorted_by_fn[:5]:
        print(f"  {r['name']}: FN={r['fn']} (TP={r['tp']} FP={r['fp']})")

    # Save results to JSON
    import json
    summary = {
        'overall': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn,
        },
        'per_image': results
    }

    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir / 'evaluation_results.json'}")


# =============================================================================
# AGGREGATE PLOTS
# =============================================================================

def plot_confidence_analysis(results: List[Dict], output_dir: str):
    """Plot confidence distribution and correlation with performance."""

    output_dir = Path(output_dir)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: F1 distribution
    ax1 = axes[0, 0]
    f1_scores = [r['f1'] for r in results]
    ax1.hist(f1_scores, bins=20, edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(f1_scores), color='r', linestyle='--',
                label=f'Mean: {np.mean(f1_scores):.3f}')
    ax1.set_xlabel('F1 Score')
    ax1.set_ylabel('Count')
    ax1.set_title('Per-Image F1 Distribution')
    ax1.legend()

    # Plot 2: TP/FP/FN distribution
    ax2 = axes[0, 1]
    tp = [r['tp'] for r in results]
    fp = [r['fp'] for r in results]
    fn = [r['fn'] for r in results]

    x = np.arange(len(results))
    width = 0.25
    ax2.bar(x - width, tp, width, label='TP', color='green', alpha=0.7)
    ax2.bar(x, fp, width, label='FP', color='red', alpha=0.7)
    ax2.bar(x + width, fn, width, label='FN', color='blue', alpha=0.7)
    ax2.set_xlabel('Image Index')
    ax2.set_ylabel('Count')
    ax2.set_title('TP/FP/FN per Image')
    ax2.legend()

    # Plot 3: Max confidence vs F1
    ax3 = axes[1, 0]
    max_confs = [r['max_conf'] for r in results]
    ax3.scatter(max_confs, f1_scores, alpha=0.6)
    ax3.set_xlabel('Max Confidence')
    ax3.set_ylabel('F1 Score')
    ax3.set_title('Max Confidence vs F1')

    # Plot 4: GT count vs performance
    ax4 = axes[1, 1]
    n_gt = [r['n_gt'] for r in results]
    ax4.scatter(n_gt, f1_scores, alpha=0.6)
    ax4.set_xlabel('Number of GT Objects')
    ax4.set_ylabel('F1 Score')
    ax4.set_title('Object Count vs F1')

    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_analysis.png', dpi=150)
    plt.close()

    print(f"Saved confidence analysis to {output_dir / 'confidence_analysis.png'}")


# =============================================================================
# UPDATED EVALUATE FUNCTION WITH VISUALIZATION
# =============================================================================

def evaluate_with_visualization(
        model,
        dataloader,
        device,
        stride: int,
        threshold: float = 0.3,
        match_radius: float = 25.0,
        visualize: bool = False,
        vis_dir: str = './visualizations',
        num_vis: int = 20,
        extract_points_fn=None
):
    """
    Evaluate model with optional visualization.

    Args:
        model: Trained model
        dataloader: Validation dataloader
        device: torch device
        stride: Model output stride
        threshold: Detection threshold
        match_radius: Matching radius in pixels
        visualize: Whether to save visualizations
        vis_dir: Directory to save visualizations
        num_vis: Number of images to visualize
        extract_points_fn: Your extract_points function

    Returns:
        Dict with metrics
    """
    model.eval()

    total_tp, total_fp, total_fn = 0, 0, 0
    all_max_conf = []
    all_results = []
    all_data = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images)

            for b in range(len(targets)):
                # Handle different target formats
                if 'points' in targets[b]:
                    gt_points = targets[b]['points'].to(device)
                    if 'n_points' in targets[b]:
                        n_gt = targets[b]['n_points'].item()
                        gt_points = gt_points[:n_gt]
                else:
                    gt_points = torch.zeros((0, 2), device=device)

                pred_conf = outputs['conf'][b]
                pred_offset = outputs['offset'][b]

                conf_sigmoid = torch.sigmoid(pred_conf)
                all_max_conf.append(conf_sigmoid.max().item())

                # Extract points
                if extract_points_fn:
                    pred_points, pred_scores = extract_points_fn(
                        pred_conf, pred_offset, threshold, stride
                    )
                else:
                    pred_points = torch.zeros((0, 2), device=device)
                    pred_scores = torch.zeros((0,), device=device)

                n_pred, n_gt = len(pred_points), len(gt_points)
                matched_gt, matched_pred = set(), set()

                if n_pred > 0 and n_gt > 0:
                    dists = torch.cdist(pred_points, gt_points)
                    for idx in dists.flatten().argsort():
                        if dists.flatten()[idx] > match_radius:
                            break
                        pi, gi = (idx // n_gt).item(), (idx % n_gt).item()
                        if pi not in matched_pred and gi not in matched_gt:
                            matched_pred.add(pi)
                            matched_gt.add(gi)

                tp = len(matched_pred)
                fp = n_pred - len(matched_pred)
                fn = n_gt - len(matched_gt)

                total_tp += tp
                total_fp += fp
                total_fn += fn

                # Store for visualization
                if visualize:
                    p = tp / max(tp + fp, 1)
                    r = tp / max(tp + fn, 1)
                    f1 = 2 * p * r / max(p + r, 1e-6)

                    img_name = targets[b].get('name', targets[b].get('image_name', f'img_{len(all_results)}'))

                    all_results.append({
                        'idx': len(all_results),
                        'name': img_name,
                        'f1': f1, 'precision': p, 'recall': r,
                        'tp': tp, 'fp': fp, 'fn': fn,
                        'n_gt': n_gt, 'n_pred': n_pred,
                        'max_conf': conf_sigmoid.max().item()
                    })

                    all_data.append({
                        'image': denormalize_image(images[b].cpu()),
                        'gt_points': gt_points.cpu().numpy(),
                        'pred_points': pred_points.cpu().numpy(),
                        'pred_scores': pred_scores.cpu().numpy(),
                        'conf_map': conf_sigmoid.cpu().numpy(),
                        'name': img_name
                    })

    # Compute metrics
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    # Visualization
    if visualize and all_data:
        vis_dir = Path(vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)

        # Sort by F1 (worst first)
        sorted_indices = sorted(range(len(all_results)), key=lambda i: all_results[i]['f1'])
        selected = sorted_indices[:num_vis]

        print(f"\nSaving {len(selected)} visualizations to {vis_dir}")

        for rank, idx in enumerate(selected):
            data = all_data[idx]
            result = all_results[idx]

            save_path = vis_dir / f'{rank:03d}_{Path(data["name"]).stem}_f1={result["f1"]:.2f}.png'

            visualize_single_prediction(
                image=data['image'],
                gt_points=data['gt_points'],
                pred_points=data['pred_points'],
                pred_scores=data['pred_scores'],
                conf_map=data['conf_map'],
                match_radius=match_radius,
                image_name=data['name'],
                save_path=str(save_path)
            )

        # Summary plots
        plot_confidence_analysis(all_results, vis_dir)
        print_evaluation_summary(all_results, vis_dir)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'avg_max_conf': np.mean(all_max_conf) if all_max_conf else 0
    }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == '__main__':
    print("""
Example usage:

from visualization import evaluate_with_visualization, extract_points

# After training, run evaluation with visualization:
metrics = evaluate_with_visualization(
    model=model,
    dataloader=val_loader,
    device=device,
    stride=model.stride,
    threshold=0.3,
    match_radius=25,
    visualize=True,
    vis_dir='./visualizations',
    num_vis=30,
    extract_points_fn=extract_points
)

print(f"F1: {metrics['f1']:.4f}")
""")