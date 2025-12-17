"""
Visualize Model Attention: See What the Model "Sees" When Classifying Points

This script helps answer:
- "Why did the model call this rock an iguana?"
- "Why did the model miss this obvious iguana?"
- "What parts of the scene does the model focus on?"

Usage:
    python visualize_attention.py \
        --checkpoint ./outputs_optimized/best.pth \
        --image /path/to/image.jpg \
        --output ./attention_maps/
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
import cv2
from pathlib import Path

# Import model
import sys
sys.path.append('.')
from two_stage_detector_optimized import TwoStagePointDetector


def load_model(checkpoint_path, device):
    """Load trained model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = TwoStagePointDetector(
        backbone='vit_large_patch16_dinov3.sat493m',
        freeze_backbone=True,
        heatmap_size=128,
        max_proposals=300,
        refine_hidden=512,
        roi_size=11,
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def preprocess_image(image_path, size=512):
    """Load and preprocess image."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    
    # Convert to tensor
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # Normalize (ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_norm = (img_array - mean) / std
    
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor, img_array


def visualize_predictions(image, points, scores, cls_probs, threshold=0.3):
    """Visualize predictions with classification confidence."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. All proposals
    axes[0].imshow(image)
    axes[0].set_title('All Proposals (before classification)')
    for i, (pt, score) in enumerate(zip(points, scores)):
        x, y = pt[0] * 512, pt[1] * 512
        circle = Circle((x, y), 5, color='yellow', fill=False, linewidth=1)
        axes[0].add_patch(circle)
    axes[0].axis('off')
    
    # 2. After classification (colored by confidence)
    axes[1].imshow(image)
    axes[1].set_title('After Classification (color = confidence)')
    for i, (pt, score, cls_prob) in enumerate(zip(points, scores, cls_probs)):
        x, y = pt[0] * 512, pt[1] * 512
        
        # Color by classification confidence
        if cls_prob > 0.7:
            color = 'green'  # Confident positive
            size = 8
        elif cls_prob < 0.3:
            color = 'red'  # Confident negative
            size = 5
        else:
            color = 'orange'  # Uncertain
            size = 6
        
        circle = Circle((x, y), size, color=color, fill=False, linewidth=2)
        axes[1].add_patch(circle)
        
        # Add confidence text for high-confidence predictions
        if cls_prob > 0.7 or cls_prob < 0.3:
            axes[1].text(x, y-10, f'{cls_prob:.2f}', 
                        color=color, fontsize=8, ha='center',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    axes[1].axis('off')
    
    # 3. Final detections (thresholded)
    axes[2].imshow(image)
    axes[2].set_title(f'Final Detections (threshold={threshold})')
    n_detections = 0
    for i, (pt, score, cls_prob) in enumerate(zip(points, scores, cls_probs)):
        final_score = score * cls_prob
        if final_score >= threshold:
            x, y = pt[0] * 512, pt[1] * 512
            circle = Circle((x, y), 10, color='lime', fill=False, linewidth=3)
            axes[2].add_patch(circle)
            axes[2].text(x, y-15, f'{final_score:.2f}', 
                        color='lime', fontsize=10, ha='center',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            n_detections += 1
    
    axes[2].text(10, 30, f'Detections: {n_detections}', 
                color='white', fontsize=14,
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig


def visualize_attention_maps(image, points, attention_weights, cls_probs, top_k=6):
    """
    Visualize attention maps for top-k most confident predictions.
    Shows: "What parts of the scene is the model looking at?"
    """
    # Select top-k confident predictions
    sorted_idx = np.argsort(np.abs(cls_probs - 0.5))[::-1][:top_k]
    
    n_cols = 3
    n_rows = (top_k + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()
    
    for idx, point_idx in enumerate(sorted_idx):
        pt = points[point_idx]
        cls_prob = cls_probs[point_idx]
        attn = attention_weights[point_idx]  # [H, W]
        
        # Overlay attention on image
        ax = axes[idx]
        ax.imshow(image)
        
        # Resize attention map to image size
        attn_resized = cv2.resize(attn, (512, 512), interpolation=cv2.INTER_LINEAR)
        
        # Show attention as heatmap overlay
        im = ax.imshow(attn_resized, cmap='jet', alpha=0.5, vmin=0, vmax=attn_resized.max())
        
        # Mark the point
        x, y = pt[0] * 512, pt[1] * 512
        ax.plot(x, y, 'w*', markersize=20, markeredgecolor='black', markeredgewidth=2)
        
        # Title with classification
        decision = "IGUANA" if cls_prob > 0.5 else "ROCK"
        confidence = cls_prob if cls_prob > 0.5 else 1 - cls_prob
        color = 'green' if cls_prob > 0.5 else 'red'
        
        ax.set_title(f'{decision} (conf={confidence:.2%})', 
                     color=color, fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for idx in range(len(sorted_idx), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Attention Maps: Where is the model looking?', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize model attention")
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint path')
    parser.add_argument('--image', required=True, help='Input image path')
    parser.add_argument('--output', default='./attention_viz/', help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.3, help='Detection threshold')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    
    # Load image
    print(f"Processing image: {args.image}")
    img_tensor, img_array = preprocess_image(args.image)
    img_tensor = img_tensor.to(device)
    
    # Forward pass with attention
    with torch.no_grad():
        # Stage 1: Proposals
        feat = model.extract_features(img_tensor)
        feat_up = torch.nn.functional.interpolate(
            feat, size=model.heatmap_size, mode='bilinear', align_corners=False
        )
        heatmap = model.stage1(feat)
        proposals, prop_scores, prop_mask = model.generate_proposals(heatmap)
        
        # Stage 2: Classification with attention
        stage2_out = model.stage2.forward_with_attention(feat_up, proposals, prop_mask)
        
        cls_logits = stage2_out['cls_logits']
        offsets = stage2_out['offsets']
        attention_weights = stage2_out['attention_weights']
        
        final_points = proposals + offsets
        final_points = final_points.clamp(0, 1)
        
        # Get classification probabilities
        cls_probs = torch.sigmoid(cls_logits)
        final_scores = prop_scores * cls_probs
    
    # Extract for visualization (batch=0)
    points = final_points[0, prop_mask[0]].cpu().numpy()
    scores = prop_scores[0, prop_mask[0]].cpu().numpy()
    cls_probs_np = cls_probs[0, prop_mask[0]].cpu().numpy()
    attention = attention_weights[0, prop_mask[0]].cpu().numpy()
    
    print(f"\nResults:")
    print(f"  Total proposals: {prop_mask[0].sum().item()}")
    print(f"  Confident positives (>0.7): {(cls_probs_np > 0.7).sum()}")
    print(f"  Confident negatives (<0.3): {(cls_probs_np < 0.3).sum()}")
    print(f"  Uncertain (0.3-0.7): {((cls_probs_np >= 0.3) & (cls_probs_np <= 0.7)).sum()}")
    print(f"  Final detections (threshold={args.threshold}): {(scores * cls_probs_np >= args.threshold).sum()}")
    
    # Visualization 1: Predictions
    print("\nGenerating prediction visualization...")
    fig1 = visualize_predictions(img_array, points, scores, cls_probs_np, args.threshold)
    fig1.savefig(output_dir / 'predictions.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'predictions.png'}")
    
    # Visualization 2: Attention maps
    print("\nGenerating attention visualizations...")
    fig2 = visualize_attention_maps(img_array, points, attention, cls_probs_np, top_k=9)
    fig2.savefig(output_dir / 'attention_maps.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'attention_maps.png'}")
    
    # Classification confidence histogram
    print("\nGenerating confidence histogram...")
    fig3, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.hist(cls_probs_np, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(0.3, color='red', linestyle='--', linewidth=2, label='Low confidence')
    ax.axvline(0.7, color='green', linestyle='--', linewidth=2, label='High confidence')
    ax.set_xlabel('Classification Probability', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Classification Confidence Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig3.savefig(output_dir / 'confidence_histogram.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'confidence_histogram.png'}")
    
    print("\n✓ Visualization complete!")
    print(f"  Output directory: {output_dir}")


if __name__ == '__main__':
    main()
