"""
P2PNet Convergence Diagnostic Script

Run this to verify your model can overfit on a single batch.
If this doesn't converge, there's a fundamental architecture/loss issue.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from herdnet_p2p_edbug import HerdNetP2P
from p2p_debug import HungarianPointLoss


# Import your fixed modules (adjust paths as needed)
# from herdnet_p2p_fixed import HerdNetP2P
# from hungarian_loss_fixed import HungarianPointLoss


def create_synthetic_batch(batch_size=1, img_size=512, n_points=5, device='cuda'):
    """Create synthetic training data with known point locations."""

    images = torch.randn(batch_size, 3, img_size, img_size, device=device)

    targets = []
    for b in range(batch_size):
        # Random points in (y, x) format (row, col)
        points = torch.randint(50, img_size - 50, (n_points, 2), dtype=torch.float32, device=device)
        labels = torch.ones(n_points, dtype=torch.long, device=device)

        targets.append({
            'points': points,  # (y, x) format
            'labels': labels
        })

    return images, targets


def diagnose_forward_pass(model, images, targets):
    """Check that forward pass produces sensible outputs."""

    model.eval()
    with torch.no_grad():
        outputs = model(images, targets)

    print("\n=== Forward Pass Diagnostics ===")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Points shape: {outputs['points'].shape}")
    print(f"Points normalized shape: {outputs['points_normalized'].shape}")

    # Check value ranges
    pts_norm = outputs['points_normalized']
    print(f"\nPoints normalized range: [{pts_norm.min():.4f}, {pts_norm.max():.4f}]")
    print(f"Expected range: [0, 1]")

    # Check logits
    logits = outputs['logits']
    probs = torch.softmax(logits.flatten(2).transpose(1, 2), dim=-1)
    fg_prob = probs[:, :, 1]
    print(f"\nForeground prob range: [{fg_prob.min():.4f}, {fg_prob.max():.4f}]")
    print(f"Mean foreground prob: {fg_prob.mean():.4f}")

    return outputs


def diagnose_loss(criterion, outputs, targets):
    """Check loss computation."""

    print("\n=== Loss Diagnostics ===")

    loss = criterion(outputs, targets)
    print(f"Total loss: {loss.item():.4f}")

    # Check individual components if available
    return loss


def overfit_single_batch(model, criterion, images, targets, n_steps=200, lr=1e-4):
    """Attempt to overfit on a single batch."""

    print("\n=== Overfitting Test ===")
    print(f"Learning rate: {lr}")
    print(f"Steps: {n_steps}")

    model.train()
    # Only optimize adapter and heads, freeze backbone
    params_to_optimize = []
    for name, param in model.named_parameters():
        if 'backbone' not in name:
            params_to_optimize.append(param)

    optimizer = torch.optim.AdamW(params_to_optimize, lr=lr)

    losses = []

    for step in range(n_steps):
        optimizer.zero_grad()

        outputs = model(images, targets)
        loss = outputs.get('loss_p2p', criterion(outputs, targets))

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        losses.append(loss.item())

        if step % 20 == 0:
            print(f"Step {step:4d}: Loss = {loss.item():.4f}")

    print(f"\nInitial loss: {losses[0]:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Loss decreased: {losses[0] - losses[-1]:.4f}")

    return losses


def visualize_predictions(model, images, targets, image_size=512):
    """Visualize predictions vs ground truth."""

    model.eval()
    with torch.no_grad():
        outputs = model(images, targets)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Get predictions
    pts_pixel = outputs['points'][0].cpu().numpy()  # [N, 2] in (x, y)
    logits = outputs['logits'][0].cpu()  # [C, H, W]
    probs = torch.softmax(logits, dim=0)[1].numpy()  # Foreground prob

    # Get GT
    gt_pts = targets[0]['points'].cpu().numpy()  # [M, 2] in (y, x)

    # Left: Probability heatmap
    axes[0].imshow(probs, cmap='hot', vmin=0, vmax=1)
    axes[0].scatter(gt_pts[:, 1], gt_pts[:, 0], c='cyan', s=100, marker='x', linewidths=2, label='GT')
    axes[0].set_title('Foreground Probability')
    axes[0].legend()

    # Right: Top-K predictions
    flat_probs = torch.softmax(logits.flatten(1), dim=0)[1]  # [H*W]
    k = min(20, len(gt_pts) * 3)
    top_k_idx = flat_probs.topk(k).indices.numpy()

    h, w = probs.shape
    top_k_y = top_k_idx // w
    top_k_x = top_k_idx % w

    # Scale to image coordinates
    scale_y = image_size / h
    scale_x = image_size / w

    axes[1].set_xlim(0, image_size)
    axes[1].set_ylim(image_size, 0)  # Flip y-axis
    axes[1].scatter(gt_pts[:, 1], gt_pts[:, 0], c='green', s=100, marker='o', label='GT', alpha=0.7)
    axes[1].scatter(top_k_x * scale_x + scale_x / 2, top_k_y * scale_y + scale_y / 2,
                    c='red', s=50, marker='^', label=f'Top-{k} Pred', alpha=0.7)
    axes[1].set_title('Point Predictions')
    axes[1].legend()
    axes[1].set_aspect('equal')

    plt.tight_layout()
    plt.savefig('p2p_diagnostic.png', dpi=150)
    plt.close()
    print("\nSaved visualization to p2p_diagnostic.png")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create synthetic data
    print("\n" + "=" * 50)
    print("Creating synthetic data...")
    images, targets = create_synthetic_batch(
        batch_size=1,
        img_size=512,
        n_points=5,  # Start with few points for overfitting test
        device=device
    )

    print(f"Image shape: {images.shape}")
    print(f"GT points:\n{targets[0]['points']}")

    # Initialize model
    print("\n" + "=" * 50)
    print("Initializing model...")

    # Option 1: Use your fixed model
    model = HerdNetP2P(
        backbone='timm/vit_large_patch16_dinov3.sat493m',
        num_classes=2,
        pretrained=True,
        freeze_backbone=True,  # Freeze for overfitting test
        hidden_dim=256
    ).to(device)

    criterion = HungarianPointLoss(cost_class=1.0, cost_point=5.0)
    model.set_criterion(criterion)

    # For testing without the actual model:
    print("NOTE: Uncomment model initialization lines and import your modules to run")
    print("This script provides the diagnostic framework.")


    # Diagnose forward pass
    outputs = diagnose_forward_pass(model, images, targets)

    # Diagnose loss
    diagnose_loss(model.criterion, outputs, targets)

    # Overfit test
    losses = overfit_single_batch(
        model,
        model.criterion,
        images,
        targets,
        n_steps=200,
        lr=1e-4
    )

    # Visualize results
    visualize_predictions(model, images, targets)

    # Plot loss curve
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Overfitting Test Loss Curve')
    plt.savefig('p2p_loss_curve.png', dpi=150)
    plt.close()
    print("Saved loss curve to p2p_loss_curve.png")

    # Summary
    print("\n" + "=" * 50)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 50)

    if losses[-1] < losses[0] * 0.5:
        print("✓ Loss decreased significantly - model can learn!")
    else:
        print("✗ Loss did not decrease enough - check architecture/loss")

    if losses[-1] < 0.5:
        print("✓ Final loss is low - overfitting successful")
    else:
        print("⚠ Final loss still high - may need more steps or tuning")


if __name__ == '__main__':
    main()