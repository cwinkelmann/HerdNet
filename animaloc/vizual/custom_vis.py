import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle
from matplotlib.patches import Circle, Rectangle
import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['model_vizual', 'plot_heatmaps', 'plot_heatmaps_combined', 'denormalize_image']

def model_vizual(image: torch.Tensor, target: torch.Tensor, output: List[torch.Tensor]) -> plt.Figure:
    """
    Create visualization comparing input images, targets, and model predictions.

    Args:
        image: Input tensor of shape (B, C, H, W) or (C, H, W)
        target: Target tensor (format depends on your task)
        output: List of 2 tensors [heatmap, class_map]

    Returns:
        matplotlib Figure for wandb logging
    """
    # Extract heatmap and class map from output
    heatmap, class_map = output[0], output[1]

    # Handle batch dimension - take first item if batch
    if len(image.shape) == 4:
        img = image[0]  # (C, H, W)
        tgt = target[0] if len(target.shape) == 4 else target
        hmap = heatmap[0] if len(heatmap.shape) == 4 else heatmap
        cmap = class_map[0] if len(class_map.shape) == 4 else class_map
    else:
        img, tgt, hmap, cmap = image, target, heatmap, class_map

    # Convert tensors to numpy and move to CPU
    img_np = _tensor_to_numpy(img)
    target_np = _tensor_to_numpy(tgt)
    heatmap_np = _tensor_to_numpy(hmap)
    classmap_np = _tensor_to_numpy(cmap)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Validation Visualization', fontsize=16)

    # Row 1: Original data
    # Original image
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')

    # Target
    if len(target_np.shape) == 3:  # Multi-class target
        axes[0, 1].imshow(target_np, cmap='tab20')
    else:  # Single channel target
        axes[0, 1].imshow(target_np, cmap='gray')
    axes[0, 1].set_title('Ground Truth')
    axes[0, 1].axis('off')

    # Target overlay on image
    axes[0, 2].imshow(img_np)
    if len(target_np.shape) == 3:
        axes[0, 2].imshow(target_np, alpha=0.5, cmap='tab20')
    else:
        axes[0, 2].imshow(target_np, alpha=0.5, cmap='Reds')
    axes[0, 2].set_title('Image + Ground Truth')
    axes[0, 2].axis('off')

    # Row 2: Predictions
    # Heatmap
    heatmap_display = heatmap_np
    if len(heatmap_display.shape) == 3:  # Multi-channel heatmap
        heatmap_display = np.max(heatmap_display, axis=0)  # Take max across channels

    im1 = axes[1, 0].imshow(heatmap_display, cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title('Predicted Heatmap')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Class map
    if len(classmap_np.shape) == 3:  # Multi-class
        classmap_display = np.argmax(classmap_np, axis=0)
    else:
        classmap_display = classmap_np

    im2 = axes[1, 1].imshow(classmap_display, cmap='tab20')
    axes[1, 1].set_title('Predicted Classes')
    axes[1, 1].axis('off')

    # Prediction overlay on image
    axes[1, 2].imshow(img_np)
    axes[1, 2].imshow(heatmap_display, alpha=0.6, cmap='hot')
    axes[1, 2].set_title('Image + Prediction')
    axes[1, 2].axis('off')

    plt.tight_layout()
    return fig


def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array for visualization."""
    # Move to CPU and detach from computation graph
    tensor = tensor.detach().cpu()

    # Convert to numpy
    array = tensor.numpy()

    # Handle different tensor formats
    if len(array.shape) == 3 and array.shape[0] in [1, 3]:  # (C, H, W)
        if array.shape[0] == 3:  # RGB image
            array = np.transpose(array, (1, 2, 0))  # (H, W, C)
            # Normalize to [0, 1] if needed
            if array.max() > 1.0:
                array = array / 255.0
        elif array.shape[0] == 1:  # Single channel
            array = array.squeeze(0)  # (H, W)

    # Ensure values are in valid range
    array = np.clip(array, 0, 1)

    return array




def denormalize_image(img_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    img_tensor: torch.Tensor of shape (3, H, W)
    Returns: numpy array of shape (H, W, 3) in range [0, 1]
    """
    mean = torch.tensor(mean).view(3, 1, 1).to(img_tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(img_tensor.device)
    img = img_tensor * std + mean
    img = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).cpu().numpy()





def plot_heatmaps(image_tensor, heatmap_tensor,
                  class_names=None, max_channels=7,
                  overlay_channel=0, alpha=0.5, show_argmax_overlay=True, cmap="inferno"):
    """
    Visualize the heatmaps next to the input image, including overlays.

    Parameters:
    - image_tensor: (3, H, W) torch.Tensor
    - heatmap_tensor: (C, H, W) torch.Tensor
    - class_names: optional list of names for each heatmap channel
    - max_channels: maximum number of heatmaps to show
    - overlay_channel: which channel to overlay onto image
    - alpha: transparency for overlays
    - show_argmax_overlay: whether to show the argmax overlay
    """
    image_np = denormalize_image(image_tensor)
    heatmap_tensor = heatmap_tensor.detach().cpu()


    H, W = image_tensor.shape[1], image_tensor.shape[2]
    heatmap_tensor = F.interpolate(heatmap_tensor.unsqueeze(0),
                                   size=(H, W), mode='bilinear', align_corners=False)[0]
    heatmap_tensor = torch.where(heatmap_tensor < 0.01, 0.0, heatmap_tensor)

    num_channels = min(heatmap_tensor.shape[0], max_channels)
    num_subplots = num_channels + 1 if show_argmax_overlay else num_channels

    img_aspect = W / H

    # Adjust figure size - reduce height, optimize width
    fig_width = 4 * (num_subplots + 1)  # Reduced from 8 to 4 per subplot
    fig_height = 4 * (1 / img_aspect) if img_aspect > 1 else 4

    fig, axes = plt.subplots(1, num_subplots+1, figsize=(fig_width, fig_height))
    # if num_subplots == 1:
    #     axes = [axes]

    # Original image
    axes[0].imshow(image_np)
    # axes[0].set_title("Image")
    # axes[0].axis("off")

    # Heatmaps
    for i in range(num_channels):
        heat = heatmap_tensor[i].numpy()
        axes[i + 1].imshow(heat, cmap=cmap)
        # title = f"Heatmap {i}" if class_names is None else class_names[i]
        # axes[i + 1].set_title(title)
        axes[i + 1].axis("off")

    # Overlay selected channel
    if show_argmax_overlay and overlay_channel < heatmap_tensor.shape[0]:
        heat_overlay = heatmap_tensor[overlay_channel].numpy()
        ax_idx = num_channels + 1

        axes[ax_idx].imshow(image_np)

        # Create a masked array to make low values transparent
        heat_masked = np.ma.masked_where(heat_overlay < 0.05, heat_overlay)

        axes[ax_idx].imshow(heat_masked, cmap="inferno", alpha=alpha)

        # axes[ax_idx].imshow(heat_overlay, cmap="inferno", alpha=alpha)
        # axes[ax_idx].set_title(f"Overlay: Channel {overlay_channel}")
        axes[ax_idx].axis("off")


    plt.tight_layout(pad=1.5)
    return fig, axes


def plot_heatmaps_combined(image_tensor, heatmap_tensor,
                           combine_method='max', alpha=0.5,
                           cmap="inferno", threshold=0.05):
    """
    Visualize the original image and the image with combined heatmap overlay.

    Parameters:
    - image_tensor: (3, H, W) torch.Tensor
    - heatmap_tensor: (C, H, W) torch.Tensor
    - combine_method: how to combine heatmaps ('max', 'mean', 'argmax')
    - alpha: transparency for overlays
    - cmap: colormap for heatmap
    - threshold: minimum value to show in overlay (for transparency)
    """
    image_np = denormalize_image(image_tensor)
    heatmap_tensor = heatmap_tensor.detach().cpu()

    H, W = image_tensor.shape[1], image_tensor.shape[2]
    try:
        heatmap_tensor = F.interpolate(heatmap_tensor.unsqueeze(0).unsqueeze(0),
                                   size=(H, W), mode='bilinear', align_corners=False)[0]
    except:
        heatmap_tensor = F.interpolate(heatmap_tensor.unsqueeze(0),
                                       size=(H, W), mode='bilinear', align_corners=False)[0]
    heatmap_tensor = torch.where(heatmap_tensor < 0.01, 0.0, heatmap_tensor)

    # Combine heatmaps across channels
    if combine_method == 'max':
        combined_heatmap = torch.max(heatmap_tensor, dim=0)[0]
    elif combine_method == 'mean':
        combined_heatmap = torch.mean(heatmap_tensor, dim=0)
    elif combine_method == 'argmax':
        # Create a heatmap showing the max activation value at each pixel
        combined_heatmap = torch.max(heatmap_tensor, dim=0)[0]
    else:
        raise ValueError("combine_method must be 'max', 'mean', or 'argmax'")

    combined_heatmap = combined_heatmap.numpy()

    # Create figure with 2 subplots
    img_aspect = W / H
    fig_width = 5  # Total width for 2 subplots
    fig_height = 5 * (1 / img_aspect) if img_aspect > 1 else 5

    fig, axes = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    #
    # # Original image
    # axes[0].imshow(image_np)
    # axes[0].set_title("Augmented Image")
    # axes[0].axis("off")
    #
    # # Image with combined heatmap overlay
    axes.imshow(image_np)

    # Create a masked array to make low values transparent
    heat_masked = np.ma.masked_where(combined_heatmap < threshold, combined_heatmap)

    im = axes.imshow(heat_masked, cmap=cmap, alpha=alpha)
    # axes.set_title(f"Augmented Annotation Heatmap")
    axes.axis("off")

    # Add colorbar
    # plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout(pad=1.5)
    return fig, axes