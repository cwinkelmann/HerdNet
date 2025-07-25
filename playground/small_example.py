import torch
import matplotlib.pyplot as plt
import numpy as np

from animaloc.data import FIDT


def visualize_fidt_output(image, target, fidt_map, save_path=None):
    """
    Visualize FIDT transformation output

    Args:
        image: Original image tensor [C, H, W]
        target: Target dict with 'points' and 'labels'
        fidt_map: Output FIDT map [num_classes, H, W]
        save_path: Optional path to save figure
    """

    # Convert tensors to numpy for plotting
    if isinstance(image, torch.Tensor):
        if image.shape[0] == 3:  # RGB
            img_np = image.permute(1, 2, 0).numpy()
        else:  # Grayscale
            img_np = image.squeeze().numpy()

    points = target['points'].numpy() if isinstance(target['points'], torch.Tensor) else target['points']
    labels = target['labels'].numpy() if isinstance(target['labels'], torch.Tensor) else target['labels']

    num_classes = fidt_map.shape[0]

    # Create subplot layout
    fig, axes = plt.subplots(2, num_classes + 1, figsize=(4 * (num_classes + 1), 8))
    if num_classes == 1:
        axes = axes.reshape(2, -1)

    # scale the image [0, 1] to [0,255] for visualization
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255

    # Plot original image with annotations
    axes[0, 0].imshow(img_np, cmap='gray' if len(img_np.shape) == 2 else None)

    # Color map for different classes
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']

    for point, label in zip(points, labels):
        color = colors[int(label - 1) % len(colors)]
        axes[0, 0].plot(point[0], point[1], 'o', color=color, markersize=8,
                        markerfacecolor=color, markeredgecolor='white', markeredgewidth=2)

    axes[0, 0].set_title('Original Image + Points')
    axes[0, 0].axis('off')

    # Plot FIDT maps for each class
    for class_idx in range(num_classes):
        fidt_class = fidt_map[class_idx].numpy()

        # Top row: FIDT heatmap
        im1 = axes[0, class_idx + 1].imshow(fidt_class, cmap='hot', interpolation='bilinear')
        axes[0, class_idx + 1].set_title(f'FIDT Map - Class {class_idx + 1}')
        axes[0, class_idx + 1].axis('off')
        plt.colorbar(im1, ax=axes[0, class_idx + 1], fraction=0.046, pad=0.04)

        # Bottom row: Thresholded peaks (for local maxima detection)
        threshold = 0.3  # Adjust based on your needs
        peaks = fidt_class > threshold

        # Show original image with detected peaks
        axes[1, class_idx + 1].imshow(img_np, cmap='gray' if len(img_np.shape) == 2 else None, alpha=0.7)

        # Find and mark local maxima
        from scipy.ndimage import maximum_filter
        local_maxima = (fidt_class == maximum_filter(fidt_class, size=3)) & (fidt_class > threshold)
        y_peaks, x_peaks = np.where(local_maxima)

        for x, y in zip(x_peaks, y_peaks):
            axes[1, class_idx + 1].plot(x, y, 's', color=colors[class_idx % len(colors)],
                                        markersize=6, markerfacecolor='none', markeredgewidth=2)

        axes[1, class_idx + 1].set_title(f'Detected Peaks - Class {class_idx + 1}\n({len(x_peaks)} peaks found)')
        axes[1, class_idx + 1].axis('off')

    # Bottom left: Combined visualization
    axes[1, 0].imshow(img_np, cmap='gray' if len(img_np.shape) == 2 else None, alpha=0.7)

    # Overlay all FIDT maps
    combined_fidt = fidt_map.sum(dim=0).numpy()
    axes[1, 0].imshow(combined_fidt, cmap='hot', alpha=0.5, interpolation='bilinear')
    axes[1, 0].set_title('Combined FIDT Maps')
    axes[1, 0].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# Example usage with your FIDT class
def test_with_sample_data():
    """Test the visualization with sample data"""

    # Create sample image and targets
    image = torch.randn(3, 256, 256)  # Random RGB image
    target = {
        'points': torch.tensor([[50, 50], [150, 80], [200, 200], [180, 180]]),
        'labels': torch.tensor([1, 1, 2, 2])
    }

    # Apply FIDT transform with different parameters for larger objects
    fidt_large = FIDT(alpha=0.02, beta=0.75, c=1.0, radius=5, num_classes=3)
    _, fidt_map_large = fidt_large(image, target)

    fidt_small = FIDT(alpha=0.02, beta=0.75, c=1.0, radius=1, num_classes=3)
    _, fidt_map_small = fidt_small(image, target)

    # Visualize both
    fig1 = visualize_fidt_output(image, target, fidt_map_small, save_path="./fidt_small.png")
    fig1.suptitle('FIDT with Small Object Parameters (radius=1)', fontsize=16)

    fig2 = visualize_fidt_output(image, target, fidt_map_large, save_path="./fidt_large.png")
    fig2.suptitle('FIDT with Large Object Parameters (radius=5)', fontsize=16)

    plt.show()

    # Print statistics
    print("Small objects FIDT - Max values per class:", fidt_map_small.max(dim=-1)[0].max(dim=-1)[0])
    print("Large objects FIDT - Max values per class:", fidt_map_large.max(dim=-1)[0].max(dim=-1)[0])


if __name__ == "__main__":
    test_with_sample_data()