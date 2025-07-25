import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import scipy.ndimage
from typing import Dict, List, Tuple, Optional


def _point_buffer(x: float, y: float, mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """Create circular buffer around point"""
    h, w = mask.shape
    y_grid, x_grid = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    distances = torch.sqrt((x_grid - x) ** 2 + (y_grid - y) ** 2)
    return distances <= radius


class FIDT:
    """Simple FIDT implementation for visualization"""

    def __init__(self, alpha: float = 0.02, beta: float = 0.75, c: float = 1.0, radius: int = 3, num_classes: int = 2):
        self.alpha = alpha
        self.beta = beta
        self.c = c
        self.radius = radius
        self.num_classes = num_classes - 1  # Exclude background

    def __call__(self, image: torch.Tensor, target: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply FIDT transformation"""
        h, w = image.shape[1], image.shape[2]
        fidt_maps = torch.zeros((self.num_classes, h, w))

        if len(target['points']) == 0:
            return image, fidt_maps

        points = target['points']
        labels = target['labels']

        # Group points by class
        class_points = {}
        for point, label in zip(points, labels):
            if label not in class_points:
                class_points[label] = []
            class_points[label].append(point)

        for class_id, class_pts in class_points.items():
            if class_id < 1 or class_id > self.num_classes:
                continue

            # Create binary mask
            mask = torch.ones((h, w))
            for x, y in class_pts:
                x, y = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
                point_buffer = _point_buffer(x, y, mask, self.radius)
                mask[point_buffer] = 0

            # Apply distance transform and FIDT formula
            dist_map = scipy.ndimage.distance_transform_edt(mask.numpy())
            dist_map = torch.from_numpy(dist_map)
            fidt_map = 1 / (torch.pow(dist_map, self.alpha * dist_map + self.beta) + self.c)
            fidt_map = torch.where(fidt_map < 0.01, 0., fidt_map)

            fidt_maps[class_id - 1] = fidt_map

        return image, fidt_maps


def load_iguana_data(csv_path: str, image_dir: str, max_samples: int = 5) -> List[Dict]:
    """Load iguana data from CSV in the specified format"""
    df = pd.read_csv(csv_path)
    print(f"Loaded CSV with {len(df)} annotations")
    print("CSV columns:", df.columns.tolist())
    print("First few rows:")
    print(df.head())

    # Group by image filename to get all points per image
    grouped = df.groupby('images')
    samples = []

    for image_name, group in grouped:
        if len(samples) >= max_samples:
            break

        image_path = Path(image_dir) / image_name
        if not image_path.exists():
            print(f"Image not found: {image_path}")
            continue

        # Extract points and labels for this image
        points = [(row['x'], row['y']) for _, row in group.iterrows()]
        labels = [row['labels'] for _, row in group.iterrows()]
        species = [row['species'] for _, row in group.iterrows()]

        samples.append({
            'image_path': image_path,
            'image_name': image_name,
            'points': points,
            'labels': labels,
            'species': species,
            'base_image': group.iloc[0]['base_images']
        })

        print(f"Loaded {image_name}: {len(points)} iguanas")

    return samples


def visualize_fidt_output(image, target, fidt_map, title="FIDT Visualization", save_path=None):
    """
    Visualize FIDT transformation output

    Args:
        image: Original image tensor [C, H, W]
        target: Target dict with 'points' and 'labels'
        fidt_map: Output FIDT map [num_classes, H, W]
        title: Title for the figure
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


# Example usage with iguana data
def visualize_iguana_data(csv_path: str, image_dir: str, max_samples: int = 3):
    """Load and visualize iguana data with FIDT transformation"""

    # Load iguana data
    samples = load_iguana_data(csv_path, image_dir, max_samples)

    if not samples:
        print("No samples loaded!")
        return

    # Test different FIDT parameters
    fidt_configs = [
        {"radius": 2, "alpha": 0.02, "name": "Small radius (r=2)"},
        {"radius": 4, "alpha": 0.02, "name": "Medium radius (r=4)"},
        {"radius": 6, "alpha": 0.02, "name": "Large radius (r=6)"},
    ]

    for sample in samples:
        print(f"\n{'=' * 60}")
        print(f"Processing: {sample['image_name']}")
        print(f"Base image: {sample['base_image']}")
        print(f"Iguanas: {len(sample['points'])}")
        print(f"Points: {sample['points']}")
        print(f"{'=' * 60}")

        # Load image
        try:
            image_pil = Image.open(sample['image_path']).convert('RGB')
            image_array = np.array(image_pil)

            # Convert to tensor format [C, H, W]
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0

        except Exception as e:
            print(f"Error loading image: {e}")
            continue

        # Prepare target format
        target = {
            'points': torch.tensor(sample['points'], dtype=torch.float32),
            'labels': torch.tensor(sample['labels'], dtype=torch.long)
        }

        # Create figure for parameter comparison
        fig, axes = plt.subplots(2, len(fidt_configs), figsize=(5 * len(fidt_configs), 10))
        if len(fidt_configs) == 1:
            axes = axes.reshape(-1, 1)

        for i, config in enumerate(fidt_configs):
            # Apply FIDT with current parameters
            fidt_transform = FIDT(
                alpha=config["alpha"],
                radius=config["radius"],
                num_classes=2  # Background + iguana
            )

            _, fidt_map = fidt_transform(image_tensor, target)

            # Plot original image with annotations
            axes[0, i].imshow(image_array)

            # Plot points
            for j, (point, label) in enumerate(zip(sample['points'], sample['labels'])):
                x, y = point
                axes[0, i].plot(x, y, 'ro', markersize=8,
                                markerfacecolor='red', markeredgecolor='white', markeredgewidth=2)
                axes[0, i].text(x + 5, y - 5, f'{j + 1}', color='white', fontweight='bold',
                                bbox=dict(boxstyle="round,pad=0.2", facecolor='red', alpha=0.7))

            axes[0, i].set_title(f'{config["name"]}\nOriginal ({len(sample["points"])} iguanas)')
            axes[0, i].axis('off')

            # Plot FIDT map
            fidt_combined = fidt_map.sum(dim=0)  # Sum all classes
            im = axes[1, i].imshow(fidt_combined.numpy(), cmap='hot', interpolation='bilinear')
            axes[1, i].set_title(f'FIDT Map\n(α={config["alpha"]}, r={config["radius"]})')
            axes[1, i].axis('off')

            # Add colorbar
            plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)

            # Find and mark local maxima
            from scipy.ndimage import maximum_filter
            threshold = 0.2
            fidt_np = fidt_combined.numpy()
            local_maxima = (fidt_np == maximum_filter(fidt_np, size=5)) & (fidt_np > threshold)
            y_peaks, x_peaks = np.where(local_maxima)

            for x, y in zip(x_peaks, y_peaks):
                axes[1, i].plot(x, y, 's', color='cyan', markersize=6,
                                markerfacecolor='none', markeredgewidth=2)

            # Print statistics
            max_val = fidt_combined.max().item()
            mean_val = fidt_combined.mean().item()
            print(f"  {config['name']}: Max={max_val:.3f}, Mean={mean_val:.3f}, Peaks detected={len(x_peaks)}")

        plt.tight_layout()
        plt.suptitle(f'FIDT Parameter Comparison: {sample["image_name"]}', fontsize=16, y=1.02)
        plt.show()

        # Create detailed visualization for best parameters (medium radius)
        best_fidt = FIDT(alpha=0.02, radius=4, num_classes=2)
        _, best_fidt_map = best_fidt(image_tensor, target)

        fig_detail = visualize_fidt_output(image_tensor, target, best_fidt_map,
                                           title=f"Detailed FIDT Analysis: {sample['image_name']}")
        plt.show()


def main():
    """Main function to run iguana FIDT visualization"""

    # Paths to your iguana data
    csv_path = "/home/christian/hnee/HerdNet/data_iguana/train_patches_640/gt.csv"
    image_dir = "/home/christian/hnee/HerdNet/data_iguana/train_patches_640"

    # Check if paths exist
    import os
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        print("Please check the path and try again.")
        return

    if not os.path.exists(image_dir):
        print(f"Image directory not found: {image_dir}")
        print("Please check the path and try again.")
        return

    print("Starting iguana FIDT visualization...")
    print(f"CSV: {csv_path}")
    print(f"Images: {image_dir}")

    # Run visualization
    visualize_iguana_data(csv_path, image_dir, max_samples=3)

    print("\nVisualization complete!")
    print("\nParameter recommendations based on results:")
    print("- For small iguanas (< 30px): radius=2-3")
    print("- For medium iguanas (30-50px): radius=4-5")
    print("- For large iguanas (> 50px): radius=6-8")
    print("- Keep alpha=0.02 for standard decay rate")


if __name__ == "__main__":
    main()