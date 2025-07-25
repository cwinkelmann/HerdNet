from pathlib import Path

import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import scipy.ndimage
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Optional, Union
import PIL.Image


def _point_buffer(x: float, y: float, mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """Create circular buffer around point"""
    h, w = mask.shape
    y_grid, x_grid = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    distances = torch.sqrt((x_grid - x) ** 2 + (y_grid - y) ** 2)
    return distances <= radius


class FIDTVisualizer:
    """Visualize FIDT transformation with different parameters"""

    def __init__(self,
                 alpha: float = 0.02,
                 beta: float = 0.75,
                 c: float = 1.0,
                 radius: int = 1):
        self.alpha = alpha
        self.beta = beta
        self.c = c
        self.radius = radius

    def create_fidt_map(self, points: List[Tuple[int, int]],
                        image_size: Tuple[int, int]) -> torch.Tensor:
        """Create FIDT map from list of points"""
        h, w = image_size

        # Create binary mask with points marked as 0
        mask = torch.ones((h, w))
        for x, y in points:
            point_buffer = _point_buffer(x, y, mask, self.radius)
            mask[point_buffer] = 0

        # Apply distance transform
        dist_map = scipy.ndimage.distance_transform_edt(mask.numpy())
        dist_map = torch.from_numpy(dist_map)

        # Apply FIDT formula
        fidt_map = 1 / (torch.pow(dist_map, self.alpha * dist_map + self.beta) + self.c)
        fidt_map = torch.where(fidt_map < 0.01, 0., fidt_map)

        return fidt_map, mask

    def visualize_comparison(self, points: List[Tuple[int, int]],
                             image_size: Tuple[int, int] = (200, 200),
                             parameter_sets: Optional[List[Dict]] = None):
        """Compare different parameter settings"""

        if parameter_sets is None:
            parameter_sets = [
                {"radius": 1, "alpha": 0.02, "name": "Small objects (original)"},
                {"radius": 3, "alpha": 0.02, "name": "Medium objects"},
                {"radius": 6, "alpha": 0.02, "name": "Large objects"},
                {"radius": 12, "alpha": 0.02, "name": "Very Large objects"},
                {"radius": 3, "alpha": 0.05, "name": "Medium objects + slower decay"}
            ]

        fig, axes = plt.subplots(2, len(parameter_sets), figsize=(4 * len(parameter_sets), 8))
        if len(parameter_sets) == 1:
            axes = axes.reshape(-1, 1)

        for i, params in enumerate(parameter_sets):
            # Update parameters
            old_params = (self.radius, self.alpha, self.beta, self.c)
            self.radius = params.get("radius", self.radius)
            self.alpha = params.get("alpha", self.alpha)
            self.beta = params.get("beta", self.beta)
            self.c = params.get("c", self.c)

            # Generate FIDT map
            fidt_map, mask = self.create_fidt_map(points, image_size)

            # Plot original points
            axes[0, i].imshow(1 - mask.numpy(), cmap='gray', alpha=0.7)
            for x, y in points:
                axes[0, i].plot(x, y, 'ro', markersize=8, markerfacecolor='red',
                                markeredgecolor='white', markeredgewidth=2)
            axes[0, i].set_title(f'{params["name"]}\nOriginal Points')
            axes[0, i].axis('off')

            # Plot FIDT map
            im = axes[1, i].imshow(fidt_map.numpy(), cmap='hot', interpolation='bilinear')
            axes[1, i].set_title(f'FIDT Map\n(α={self.alpha}, r={self.radius})')
            axes[1, i].axis('off')
            plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)

            # Restore parameters
            self.radius, self.alpha, self.beta, self.c = old_params

        plt.tight_layout()
        return fig

    def visualize_3d_surface(self, points: List[Tuple[int, int]],
                             image_size: Tuple[int, int] = (100, 100),
                             crop_region: Optional[Tuple[int, int, int, int]] = None):
        """Create 3D surface plot of FIDT peaks"""

        fidt_map, _ = self.create_fidt_map(points, image_size)

        # Crop region for better visualization if specified
        if crop_region:
            x1, y1, x2, y2 = crop_region
            fidt_map = fidt_map[y1:y2, x1:x2]
            # Adjust point coordinates
            points = [(x - x1, y - y1) for x, y in points if x1 <= x <= x2 and y1 <= y <= y2]

        h, w = fidt_map.shape
        x = np.arange(w)
        y = np.arange(h)
        X, Y = np.meshgrid(x, y)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Create surface plot
        surf = ax.plot_surface(X, Y, fidt_map.numpy(), cmap='hot',
                               alpha=0.8, antialiased=True)

        # Mark original points
        for x, y in points:
            if 0 <= x < w and 0 <= y < h:
                ax.scatter([x], [y], [fidt_map[y, x].item()],
                           color='blue', s=100, alpha=1.0)

        ax.set_title(f'FIDT 3D Surface (α={self.alpha}, β={self.beta}, radius={self.radius})')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_zlabel('FIDT Value')

        plt.colorbar(surf, shrink=0.5, aspect=5)
        return fig

    def analyze_peak_properties(self, points: List[Tuple[int, int]],
                                image_size: Tuple[int, int] = (200, 200)):
        """Analyze properties of FIDT peaks"""

        fidt_map, _ = self.create_fidt_map(points, image_size)

        print(f"FIDT Analysis (α={self.alpha}, β={self.beta}, c={self.c}, radius={self.radius})")
        print("-" * 60)
        print(f"Max value: {fidt_map.max():.4f}")
        print(f"Mean value: {fidt_map.mean():.4f}")
        print(f"Non-zero pixels: {(fidt_map > 0.01).sum().item()}")

        # Analyze individual peaks
        for i, (x, y) in enumerate(points):
            peak_value = fidt_map[y, x].item()
            print(f"Point {i + 1} at ({x}, {y}): peak value = {peak_value:.4f}")

            # Check if it's a local maximum
            local_region = fidt_map[max(0, y - 2):min(image_size[0], y + 3),
                           max(0, x - 2):min(image_size[1], x + 3)]
            is_local_max = peak_value == local_region.max().item()
            print(f"  Is local maximum: {is_local_max}")

        return fidt_map


# Example usage and comparison
def demo_fidt_visualization():
    """Demonstrate FIDT visualization with different scenarios"""

    # Test case 1: Sparse objects
    sparse_points = [(50, 50), (150, 50), (100, 150)]

    # Test case 2: Dense cluster
    dense_points = [(80, 80), (85, 82), (78, 85), (83, 88), (90, 85)]

    # Test case 3: Mixed density
    mixed_points = [(30, 30), (170, 30), (100, 100), (95, 105), (105, 95), (100, 110)]

    real_points_path = Path("/home/christian/hnee/HerdNet/data_iguana/train_patches_640/gt.csv")
    df_real_points = pd.read_csv(real_points_path)
    df_real_points = df_real_points[df_real_points["images"] == "FMO03___DJI_0466_36.JPG"]
    real_points = [(row['x'], row['y']) for _, row in df_real_points.iterrows()]

    scenarios = [
        #(sparse_points, "Sparse Objects"),
        # (dense_points, "Dense Cluster"),
        #(mixed_points, "Mixed Density")
        # Add more scenarios as needed
        (real_points, "Real-World Data Example")
    ]



    for points, title in scenarios:
        print(f"\n{title}")
        print("=" * 50)

        # Compare different parameter settings
        visualizer = FIDTVisualizer()
        fig1 = visualizer.visualize_comparison(points, image_size=(640, 640))
        fig1.suptitle(f'{title} - Parameter Comparison', fontsize=16)

        # 3D visualization with medium object settings
        visualizer_3d = FIDTVisualizer(radius=3, alpha=0.02)
        fig2 = visualizer_3d.visualize_3d_surface(points, image_size=(640, 640))
        fig2.suptitle(f'{title} - 3D Surface View', fontsize=16)

        # Analyze peak properties
        visualizer_3d.analyze_peak_properties(points, image_size=(640, 640))

        plt.show()


if __name__ == "__main__":
    demo_fidt_visualization()