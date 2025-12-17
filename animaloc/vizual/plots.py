__copyright__ = \
    """
    Copyright (C) 2024 University of Liège, Gembloux Agro-Bio Tech, Forest Is Life
    All rights reserved.

    This source code is under the MIT License.

    Please contact the author Alexandre Delplanque (alexandre.delplanque@uliege.be) for any questions.

    Last modification: March 18, 2024
    """
__author__ = "Alexandre Delplanque"
__license__ = "MIT License"
__version__ = "0.2.1"

import itertools
import random
import typing
from pathlib import Path
from typing import Any
from typing import Optional, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn.functional as F
import wandb
from loguru import logger
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from sklearn.decomposition import PCA
from torch import Tensor

from animaloc.vizual.custom_vis import plot_heatmaps, denormalize_image

__all__ = ['PlotPrecisionRecall', 'Visualiser', 'HeatMapVisualizer', 'visualize_sample', "DebugVisualizer"]

class PlotPrecisionRecall:

    def __init__(
        self,
        figsize: tuple = (7,7), 
        legend: bool = False, 
        seed: int = 1
        ) -> None:
        
        self.figsize = figsize
        self.legend = legend
        self.seed = seed

        self._data = []
        self._labels = []

    def feed(self, recalls: list, precisions: list, label: Optional[str] = None) -> None:
        # recalls.append(recalls[-1])
        # precisions.append(0)
        self._data.append((recalls, precisions))
        self._labels.append(label)
    
    def plot(self) -> None:
        
        random.seed(self.seed)
        colors = self._gen_colors(len(self._data))
        
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(1,1,1)
        ax.set_xlim(0,1.02)
        ax.set_ylim(0,1.02)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')

        markers = self._markers
        for i, (recall, precision) in enumerate(self._data):
            ax.plot(recall, precision,
                color=colors[i],
                marker=next(markers),
                markevery=0.1,
                alpha=0.7,
                label=self._labels[i])
        
        if self.legend:
            lg = plt.legend(bbox_to_anchor=(1.04,1), loc='upper left')
        
        self.fig = fig
    
    def save(self, path: Path) -> None:
        if 'fig' not in self.__dict__:
            self.plot()

        self.fig.savefig(path, dpi=300, format='png', bbox_inches='tight')

    def _gen_colors(self, n: int) -> list:

        colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
            for i in range(n)]

        return colors
    
    @property
    def _markers(self) -> itertools.cycle:
        return itertools.cycle(('^','o','s','x','D','v','>'))





class Visualiser(typing.Callable):
    def __init__(self, output_path: Path):
        self.output_path = output_path
        Path(self.output_path).mkdir(parents=True, exist_ok=True)

class HeatMapVisualizer(Visualiser):
    def __init__(self, output_path, down_ratio):
        super().__init__(output_path)
        self.down_ratio = down_ratio

    def __call__(self, image: Tensor, target: Dict,
                 output: typing.Tuple[Tensor, Tensor],
                 epoch: int,
                 output_name: str = 'heatmap.png',
                 visualise_predictions: pd.DataFrame | None = None
                 ):

        """
        Visualizes a sample image, target points, and model output.

        Args:
            image: Input image tensor [B, C, H, W] (normalized)
            target: Target dictionary containing 'points', 'labels'
            output: Model output (optional) - could be predictions, heatmaps, etc.

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        output_name = f"heatmap_overlay_{target['original_image_name'][0][0]}_{epoch}.png"

        fig = visualize_sample(image, target, output)
        Path(self.output_path).mkdir(parents=True, exist_ok=True)

        fig.savefig(Path(self.output_path) / output_name )
        wandb.log({output_name: wandb.Image(fig)})
        plt.close(fig)


        output_name = f"heatmap_target_{target['original_image_name'][0][0]}_{epoch}.png"
        heatmap_fig = visualise_full_res_heatmap(image,
                                                 target,
                                                 output
                                   )
        if visualise_predictions is not None and len(visualise_predictions):


            for idx, row in visualise_predictions.iterrows():
                y = row['loc'][0] * self.down_ratio
                x = row['loc'][1] * self.down_ratio
                circ = Circle((x, y), radius=2, color='white', fill=False, linewidth=2)
                heatmap_fig.axes[0].add_patch(circ)

                # Add text label next to the circle
                heatmap_fig.axes[0].text(x + 10, y, f"sc: {row['scores']:.2f}, ds: {row['dscores']:.2f}",
                                         color='white', fontsize=8, va='center',
                                         bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))

        wandb.log({output_name: wandb.Image(heatmap_fig)})
        heatmap_fig.savefig(Path(self.output_path) / output_name)
        plt.close(heatmap_fig)
        return fig



#
# class DebugVisualizer(Visualiser):
#
#
#     def __init__(self, output_path: Path):
#         super().__init__(Path(output_path))
#
#
#     def __call__(self, debug_data, image: Tensor, target: Dict,
#                  output: typing.Tuple[Tensor, Tensor],
#                  epoch: int,
#                  output_name: str = 'heatmap.png',
#                  visualise_predictions: pd.DataFrame | None = None
#                  ):
#
#         """
#         Visualizes a sample image, target points, and model output.
#
#         Args:
#             image: Input image tensor [B, C, H, W] (normalized)
#             target: Target dictionary containing 'points', 'labels'
#             output: Model output (optional) - could be predictions, heatmaps, etc.
#
#         Returns:
#             matplotlib.figure.Figure: The created figure
#         """
#
#
#         fig = self.visualize_debug(debug_data, image, save_path= f"debug_pca_{target['original_image_name'][0][0]}_{epoch}.png")
#
#         return fig
#
#     def visualize_debug(self, debug_data, img_tensor, save_path="debug_layers.png"):
#         """
#         Runs the model in debug mode and saves a grid of all internal layers.
#         img_tensor: [1, 3, H, W] normalized
#         """
#         # model.eval()
#          # with torch.no_grad():
#         #     # Get debug dictionary
#         #     debug_data = model(img_tensor, debug=True)
#
#         # We will plot: Input, Prediction, Fused, and Backbones
#         # Prepare figure
#         num_layers = len(debug_data['backbone'])
#         cols = num_layers + 3  # Input, Pred, Fused, + Layers
#         fig, axes = plt.subplots(2, cols, figsize=(4 * cols, 8))
#
#         # 1. Plot Input
#         input_img = img_tensor[0].permute(1, 2, 0).cpu().numpy()
#         # Simple denorm for visualization (approximate)
#         input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())
#         axes[0, 0].imshow(input_img)
#         axes[0, 0].set_title("Input")
#         axes[1, 0].axis('off')
#
#         # 2. Plot Prediction
#         pred = debug_data['prediction'][0, 0].cpu().numpy()
#         axes[0, 1].imshow(pred, cmap='jet')
#         axes[0, 1].set_title("Prediction")
#         axes[1, 1].axis('off')
#
#         # 3. Plot Fused Feature (PCA)
#         # The 'fused' map is what the head sees. If this is blurry, the head fails.
#         fused = debug_data['fused'][0].cpu().numpy()  # [C, H, W]
#         self.show_pca(fused, axes[0, 2])
#         axes[0, 2].set_title("Fused (PCA)")
#
#         # Plot Fused Feature (Activation Energy)
#         # L2 Norm shows WHERE the features are strong
#         activation = np.linalg.norm(fused, axis=0)
#         axes[1, 2].imshow(activation, cmap='magma')
#         axes[1, 2].set_title("Fused (Energy)")
#
#         # 4. Plot Backbone Layers
#         for i, (name, feat_tensor) in enumerate(debug_data['backbone'].items()):
#             feat = feat_tensor[0].cpu().numpy()  # [C, H, W]
#             col_idx = i + 3
#
#             # Row 0: PCA (Semantic View)
#             self.show_pca(feat, axes[0, col_idx])
#             axes[0, col_idx].set_title(f"{name} (PCA)")
#
#             # Row 1: Energy (Activity View)
#             activation = np.linalg.norm(feat, axis=0)
#             axes[1, col_idx].imshow(activation, cmap='magma')
#             axes[1, col_idx].set_title(f"{name} (Energy)")
#
#         plt.tight_layout()
#         plt.savefig(self.output_path / save_path)
#         wandb.log({save_path: wandb.Image(fig)})
#         logger.info(f"Debug Image saved to {self.output_path / save_path}")
#         plt.close(fig)
#         return fig
#
#     def show_pca(self, feature_map, ax):
#         """
#         Project High-Dim feature map [C, H, W] to [H, W, 3] RGB using PCA.
#         """
#         C, H, W = feature_map.shape
#         # Flatten spatial dims: [C, N]
#         flat_feat = feature_map.reshape(C, -1).transpose()  # [N, C]
#
#         # PCA to 3 components
#         pca = PCA(n_components=3)
#         rgb = pca.fit_transform(flat_feat)  # [N, 3]
#
#         # Normalize to 0-1
#         rgb = (rgb - rgb.min(0)) / (rgb.max(0) - rgb.min(0))
#
#         # Reshape back to image
#         rgb_img = rgb.reshape(H, W, 3)
#         ax.imshow(rgb_img)


"""
Debug Visualizer for HerdNetDINOv2

This module provides visualization tools for debugging the HerdNetDINOv2 model.
It shows intermediate features, attention maps, and predictions.

Usage:
    visualizer = DebugVisualizer(output_path=Path("./debug_outputs"))

    # In training loop:
    debug_output = model(images, debug=True)
    visualizer(debug_data=debug_output, 
               image=images, 
               target=targets,
               output=(debug_output['prediction'], debug_output['classification']),
               epoch=epoch,
               visualise_predictions=predictions_df)
"""

from pathlib import Path
import typing
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
from torch import Tensor

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from loguru import logger


class DebugVisualizer:
    """Visualizer for debugging HerdNetDINOv2 intermediate features.

    Shows:
    - Input image
    - Final prediction
    - Fused features (PCA and energy)
    - Multi-scale backbone features
    - Attention maps
    """

    def __init__(self, output_path: Path):
        """
        Args:
            output_path: Directory to save debug visualizations
        """
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def __call__(
            self,
            debug_data: Dict,
            image: Tensor,
            target: Dict,
            output: Tuple[Tensor, Tensor],
            epoch: int,
            output_name: str = 'heatmap.png',
            visualise_predictions: pd.DataFrame | None = None
    ):
        """
        Main entry point for visualization.

        Args:
            debug_data: Debug dictionary from model forward pass
            image: Input image tensor [B, C, H, W] (normalized)
            target: Target dictionary containing 'points', 'labels', etc.
            output: Model output tuple (heatmap, classification)
            epoch: Current epoch number
            output_name: Base name for output file (not used in current implementation)
            visualise_predictions: DataFrame of predictions (optional)

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Get original image name if available
        if 'original_image_name' in target:
            img_name = target['original_image_name'][0]
            if isinstance(img_name, (list, tuple)):
                img_name = img_name[0]
        else:
            img_name = "image"

        save_path = f"debug_{img_name}_epoch{epoch}.png"

        fig = self.visualize_debug(
            debug_data,
            image,
            save_path=save_path
        )

        return fig

    def visualize_debug(
            self,
            debug_data: Dict,
            img_tensor: Tensor,
            save_path: str = "debug_layers.png"
    ):
        """
        Create comprehensive debug visualization showing all internal layers.

        Args:
            debug_data: Dictionary from model(x, debug=True) containing:
                - prediction: Final heatmap
                - fused: Fused features
                - backbone: Dict of multi-scale features
                - attention_heatmaps: Attention maps
                - bottleneck: Bottleneck features
            img_tensor: Input image [B, 3, H, W] normalized
            save_path: Output filename

        Returns:
            matplotlib.figure.Figure
        """
        # Extract data
        num_backbone_layers = len(debug_data['backbone'])

        # Calculate grid size
        # Row 1: Input | Prediction | Fused | Bottleneck | Backbone layers
        # Row 2: (empty) | (empty) | Fused Energy | Bottleneck Energy | Backbone energies
        # Row 3: (optional) Attention maps
        has_attention = 'attention_heatmaps' in debug_data and debug_data['attention_heatmaps']

        cols = 4 + num_backbone_layers  # Input, Pred, Fused, Bottleneck, + Layers
        rows = 3 if has_attention else 2

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

        # Handle case where axes might not be 2D
        if rows == 1:
            axes = axes.reshape(1, -1)

        # =====================================================================
        # Row 0: Main visualizations (PCA for feature maps)
        # =====================================================================

        # Column 0: Input Image
        input_img = img_tensor[0].permute(1, 2, 0).cpu().numpy()
        # Denormalize for visualization
        input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min() + 1e-8)
        axes[0, 0].imshow(input_img)
        axes[0, 0].set_title("Input", fontsize=10)
        axes[0, 0].axis('off')

        # Column 1: Prediction
        pred = debug_data['prediction'][0, 0].cpu().numpy()
        im = axes[0, 1].imshow(pred, cmap='jet')
        axes[0, 1].set_title(f"Prediction\n(max: {pred.max():.3f})", fontsize=10)
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

        # Column 2: Fused Features (PCA)
        fused = debug_data['fused'][0].cpu().numpy()  # [C, H, W]
        self.show_pca(fused, axes[0, 2])
        axes[0, 2].set_title(f"Fused Features (PCA)\nShape: {fused.shape}", fontsize=10)

        # Column 3: Bottleneck Features (PCA)
        bottleneck = debug_data['bottleneck'][0].cpu().numpy()  # [C, H, W]
        self.show_pca(bottleneck, axes[0, 3])
        axes[0, 3].set_title(f"Bottleneck (PCA)\nShape: {bottleneck.shape}", fontsize=10)

        # Columns 4+: Backbone Layers (PCA)
        for i, (name, feat_tensor) in enumerate(sorted(debug_data['backbone'].items())):
            feat = feat_tensor[0].cpu().numpy()  # [C, H, W]
            col_idx = i + 4

            self.show_pca(feat, axes[0, col_idx])
            axes[0, col_idx].set_title(f"{name} (PCA)\nShape: {feat.shape}", fontsize=10)

        # =====================================================================
        # Row 1: Energy/Activation maps
        # =====================================================================

        # Columns 0-1: Empty
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')

        # Column 2: Fused Energy
        fused_energy = np.linalg.norm(fused, axis=0)
        im = axes[1, 2].imshow(fused_energy, cmap='magma')
        axes[1, 2].set_title(f"Fused Energy\n(mean: {fused_energy.mean():.2f})", fontsize=10)
        axes[1, 2].axis('off')
        plt.colorbar(im, ax=axes[1, 2], fraction=0.046)

        # Column 3: Bottleneck Energy
        bottleneck_energy = np.linalg.norm(bottleneck, axis=0)
        im = axes[1, 3].imshow(bottleneck_energy, cmap='magma')
        axes[1, 3].set_title(f"Bottleneck Energy\n(mean: {bottleneck_energy.mean():.2f})", fontsize=10)
        axes[1, 3].axis('off')
        plt.colorbar(im, ax=axes[1, 3], fraction=0.046)

        # Columns 4+: Backbone Energy
        for i, (name, feat_tensor) in enumerate(sorted(debug_data['backbone'].items())):
            feat = feat_tensor[0].cpu().numpy()  # [C, H, W]
            col_idx = i + 4

            activation = np.linalg.norm(feat, axis=0)
            im = axes[1, col_idx].imshow(activation, cmap='magma')
            axes[1, col_idx].set_title(f"{name} Energy\n(mean: {activation.mean():.2f})", fontsize=10)
            axes[1, col_idx].axis('off')
            plt.colorbar(im, ax=axes[1, col_idx], fraction=0.046)

        # =====================================================================
        # Row 2: Attention maps (if available)
        # =====================================================================
        if has_attention:
            # Column 0-1: Empty
            axes[2, 0].axis('off')
            axes[2, 1].axis('off')

            # Columns 2+: Attention heatmaps
            attention_items = sorted(debug_data['attention_heatmaps'].items())
            for i, (name, attn_tensor) in enumerate(attention_items):
                if i + 2 < cols:
                    attn = attn_tensor[0, 0].cpu().numpy()  # [H, W]
                    im = axes[2, i + 2].imshow(attn, cmap='viridis')
                    axes[2, i + 2].set_title(f"Attention: {name}", fontsize=10)
                    axes[2, i + 2].axis('off')
                    plt.colorbar(im, ax=axes[2, i + 2], fraction=0.046)

            # Fill remaining with empty
            for col_idx in range(len(attention_items) + 2, cols):
                axes[2, col_idx].axis('off')

        # Add overall title with temperature info
        if 'temperature' in debug_data:
            fig.suptitle(f"HerdNetDINOv2 Debug Visualization (Temperature: {debug_data['temperature']:.2f})",
                         fontsize=14, y=0.995)

        plt.tight_layout()

        # Save
        save_full_path = self.output_path / save_path
        plt.savefig(save_full_path, dpi=150, bbox_inches='tight')
        logger.info(f"Debug visualization saved to {save_full_path}")

        # Log to wandb if available
        if WANDB_AVAILABLE:
            try:
                wandb.log({save_path: wandb.Image(fig)})
            except:
                pass  # wandb not initialized or error

        plt.close(fig)
        return fig

    def show_pca(self, feature_map: np.ndarray, ax: plt.Axes):
        """
        Project high-dimensional feature map to RGB using PCA.

        Args:
            feature_map: Feature map [C, H, W]
            ax: Matplotlib axis to plot on
        """
        C, H, W = feature_map.shape

        # Flatten spatial dimensions: [C, N] -> [N, C]
        flat_feat = feature_map.reshape(C, -1).transpose()

        # Handle edge case: too few samples for PCA
        if flat_feat.shape[0] < 3:
            # Just show grayscale mean
            mean_feat = feature_map.mean(axis=0)
            mean_feat = (mean_feat - mean_feat.min()) / (mean_feat.max() - mean_feat.min() + 1e-8)
            ax.imshow(mean_feat, cmap='gray')
            ax.axis('off')
            return

        # PCA to 3 components
        n_components = min(3, C, flat_feat.shape[0])
        pca = PCA(n_components=n_components)

        try:
            rgb = pca.fit_transform(flat_feat)  # [N, n_components]

            # If we got fewer than 3 components, pad with zeros
            if n_components < 3:
                rgb = np.pad(rgb, ((0, 0), (0, 3 - n_components)), mode='constant')

            # Normalize to 0-1
            for i in range(3):
                if rgb[:, i].max() > rgb[:, i].min():
                    rgb[:, i] = (rgb[:, i] - rgb[:, i].min()) / (rgb[:, i].max() - rgb[:, i].min())

            # Reshape back to image
            rgb_img = rgb.reshape(H, W, 3)
            ax.imshow(rgb_img)

        except Exception as e:
            # Fallback: show grayscale mean
            logger.warning(f"PCA failed: {e}, using mean instead")
            mean_feat = feature_map.mean(axis=0)
            mean_feat = (mean_feat - mean_feat.min()) / (mean_feat.max() - mean_feat.min() + 1e-8)
            ax.imshow(mean_feat, cmap='gray')

        ax.axis('off')


# =============================================================================
# Example Usage
# =============================================================================

# if __name__ == "__main__":
#     """
#     Example usage of the debug visualizer.
#     """
#     import torch
#
#     # Create dummy debug data (simulating model output)
#     B, C, H, W = 1, 256, 64, 64
#
#     debug_data = {
#         'prediction': torch.randn(B, 1, 128, 128).sigmoid(),
#         'classification': torch.randn(B, 2, 16, 16),
#         'fused': torch.randn(B, C, H, W),
#         'bottleneck': torch.randn(B, C, H, W),
#         'backbone': {
#             'scale_0': torch.randn(B, 256, H * 2, W * 2),
#             'scale_1': torch.randn(B, 512, H, W),
#             'scale_2': torch.randn(B, 1024, H // 2, W // 2),
#         },
#         'attention_heatmaps': {
#             'scale_0': torch.randn(B, 1, H * 2, W * 2).sigmoid(),
#             'scale_1': torch.randn(B, 1, H, W).sigmoid(),
#             'scale_2': torch.randn(B, 1, H // 2, W // 2).sigmoid(),
#         },
#         'temperature': 2.0,
#     }
#
#     # Create visualizer
#     visualizer = DebugVisualizer(output_path=Path("./debug_outputs"))
#
#     # Visualize
#     img = torch.randn(B, 3, 512, 512)
#     target = {'original_image_name': ['test_image']}
#
#     fig = visualizer(
#         debug_data=debug_data,
#         image=img,
#         target=target,
#         output=(debug_data['prediction'], debug_data['classification']),
#         epoch=0
#     )
#
#     print("Debug visualization created successfully!")

def visualize_sample(image: Tensor, target: Dict, output: typing.Tuple[Tensor, Tensor],):
    """
    Visualization function compatible with animaloc.vizual.plots interface.

    Args:
        image: Input image tensor [B, C, H, W] (normalized)
        target: Target dictionary containing 'points', 'labels'
        output: Model output (optional) - could be predictions, heatmaps, etc.

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    cls_map = output[1]
    obj_heatmap = output[0]

    image = image.squeeze(0)
    obj_heatmap = obj_heatmap.squeeze(0)
    cls_heatmap = cls_map.squeeze(0)

    fig, axes = plot_heatmaps(image, obj_heatmap,
                              class_names=None,
                              max_channels=2,
                              overlay_channel=0,
                              alpha=0.5,
                              show_argmax_overlay=True)

    return fig

def visualise_full_res_heatmap(
    image: Tensor,
    target: Dict[str, Any],
    output: typing.Tuple[Tensor, Tensor],

) -> Figure:
    """
    Visualizes a full resolution heatmap with the input image and target points.

    Args:
        image (Tensor): Input image tensor [B, C, H, W] (normalized)
        target (Dict[str, Any]): Target dictionary containing 'points', 'labels'
        output (Tuple[Tensor, Tensor]): Model output (heatmap, class map)
        output_name (str): Name of the output file
        output_path (str): Path to save the output file
    """


    cls_map = output[1]
    obj_heatmap = output[0]


    image_tensor = image.squeeze(0)
    obj_heatmap = obj_heatmap.squeeze(0)
    # TODO estimate the down_ratio from the model or dataset

    image_np = denormalize_image(image_tensor)
    heatmap_tensor = obj_heatmap.detach().cpu()

    H, W = image_tensor.shape[1], image_tensor.shape[2]
    HH, HW = obj_heatmap.detach().cpu().shape[1], obj_heatmap.detach().cpu().shape[2]

    dr_y = H / HH
    dr_x = W / HW

    heatmap_tensor = F.interpolate(heatmap_tensor.unsqueeze(0),
                                   size=(H, W), mode='bilinear', align_corners=False)[0]

    heatmap_np = heatmap_tensor.squeeze(0).cpu().numpy()

    aspect_ratio = W / H

    # Set a reasonable maximum size and scale appropriately
    max_size = 10  # Reduced from 20

    if aspect_ratio > 1:
        # Wide image - limit width, scale height
        fig_width = max_size
        fig_height = max_size / aspect_ratio
    else:
        # Tall image - limit height, scale width
        fig_height = max_size
        fig_width = max_size * aspect_ratio

    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))


    ax.imshow(image_np)
    ax.imshow(heatmap_np, cmap='jet', alpha=0.5)

    points = target["points"].squeeze(0).cpu().numpy()
    points = points * dr_x
    # plot these points
    for (x, y) in points:
        ax.plot(x, y, '+', markersize=16, markeredgewidth=2.0, markeredgecolor='w')
    # ax.set_title(f"{target['original_image_name'][0][0]} with Heatmap Overlay")
    ax.axis("off")

    plt.tight_layout(pad=0.2)
    # plt.show()
    return fig