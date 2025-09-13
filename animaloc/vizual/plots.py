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

import typing
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import random
import itertools

from typing import Optional, Dict

import wandb
from matplotlib.figure import Figure
from torch import Tensor


from animaloc.vizual.custom_vis import plot_heatmaps, denormalize_image

__all__ = ['PlotPrecisionRecall', 'Visualiser', 'HeatMapVisualizer', 'visualize_sample']

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


import matplotlib.pyplot as plt
import torch
import numpy as np
from matplotlib.patches import Circle
from typing import Any
from torchvision.transforms import ToPILImage

class Visualiser:
    def __init__(self, output_path: str):
        self.output_path = output_path
        Path(self.output_path).mkdir(parents=True, exist_ok=True)

class HeatMapVisualizer(Visualiser):
    def __init__(self, output_path):
        super().__init__(output_path)

    def __call__(self, image: Tensor, target: Dict,
                 output: typing.Tuple[Tensor, Tensor],
                 epoch: int,
                 output_name: str = 'heatmap.png',
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


        output_name = f"heatmap_{target['original_image_name'][0][0]}_{epoch}.png"
        heatmap_fig = visualise_full_res_heatmap(image,
                                                 target,
                                                 output
                                   )
        wandb.log({output_name: wandb.Image(heatmap_fig)})
        heatmap_fig.savefig(Path(self.output_path) / output_name)
        plt.close(heatmap_fig)
        return fig

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
    max_size = 6  # Reduced from 20

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
    ax.set_title(f"{target['original_image_name'][0][0]} with Heatmap Overlay")
    ax.axis("off")

    plt.tight_layout(pad=0.2)

    return fig