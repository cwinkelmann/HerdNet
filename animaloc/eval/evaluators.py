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

import numpy as np
import pandas as pd
import torch
import pandas
import os
import numpy
try:
    import wandb
except ImportError:
    wandb = None
import matplotlib
from matplotlib import pyplot as plt
from typing import List, Dict, Any

from animaloc.vizual import Visualiser

matplotlib.use('Agg')

from typing import Any, Optional, Dict, List, Callable

import torch.nn.functional as F

from ..utils.logger import CustomLogger

from .stitchers import Stitcher
from .metrics import Metrics
from .lmds import HerdNetLMDS

from ..utils.registry import Registry
from loguru import logger as loguru_logger
EVALUATORS = Registry('evaluators', module_key='animaloc.eval.evaluators')

__all__ = ['EVALUATORS', *EVALUATORS.registry_names]

@EVALUATORS.register()
class Evaluator:
    ''' Base class for evaluators '''

    def __init__(
        self,
        model: torch.nn.Module, 
        dataloader: torch.utils.data.DataLoader,
        metrics: Metrics,
        device_name: str = 'cuda', 
        print_freq: int = 10,
        stitcher: Optional[Stitcher] = None,
        vizual_fn: Optional[Visualiser] = None,
        vizual_debug_fn: Optional[Visualiser] = None,
        work_dir: Optional[str] = None,
        header: Optional[str] = None
        ):
        '''
        Args:
            model (torch.nn.Module): CNN detection model to evaluate, that takes as 
                input tensor image and returns output and loss as tuple.
            dataloader (torch.utils.data.DataLoader): a pytorch's DataLoader that returns a tensor
                image and target.
            metrics (Metrics): Metrics instance used to compute model performances.
            device_name (str): the device name on which tensors will be allocated ('cpu' or  
                'cuda').
                Defaults to 'cuda'.
            print_freq (int, optional): define the frequency at which the logs will be
                printed and/or recorded. 
                Defaults to 10.
            stitcher (Stitcher, optional): optional Stitcher class instance to evaluate over
                large images. The specified dataloader should thus be composed of large images 
                for the use of this algorithm to make sense.
                Defaults to None.
            vizual_fn (callable, optional): a model specific function that will be use for plotting
                samples in Weights & Biases during validation. It must take 'image', 'target', and 
                'output' as arguments and return a matplotlib figure. Defaults to None.
            work_dir (str, optional): directory where logs (and results) will be saved. If
                None is given, results and logs files will be saved in current working 
                directory.
                Defaults to None.
            header (str, optional): string put at the beginning of the printed logs
                Defaults to None
        '''
        
        assert isinstance(model, torch.nn.Module), \
            'model argument must be an instance of nn.Module'
        
        assert isinstance(dataloader, torch.utils.data.DataLoader), \
            'dataset argument must be an instance of torch.utils.data.DataLoader'

        assert isinstance(metrics, Metrics), \
            'metrics argument must be an instance of Metrics'
        
        assert isinstance(stitcher, (type(None), Stitcher)), \
            'stitcher argument must be an instance of Stitcher class'
        
        assert callable(vizual_fn) or isinstance(vizual_fn, type(None)), \
            f'vizual_fn argument must be a callable function, got \'{type(vizual_fn)}\''

        assert callable(vizual_debug_fn) or isinstance(vizual_debug_fn, type(None)), \
            f'vizual_debug_fn argument must be a callable function, got \'{type(vizual_debug_fn)}\''
        
        self.model = model
        self.dataloader = dataloader
        self.metrics = metrics
        self.device = torch.device(device_name)
        self.print_freq = print_freq
        self.stitcher = stitcher
        self.vizual_fn = vizual_fn
        self.vizual_debug_fn = vizual_debug_fn
        self.current_epoch = None
        self.work_dir = work_dir
        if self.work_dir is None:
            self.work_dir = os.getcwd()

        self.header = header

        self._stored_metrics = None

        self.logs_filename = 'evaluation'

    def prepare_data(self, images: Any, targets: Any) -> tuple:
        ''' Method to prepare the data before feeding to the model. 
        Can be overriden by subclasses.

        Args:
            images (Any)
            targets (Any)
        
        Returns:
            tuple
        '''
        
        return images.to(self.device), targets.to(self.device)
    
    def prepare_feeding(self, targets: Any, output: Any) -> dict:
        ''' Method to prepare targets and output before feeding to the Metrics instance. 
        Can be overriden by subclasses.

        Args:
            targets (Any)
            output (Any)
        
        Returns:
            dict
        '''
        
        return dict(gt = targets, preds = output)
    
    def post_stitcher(self, output: torch.Tensor) -> Any:
        ''' Method to post-treat the output of the stitcher.
        Can be overriden by subclasses.

        Args:
            output (torch.Tensor): output of Stitcher call
        
        Returns:
            Any
        '''
        return output
    
    @torch.no_grad()
    def evaluate(self, returns: str = 'recall', wandb_flag: bool = False, viz: bool = False,
        log_meters: bool = True, dont_finish=False) -> float:
        ''' Evaluate the model
        
        Args:
            returns (str, optional): metric to be returned. Possible values are:
                'recall', 'precision', 'f1_score', 'f2_score' 'mse', 'mae', 'rmse', 'accuracy'
                and 'mAP'. Defauts to 'recall'
            wandb_flag (bool, optional): set to True to log on Weight & Biases. 
                Defaults to False.
            viz (bool, optional): set to True to save vizual predictions on original
                images. Defaults to False.
            log_meters (bool, optional): set to False to disable meters logging. 
                Defaults to True.
        
        Returns:
            float
        '''
        
        self.model.eval()

        self.metrics.flush()

        logger = CustomLogger(delimiter=' ', filename=self.logs_filename, work_dir=self.work_dir)
        iter_metrics = self.metrics.copy()

        for i, (images, targets) in enumerate(logger.log_every(self.dataloader, self.print_freq, self.header)):

            debug_output = None
            
            # loguru_logger.info(f'[{i}/{len(self.dataloader)}], {targets["image_name"]} ')
            images, targets = self.prepare_data(images, targets)
            if len(images) > 1:
                loguru_logger.warning(f"A batch size larger than 1 is not supported, got {len(images)} images in the batch.")
            
            # TODO if the image is just 512px the stitcher should not be used but is anyway
            if self.stitcher is not None:
                model_output = self.stitcher(images[0]) # remove batch dimension
                model_output = self.post_stitcher(model_output)
            else:
                # output, _ = self.model(images, targets)  
                model_output, _ = self.model(images)
                if self.vizual_debug_fn is not None:
                    debug_output = self.model.model( images, debug=True)

            # the model output is a list of 2 tensors, one heatmap one class map
            output_prediction = self.prepare_feeding(targets, model_output)

            if viz and self.vizual_fn is not None:
                if i % self.print_freq == 0 or i == len(self.dataloader) - 1:
                    self._vizual(image=images,
                                 target=targets,
                                 output=model_output,
                                 visualise_predictions=pd.DataFrame(output_prediction["preds"]))
                    if debug_output:
                        self._vizual_debug(debug_output=debug_output,
                                           image=images,
                                           target=targets,
                                           output=model_output,
                                           visualise_predictions=pd.DataFrame(output_prediction["preds"]))

            # for each image feed outputs and aggregate metrics, should look like
            """
            {'est_count': [7, 0, 0, 0, 0, 0, 0], 
            'gt': {'labels': [1], 
            'loc': [[1596.0, 1747.0]]}, 
            'preds': {'dscores': [0.19435586035251617, 0.27331411838531494, 0.18535958230495453, 0.23847505450248718, 0.30953675508499146, 0.2966581881046295, 0.31906700134277344], 
            'labels': [1, 1, 1, 1, 1, 1, 1], 
            'loc': [[9.0, 1181.0], [174.0, 187.0], [182.0, 1025.0], [423.0, 1246.0], [581.0, 840.0], [1007.0, 54.0], [1593.0, 1744.0]], 
            'scores': [0.8123772740364075, 0.970843493938446, 0.9492995738983154, 0.9641066193580627, 0.9905760288238525, 0.9475813508033752, 0.9999991655349731]}}
            """


            iter_metrics.feed(**output_prediction)
            iter_metrics.aggregate()

            # print("\n" + iter_metrics.density_report())
            # print("\n" + iter_metrics.occupancy_report())
            #
            # # Show DataFrame
            # print("\nDensity DataFrame:")
            # print(iter_metrics.density_dataframe().to_string(index=False))
            #
            # print("\nTile Results (first 5):")
            # print(iter_metrics.tile_results_dataframe().head().to_string(index=False))

            if log_meters:
                logger.add_meter('n', sum(iter_metrics.tp) + sum(iter_metrics.fn) + sum(iter_metrics.fp))
                logger.add_meter('tp', sum(iter_metrics.tp))
                logger.add_meter('fp', sum(iter_metrics.fp))
                logger.add_meter('fn', sum(iter_metrics.fn))
                logger.add_meter('recall', round(iter_metrics.recall(), 2))
                logger.add_meter('precision', round(iter_metrics.precision(), 2))
                logger.add_meter('f1_score', round(iter_metrics.fbeta_score(), 2))
                logger.add_meter('f2_score', round(iter_metrics.fbeta_score(beta=2), 2))
                logger.add_meter('f5_score', round(iter_metrics.fbeta_score(beta=5), 2))
                logger.add_meter('MAE', round(iter_metrics.mae(), 2))
                logger.add_meter('ME', round(iter_metrics.me(), 2))
                logger.add_meter('MSE', round(iter_metrics.mse(), 2))
                logger.add_meter('RMSE', round(iter_metrics.rmse(), 2))
                logger.add_meter('avg_score', round(iter_metrics.avg_score(), 2))
                logger.add_meter('avg_dscore', round(iter_metrics.avg_dscore(), 3))

            if wandb_flag:
                wandb.log({
                    'n': sum(iter_metrics.tp) + sum(iter_metrics.fn) + sum(iter_metrics.fp),
                    'tp': sum(iter_metrics.tp),
                    'fp': sum(iter_metrics.fp),
                    'fn': sum(iter_metrics.fn),
                    'recall': iter_metrics.recall(),
                    'precision': iter_metrics.precision(),
                    'f1_score': iter_metrics.fbeta_score(),
                    'f2_score': iter_metrics.fbeta_score(beta=2),
                    'f5_score': iter_metrics.fbeta_score(beta=5),
                    'MAE': iter_metrics.mae(),
                    'ME': iter_metrics.me(),
                    'MSE': iter_metrics.mse(),
                    'RMSE': iter_metrics.rmse(),
                    'avg_score': iter_metrics.avg_score(),
                    'avg_dscore': iter_metrics.avg_dscore(),

                    })

            iter_metrics.flush()

            self.metrics.feed(**output_prediction)
        
        self._stored_metrics = self.metrics.copy()

        mAP = numpy.mean([self.metrics.ap(c) for c in range(1, self.metrics.num_classes)]).item()
        
        self.metrics.aggregate()

        if wandb_flag:
            wandb.run.summary['recall'] =  self.metrics.recall()
            wandb.run.summary['precision'] =  self.metrics.precision()
            wandb.run.summary['f1_score'] =  self.metrics.fbeta_score()
            wandb.run.summary['f2_score'] =  self.metrics.fbeta_score(beta=2)
            wandb.run.summary['f5_score'] =  self.metrics.fbeta_score(beta=5)
            wandb.run.summary['MAE'] =  self.metrics.mae()
            wandb.run.summary['ME'] =  iter_metrics.me()

            wandb.run.summary['MSE'] =  self.metrics.mse()
            wandb.run.summary['RMSE'] =  self.metrics.rmse()
            wandb.run.summary['accuracy'] =  self.metrics.accuracy()
            wandb.run.summary['mAP'] =  mAP
            wandb.run.summary['tp'] =  sum(self.metrics.tp)
            wandb.run.summary['fn'] =  sum(self.metrics.fn)
            wandb.run.summary['fp'] =  sum(self.metrics.fp)
            wandb.run.summary['n'] =  sum(self.metrics.fp) +  sum(self.metrics.fn) + sum(self.metrics.tp)
            wandb.run.summary['avg_score'] =  self.metrics.avg_score()
            wandb.run.summary['avg_dscore'] =  self.metrics.avg_dscore()



            print(f"Wandb summary: {wandb.run.summary}")

            if dont_finish:
                loguru_logger.info("wandb.run.finish() has been disabled")
            else:
                wandb.run.finish()

        if returns == 'recall':
            return self.metrics.recall()
        elif returns == 'precision':
            return self.metrics.precision()
        elif returns == 'f1_score':
            return self.metrics.fbeta_score()
        elif returns == 'f2_score':
            return self.metrics.fbeta_score(beta=2)
        elif returns == 'f5_score':
            return self.metrics.fbeta_score(beta=5)
        elif returns == 'mse':
            return self.metrics.mse()
        elif returns == 'mse':
            return self.metrics.me()
        elif returns == 'mae':
            return self.metrics.mae()
        elif returns == 'me':
            return self.metrics.me()
        elif returns == 'rmse':
            return self.metrics.rmse()
        elif returns == 'accuracy':
            return self.metrics.accuracy()
        elif returns == 'mAP':
            return mAP
        else:
            raise ValueError(f'Unknown return value: {returns}. Possible values are: '
                             '\'recall\', \'precision\', \'f1_score\', \'f2_score\', '
                             '\'f5_score\', \'mse\', \'mae\',\'me\', \'rmse\', \'accuracy\' and \'mAP\'.')
    
    @property
    def results(self) -> pandas.DataFrame:
        ''' Returns metrics by class (recall, precision, f1_score, mse, mae, and rmse) 
        in a pandas dataframe '''
        
        assert self._stored_metrics is not None, \
            'No metrics have been stored, please use the evaluate method first.'
        
        metrics_cpy = self._stored_metrics.copy()
        
        res = []
        for c in range(1, metrics_cpy.num_classes):
            metrics = {
                'class': str(c),
                'n': metrics_cpy.tp[c-1] + metrics_cpy.fn[c-1],
                'recall': metrics_cpy.recall(c),
                'precision': metrics_cpy.precision(c),
                'f1_score': metrics_cpy.fbeta_score(c),
                'confusion': metrics_cpy.confusion(c), 
                'mae': metrics_cpy.mae(c),
                'me': metrics_cpy.me(c),
                'mse': metrics_cpy.mse(c),
                'rmse': metrics_cpy.rmse(c),
                'ap': metrics_cpy.ap(c),
            }
            res.append(metrics)
        
        metrics_cpy.aggregate()
        res.append({
            'class': 'binary',
            'n': metrics_cpy.tp[0] + metrics_cpy.fn[0],
            'recall': metrics_cpy.recall(),
            'precision': metrics_cpy.precision(),
            'f1_score': metrics_cpy.fbeta_score(),
            'confusion': metrics_cpy.confusion(),
            'mae': metrics_cpy.mae(),
            'me': metrics_cpy.me(),
            'mse': metrics_cpy.mse(),
            'rmse': metrics_cpy.rmse(),
            'ap': metrics_cpy.ap()
        })

        return pandas.DataFrame(data = res)
    
    @property
    def detections(self) -> pandas.DataFrame:
        ''' Returns detections (image id, location, label and score) in a pandas
        dataframe '''
        
        assert self._stored_metrics is not None, \
            'No detections have been stored, please use the evaluate method first.'

        img_names = self.dataloader.dataset._img_names
        dets = self._stored_metrics.detections
        for det in dets:
            det['images'] = img_names[det['images']]

        return pandas.DataFrame(data = dets)
    
    def _vizual(self, image: Any, target: Any, output: Any, visualise_predictions: pd.DataFrame = None) -> None:
        fig = self.vizual_fn(image=image,
                             target=target,
                             output=output,
                             epoch=self.current_epoch,
                             visualise_predictions=visualise_predictions)

        return fig

    def _vizual_debug(self, debug_output, image: Any, target: Any, output: Any,
                      visualise_predictions: pd.DataFrame = None) -> None:
        debug_fig = self.vizual_debug_fn(debug_data=debug_output,
                                   image=image,
                                   target=target,
                                   output=output,
                                   epoch=self.current_epoch,
                                   visualise_predictions=visualise_predictions)

        return debug_fig



@EVALUATORS.register()
class HerdNetEvaluator(Evaluator):

    def __init__(self, model: torch.nn.Module,
                 dataloader: torch.utils.data.DataLoader,
                 metrics: Metrics,
                 lmds_kwargs: dict = {'kernel_size': (3, 3)},
                 device_name: str = 'cuda',
                 print_freq: int = 10,
                 stitcher: Optional[Stitcher] = None,
                 vizual_fn: Optional[Callable] = None,
                 vizual_debug_fn: Optional[Callable] = None,
                 work_dir: Optional[str] = None,
                 header: Optional[str] = None
        ) -> None:
        super().__init__(model,
                         dataloader,
                         metrics,
                         device_name=device_name,
                         print_freq=print_freq,
                        vizual_fn=vizual_fn,
                         stitcher=stitcher,
                         work_dir=work_dir,
                         header=header,

                        vizual_debug_fn=vizual_debug_fn,

                         )

        self.lmds_kwargs = lmds_kwargs

    def prepare_data(self, images: Any, targets: Any) -> tuple:        
        return images.to(self.device), targets
    
    def post_stitcher(self, output: torch.Tensor) -> Any:
        heatmap = output[:,:1,:,:]
        clsmap = output[:,1:,:,:]
        return heatmap, clsmap

    def prepare_feeding(self, targets: Dict[str, torch.Tensor], output: List[torch.Tensor]) -> dict:

        gt_coords = [p[::-1] for p in targets['points'].squeeze(0).tolist()]
        gt_labels = targets['labels'].squeeze(0).tolist()
        
        gt = dict(
            loc = gt_coords,
            labels = gt_labels
        )

        up = True
        if self.stitcher is not None:
            up = False


        # TODO I still don't understand why the up parameter is set differently depending on the stitcher
        if "up" in self.lmds_kwargs.keys():
            lmds = HerdNetLMDS(**self.lmds_kwargs)
        elif self.stitcher is not None:
            lmds = HerdNetLMDS(up=False, **self.lmds_kwargs)
        else:
            lmds = HerdNetLMDS(up=True, **self.lmds_kwargs)

        #
        # lmds = HerdNetLMDS(**self.lmds_kwargs)
        counts, locs, labels, scores, dscores = lmds(output)
        
        preds = dict(
            loc = locs[0],
            labels = labels[0],
            scores = scores[0],
            dscores = dscores[0]
        )
        
        return dict(gt = gt, preds = preds, est_count = counts[0])

@EVALUATORS.register()
class DensityMapEvaluator(Evaluator):
  
    def prepare_data(self, images: Any, targets: Any) -> tuple:        
        return images.to(self.device), targets

    def prepare_feeding(self, targets: Dict[str, torch.Tensor], output: torch.Tensor) -> dict:

        gt_coords = [p[::-1] for p in targets['points'].squeeze(0).tolist()]
        gt_labels = targets['labels'].squeeze(0).tolist()
        
        gt = dict(loc = gt_coords, labels = gt_labels)
        preds = dict(loc = [], labels = [], scores = [])

        _, idx = torch.max(output, dim=1)
        masks = F.one_hot(idx, num_classes=output.shape[1]).permute(0,3,1,2)
        output = (output * masks)
        est_counts = output[0].sum(2).sum(1).tolist()
        
        return dict(gt = gt, preds = preds, est_count = est_counts)

@EVALUATORS.register()
class FasterRCNNEvaluator(Evaluator):

    def prepare_data(self, images: List[torch.Tensor], targets: List[dict]) -> tuple:
        images = list(image.to(self.device) for image in images)    
        targets = [{k: v.to(self.device) for k, v in t.items() if torch.is_tensor(v)} 
                            for t in targets]
        return images, targets
    
    def post_stitcher(self, output: dict) -> list:
        return [output]
    
    def prepare_feeding(self, targets: List[dict], output: List[dict]) -> dict:

        targets, output = targets[0], output[0]

        gt = dict(
            loc = targets['boxes'].tolist(),
            labels = targets['labels'].tolist()
            )
        
        preds = dict(
            loc = output['boxes'].tolist(),
            labels = output['labels'].tolist(),
            scores = output['scores'].tolist()
            )

        num_classes = self.metrics.num_classes - 1
        counts = [preds['labels'].count(i+1) for i in range(num_classes)]

        return dict(gt = gt, preds = preds, est_count = counts)





@EVALUATORS.register()
class P2PNetEvaluator(Evaluator):
    """
    Evaluator specialized for Point-to-Point (P2P) Networks.

    Handles both output formats:
    - Dense grid: logits [B, C, H, W], points [B, H*W, 2]
    - Sparse query: logits [B, N, C], points [B, N, 2]
    """

    def __init__(
            self,
            model: torch.nn.Module,
            dataloader: torch.utils.data.DataLoader,
            metrics: Metrics,
            device_name: str = 'cuda',
            print_freq: int = 10,
            stitcher: Optional[Stitcher] = None,
            vizual_fn: Optional[Callable] = None,
            vizual_debug_fn: Optional[Callable] = None,
            work_dir: Optional[str] = None,
            header: Optional[str] = None,
            confidence_threshold: float = 0.1,
    ) -> None:
        super().__init__(
            model,
            dataloader,
            metrics,
            device_name=device_name,
            print_freq=print_freq,
            vizual_fn=vizual_fn,
            stitcher=stitcher,
            work_dir=work_dir,
            header=header,
            vizual_debug_fn=vizual_debug_fn,
        )
        self.confidence_threshold = confidence_threshold

    def prepare_data(self, images: torch.Tensor, targets: List[dict]) -> tuple:
        """Move images and target tensors to device."""
        images = images.to(self.device)

        targets_on_device = []
        for t in targets:
            t_on_device = {}
            for k, v in t.items():
                if torch.is_tensor(v):
                    t_on_device[k] = v.to(self.device)
                else:
                    t_on_device[k] = v
            targets_on_device.append(t_on_device)

        return images, targets_on_device

    def _get_tensor(self, output: Dict, *keys) -> Optional[torch.Tensor]:
        """Safely get tensor from dict, trying multiple keys."""
        for key in keys:
            if key in output and output[key] is not None:
                return output[key]
        return None

    def _detect_format(self, output: Dict) -> str:
        """Detect whether output is from dense or sparse model."""
        logits = self._get_tensor(output, 'logits', 'pred_logits')

        if logits is None:
            raise ValueError("No 'logits' or 'pred_logits' in output")

        if logits.dim() == 4:
            return 'dense'  # [B, C, H, W]
        elif logits.dim() == 3:
            return 'sparse'  # [B, N, C]
        else:
            raise ValueError(f"Unexpected logits shape: {logits.shape}")

    def _extract_predictions(self, output: Dict) -> tuple:
        """
        Extract predictions from model output, handling both formats.

        Returns:
            pred_points: [N, 2] tensor of (x, y) pixel coordinates
            pred_scores: [N] tensor of confidence scores
            pred_labels: [N] tensor of class labels
        """
        fmt = self._detect_format(output)

        logits = self._get_tensor(output, 'logits', 'pred_logits')
        points = self._get_tensor(output, 'points', 'pred_points')

        if fmt == 'dense':
            # Dense: logits [B, C, H, W], points [B, H*W, 2]
            B, C, H, W = logits.shape

            # Flatten logits: [B, C, H, W] -> [B, H*W, C]
            logits_flat = logits.flatten(2).transpose(1, 2)  # [B, H*W, C]

            # Get probabilities and scores
            probs = F.softmax(logits_flat, dim=-1)  # [B, H*W, C]

            # For binary case: score is P(foreground)
            # For multiclass: score is max P(any foreground class)
            if C == 2:
                scores = probs[0, :, 1]  # [H*W]
                labels = torch.ones(H * W, dtype=torch.long, device=logits.device)
            else:
                # Multiclass: take max over non-background classes
                fg_probs = probs[0, :, 1:]  # [H*W, C-1]
                scores, class_idx = fg_probs.max(dim=-1)  # [H*W]
                labels = class_idx + 1  # Shift to account for background class

            # Points should be [B, H*W, 2]
            pred_points = points[0]  # [H*W, 2]

        else:
            # Sparse: logits [B, N, C], points [B, N, 2]
            B, N, C = logits.shape

            probs = F.softmax(logits, dim=-1)  # [B, N, C]

            if C == 2:
                scores = probs[0, :, 1]  # [N]
                labels = torch.ones(N, dtype=torch.long, device=logits.device)
            else:
                fg_probs = probs[0, :, 1:]  # [N, C-1]
                scores, class_idx = fg_probs.max(dim=-1)  # [N]
                labels = class_idx + 1

            pred_points = points[0]  # [N, 2]

        return pred_points, scores, labels

    """
    DIAGNOSTIC prepare_feeding for P2PNetEvaluator

    Replace your prepare_feeding method with this one to see what's happening.
    """

    import torch
    import torch.nn.functional as F
    import numpy as np
    from typing import List, Any

    def prepare_feeding(self, targets: List[dict], output: Any) -> dict:
        """
        Diagnostic version with extensive logging to identify issues.
        """

        def _get(d, *keys):
            for k in keys:
                if k in d and d[k] is not None:
                    return d[k]
            return None

        output_dict = output[0] if isinstance(output, list) else output
        targets_dict = targets[0]

        # ==================== GROUND TRUTH ====================
        gt_points = targets_dict.get('points', torch.empty((0, 2))).cpu()
        gt_labels = targets_dict.get('labels', torch.empty((0,)).long()).cpu().tolist()

        gt_coords = gt_points.tolist()  # Keep as-is, already (x, y)


        print(f"\n{'=' * 60}")
        print(f"DIAGNOSTIC: prepare_feeding")
        print(f"{'=' * 60}")
        print(f"GT points (first 3): {gt_coords[:3]}")
        print(f"GT labels: {gt_labels[:5]}...")
        print(f"Total GT: {len(gt_coords)}")

        # ==================== PREDICTIONS ====================
        logits = _get(output_dict, 'logits', 'pred_logits')
        points = _get(output_dict, 'points', 'pred_points')

        if logits is None or points is None:
            print(f"ERROR: logits is None: {logits is None}, points is None: {points is None}")
            print(f"Available keys: {output_dict.keys()}")
            return dict(
                gt=dict(loc=gt_coords, labels=gt_labels),
                preds=dict(loc=[], labels=[], scores=[]),
                est_count=[0]
            )

        print(f"\nLogits shape: {logits.shape}")
        print(f"Points shape: {points.shape}")

        # ==================== EXTRACT SCORES ====================
        if logits.dim() == 4:  # Dense: [B, C, H, W]
            B, C, H, W = logits.shape
            logits_flat = logits.flatten(2).transpose(1, 2)
            probs = F.softmax(logits_flat, dim=-1)
            scores = probs[0, :, 1] if C == 2 else probs[0, :, 1:].max(dim=-1)[0]
            pred_points = points[0]
            print(f"Format: DENSE, grid {H}x{W} = {H * W} positions")
        else:  # Sparse: [B, N, C]
            B, N, C = logits.shape
            probs = F.softmax(logits, dim=-1)
            scores = probs[0, :, 1] if C == 2 else probs[0, :, 1:].max(dim=-1)[0]
            pred_points = points[0]
            print(f"Format: SPARSE, {N} queries")

        pred_points = pred_points.cpu()
        scores = scores.cpu()

        # ==================== SCORE STATISTICS ====================
        print(f"\n--- Score Statistics ---")
        print(f"  Min: {scores.min():.4f}")
        print(f"  Max: {scores.max():.4f}")
        print(f"  Mean: {scores.mean():.4f}")
        print(f"  Std: {scores.std():.4f}")
        print(f"  Above 0.5: {(scores > 0.5).sum().item()}")
        print(f"  Above 0.3: {(scores > 0.3).sum().item()}")
        print(f"  Above 0.1: {(scores > 0.1).sum().item()}")
        print(f"  Above 0.05: {(scores > 0.05).sum().item()}")

        # ==================== COORDINATE STATISTICS ====================
        print(f"\n--- Coordinate Statistics ---")
        print(f"  Pred points range X: [{pred_points[:, 0].min():.1f}, {pred_points[:, 0].max():.1f}]")
        print(f"  Pred points range Y: [{pred_points[:, 1].min():.1f}, {pred_points[:, 1].max():.1f}]")

        if len(gt_coords) > 0:
            gt_arr = np.array(gt_coords)
            print(f"  GT points range X: [{gt_arr[:, 0].min():.1f}, {gt_arr[:, 0].max():.1f}]")
            print(f"  GT points range Y: [{gt_arr[:, 1].min():.1f}, {gt_arr[:, 1].max():.1f}]")

        # ==================== DISTANCE TO GT ====================
        if len(gt_coords) > 0:
            print(f"\n--- Distance to GT (for top-10 scoring predictions) ---")
            top_k = min(10, len(scores))
            top_scores, top_idx = scores.topk(top_k)
            top_points = pred_points[top_idx].numpy()
            gt_arr = np.array(gt_coords)

            for i, (idx, score, pt) in enumerate(zip(top_idx, top_scores, top_points)):
                # Find nearest GT
                dists = np.sqrt(((gt_arr - pt) ** 2).sum(axis=1))
                nearest_dist = dists.min()
                print(f"  Pred {i}: score={score:.4f}, coord=({pt[0]:.1f}, {pt[1]:.1f}), "
                      f"nearest GT dist={nearest_dist:.1f}px")

        # ==================== APPLY THRESHOLD ====================
        threshold = getattr(self, 'confidence_threshold', 0.1)
        mask = scores >= threshold
        n_pass = mask.sum().item()

        print(f"\n--- Thresholding (threshold={threshold}) ---")
        print(f"  Predictions passing: {n_pass}")

        pred_points_filtered = pred_points[mask].tolist()
        pred_scores_filtered = scores[mask].tolist()
        pred_points = pred_points.cpu()
        scores = scores.cpu()

        # ADD THIS DIAGNOSTIC:
        print(f"\n=== COORDINATE DIAGNOSTIC ===")
        print(f"Scores > 0.1: {(scores > 0.1).sum().item()} / {len(scores)}")
        print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        print(f"Pred coords mean: ({pred_points[:, 0].mean():.1f}, {pred_points[:, 1].mean():.1f})")
        print(f"Pred coords std: ({pred_points[:, 0].std():.1f}, {pred_points[:, 1].std():.1f})")
        print(f"Pred coords range X: [{pred_points[:, 0].min():.1f}, {pred_points[:, 0].max():.1f}]")
        print(f"Pred coords range Y: [{pred_points[:, 1].min():.1f}, {pred_points[:, 1].max():.1f}]")
        print(f"First 5 pred coords: {pred_points[:5].tolist()}")
        print(f"GT coords (first 5): {gt_coords[:5]}")
        print(f"PointsMetrics radius: {self.metrics.threshold}")



        # ==================== CHECK MATCHING RADIUS ====================
        # The PointsMetrics uses a radius threshold for matching
        # If predictions are further than this radius from GT, they won't match!
        if hasattr(self, 'metrics') and hasattr(self.metrics, 'threshold'):
            matching_radius = self.metrics.threshold
            print(f"\n--- Matching Radius Check ---")
            print(f"  PointsMetrics radius: {matching_radius}")

            if n_pass > 0 and len(gt_coords) > 0:
                pred_arr = np.array(pred_points_filtered)
                gt_arr = np.array(gt_coords)

                # For each prediction, check if ANY GT is within radius
                n_within_radius = 0
                for pt in pred_arr:
                    dists = np.sqrt(((gt_arr - pt) ** 2).sum(axis=1))
                    if dists.min() <= matching_radius:
                        n_within_radius += 1

                print(f"  Predictions within matching radius of ANY GT: {n_within_radius}/{n_pass}")
                if n_within_radius == 0:
                    print(f"  ⚠️  NO predictions are close enough to GT to be matched!")
                    print(f"  This is why recall/precision are 0!")

        print(f"{'=' * 60}\n")

        # ==================== RETURN ====================
        num_fg_classes = self.metrics.num_classes - 1 if hasattr(self, 'metrics') else 1
        est_count = None  # Simple count for single class

        return dict(
            gt=dict(loc=gt_coords, labels=gt_labels),
            preds=dict(
                loc=pred_points_filtered,
                labels=[1] * len(pred_scores_filtered),
                scores=pred_scores_filtered,
            ),
            est_count=est_count
        )
    def post_stitcher(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Handle output from stitcher (tiled inference)."""
        # Stitcher should return merged predictions in same format
        return output

    @torch.no_grad()
    def evaluate(
            self,
            returns: str = 'recall',
            wandb_flag: bool = False,
            viz: bool = False,
            log_meters: bool = True,
            dont_finish: bool = False,
    ) -> float:
        """Evaluate the P2PNet model."""

        self.model.eval()
        self.metrics.flush()

        logger = CustomLogger(
            delimiter=' ',
            filename=self.logs_filename,
            work_dir=self.work_dir
        )
        iter_metrics = self.metrics.copy()

        for i, (images, targets) in enumerate(
                logger.log_every(self.dataloader, self.print_freq, self.header)
        ):
            # 1. Prepare data
            images, targets = self.prepare_data(images, targets)

            # 2. Forward pass
            if self.stitcher is not None:
                model_output = self.stitcher(images[0])
                model_output = self.post_stitcher(model_output)
            else:
                model_output = self.model(images, targets)

            # 3. Prepare for metrics
            output_prediction = self.prepare_feeding(targets, model_output)

            # 4. Visualization
            if viz and self.vizual_fn is not None:
                if i % self.print_freq == 0 or i == len(self.dataloader) - 1:
                    self._vizual(
                        image=images,
                        target=targets,
                        output=model_output,
                        visualise_predictions=pd.DataFrame(output_prediction["preds"])
                    )

            # 5. Update metrics
            iter_metrics.feed(**output_prediction)
            iter_metrics.aggregate()

            if log_meters:
                logger.add_meter('n', sum(iter_metrics.tp) + sum(iter_metrics.fn) + sum(iter_metrics.fp))
                logger.add_meter('tp', sum(iter_metrics.tp))
                logger.add_meter('fp', sum(iter_metrics.fp))
                logger.add_meter('fn', sum(iter_metrics.fn))
                logger.add_meter('recall', round(iter_metrics.recall(), 2))
                logger.add_meter('precision', round(iter_metrics.precision(), 2))
                logger.add_meter('f1_score', round(iter_metrics.fbeta_score(), 2))
                logger.add_meter('f2_score', round(iter_metrics.fbeta_score(beta=2), 2))
                logger.add_meter('MAE', round(iter_metrics.mae(), 2))
                logger.add_meter('RMSE', round(iter_metrics.rmse(), 2))

            if wandb_flag:
                wandb.log({
                    'n': sum(iter_metrics.tp) + sum(iter_metrics.fn) + sum(iter_metrics.fp),
                    'tp': sum(iter_metrics.tp),
                    'fp': sum(iter_metrics.fp),
                    'fn': sum(iter_metrics.fn),
                    'recall': iter_metrics.recall(),
                    'precision': iter_metrics.precision(),
                    'f1_score': iter_metrics.fbeta_score(),
                    'MAE': iter_metrics.mae(),
                    'RMSE': iter_metrics.rmse(),
                })

            iter_metrics.flush()
            self.metrics.feed(**output_prediction)

        # Final aggregation
        self._stored_metrics = self.metrics.copy()
        mAP = np.mean([
            self.metrics.ap(c) for c in range(1, self.metrics.num_classes)
        ]).item()
        self.metrics.aggregate()

        # Log final results
        if wandb_flag:
            wandb.run.summary['recall'] = self.metrics.recall()
            wandb.run.summary['precision'] = self.metrics.precision()
            wandb.run.summary['f1_score'] = self.metrics.fbeta_score()
            wandb.run.summary['MAE'] = self.metrics.mae()
            wandb.run.summary['RMSE'] = self.metrics.rmse()
            wandb.run.summary['mAP'] = mAP
            wandb.run.summary['tp'] = sum(self.metrics.tp)
            wandb.run.summary['fn'] = sum(self.metrics.fn)
            wandb.run.summary['fp'] = sum(self.metrics.fp)

            if not dont_finish:
                wandb.run.finish()

        # Return requested metric
        metric_map = {
            'recall': self.metrics.recall,
            'precision': self.metrics.precision,
            'f1_score': self.metrics.fbeta_score,
            'f2_score': lambda: self.metrics.fbeta_score(beta=2),
            'mae': self.metrics.mae,
            'mse': self.metrics.mse,
            'rmse': self.metrics.rmse,
            'accuracy': self.metrics.accuracy,
            'mAP': lambda: mAP,
        }

        if returns in metric_map:
            return metric_map[returns]()
        else:
            raise ValueError(f"Unknown return metric: {returns}")