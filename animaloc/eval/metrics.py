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

from collections import defaultdict
from dataclasses import dataclass

import math
import copy

import numpy as np
import pandas as pd
import sklearn.neighbors
import numpy
from loguru import logger

from sklearn.metrics import confusion_matrix
from itertools import tee
from typing import Optional, List, Dict

from ..data import BoundingBox
from .utils import bboxes_iou

from ..utils.registry import Registry

METRICS = Registry('metrics', module_key='animaloc.eval.metrics')

__all__ = ['METRICS', *METRICS.registry_names]

@METRICS.register()
class Metrics:
    '''
    Class to accumulate classification, detection and counting metrics, i.e.: 

        - Precision
        - Recall
        - F-beta score
        - Mean Absolute Error (MAE)
        - Mean Squared Error (MSE)
        - Root Mean Squared Error (RMSE)
        - Average Precision (AP)
        - Interclass confusion
        - Classification accuracy

    for binary or multiclass model.

    First, instanciate with a data matching threshold (e.g. IoU, radius),
    then feed the object with ground truth and predictions. You can retrieve 
    any metric at any time by calling the corresponding attribute.
    '''
    
    def __init__(self, threshold: float, num_classes: int = 2) -> None:
        '''
        Args:
            threshold (float): data matching threshold
            num_classes (int, optional): number of classes, background included. 
                Defaults to 2 (binary case).
        '''
        
        self.threshold = threshold
        self.num_classes = num_classes

        self.detections = []
        self.predictions = []
        self.idx = 0

        self.tp = self._init_attr()
        self.fp = self._init_attr()
        self.fn = self._init_attr()
    
        self._sum_absolute_error = self._init_attr()
        self._sum_squared_error = self._init_attr()
        self._sum_error = self._init_attr()
        self._n_calls = self._init_attr()
        self._agg_sum_absolute_error = 0
        self._agg_sum_squared_error = 0
        self._agg_sum_error = 0
        self._total_calls = 0
        self._total_count = self._init_attr()

        self._ap_tables = self._init_attr(val=[])
        
        self.confusion_matrix = numpy.zeros((self.num_classes-1,self.num_classes-1))
        self._confusion_matrix = self.confusion_matrix

    def feed(self, gt: dict, preds: dict, est_count: Optional[list] = None) -> None:
        '''
        Feed the object with ground truth and predictions and returns
        specified metrics optionally.

        Args:
            gt (dict): ground truth containing a dict with 'loc' and 'labels' 
                keys and list as values.
            preds (dict): predictions containing a dict with 'loc' and 'labels' 
                keys and list as values. Can contain an optional 'scores' key. 
            est_count (list, optional): list containing estimated count for each
                class, background excluded. Defaults to None.
        '''

        assert isinstance(est_count, (type(None), list))
        for o in [gt, preds]:
            assert isinstance(o, dict)
            assert len(o['loc']) == len(o['labels'])
        
        self.score_flag = 0
        if 'scores' in preds.keys():
            self.score_flag = 1
            assert len(preds['scores']) == len(preds['loc'])
        
        if len(gt['loc']) == 0:
            self._no_gt(gt, preds)
        
        if len(preds['loc']) == 0:
            self._no_preds(gt, preds)
        
        if len(gt['loc']) > 0 and len(preds['loc']) > 0:
            self.matching(gt, preds)
        
        if est_count is not None:
            gt_count = self._init_attr(0)
            if len(gt['loc']) > 0:
                gt_count = [gt['labels'].count(i+1) for i in range(self.num_classes-1)]

            self._update_errors(gt_count, est_count)
            self._update_calls(gt_count, est_count)
        
            self._total_calls += 1
            self._total_count = [self._total_count[i] + count for i, count in enumerate(est_count)]
        
        self._store_detections(preds, est_count)
        self.idx += 1
    
    def matching(self, gt: dict, preds: dict) -> None:
        ''' Method to match ground truth and predictions.
        To be overriden by subclasses
        
        Args:
            gt (dict): ground truth containing a dict with 'loc' and 'labels' 
                keys and list as values.
            preds (dict): predictions containing a dict with 'loc' and 'labels' 
                keys and list as values. Can contain an optional 'scores' key. 
        '''
        pass

    def copy(self):
        clone = copy.deepcopy(self)
        return clone
    
    def flush(self) -> None:
        ''' Flush the object '''

        self.detections = []
        self.idx = 0
        
        self.tp = self._init_attr()
        self.fp = self._init_attr()
        self.fn = self._init_attr()
    
        self._sum_absolute_error = self._init_attr()
        self._sum_squared_error = self._init_attr()
        self._n_calls = self._init_attr()
        self._agg_sum_absolute_error = 0
        self._agg_sum_squared_error = 0
        self._total_calls = 0
        self._total_count = self._init_attr()

        self._ap_tables = self._init_attr(val=[])

        self.confusion_matrix = numpy.zeros((self.num_classes-1, self.num_classes-1))
        self._confusion_matrix = self.confusion_matrix
    
    def aggregate(self) -> None:
        ''' Aggregate the metrics.

        By default, the classes are aggregated into a single class and the metrics are 
        therefore relative to the object vs. background configuration.
        '''

        inter = int(self._confusion_matrix.sum()) - sum(self.tp)
        
        self.fp = [sum(self.fp) - inter]
        self.fn = [sum(self.fn) - inter]
        self.tp = [int(self._confusion_matrix.sum())]
        self._sum_absolute_error = [self._agg_sum_absolute_error]
        self._sum_squared_error = [self._agg_sum_squared_error]
        self._n_calls = [self._total_calls]
        self._ap_tables = [[[1,*x[1:]] for x in sum(self._ap_tables, [])]]
        self._confusion_matrix = numpy.array([[1.]])
        self._total_count = [sum(self._total_count)]
        # self.predictions =

    def precision(self, c: int = 1) -> float:
        ''' Precision 
        Args:
            c (int, optional): class id. Defaults to 1.
        
        Returns:
            float
        '''

        c = c - 1
        if self.tp[c] > 0:
            return float(self.tp[c] / (self.tp[c] + self.fp[c]))
        else:
            return float(0)
    
    def recall(self, c: int = 1) -> float:
        ''' Recall 
        Args:
            c (int, optional): class id. Defaults to 1.
        
        Returns:
            float
        '''
        
        c = c - 1
        if self.tp[c] > 0:
            return float(self.tp[c] / (self.tp[c] + self.fn[c]))
        else:
            return float(0)
    
    def fbeta_score(self, c: int = 1, beta: int = 1) -> float:
        ''' F-beta score 
        Args:
            c (int, optional): class id. Defaults to 1.
            beta (int, optional): beta value. Defaults to 1.
        
        Returns:
            float
        '''
        
        if self.tp[c-1] > 0:
            return float(
                (1 + beta**2)*self.precision(c)*self.recall(c) / 
                ((beta**2)*self.precision(c) + self.recall(c))
                )
        else:
            return float(0)        
    
    def mae(self, c: int = 1) -> float:
        ''' Mean Absolute Error 
        Args:
            c (int, optional): class id. Defaults to 1.
        
        Returns:
            float
        '''

        c = c - 1
        return float(self._sum_absolute_error[c] / self._n_calls[c]) \
            if self._n_calls[c] else 0.
    
    def mse(self, c: int = 1) -> float:
        ''' Mean Squared Error 
        Args:
            c (int, optional): class id. Defaults to 1.
        
        Returns:
            float
        '''
        c = c - 1
        return float(self._sum_squared_error[c] / self._n_calls[c]) \
            if self._n_calls[c] else 0.

    def me(self, c: int = 1) -> float:
        ''' Mean Error
        Args:
            c (int, optional): class id. Defaults to 1.

        Returns:
            float
        '''
        c = c - 1
        return float(self._sum_error[c] / self._n_calls[c]) \
            if self._n_calls[c] else 0.
    
    def rmse(self, c: int = 1) -> float:
        ''' Root Mean Squared Error 
        Args:
            c (int, optional): class id. Defaults to 1.
        
        Returns:
            float
        '''
        return float(math.sqrt(self.mse(c)))
    
    def ap(self, c: int = 1) -> float:
        ''' Average Precision
        Args: 
            c (int, optional): class id. Defaults to 1.
        
        Returns:
            float
        '''

        recalls, precisions = self.rec_pre_lists(c)
        
        if len(recalls) == 0 or len(precisions) == 0:
            return 0.
        else:
            return self._compute_AP(recalls, precisions)
    
    def rec_pre_lists(self, c: int = 1) -> tuple:
        ''' Recalls and Precisions lists
        Args: 
            c (int, optional): class id. Defaults to 1.
        
        Returns:
            tuple
                recalls and precisions
        '''

        c = c - 1

        if len(self._ap_tables[c]) == 0:
            return [], []

        else:
            n_gt = self.fn[c] + self.tp[c]

            sorted_table = sorted(self._ap_tables[c], key=lambda x: x[1], reverse=True)
            sorted_table = numpy.array(sorted_table)
            sorted_table[:,2] = numpy.cumsum(sorted_table[:,2], axis=0)
            sorted_table[:,3] = numpy.cumsum(sorted_table[:,3], axis=0)

            precisions = sorted_table[:,2] / (sorted_table[:,2]+sorted_table[:,3])
            recalls = sorted_table[:,2] / n_gt

            return recalls.tolist(), precisions.tolist()
    
    def confusion(self, c: int = 1) -> float:
        ''' Interclass confusion
        Args: 
            c (int, optional): class id. Defaults to 1.
        
        Returns:
            float
                interclass confusion
        '''
        c = c - 1
        cm_row = self._confusion_matrix[c]
        p = cm_row[c]/sum(cm_row) if sum(cm_row) else 0.
        return 1 - p
    
    def accuracy(self) -> float:
        ''' Classification accuracy 
        
        Returns:
            float
        '''

        N = self.confusion_matrix.sum()
        tp = self.confusion_matrix.diagonal().sum()
        if N > 0:
            return tp / N
        else:
            return 0.

    def avg_score(self) -> float:
        ''' Average score of predictions

        Returns:
            float
        '''
        if len(self.detections) > 0:
            return float(np.mean([det.get("scores", 0) for det in self.detections]))
        else:
            return 0.0

    def avg_dscore(self) -> float:
        ''' Average detection score of predictions

        Returns:
            float
        '''
        if len(self.detections) > 0:
            return float(np.mean([det.get("dscores", 0) for det in self.detections]))
        else:
            return 0.0
    def total_count(self, c: int = 1) -> float:
        ''' Total class count
        Args: 
            c (int, optional): class id. Defaults to 1.
        
        Returns:
            float
                count
        '''
        c = c - 1
        return self._total_count[c]
        
    def _init_attr(self, val: int = 0) -> list:
        return [val] * (self.num_classes - 1)
    
    def _update_calls(self, gt_count: list, est_count: list):

        for i, _ in enumerate(self._n_calls):
            if gt_count[i] != 0 or est_count[i] != 0:
                self._n_calls[i] += 1

    def _update_errors(self, gt_count: list, est_count: list):

        for i, (count, est) in enumerate(zip(gt_count, est_count)):
            error = abs(count - est)
            squared_error = error**2
            signed_error = est - count

            self._sum_absolute_error[i] += error
            self._sum_squared_error[i] += squared_error
            self._sum_error[i] += signed_error
        
        agg_error = abs(sum(gt_count) - sum(est_count))
        agg_squared_error = agg_error**2
        agg_signed_error = sum(est_count) - sum(gt_count)

        self._agg_sum_absolute_error += agg_error
        self._agg_sum_squared_error += agg_squared_error
        self._agg_sum_error += agg_signed_error
    
    def _no_gt(self, gt: dict, preds: dict) -> None:

        for c in range(1, self.num_classes):
            n_pred = len([lab for lab in preds['labels'] if lab == c])

            self.tp[c-1] += 0
            self.fp[c-1] += n_pred
            self.fn[c-1] += 0

            if self.score_flag:
                preds_fp = [[preds['labels'][i],preds['scores'][i],0,1]
                                  for i, _ in enumerate(preds['labels'])
                                  if preds['labels'][i] == c]

                self._ap_tables[c-1] = [*self._ap_tables[c-1], *preds_fp]
    
    def _no_preds(self, gt: dict, preds: dict) -> None:

        for c in range(1, self.num_classes):
            n_gt = len([lab for lab in gt['labels'] if lab == c])

            self.tp[c-1] += 0
            self.fp[c-1] += 0
            self.fn[c-1] += n_gt
    
    def _compute_AP(self, recalls: list, precisions: list) -> float:
        '''
        Compute the VOC Average Precision
        Code from: https://github.com/Cartucho/mAP
        (adapted from official matlab code VOC2012)
        '''

        recalls.insert(0, 0.0)
        recalls.append(1.0)
        precisions.insert(0, 0.0) 
        precisions.append(0.0) 

        mrec, mpre = recalls[:], precisions[:]

        for i in range(len(mpre)-2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i+1])

        i_list = []
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i-1]:
                i_list.append(i) 

        ap = 0.0
        for i in i_list:
            ap += ((mrec[i]-mrec[i-1])*mpre[i])
        
        return ap
    
    def _store_detections(self, preds: dict, est_count: Optional[list] = None) -> None:
        ''' Store detections internally (1 row = 1 detection) '''

        m = map(dict, zip( * [
            [(k, v) for v in value]
            for k, value in preds.items()
            ]))
        m, m_copy = tee(m)
        
        counts = {}
        if est_count is not None:
            counts = {f'count_{i+1}': x for i, x in enumerate(est_count)}

        if len([x for x in m_copy]) > 0:
            for det in m:
                self.detections.append({'images': self.idx, **det, **counts})
        else:
            self.detections.append({'images': self.idx, **counts})
    
@METRICS.register()
class PointsMetrics(Metrics):
    ''' Metrics class for points (must be in (x,y) format) '''

    def __init__(self, radius: float, num_classes: int = 2) -> None:
        '''
        Args:
            radius (float): distance between ground truth and predicted point
                from which a point is characterizd as true positive
            num_classes (int, optional): number of classes, background included. 
                Defaults to 2 (binary case).
        '''
        super().__init__(threshold=radius, num_classes=num_classes)
    
    def matching(self, gt: dict, preds: dict) -> None:
        """
        Matching ground truth and predictions to determine true positives, false positives
        # TODO this matching is not a good. Use HungarianMatching here as well.
        """
        assert gt.keys() == {'loc', 'labels'}
        assert preds.keys() ==  {'loc', 'labels', 'scores', 'dscores'}
        
        if 'dscores' not in preds.keys():
            # preds['dscores'] = []
            # logger.warning(f"dscores is not set, which is on purpose for p2p")
            pass
        
        # matching
        dist = sklearn.neighbors.NearestNeighbors(n_neighbors=1, metric='euclidean').fit(preds['loc'])
        dist, idx = dist.kneighbors(gt['loc'])
        match_gt = [(k, d, i) for k, (d, i) in enumerate(zip(dist[:,0], idx[:,0]))]

        # sort according to distance
        match_gt = sorted(match_gt, key = lambda tup: tup[1])              
        # discard duplicates
        k_discard, i_discard = [], []
        filter_match_gt = []
        for k, d, i in match_gt:
            if k not in k_discard and i not in i_discard:
                filter_match_gt.append((k,d,i))
                k_discard.append(k), i_discard.append(i)

        # threshold
        filter_match_gt = [(k, d, i) for k, d, i in filter_match_gt if d <= self.threshold]

        if len(filter_match_gt) == 0:
            # logger.error(f'NO Prediction is matched to any Ground Truth: filter_match_gt: {filter_match_gt}')
            pass
        # confusion matrix
        y_true = [gt['labels'][k] for k, d, i in filter_match_gt]
        y_pred = [preds['labels'][i] for k, d, i in filter_match_gt]

        try:
            self._confusion_matrix += confusion_matrix(
                y_true, y_pred, labels=list(range(1, self.num_classes)))
        except ValueError:
            # logger.warning('Confusion matrix is empty')
            pass

        for c in range(1, self.num_classes):
            n_gt = len([lab for lab in gt['labels'] if lab == c])
            n_pred = len([lab for lab in preds['labels'] if lab == c])

            lab_match = [(d, i) for k, d, i in filter_match_gt 
                            if gt['labels'][k] == preds['labels'][i] == c]

            tp = len(lab_match)
            self.tp[c-1] += tp
            self.fp[c-1] += (n_pred - tp)
            self.fn[c-1] += (n_gt - tp)

            if self.score_flag:
                tp_ids = [i for d, i in lab_match]
                preds_tp = [[preds['labels'][i],preds['scores'][i],1,0]
                              for _, i in lab_match]
                preds_fp = [[preds['labels'][i],preds['scores'][i],0,1]
                              for i, _ in enumerate(preds['labels'])
                              if preds['labels'][i] == c and i not in tp_ids]

                self._ap_tables[c-1] = [*self._ap_tables[c-1], *preds_tp, *preds_fp]
    
    def _store_detections(self, preds: dict, est_count: Optional[list] = None) -> None:

        m = map(dict, zip( * [
            [(k, v) for v in value]
            for k, value in preds.items()
            ]))
        m, m_copy = tee(m)
        
        counts = {}
        if est_count is not None:
            counts = {f'count_{i+1}': x for i, x in enumerate(est_count)}

        if len([x for x in m_copy]) > 0:
            for det in m:
                y, x = det['loc']
                det.update(dict(x=x, y=y))
                _ = det.pop('loc')
                self.detections.append({'images': self.idx, **det, **counts})
        else:
            self.detections.append({'images': self.idx, **counts})


@dataclass
class DensityBinStats:
    """Statistics for a single density bin."""
    tp: int = 0
    fp: int = 0
    fn: int = 0
    n_tiles: int = 0
    total_gt: int = 0
    total_pred: int = 0
    sum_abs_error: float = 0.0
    sum_squared_error: float = 0.0
    sum_signed_error: float = 0.0

    @property
    def precision(self) -> float:
        if self.tp + self.fp == 0:
            return 0.0
        return self.tp / (self.tp + self.fp)

    @property
    def recall(self) -> float:
        if self.tp + self.fn == 0:
            return 0.0
        return self.tp / (self.tp + self.fn)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    @property
    def f2(self) -> float:
        """F2 score (weights recall higher than precision)."""
        return self.fbeta(beta=2.0)

    def fbeta(self, beta: float = 1.0) -> float:
        """F-beta score with configurable beta."""
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return (1 + beta ** 2) * p * r / (beta ** 2 * p + r)

    @property
    def mae(self) -> float:
        """Mean Absolute Error."""
        if self.n_tiles == 0:
            return 0.0
        return self.sum_abs_error / self.n_tiles

    @property
    def me(self) -> float:
        """Mean signed error (bias). Positive = overcounting."""
        if self.n_tiles == 0:
            return 0.0
        return self.sum_signed_error / self.n_tiles

    @property
    def rmse(self) -> float:
        """Root Mean Squared Error."""
        if self.n_tiles == 0:
            return 0.0
        return np.sqrt(self.sum_squared_error / self.n_tiles)

    def __repr__(self) -> str:
        return (
            f"DensityBinStats(n_tiles={self.n_tiles}, "
            f"tp={self.tp}, fp={self.fp}, fn={self.fn}, "
            f"precision={self.precision:.3f}, recall={self.recall:.3f}, f1={self.f1:.3f}, "
            f"mae={self.mae:.2f}, me={self.me:+.2f})"
        )


@METRICS.register()
class DensityAwarePointsMetrics(PointsMetrics):
    """
    Point detection metrics stratified by ground truth density.

    Extends PointsMetrics to track:
    - Per-density bin metrics (F1, precision, recall, MAE, RMSE)
    - Binary occupancy accuracy (empty vs occupied)
    - Per-tile detailed results

    Fully compatible with existing animaloc evaluation pipeline.
    All parent methods (precision, recall, fbeta_score, mae, rmse, ap, etc.)
    work exactly as before.

    Args:
        radius: Matching radius for point detection (pixels)
        num_classes: Number of classes including background (default 2 for binary)
        max_density: Maximum density to track individually (higher grouped as "N+")

    Example:
        # Drop-in replacement for PointsMetrics
        metrics = DensityAwarePointsMetrics(radius=25, num_classes=2, max_density=15)

        for gt, preds in data:
            metrics.feed(gt=gt, preds=preds, est_count=[len(preds['loc'])])

        # Standard metrics (inherited from PointsMetrics)
        print(f"Precision: {metrics.precision()}")
        print(f"Recall: {metrics.recall()}")
        print(f"F1: {metrics.fbeta_score()}")
        print(f"MAE: {metrics.mae()}")
        print(f"AP: {metrics.ap()}")

        # Density-stratified metrics (new)
        print(metrics.density_report())
        print(metrics.occupancy_report())
        df = metrics.density_dataframe()
    """

    def __init__(
        self,
        radius: float = 25.0,
        num_classes: int = 2,
        max_density: int = 15
    ):
        # Initialize parent class
        super().__init__(radius=radius, num_classes=num_classes)

        self.max_density = max_density

        # Initialize density tracking
        self._init_density_tracking()

    def _init_density_tracking(self):
        """Initialize or reset density tracking structures."""
        # Per-density statistics
        self.density_stats: Dict[str, DensityBinStats] = defaultdict(DensityBinStats)

        # Binary occupancy statistics
        self.occupancy_tp = 0  # Correctly predicted occupied
        self.occupancy_tn = 0  # Correctly predicted empty
        self.occupancy_fp = 0  # Predicted occupied, actually empty
        self.occupancy_fn = 0  # Predicted empty, actually occupied

        # Per-tile results for detailed analysis
        self.tile_results: List[Dict] = []

    def _get_density_key(self, count: int) -> str:
        """Convert count to density bin key."""
        if count > self.max_density:
            return f"{self.max_density}+"
        return str(count)

    def flush(self) -> None:
        """Reset all statistics including density tracking."""
        # Call parent flush
        super().flush()

        # Reset density tracking
        self._init_density_tracking()

    def copy(self):
        """Create a deep copy including density stats."""
        return copy.deepcopy(self)

    def feed(self, gt: dict, preds: dict, est_count: Optional[list] = None) -> None:
        """
        Feed ground truth and predictions for one tile.

        Extends parent feed() to also track density-stratified metrics.

        Args:
            gt: Dict with 'loc' (list of (x,y) or (y,x)) and 'labels'
            preds: Dict with 'loc', 'labels', and optionally 'scores', 'dscores'
            est_count: Optional estimated count per class
        """
        # Get counts before processing
        n_gt = len(gt.get('loc', []))
        n_pred = len(preds.get('loc', []))

        # Store current cumulative TP/FP/FN before parent processes this tile
        prev_tp = sum(self.tp)
        prev_fp = sum(self.fp)
        prev_fn = sum(self.fn)

        # Store tile index before parent increments it
        tile_idx = self.idx

        # Call parent feed - this does all the matching and updates tp/fp/fn
        super().feed(gt, preds, est_count)

        # Calculate per-tile TP/FP/FN as delta from cumulative stats
        tile_tp = sum(self.tp) - prev_tp
        tile_fp = sum(self.fp) - prev_fp
        tile_fn = sum(self.fn) - prev_fn

        # Update density tracking
        self._update_density_stats(
            n_gt=n_gt,
            n_pred=n_pred,
            tp=tile_tp,
            fp=tile_fp,
            fn=tile_fn,
            tile_idx=tile_idx,
            preds=preds
        )

    def _update_density_stats(
        self,
        n_gt: int,
        n_pred: int,
        tp: int,
        fp: int,
        fn: int,
        tile_idx: int,
        preds: dict
    ):
        """Update density-stratified statistics after matching."""
        # Get density bin
        density_key = self._get_density_key(n_gt)

        # Update density-specific stats
        stats = self.density_stats[density_key]
        stats.tp += tp
        stats.fp += fp
        stats.fn += fn
        stats.n_tiles += 1
        stats.total_gt += n_gt
        stats.total_pred += n_pred

        # Count error
        count_error = n_pred - n_gt
        stats.sum_abs_error += abs(count_error)
        stats.sum_squared_error += count_error ** 2
        stats.sum_signed_error += count_error

        # Update binary occupancy stats
        gt_occupied = n_gt > 0
        pred_occupied = n_pred > 0

        if gt_occupied and pred_occupied:
            self.occupancy_tp += 1
        elif not gt_occupied and not pred_occupied:
            self.occupancy_tn += 1
        elif not gt_occupied and pred_occupied:
            self.occupancy_fp += 1
        else:  # gt_occupied and not pred_occupied
            self.occupancy_fn += 1

        # Store tile result for detailed analysis
        pred_scores = preds.get('scores', [])
        self.tile_results.append({
            'idx': tile_idx,
            'gt_count': n_gt,
            'pred_count': n_pred,
            'density_bin': density_key,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': tp / max(tp + fp, 1),
            'recall': tp / max(tp + fn, 1),
            'gt_occupied': gt_occupied,
            'pred_occupied': pred_occupied,
            'correct_occupancy': gt_occupied == pred_occupied,
            'count_error': count_error,
            'abs_count_error': abs(count_error),
            'avg_score': float(np.mean(pred_scores)) if pred_scores else 0.0,
        })

    # ==================== Occupancy Metrics ====================

    def occupancy_accuracy(self) -> float:
        """Binary accuracy: correctly classified as empty or occupied."""
        total = self.occupancy_tp + self.occupancy_tn + self.occupancy_fp + self.occupancy_fn
        if total == 0:
            return 0.0
        return (self.occupancy_tp + self.occupancy_tn) / total

    def occupancy_precision(self) -> float:
        """Precision for occupied class."""
        if self.occupancy_tp + self.occupancy_fp == 0:
            return 0.0
        return self.occupancy_tp / (self.occupancy_tp + self.occupancy_fp)

    def occupancy_recall(self) -> float:
        """Recall for occupied class (sensitivity)."""
        if self.occupancy_tp + self.occupancy_fn == 0:
            return 0.0
        return self.occupancy_tp / (self.occupancy_tp + self.occupancy_fn)

    def occupancy_specificity(self) -> float:
        """Specificity: correctly identified empty tiles."""
        if self.occupancy_tn + self.occupancy_fp == 0:
            return 0.0
        return self.occupancy_tn / (self.occupancy_tn + self.occupancy_fp)

    def occupancy_f1(self) -> float:
        """F1 score for occupied class."""
        p, r = self.occupancy_precision(), self.occupancy_recall()
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    # ==================== Per-Density Metrics ====================

    def density_precision(self, density: int) -> float:
        """Get precision for a specific density bin."""
        key = self._get_density_key(density)
        return self.density_stats[key].precision()

    def density_recall(self, density: int) -> float:
        """Get recall for a specific density bin."""
        key = self._get_density_key(density)
        return self.density_stats[key].recall()

    def density_f1(self, density: int) -> float:
        """Get F1 for a specific density bin."""
        key = self._get_density_key(density)
        return self.density_stats[key].f1()

    def density_mae(self, density: int) -> float:
        """Get MAE for a specific density bin."""
        key = self._get_density_key(density)
        return self.density_stats[key].mae()

    def get_density_metrics(self, density: int) -> Dict[str, float]:
        """Get all metrics for a specific density bin."""
        key = self._get_density_key(density)
        stats = self.density_stats.get(key, DensityBinStats())

        return {
            'density': key,
            'n_tiles': stats.n_tiles,
            'total_gt': stats.total_gt,
            'total_pred': stats.total_pred,
            'tp': stats.tp,
            'fp': stats.fp,
            'fn': stats.fn,
            'precision': stats.precision(),
            'recall': stats.recall(),
            'f1': stats.f1(),
            'f2': stats.fbeta(beta=2),
            'mae': stats.mae(),
            'me': stats.me(),
            'rmse': stats.rmse(),
        }

    # ==================== Reports ====================

    def density_report(self) -> str:
        """Generate a formatted report of per-density metrics."""
        lines = []
        lines.append("=" * 115)
        lines.append("DENSITY-STRATIFIED METRICS")
        lines.append("=" * 115)
        lines.append(
            f"{'Density':<10} {'Tiles':>8} {'GT':>8} {'Pred':>8} "
            f"{'TP':>6} {'FP':>6} {'FN':>6} "
            f"{'Prec':>7} {'Rec':>7} {'F1':>7} {'MAE':>7} {'ME':>7}"
        )
        lines.append("-" * 115)

        # Sort density keys
        keys = sorted(
            self.density_stats.keys(),
            key=lambda x: int(x.rstrip('+')) if x.rstrip('+').isdigit() else 999
        )

        # Aggregate stats for overall row
        overall = DensityBinStats()

        for key in keys:
            stats = self.density_stats[key]
            overall.tp += stats.tp
            overall.fp += stats.fp
            overall.fn += stats.fn
            overall.n_tiles += stats.n_tiles
            overall.total_gt += stats.total_gt
            overall.total_pred += stats.total_pred
            overall.sum_abs_error += stats.sum_abs_error
            overall.sum_squared_error += stats.sum_squared_error
            overall.sum_signed_error += stats.sum_signed_error

            lines.append(
                f"{key:<10} {stats.n_tiles:>8} {stats.total_gt:>8} {stats.total_pred:>8} "
                f"{stats.tp:>6} {stats.fp:>6} {stats.fn:>6} "
                f"{stats.precision:>7.3f} {stats.recall:>7.3f} {stats.f1:>7.3f} "
                f"{stats.mae:>7.2f} {stats.me:>+7.2f}"
            )

        lines.append("-" * 115)
        lines.append(
            f"{'OVERALL':<10} {overall.n_tiles:>8} {overall.total_gt:>8} {overall.total_pred:>8} "
            f"{overall.tp:>6} {overall.fp:>6} {overall.fn:>6} "
            f"{overall.precision:>7.3f} {overall.recall:>7.3f} {overall.f1:>7.3f} "
            f"{overall.mae:>7.2f} {overall.me:>+7.2f}"
        )
        lines.append("=" * 115)

        return "\n".join(lines)

    def occupancy_report(self) -> str:
        """Generate a formatted report of binary occupancy metrics."""
        lines = []
        lines.append("=" * 60)
        lines.append("BINARY OCCUPANCY CLASSIFICATION")
        lines.append("=" * 60)
        lines.append(f"                    Predicted")
        lines.append(f"                  Empty    Occupied")
        lines.append(f"Actual Empty     {self.occupancy_tn:>6}    {self.occupancy_fp:>6}")
        lines.append(f"Actual Occupied  {self.occupancy_fn:>6}    {self.occupancy_tp:>6}")
        lines.append("-" * 60)
        lines.append(f"Accuracy:    {self.occupancy_accuracy():.4f}")
        lines.append(f"Precision:   {self.occupancy_precision():.4f}  (occupied class)")
        lines.append(f"Recall:      {self.occupancy_recall():.4f}  (occupied class)")
        lines.append(f"Specificity: {self.occupancy_specificity():.4f}  (empty class)")
        lines.append(f"F1 Score:    {self.occupancy_f1():.4f}  (occupied class)")
        lines.append("=" * 60)

        return "\n".join(lines)

    def density_dataframe(self) -> pd.DataFrame:
        """Convert per-density metrics to a pandas DataFrame."""
        rows = []

        keys = sorted(
            self.density_stats.keys(),
            key=lambda x: int(x.rstrip('+')) if x.rstrip('+').isdigit() else 999
        )

        for key in keys:
            stats = self.density_stats[key]
            rows.append({
                'density': key,
                'n_tiles': stats.n_tiles,
                'total_gt': stats.total_gt,
                'total_pred': stats.total_pred,
                'tp': stats.tp,
                'fp': stats.fp,
                'fn': stats.fn,
                'precision': stats.precision,
                'recall': stats.recall,
                'f1': stats.f1,
                'f2': stats.fbeta(beta=2),
                'mae': stats.mae,
                'me': stats.me,
                'rmse': stats.rmse,
            })

        return pd.DataFrame(rows)

    def tile_results_dataframe(self) -> pd.DataFrame:
        """Get per-tile results as DataFrame for detailed analysis."""
        return pd.DataFrame(self.tile_results)

    def density_summary_dict(self) -> Dict[str, float]:
        """Get density-related summary statistics as a dictionary."""
        # Aggregate overall from density bins
        overall = DensityBinStats()
        for stats in self.density_stats.values():
            overall.tp += stats.tp
            overall.fp += stats.fp
            overall.fn += stats.fn
            overall.n_tiles += stats.n_tiles
            overall.total_gt += stats.total_gt
            overall.total_pred += stats.total_pred
            overall.sum_abs_error += stats.sum_abs_error
            overall.sum_squared_error += stats.sum_squared_error
            overall.sum_signed_error += stats.sum_signed_error

        return {
            # Density-based overall (should match parent's metrics)
            'density_precision': overall.precision,
            'density_recall': overall.recall,
            'density_f1': overall.f1,
            'density_f2': overall.fbeta(beta=2),
            'density_mae': overall.mae,
            'density_me': overall.me,
            'density_rmse': overall.rmse,
            'density_n_tiles': overall.n_tiles,

            # Occupancy metrics
            'occupancy_accuracy': self.occupancy_accuracy(),
            'occupancy_precision': self.occupancy_precision(),
            'occupancy_recall': self.occupancy_recall(),
            'occupancy_specificity': self.occupancy_specificity(),
            'occupancy_f1': self.occupancy_f1(),
            'occupancy_tp': self.occupancy_tp,
            'occupancy_tn': self.occupancy_tn,
            'occupancy_fp': self.occupancy_fp,
            'occupancy_fn': self.occupancy_fn,
        }


@METRICS.register()
class BoxesMetrics(Metrics):
    ''' Metrics class for bounding boxes 
    (must be in (x_min, y_min, x_max, y_max) format '''

    def __init__(self, iou: float, num_classes: int = 2) -> None:
        '''
        Args:
            iou (float): Intersect-over-Union (IoU) threshold used to define a true
                positive.
            num_classes (int, optional): number of classes, background included. 
                Defaults to 2 (binary case).
        '''
        super().__init__(threshold=iou, num_classes=num_classes)
    
    def matching(self, gt: dict, preds: dict) -> None:

        ious, idx = self._most_overlapping_boxes(gt['loc'], preds['loc'])
        match_gt = [(k, iou, i) for k, (iou, i) in enumerate(zip(ious, idx)) 
                        if iou >= self.threshold]
        
        # confusion matrix
        y_true = [gt['labels'][k] for k, d, i in match_gt]
        y_pred = [preds['labels'][i] for k, d, i in match_gt]

        self._confusion_matrix += confusion_matrix(
            y_true, y_pred, labels=list(range(1, self.num_classes)))
        
        for c in range(1, self.num_classes):
            n_gt = len([lab for lab in gt['labels'] if lab == c])
            n_pred = len([lab for lab in preds['labels'] if lab == c])

            lab_match = [(d, i) for k, d, i in match_gt 
                            if gt['labels'][k] == preds['labels'][i] == c]

            tp = len(lab_match)
            self.tp[c-1] += tp
            self.fp[c-1] += (n_pred - tp)
            self.fn[c-1] += (n_gt - tp)

            if self.score_flag:
                tp_ids = [i for d, i in lab_match]
                preds_tp = [[preds['labels'][i],preds['scores'][i],1,0]
                              for _, i in lab_match]
                preds_fp = [[preds['labels'][i],preds['scores'][i],0,1]
                              for i, _ in enumerate(preds['labels'])
                              if preds['labels'][i] == c and i not in tp_ids]

                self._ap_tables[c-1] = [*self._ap_tables[c-1], *preds_tp, *preds_fp]
    
    def _most_overlapping_boxes(
        self, 
        gt_boxes: List[tuple], 
        preds_boxes: List[tuple], 
        ) -> tuple:
        
        gt_boxes = [BoundingBox(*coord) for coord in gt_boxes]
        preds_boxes = [BoundingBox(*coord) for coord in preds_boxes]

        iou_matrix = bboxes_iou(gt_boxes, preds_boxes)

        match_idx = []
        ious = []
        for row in iou_matrix:
            filt_row = [(k, elem) for k, elem in enumerate(row) if k not in match_idx]
            if len(filt_row) > 0:
                idx, iou_max = max(filt_row, key=lambda item:item[1])

                match_idx.append(idx)
                ious.append(iou_max)
        
        return ious, match_idx
    
    def _store_detections(self, preds: dict, est_count: Optional[list] = None) -> None:

        m = map(dict, zip( * [
            [(k, v) for v in value]
            for k, value in preds.items()
            ]))
        m, m_copy = tee(m)
        
        counts = {}
        if est_count is not None:
            counts = {f'count_{i+1}': x for i, x in enumerate(est_count)}

        if len([x for x in m_copy]) > 0:
            for det in m:
                x_min, y_min, x_max, y_max = det['loc']
                det.update(dict(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max))
                _ = det.pop('loc')
                self.detections.append({'images': self.idx, **det, **counts})
        else:
            self.detections.append({'images': self.idx, **counts})

@METRICS.register()
class ImageLevelMetrics(Metrics):
    ''' Metrics class for image-level classification '''

    def __init__(self, num_classes: int = 2) -> None:
        num_classes = num_classes + 1 # for convenience
        super().__init__(0, num_classes)
    
    def feed(self, gt: int, pred: int) -> tuple:
        '''
        Args:
            gt (int): numeric ground truth label
            pred (int): numeric predicted label
        '''

        gt = dict(labels=[gt], loc=[(0,0)])
        preds = dict(labels=[pred], loc=[(0,0)])
        
        super().feed(gt, preds)
    
    def matching(self, gt: dict, pred: dict) -> None:
        gt_lab = gt['labels'][0]
        p_lab = pred['labels'][0]

        if gt_lab == p_lab:
            self.tp[gt_lab-1] += 1
        else:
            self.fp[p_lab-1] += 1
            self.fn[gt_lab-1] += 1
        
        self._confusion_matrix += confusion_matrix(
            [gt_lab], [p_lab], labels=list(range(self.num_classes-1)))