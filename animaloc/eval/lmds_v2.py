import torch
import numpy as np
import torch.nn.functional as F
from typing import Tuple, List, Optional

__all__ = ['LMDS', 'HerdNetLMDS']


class LMDS:
    ''' Local Maxima Detection Strategy (Robust Version) '''

    def __init__(
            self,
            kernel_size: tuple = (3, 3),
            adapt_ts: float = 0.4,  # Adaptive factor (Relative)
            score_threshold: float = 0.5,  # <--- NEW: Hard Floor (Absolute)
            neg_ts: float = 0.1  # Negative sample threshold
    ) -> None:

        assert kernel_size[0] == kernel_size[1], 'Kernel must be square'
        assert kernel_size[0] % 2 != 0, 'Kernel size must be odd'

        self.kernel_size = tuple(kernel_size)
        self.adapt_ts = adapt_ts
        self.score_threshold = score_threshold
        self.neg_ts = neg_ts
        self.padding = self.kernel_size[0] // 2

    def __call__(self, est_map: torch.Tensor) -> Tuple[list, list, list, list]:
        """
        Input: est_map [B, Classes, H, W]
        """
        batch_size, classes = est_map.shape[:2]

        # Output storage
        b_counts, b_labels, b_scores, b_locs = [], [], [], []

        for b in range(batch_size):
            counts, labels, scores, locs = [], [], [], []
            for c in range(classes):
                # Process each class channel
                count, loc, score = self._lmds(est_map[b][c])

                counts.append(count)
                if count > 0:
                    labels.extend([c + 1] * count)
                    scores.extend(score)
                    locs.extend(loc)

            b_counts.append(counts)
            b_labels.append(labels)
            b_scores.append(scores)
            b_locs.append(locs)

        return b_counts, b_locs, b_labels, b_scores

    def _lmds(self, est_map: torch.Tensor) -> Tuple[int, list, list]:
        ''' Single channel processing. Shape: est_map = [H,W] '''

        # 1. Global Max for Adaptive Thresholding
        est_map_max = est_map.max().item()

        # 2. Early Exit: Negative Sample Check
        # If the highest peak in the image is garbage, return empty immediately.
        if est_map_max < self.neg_ts:
            return 0, [], []

        # 3. Local Maxima Extraction (Peak Finding)
        # Keep tensor on device!
        est_map_unsqueezed = est_map.unsqueeze(0).unsqueeze(0)
        keep = F.max_pool2d(est_map_unsqueezed, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        is_peak = (keep == est_map_unsqueezed).squeeze()  # [H, W]

        # 4. Calculate Robust Threshold
        # Threshold = Max(Adaptive, Absolute)
        #
        adaptive_threshold = self.adapt_ts * est_map_max
        final_threshold = max(adaptive_threshold, self.score_threshold)

        # 5. Apply Filter
        # Keep if (Is a Peak) AND (Value > Threshold)
        keep_mask = is_peak & (est_map > final_threshold)

        # 6. Extract Coordinates and Scores
        # using torch.nonzero is much faster than numpy conversions
        locs_tensor = torch.nonzero(keep_mask, as_tuple=False)  # [N, 2] (y, x)

        if locs_tensor.numel() == 0:
            return 0, [], []

        # Get values at those coordinates
        scores_tensor = est_map[keep_mask]

        count = locs_tensor.shape[0]

        return count, locs_tensor.tolist(), scores_tensor.tolist()


class HerdNetLMDS(LMDS):

    def __init__(
            self,
            up: bool = True,
            kernel_size: tuple = (3, 3),
            adapt_ts: float = 0.4,
            score_threshold: float = 0.3,  # Add this here
            neg_ts: float = 0.1,
            scale_factor: int = 16
    ) -> None:

        super().__init__(
            kernel_size=kernel_size,
            adapt_ts=adapt_ts,
            score_threshold=score_threshold,
            neg_ts=neg_ts
        )

        self.up = up
        self.scale_factor = scale_factor

    def __call__(self, outputs: List[torch.Tensor]) -> Tuple[list, list, list, list, list]:
        '''
        Inputs: heatmap [B,1,H,W], clsmap [B,C,H/16,W/16]
        '''
        heatmap, clsmap = outputs

        # 1. Upsample Class Map (Nearest is fine for segmentation masks)
        if self.up and self.scale_factor > 1:
            clsmap = F.interpolate(clsmap, scale_factor=self.scale_factor, mode='nearest')

        # 2. Get Classification Probabilities
        # dim=1 is channels. [B, C+1, H, W] -> [B, C, H, W] (remove background class 0)
        cls_scores_all = torch.softmax(clsmap, dim=1)
        cls_probs = cls_scores_all[:, 1:, :, :]

        # 3. LMDS on Heatmap (Objectness)
        batch_size, _, _, _ = heatmap.shape
        # output storage
        b_counts, b_labels, b_scores, b_locs, b_dscores = [], [], [], [], []

        for b in range(batch_size):
            # Run LMDS on the heatmap (single channel 0)
            # This returns valid object locations based on your thresholds
            count, locs, dscores = self._lmds(heatmap[b][0])

            if count == 0:
                b_labels.append([])
                b_scores.append([])
                b_locs.append([])
                b_counts.append([0] * cls_probs.shape[1])
                b_dscores.append([])
                continue

            # Convert to tensors for indexing
            locs_tensor = torch.tensor(locs, device=heatmap.device, dtype=torch.long)
            h_idx, w_idx = locs_tensor[:, 0], locs_tensor[:, 1]

            # 4. Extract Class Labels at Object Locations
            # We look at the Argmax of the class probabilities at the found coordinates
            # cls_probs[b] is [C, H, W]

            # Get class index (0 = Cow, 1 = Sheep, etc.)
            # We add 1 because class 0 was background (removed earlier)
            cls_vectors = cls_probs[b][:, h_idx, w_idx]  # [C, N_objects]
            obj_cls_idx = torch.argmax(cls_vectors, dim=0)  # [N_objects]

            labels = (obj_cls_idx + 1).tolist()  # Convert back to 1-based indexing

            # 5. Extract Classification Score
            # "How confident are we that this object is a Cow?"
            # We gather the probability of the chosen class
            # dim=0 is classes
            scores = torch.gather(cls_vectors, 0, obj_cls_idx.unsqueeze(0)).squeeze(0).tolist()

            # Count per class
            num_classes = cls_probs.shape[1]
            counts_per_class = [labels.count(i) for i in range(1, num_classes + 1)]

            b_labels.append(labels)
            b_scores.append(scores)  # Class Confidence (e.g., 0.95)
            b_locs.append(locs)  # Coordinates
            b_counts.append(counts_per_class)
            b_dscores.append(dscores)  # Objectness Confidence (e.g., 0.45)

        return b_counts, b_locs, b_labels, b_scores, b_dscores