import argparse
import os
import PIL
import torchvision
import numpy
import cv2
import pandas


if __name__ == "__main__":

    print("hello")

    from animaloc.models import HerdNet
    from torch import Tensor
    from animaloc.models import LossWrapper
    from animaloc.train.losses import FocalLoss
    from torch.nn import CrossEntropyLoss

    herdnet = HerdNet(num_classes=2, down_ratio=2)

    weight = Tensor([0.1, 1.0, 2.0, 1.0, 6.0, 12.0, 1.0])

    losses = [
        {'loss': FocalLoss(reduction='mean'), 'idx': 0, 'idy': 0, 'lambda': 1.0, 'name': 'focal_loss'},
        {'loss': CrossEntropyLoss(reduction='mean', weight=weight), 'idx': 1, 'idy': 1, 'lambda': 1.0,
         'name': 'ce_loss'}
    ]

    herdnet = LossWrapper(herdnet, losses=losses)