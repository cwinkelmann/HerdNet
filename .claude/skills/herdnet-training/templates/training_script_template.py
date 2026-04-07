"""
HerdNet Training Script Template
=================================
Complete training pipeline using animaloc APIs.
Trains a HerdNetTimmDLA model on point annotations.

Usage:
    python training_script_template.py

Customize:
    1. Update DATA_CONFIG paths for your dataset
    2. Adjust MODEL_CONFIG for your species (num_classes, backbone)
    3. Adjust TRAIN_CONFIG for your hardware (batch_size, device)
    4. Adjust EVAL_CONFIG matching_radius for your animal size
"""

import os
import torch
import pandas as pd
from torch.utils.data import DataLoader

# animaloc imports
from animaloc.models import HerdNetTimmDLA
from animaloc.models.utils import LossWrapper
from animaloc.datasets import CSVDataset
from animaloc.train.trainers import Trainer
from animaloc.eval.evaluators import HerdNetEvaluator
from animaloc.eval.stitchers import HerdNetStitcher
from animaloc.eval.metrics import Metrics
from animaloc.data.transforms import FIDT, DownSample, MultiTransformsWrapper

# Optional: Weights & Biases
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# =============================================================================
# CONFIGURATION -- EDIT THESE FOR YOUR PROJECT
# =============================================================================

# Data paths
DATA_CONFIG = {
    'train_csv': '/path/to/train/annotations.csv',       # TODO: Update
    'train_dir': '/path/to/train/images/',                # TODO: Update
    'val_csv': '/path/to/val/annotations.csv',            # TODO: Update
    'val_dir': '/path/to/val/images/',                    # TODO: Update
    'num_classes': 3,           # background(0) + iguana(1) + hard_negative(2)
    'img_size': (512, 512),
}

# Model architecture
MODEL_CONFIG = {
    'backbone': 'timm/dla34',   # DLA-34 is optimal for iguana data
    'num_classes': 3,           # Must match DATA_CONFIG
    'down_ratio': 4,            # Optimal: 4 (not 2)
    'head_conv': 64,            # Optimal: 64 (not 128)
    'pretrained': True,         # Use ImageNet pretrained weights
}

# Training hyperparameters
TRAIN_CONFIG = {
    'lr': 1e-4,                 # Head learning rate
    'backbone_lr': 1e-6,        # Backbone learning rate (much lower)
    'weight_decay': 3.25e-4,    # Optimal: 3.25e-4
    'warmup_iters': 100,        # Linear warmup iterations
    'epochs': 20,
    'batch_size': 2,
    'num_workers': 4,
    'early_stopping': True,
    'patience': 10,
}

# Loss configuration
LOSS_CONFIG = {
    'focal_beta': 5,            # Optimal: 5 (default was 4)
    'focal_alpha': 2,
    'ce_weights': [0.1, 4.0, 1.0],  # [background, iguana, hard_neg]
}

# Evaluation configuration
EVAL_CONFIG = {
    'matching_radius': 75,      # CRITICAL: 75px for iguanas (not 25px default!)
    'lmds_kernel': (5, 5),      # Optimal: (5,5)
    'lmds_adapt_ts': 0.5,       # Optimal: 0.5
    'stitcher_overlap': 120,
}

# W&B configuration
WANDB_CONFIG = {
    'enabled': False,            # Set to True to enable W&B logging
    'project': 'herdnet',
    'entity': 'your_team',
    'run_name': 'experiment_01',
}


# =============================================================================
# DEVICE SELECTION
# =============================================================================

def get_device():
    """Select best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def main():
    device = get_device()
    print(f"Using device: {device}")

    # ---- 1. Build Model ----
    model = HerdNetTimmDLA(
        backbone=MODEL_CONFIG['backbone'],
        num_classes=MODEL_CONFIG['num_classes'],
        down_ratio=MODEL_CONFIG['down_ratio'],
        head_conv=MODEL_CONFIG['head_conv'],
        pretrained=MODEL_CONFIG['pretrained'],
        debug=True,
    )
    model.check_trainable_parameters()

    # ---- 2. Build Losses ----
    from animaloc.train.losses import FocalLoss
    losses = [
        {
            'idx': 0, 'idy': 0,
            'name': 'focal_loss',
            'lambda': 1.0,
            'loss': FocalLoss(
                alpha=LOSS_CONFIG['focal_alpha'],
                beta=LOSS_CONFIG['focal_beta'],
                reduction='mean',
            ),
        },
        {
            'idx': 1, 'idy': 1,
            'name': 'ce_loss',
            'lambda': 1.0,
            'loss': torch.nn.CrossEntropyLoss(
                weight=torch.tensor(LOSS_CONFIG['ce_weights'], dtype=torch.float32),
                reduction='mean',
            ),
        },
    ]
    wrapped_model = LossWrapper(model, losses, mode='module')

    # ---- 3. Build Optimizer with Differential Learning Rates ----
    param_groups = [
        {'params': model.backbone.parameters(), 'lr': TRAIN_CONFIG['backbone_lr']},
        {'params': model.loc_head.parameters(), 'lr': TRAIN_CONFIG['lr']},
        {'params': model.cls_head.parameters(), 'lr': TRAIN_CONFIG['lr']},
        {'params': model.dla_up.parameters(), 'lr': TRAIN_CONFIG['lr']},
        {'params': model.bottleneck_conv.parameters(), 'lr': TRAIN_CONFIG['lr']},
    ]
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=TRAIN_CONFIG['weight_decay'],
    )

    # ---- 4. Build Datasets ----
    import albumentations as A

    train_albu = [
        A.RandomCrop(height=512, width=512, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(p=1.0),
    ]

    val_albu = [A.Normalize(p=1.0)]

    train_end = [
        MultiTransformsWrapper([
            FIDT(
                num_classes=DATA_CONFIG['num_classes'],
                down_ratio=MODEL_CONFIG['down_ratio'],
                radius=2,
            ),
            # PointsToMask for classification head target
            # (add if using MultiTransformsWrapper)
        ]),
    ]

    val_end = [
        DownSample(
            down_ratio=MODEL_CONFIG['down_ratio'],
            anno_type='point',
        ),
    ]

    train_dataset = CSVDataset(
        csv_file=DATA_CONFIG['train_csv'],
        root_dir=DATA_CONFIG['train_dir'],
        albu_transforms=train_albu,
        end_transforms=train_end,
    )

    val_dataset = CSVDataset(
        csv_file=DATA_CONFIG['val_csv'],
        root_dir=DATA_CONFIG['val_dir'],
        albu_transforms=val_albu,
        end_transforms=val_end,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=True,
        num_workers=TRAIN_CONFIG['num_workers'],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Must be 1 for evaluation
        shuffle=False,
        num_workers=TRAIN_CONFIG['num_workers'],
    )

    # ---- 5. Build Evaluator ----
    metrics = Metrics(
        threshold=EVAL_CONFIG['matching_radius'],
        num_classes=DATA_CONFIG['num_classes'],
    )

    stitcher = HerdNetStitcher(
        model=wrapped_model,
        size=DATA_CONFIG['img_size'],
        overlap=EVAL_CONFIG['stitcher_overlap'],
        down_ratio=MODEL_CONFIG['down_ratio'],
        up=False,
        reduction='mean',
        device_name=str(device),
    )

    evaluator = HerdNetEvaluator(
        model=wrapped_model,
        dataloader=val_loader,
        metrics=metrics,
        lmds_kwargs={
            'kernel_size': EVAL_CONFIG['lmds_kernel'],
            'adapt_ts': EVAL_CONFIG['lmds_adapt_ts'],
            'scale_factor': 1,
            'up': False,  # False when using stitcher
        },
        device_name=str(device),
        print_freq=50,
        stitcher=stitcher,
        work_dir='./output',
        header='Validation',
    )

    # ---- 6. Build Trainer ----
    trainer = Trainer(
        model=wrapped_model,
        train_dataloader=train_loader,
        optimizer=optimizer,
        num_epochs=TRAIN_CONFIG['epochs'],
        auto_lr={'mode': 'max', 'patience': 15, 'min_lr': 1e-7},
        val_dataloader=val_loader,
        evaluator=evaluator,
        work_dir='./output',
        device_name=str(device),
        print_freq=20,
        valid_freq=1,
        early_stopping=TRAIN_CONFIG['early_stopping'],
        patience=TRAIN_CONFIG['patience'],
    )

    # ---- 7. Optional: Initialize W&B ----
    if WANDB_CONFIG['enabled'] and WANDB_AVAILABLE:
        wandb.init(
            project=WANDB_CONFIG['project'],
            entity=WANDB_CONFIG['entity'],
            name=WANDB_CONFIG['run_name'],
            config={**MODEL_CONFIG, **TRAIN_CONFIG, **LOSS_CONFIG, **EVAL_CONFIG},
        )

    # ---- 8. Train! ----
    print("\n" + "=" * 60)
    print("Starting training...")
    print(f"  Epochs: {TRAIN_CONFIG['epochs']}")
    print(f"  Batch size: {TRAIN_CONFIG['batch_size']}")
    print(f"  LR (head): {TRAIN_CONFIG['lr']}")
    print(f"  LR (backbone): {TRAIN_CONFIG['backbone_lr']}")
    print(f"  Matching radius: {EVAL_CONFIG['matching_radius']}px")
    print("=" * 60 + "\n")

    best_metric = trainer.start(
        wandb_flag=WANDB_CONFIG['enabled'] and WANDB_AVAILABLE,
        checkpoints='best',
        select_mode='max',
        validate_on='f1_score',
        returns='f1_score',
    )

    print(f"\nTraining complete! Best F1: {best_metric:.3f}")
    print(f"Best model saved to: ./output/best_model.pth")


if __name__ == '__main__':
    main()
