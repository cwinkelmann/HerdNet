# Training Agent

## Role
Guide users through HerdNet training, including Trainer setup, LossWrapper configuration, learning rate management, W&B integration, log interpretation, and failure diagnosis.

## Training Architecture

The training pipeline follows this flow:

```
Config (Hydra)
    |
    +--> Build CSVDataset (train + val)
    +--> Build Model (HerdNetTimmDLA)
    +--> Build Losses (FocalLoss + CrossEntropyLoss)
    +--> Wrap with LossWrapper
    +--> Build Optimizer (AdamW with differential LR)
    +--> Build Evaluator (HerdNetEvaluator)
    +--> Create Trainer
    +--> trainer.start(wandb_flag=True)
```

Source: `animaloc/utils/train.py` -- `main()` function

## LossWrapper

Source: `animaloc/models/utils.py`

LossWrapper wraps the model and its loss functions into a single `nn.Module`:

```python
LossWrapper(
    model: nn.Module,       # The HerdNet model
    losses: list,           # List of loss dicts
    mode: str = 'module'    # 'module' = loss only during train, output+loss during eval
)
```

### Loss Dict Format
Each loss in the list is a dict with:
```python
{
    'idx': 0,              # output_idx: which model output this loss applies to
    'idy': 0,              # target_idx: which target tensor this loss applies to
    'name': 'focal_loss',  # Print name for logging
    'lambda': 1.0,         # Loss weighting factor (lambda_const)
    'loss': FocalLoss(...)  # The actual loss module
}
```

### Mode Behavior
- `'module'` (default): During `model.train()`, returns only loss dict. During `model.eval()`, returns `(output, loss_dict)`.
- `'loss_only'`: Always returns only loss dict
- `'preds_only'`: Always returns only predictions
- `'both'`: Always returns `(output, loss_dict)`

## Trainer Class

Source: `animaloc/train/trainers.py`

```python
Trainer(
    model: nn.Module,                    # LossWrapper-wrapped model
    train_dataloader: DataLoader,
    optimizer: Optimizer,
    num_epochs: int,
    lr_milestones: list = None,          # Epoch indices for LR step decay
    auto_lr: bool | dict = False,        # ReduceLROnPlateau config
    adaloss: str = None,                 # Adaptive loss parameter name
    val_dataloader: DataLoader = None,
    val_loss_dataloader: DataLoader = None,
    evaluator: Evaluator = None,         # HerdNetEvaluator for validation
    vizual_fn: callable = None,
    work_dir: str = None,
    device_name: str = 'cuda',
    print_freq: int = 50,
    valid_freq: int = 1,                 # Validate every N epochs
    csv_logger: bool = False,
    early_stopping: bool = False,
    patience: int = 10,
    min_delta: float = 0.0,
    restore_best_weights: bool = True,
    wandb_artifact_upload: bool = False,
)
```

### Starting Training
```python
trainer.start(
    wandb_flag: bool = False,       # Log to W&B
    checkpoints: str = 'best',      # 'best', 'all', or None
    select_mode: str = 'max',       # 'max' for F1 (higher is better)
    validate_on: str = 'f1_score',  # Metric to select best checkpoint
    returns: str = 'f1_score',      # Metric to return from evaluation
    viz: bool = False,              # Save visualizations
)
```

## Optimizer Configuration

### Differential Learning Rates
HerdNet uses different learning rates for the backbone and heads:

```python
# From the training utility
param_groups = [
    {'params': model.backbone.parameters(), 'lr': backbone_lr},  # e.g., 1e-6
    {'params': model.loc_head.parameters(), 'lr': head_lr},      # e.g., 1e-4
    {'params': model.cls_head.parameters(), 'lr': head_lr},      # e.g., 1e-4
    {'params': model.dla_up.parameters(), 'lr': head_lr},        # e.g., 1e-4
    {'params': model.bottleneck_conv.parameters(), 'lr': head_lr},
]

optimizer = torch.optim.AdamW(param_groups, weight_decay=3.25e-4)
```

### Optimal Hyperparameters (Miesner 2025)
| Parameter | Optimal | Default | Notes |
|-----------|---------|---------|-------|
| lr (head) | 1e-4 | 1e-3 | Lower for stability |
| backbone_lr | 1e-6 | 1e-5 | Much lower to preserve pretrained features |
| weight_decay | 3.25e-4 | 1.6e-4 | Stronger regularization |
| warmup_iters | 100 | 0 | Prevents early NaN |
| batch_size | 2-5 | - | Depends on GPU memory |
| epochs | 20 | - | Convergence typically at 8-11 |

## Warmup

Linear warmup ramps the learning rate from 0 to `lr` over `warmup_iters` iterations (not epochs). This prevents gradient explosion in the early training phase.

```
LR
|     _______________
|    /
|   /
|  /
| /
|/__________________ iterations
 0  warmup_iters
```

The warmup implementation is in the `Trainer._warmup()` method. It linearly interpolates:
```python
lr = base_lr * (current_iter / warmup_iters)
```

**Recommendation**: Use `warmup_iters=100` for most cases. Increase to 1500 for larger datasets or higher base learning rates.

## Learning Rate Scheduling

### ReduceLROnPlateau (auto_lr)
```yaml
auto_lr:
  mode: 'max'          # 'max' when monitoring F1 (higher is better)
  patience: 15         # Wait N epochs before reducing LR
  threshold: 1e-4      # Minimum change to qualify as improvement
  threshold_mode: 'rel' # Relative threshold
  cooldown: 10         # Wait N epochs after reduction before checking again
  min_lr: 1e-7         # Don't reduce below this
  verbose: True
```

### MultiStepLR (lr_milestones)
Alternative: specify epoch indices where LR is multiplied by 0.1:
```python
lr_milestones: [10, 15, 18]  # LR drops at epochs 10, 15, 18
```

## W&B Integration

The training script integrates with Weights & Biases for experiment tracking:

```yaml
# In main.yaml
wandb_flag: True
wandb_project: 'herdnet'
wandb_entity: 'your_team'
wandb_run: 'experiment_name'
wandb_tags: ["dla34", "floreana", "dr4"]
```

### What Gets Logged
- **Per-iteration**: Training loss (focal_loss + ce_loss + total_loss)
- **Per-epoch validation**: tp, fp, fn, recall, precision, f1_score, f2_score, MAE, ME, MSE, RMSE, avg_score, avg_dscore
- **Run summary**: Final aggregated metrics, mAP
- **Config**: Full Hydra config is logged to W&B config panel

### Reading W&B Logs
Key metrics to monitor:
1. **total_loss**: Should decrease steadily. If NaN, check LR.
2. **f1_score**: Primary metric. Should increase. Plateau = convergence.
3. **recall vs precision**: If recall >> precision, too many false positives. If precision >> recall, too many false negatives.
4. **MAE / RMSE**: Counting error metrics. Lower is better.

## Training Log Interpretation

### Healthy Training Run
```
Epoch 1: total_loss=0.45, f1=0.12  -- Model learning basic features
Epoch 3: total_loss=0.22, f1=0.55  -- Rapid improvement
Epoch 5: total_loss=0.15, f1=0.78  -- Good progress
Epoch 8: total_loss=0.11, f1=0.89  -- Approaching convergence
Epoch 11: total_loss=0.10, f1=0.93 -- Converged, near optimal
Epoch 15: total_loss=0.09, f1=0.93 -- Stable, can stop
```

### Overfitting Pattern
```
Epoch 5: train_loss=0.12, val_f1=0.80
Epoch 8: train_loss=0.06, val_f1=0.82  -- Still improving
Epoch 11: train_loss=0.02, val_f1=0.78 -- val_f1 dropping = OVERFITTING
```
Fix: Switch to DLA-34, reduce head_conv to 64, add weight_decay.

## Early Stopping

```python
Trainer(
    ...
    early_stopping=True,
    patience=10,              # Stop after 10 epochs without improvement
    min_delta=0.0,            # Minimum change to count as improvement
    restore_best_weights=True # Load best checkpoint when stopping
)
```

## Gradient Clipping

Recommended for stability:
```python
# In training loop
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## Checkpointing

The Trainer saves checkpoints based on the `checkpoints` parameter:
- `'best'`: Only save when validation metric improves
- `'all'`: Save after every validation epoch
- `None`: Don't save checkpoints

Checkpoint files are saved in `work_dir` (or Hydra output directory).

## Common Training Failures and Diagnosis

### Loss is NaN
**Symptoms**: Loss becomes NaN within first few iterations
**Causes**:
1. Learning rate too high (most common)
2. No warmup period
3. FIDT targets contain invalid values (extremely rare)
4. Numerical overflow in FocalLoss

**Fix**:
```yaml
training_settings:
  lr: 1.0e-4          # Reduce from 1e-3
  warmup_iters: 100    # Add warmup
```

### F1 Stuck at 0
**Symptoms**: F1 score is 0.0 for all epochs, but loss is decreasing
**Causes**:
1. **matching_radius too small** (most common): Default is 25px, but iguanas need 75px
2. LMDS parameters wrong: adapt_ts too high, kernel_size too small
3. down_ratio mismatch between model and evaluator

**Fix**:
```yaml
evaluator:
  threshold: 75                  # Was 25 -- increase matching radius
  kwargs:
    lmds_kwargs:
      kernel_size: [5, 5]       # Was [3, 3]
      adapt_ts: 0.5             # Was 0.3
```

### Loss Not Decreasing
**Symptoms**: Loss stays flat or oscillates
**Causes**:
1. down_ratio mismatch between model and FIDT transform
2. Learning rate too low
3. Data loading issue (images not matching annotations)

**Fix**: Check that `${model.kwargs.down_ratio}` resolves correctly in FIDT config. Print resolved config with `--cfg job`.

### Overfitting
**Symptoms**: Training loss decreases, validation F1 drops after early peak
**Causes**:
1. Backbone too large (DLA-60/102/169)
2. head_conv too large
3. Too many epochs
4. Insufficient augmentation

**Fix**:
```yaml
model:
  kwargs:
    backbone: 'timm/dla34'    # Not dla60+
    head_conv: 64              # Not 128
training_settings:
  weight_decay: 3.25e-4       # Increase regularization
  epochs: 15                   # Reduce from 100
```

### Out of Memory
**Symptoms**: CUDA OOM error
**Fix**:
1. Increase `down_ratio` to 4 (from 2)
2. Reduce `batch_size` to 1 or 2
3. Reduce `num_workers`
4. Use DLA-34 (not larger backbones)

### Slow Convergence
**Symptoms**: F1 improves very slowly, needs many epochs
**Causes**:
1. No warmup (optimizer overshoots early, then slowly recovers)
2. backbone_lr same as head_lr
3. No pretrained weights

**Fix**: Add warmup, use differential LR (backbone_lr << head_lr), enable pretrained=True.

## Complete Training Script (Minimal)

```python
import torch
from torch.utils.data import DataLoader
from animaloc.models import HerdNetTimmDLA
from animaloc.models.utils import LossWrapper
from animaloc.datasets import CSVDataset
from animaloc.train.trainers import Trainer

# 1. Model
model = HerdNetTimmDLA(
    backbone='timm/dla34', num_classes=3,
    down_ratio=4, head_conv=64, pretrained=True
)

# 2. Losses
losses = [
    {'idx': 0, 'idy': 0, 'name': 'focal', 'lambda': 1.0,
     'loss': FocalLoss(alpha=2, beta=5, reduction='mean')},
    {'idx': 1, 'idy': 1, 'name': 'ce', 'lambda': 1.0,
     'loss': torch.nn.CrossEntropyLoss(
         weight=torch.tensor([0.1, 4.0, 1.0]), reduction='mean')},
]
wrapped_model = LossWrapper(model, losses)

# 3. Optimizer with differential LR
params = [
    {'params': model.backbone.parameters(), 'lr': 1e-6},
    {'params': model.loc_head.parameters(), 'lr': 1e-4},
    {'params': model.cls_head.parameters(), 'lr': 1e-4},
    {'params': model.dla_up.parameters(), 'lr': 1e-4},
    {'params': model.bottleneck_conv.parameters(), 'lr': 1e-4},
]
optimizer = torch.optim.AdamW(params, weight_decay=3.25e-4)

# 4. Trainer
trainer = Trainer(
    model=wrapped_model,
    train_dataloader=train_loader,
    optimizer=optimizer,
    num_epochs=20,
    val_dataloader=val_loader,
    evaluator=evaluator,
    device_name='cuda',
    early_stopping=True,
    patience=10,
)

# 5. Start
trainer.start(wandb_flag=True, checkpoints='best',
              select_mode='max', validate_on='f1_score')
```
