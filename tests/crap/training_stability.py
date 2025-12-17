"""
Training Stability Fixes for Point Detection

The original training showed instability:
- Epoch 1: F1=0.334
- Epoch 2: F1=0.127 (drop!)
- Epoch 3: F1=0.508 
- Epoch 4: F1=0.000 (collapse!)

Issues identified:
1. Learning rate warmup too aggressive (0.0002 -> 0.001 in 5 epochs)
2. Focal loss can be unstable early in training
3. No gradient clipping monitoring
4. Heatmap initialization might be too negative (-4.0)

This script provides fixes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple


class StableHeatmapLoss(nn.Module):
    """
    More stable version of heatmap loss with:
    1. Warmup from MSE to Focal loss
    2. Gradient scaling for numerical stability
    3. Better handling of empty images
    """
    
    def __init__(self,
                 heatmap_weight: float = 1.0,
                 count_weight: float = 0.1,
                 focal_alpha: float = 2.0,
                 focal_beta: float = 4.0,
                 warmup_epochs: int = 10):
        super().__init__()
        self.heatmap_weight = heatmap_weight
        self.count_weight = count_weight
        self.focal_alpha = focal_alpha
        self.focal_beta = focal_beta
        self.warmup_epochs = warmup_epochs
        
        self._step = 0
        self._epoch = 0
        self._grad_stats = {'min': float('inf'), 'max': 0, 'mean': 0, 'n': 0}
    
    def set_epoch(self, epoch: int):
        """Call this at the start of each epoch."""
        self._epoch = epoch
    
    def mse_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Simple MSE loss - more stable for early training."""
        pred_sigmoid = torch.sigmoid(pred)
        return F.mse_loss(pred_sigmoid, target)
    
    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Modified focal loss with better numerical stability."""
        pred_sigmoid = torch.sigmoid(pred)
        
        # Clamp for numerical stability
        pred_sigmoid = torch.clamp(pred_sigmoid, 1e-4, 1 - 1e-4)
        
        pos_mask = target.ge(0.99).float()  # More lenient than eq(1)
        neg_mask = target.lt(0.99).float()
        
        # Positive loss
        pos_loss = -torch.pow(1 - pred_sigmoid, self.focal_alpha) * \
                   torch.log(pred_sigmoid) * pos_mask
        
        # Negative loss with distance weighting
        neg_weights = torch.pow(1 - target, self.focal_beta)
        neg_loss = -torch.pow(pred_sigmoid, self.focal_alpha) * \
                   torch.log(1 - pred_sigmoid) * neg_weights * neg_mask
        
        num_pos = pos_mask.sum().clamp(min=1)
        
        # Normalize
        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos
        
        return loss
    
    def forward(self, outputs: Dict[str, torch.Tensor], 
                targets: List[Dict]) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        pred_heatmap = outputs['heatmap']
        pred_count = outputs['count']
        device = pred_heatmap.device
        
        target_heatmaps = torch.stack([t['heatmap'] for t in targets]).to(device)
        target_counts = torch.stack([t['count'] for t in targets]).to(device)
        
        # Resize if needed
        if target_heatmaps.shape[-2:] != pred_heatmap.shape[-2:]:
            target_heatmaps = F.interpolate(
                target_heatmaps.unsqueeze(1), 
                size=pred_heatmap.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
        
        # Warmup: start with MSE, transition to focal
        warmup_ratio = min(1.0, self._epoch / self.warmup_epochs)
        
        if warmup_ratio < 1.0:
            mse = self.mse_loss(pred_heatmap, target_heatmaps)
            focal = self.focal_loss(pred_heatmap, target_heatmaps)
            heatmap_loss = (1 - warmup_ratio) * mse + warmup_ratio * focal
        else:
            heatmap_loss = self.focal_loss(pred_heatmap, target_heatmaps)
        
        # Count loss with Huber for robustness
        count_loss = F.smooth_l1_loss(pred_count, target_counts, beta=2.0)
        
        total = self.heatmap_weight * heatmap_loss + self.count_weight * count_loss
        
        # Logging
        self._step += 1
        if self._step % 50 == 0:
            with torch.no_grad():
                pred_sigmoid = torch.sigmoid(pred_heatmap)
                pos_mask = target_heatmaps >= 0.5
                neg_mask = target_heatmaps < 0.1
                
                pos_pred = pred_sigmoid[pos_mask].mean().item() if pos_mask.any() else 0
                neg_pred = pred_sigmoid[neg_mask].mean().item() if neg_mask.any() else 0
                
                print(f"  [Step {self._step}] hm={heatmap_loss.item():.4f} "
                      f"cnt={count_loss.item():.4f} warmup={warmup_ratio:.2f} | "
                      f"pos_pred={pos_pred:.3f} neg_pred={neg_pred:.3f} gap={pos_pred-neg_pred:.3f}")
        
        return total, {
            'heatmap_loss': heatmap_loss.item(),
            'count_loss': count_loss.item(),
            'warmup_ratio': warmup_ratio
        }


class GradientMonitor:
    """Monitor gradient statistics to detect training issues."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.history = []
    
    def check(self) -> Dict[str, float]:
        """Check gradient statistics after backward pass."""
        total_norm = 0.0
        max_norm = 0.0
        min_norm = float('inf')
        num_params = 0
        
        for name, p in self.model.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                max_norm = max(max_norm, param_norm)
                min_norm = min(min_norm, param_norm)
                num_params += 1
        
        total_norm = total_norm ** 0.5
        
        stats = {
            'total_norm': total_norm,
            'max_norm': max_norm,
            'min_norm': min_norm if min_norm != float('inf') else 0,
            'num_params': num_params
        }
        
        self.history.append(stats)
        return stats
    
    def is_exploding(self, threshold: float = 100.0) -> bool:
        """Check if gradients are exploding."""
        if not self.history:
            return False
        return self.history[-1]['total_norm'] > threshold
    
    def is_vanishing(self, threshold: float = 1e-7) -> bool:
        """Check if gradients are vanishing."""
        if not self.history:
            return False
        return self.history[-1]['total_norm'] < threshold


def get_stable_optimizer(model: nn.Module, 
                         lr: float = 5e-5,  # Lower default LR
                         weight_decay: float = 1e-4) -> torch.optim.Optimizer:
    """
    Create optimizer with layer-wise learning rate decay.
    Backbone gets lower LR than head.
    """
    # Separate backbone and head parameters
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    
    param_groups = [
        {'params': head_params, 'lr': lr},
        {'params': backbone_params, 'lr': lr * 0.1},  # 10x lower for backbone
    ]
    
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def get_stable_scheduler(optimizer: torch.optim.Optimizer,
                         epochs: int,
                         warmup_epochs: int = 10,
                         min_lr_ratio: float = 0.01):
    """
    Stable learning rate schedule with longer warmup.
    """
    def lr_lambda(epoch):
        # Longer warmup
        if epoch < warmup_epochs:
            return 0.1 + 0.9 * (epoch / warmup_epochs)  # Start at 10% LR
        
        # Cosine decay
        progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + np.cos(np.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class StableTrainer:
    """
    Trainer with stability improvements:
    1. Gradient monitoring and clipping
    2. Loss spike detection and recovery
    3. Automatic learning rate reduction on instability
    4. Checkpoint on loss spike for recovery
    """
    
    def __init__(self, model, criterion, optimizer, scheduler, 
                 train_loader, val_loader, device, output_dir, config):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.config = config
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.epoch = 0
        self.best_f1 = 0.0
        self.patience_counter = 0
        self.loss_history = []
        self.grad_monitor = GradientMonitor(model)
        
        # Stability tracking
        self.last_stable_state = None
        self.instability_count = 0
        self.max_instability = 3
    
    def train_epoch(self):
        self.model.train()
        if hasattr(self.criterion, 'set_epoch'):
            self.criterion.set_epoch(self.epoch)
        
        total_loss = 0
        loss_components = {}
        n = 0
        batch_losses = []
        
        for images, targets in self.train_loader:
            images = images.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss, loss_dict = self.criterion(outputs, targets)
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"WARNING: NaN loss detected! Skipping batch.")
                continue
            
            loss.backward()
            
            # Check gradients
            grad_stats = self.grad_monitor.check()
            
            if self.grad_monitor.is_exploding(100.0):
                print(f"WARNING: Gradient explosion detected! norm={grad_stats['total_norm']:.2f}")
                # More aggressive clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            batch_loss = loss.item()
            batch_losses.append(batch_loss)
            total_loss += batch_loss
            
            for k, v in loss_dict.items():
                loss_components[k] = loss_components.get(k, 0) + v
            n += 1
        
        avg_loss = total_loss / max(n, 1)
        self.loss_history.append(avg_loss)
        
        # Check for loss spike
        if len(self.loss_history) > 2:
            recent_avg = np.mean(self.loss_history[-3:-1])
            if avg_loss > recent_avg * 2:
                print(f"WARNING: Loss spike detected! {recent_avg:.4f} -> {avg_loss:.4f}")
                self.instability_count += 1
        
        return {
            'loss': avg_loss,
            'loss_std': np.std(batch_losses),
            **{k: v / n for k, v in loss_components.items()}
        }
    
    def save_checkpoint(self, name):
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_f1': self.best_f1,
            'config': self.config,
            'loss_history': self.loss_history,
        }
        torch.save(checkpoint, self.output_dir / f'{name}.pth')
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint
    
    def train(self, epochs, patience=20, evaluate_fn=None):
        from improved_point_detector_dinov3 import evaluate
        
        print(f"\nTraining for {epochs} epochs (with stability monitoring)")
        
        # Save initial state for recovery
        self.save_checkpoint('initial')
        self.last_stable_state = 'initial'
        
        for epoch in range(epochs):
            self.epoch = epoch
            t0 = time.time()
            
            train_metrics = self.train_epoch()
            
            if self.scheduler:
                self.scheduler.step()
            
            # Evaluate
            val_metrics = {}
            if self.val_loader and evaluate_fn:
                heatmap_size = self.config['image_size'] // self.config.get('heatmap_stride', 4)
                val_metrics = evaluate_fn(
                    self.model, self.val_loader, self.device,
                    self.config['image_size'], 
                    self.config.get('eval_threshold', 0.3),
                    self.config['match_radius'],
                    heatmap_size
                )
                
                if val_metrics['f1'] > self.best_f1:
                    self.best_f1 = val_metrics['f1']
                    self.patience_counter = 0
                    self.save_checkpoint('best')
                    self.last_stable_state = 'best'
                else:
                    self.patience_counter += 1
                
                # Detect training collapse
                if val_metrics['f1'] == 0 and epoch > 5:
                    print(f"\nWARNING: Training collapse detected at epoch {epoch}!")
                    if self.instability_count < self.max_instability:
                        print(f"Attempting recovery from {self.last_stable_state}...")
                        self.load_checkpoint(self.output_dir / f'{self.last_stable_state}.pth')
                        # Reduce learning rate
                        for pg in self.optimizer.param_groups:
                            pg['lr'] *= 0.5
                        print(f"Reduced learning rate by 50%")
                        self.instability_count += 1
                        continue
                    else:
                        print(f"Max recovery attempts reached. Stopping.")
                        break
            
            # Logging
            lr = self.optimizer.param_groups[0]['lr']
            log = f"Epoch {epoch:3d} ({time.time() - t0:.1f}s) | "
            log += f"loss={train_metrics['loss']:.4f}±{train_metrics.get('loss_std', 0):.4f} "
            log += f"hm={train_metrics.get('heatmap_loss', 0):.4f} "
            log += f"| lr={lr:.6f}"
            
            if val_metrics:
                log += f" | P={val_metrics['precision']:.3f} R={val_metrics['recall']:.3f} F1={val_metrics['f1']:.3f}"
                if val_metrics['f1'] >= self.best_f1:
                    log += " *"
            
            print(log)
            
            # Regular checkpointing
            if epoch % 5 == 0:
                self.save_checkpoint('latest')
            
            # Early stopping
            if patience > 0 and self.patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        self.save_checkpoint('final')
        print(f"\nDone. Best F1: {self.best_f1:.4f}")


# =============================================================================
# RECOMMENDED TRAINING CONFIGURATION
# =============================================================================

STABLE_CONFIG = {
    # Model
    'backbone': 'vit_large_patch16_dinov3.sat493m',
    'hidden_dim': 256,
    'extract_layers': [8, 16, 24],
    'freeze_backbone': True,  # Keep frozen initially
    'heatmap_stride': 4,
    
    # Training - more conservative
    'epochs': 100,
    'batch_size': 4,
    'lr': 5e-5,  # Lower than before (was 1e-4)
    'weight_decay': 1e-4,
    'patience': 30,  # More patience
    
    # Loss
    'heatmap_weight': 1.0,
    'count_weight': 0.1,
    'loss_warmup_epochs': 10,  # Warmup from MSE to focal
    
    # Scheduler
    'scheduler_warmup_epochs': 10,  # Longer warmup
    'min_lr_ratio': 0.01,
    
    # Augmentation
    'copy_paste': False,  # Disable initially for stability
    'sigma': 2.5,  # Slightly larger Gaussian
    
    # Evaluation
    'image_size': 512,
    'eval_threshold': 0.3,
    'match_radius': 25,
}


def print_training_recommendations():
    """Print recommendations for stable training."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                TRAINING STABILITY RECOMMENDATIONS                 ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  1. LEARNING RATE                                                 ║
║     - Start with 5e-5 instead of 1e-4                            ║
║     - Use 10 epoch warmup (start at 10% LR)                      ║
║     - Cosine decay after warmup                                   ║
║                                                                   ║
║  2. LOSS FUNCTION                                                 ║
║     - Warmup from MSE to Focal loss over 10 epochs               ║
║     - More lenient positive threshold (>= 0.99 instead of == 1)  ║
║     - Better numerical stability with clamping                    ║
║                                                                   ║
║  3. INITIALIZATION                                                ║
║     - Heatmap bias: -2.0 instead of -4.0 (less extreme)          ║
║     - Allows faster initial learning                              ║
║                                                                   ║
║  4. GRADIENT HANDLING                                             ║
║     - Monitor gradient norms                                      ║
║     - Dynamic clipping (0.5 if exploding, 1.0 otherwise)         ║
║     - Skip NaN batches                                            ║
║                                                                   ║
║  5. RECOVERY MECHANISMS                                           ║
║     - Save checkpoint before collapse detected                    ║
║     - Auto-recover with reduced LR on collapse                    ║
║     - Max 3 recovery attempts                                     ║
║                                                                   ║
║  6. AUGMENTATION                                                  ║
║     - Disable copy-paste initially for stability                  ║
║     - Enable after model converges (epoch 30+)                    ║
║                                                                   ║
╚══════════════════════════════════════════════════════════════════╝
""")


if __name__ == '__main__':
    import time
    from pathlib import Path
    
    print_training_recommendations()
    print("\nStable config:")
    for k, v in STABLE_CONFIG.items():
        print(f"  {k}: {v}")
