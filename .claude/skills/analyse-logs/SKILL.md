---
name: analyse-logs
description: Deep analysis of training logs — loss trends, convergence, anomalies, and recommendations. Use when the user wants to understand training results or compare runs.
argument-hint: [log file or directory path]
---

# Analyse Training Logs

Perform a thorough analysis of training logs and present actionable findings.

## Steps

1. Find log files:
   - If `$0` is a directory, look for `*_training.log` and `*_validation.log` inside it
   - If `$0` is a file, use it directly
   - Otherwise find all logs in the current directory and subdirectories
2. Spawn an agent to read the full log(s) and analyse:
   - **Loss curve**: Did training loss decrease consistently? Any spikes or plateaus?
   - **Validation metrics**: f1_score, recall, precision, MAE per epoch. Which epoch was best?
   - **Convergence**: Did the model converge? Was early stopping triggered?
   - **Learning rate**: Was auto_lr reduction triggered? At which epoch?
   - **Anomalies**: NaN values, loss explosions, memory warnings
   - **Timing**: Time per epoch, total training time
3. Present a concise summary:
   - Best epoch and its metrics
   - Whether training converged or needs more epochs
   - Specific recommendations (adjust LR, more data, different augmentation, etc.)
4. If multiple runs are available, compare them side by side.
