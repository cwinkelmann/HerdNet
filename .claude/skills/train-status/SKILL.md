---
name: train-status
description: Check the status and progress of a running or completed training job
argument-hint: [optional log file path]
---

# Training Status

Check whether training is running and report current progress.

## Steps

1. Check if a training process is active:
   ```
   ps aux | grep 'tools/train.py' | grep -v grep
   ```
2. Find the output directory:
   - If `$0` is given, use that path as the log file and look for sibling logs
   - Otherwise find the most recent Hydra output dir: `ls -dt output/*/202*/*/*/ | head -1`
   - The key log file is the **loguru training log** (`YYYYMMDD_training.log`) inside the output dir — it contains aggregate F1 scores
3. Extract the **aggregate F1 table** from the loguru training log:
   - Search for lines matching: `f1_score: <value>` (logged by the trainer after each validation epoch)
   - Also extract "Best model" save lines to show which epoch was best
   - Present as a markdown table with columns: **Epoch | F1 Score | Val Loss | Best?**
4. Read the last 20 lines of `train.log` (Hydra log) for current training progress (epoch, step, loss).
5. Summarize concisely:
   - Running or finished?
   - Current epoch / total epochs
   - Latest training loss (focal_loss, ce_loss)
   - **F1 score progression table** (from all validation epochs so far)
   - Current best F1 and which epoch
   - Any errors or warnings
6. If multiple training runs are queued, report their status too.
