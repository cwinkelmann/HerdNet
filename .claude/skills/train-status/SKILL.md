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
2. Find the log file:
   - If `$0` is given, use that path
   - Otherwise find the most recent: `ls -t *_training.log | head -1`
3. Read the last 50 lines of the log file.
4. Summarize concisely:
   - Running or finished?
   - Current epoch / total epochs
   - Latest loss values (focal_loss, ce_loss)
   - Latest validation metrics (f1_score, recall, precision, MAE)
   - Any errors or warnings
5. If a `*_validation.log` exists alongside, read its tail too for evaluation details.
