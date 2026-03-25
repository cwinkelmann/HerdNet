---
name: read-logs
description: Read and display training or validation log files. Use when the user asks to see logs, check output, or review what happened during training.
argument-hint: [optional log file path or glob pattern]
---

# Read Logs

Find and display log contents from training runs.

## Steps

1. Find log files:
   - If `$0` is given, use that path directly
   - Otherwise list all available logs: `ls -lt *_training.log *_validation.log 2>/dev/null` and also check subdirectories like `best_models/*/`, `tools/`
2. If multiple logs found, show the user the list and ask which one, or default to the most recent.
3. Read the log file:
   - If the file is short (<200 lines), show all of it
   - If long, show the first 20 lines (header/config) and last 50 lines (final results)
4. Highlight key information:
   - Final loss values
   - Best validation metrics
   - Errors or warnings
   - Training duration
