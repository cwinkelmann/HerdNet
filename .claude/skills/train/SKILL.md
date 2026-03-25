---
name: train
description: Start a HerdNet training run in the background with Hydra config overrides
argument-hint: [config-name] [key=value overrides...]
---

# Train HerdNet

Run a training job in the background so the user is not blocked.

## Steps

1. If `$0` is provided, use it as the config name. Otherwise ask the user which config to use (list available ones from `configs/demo/` or `configs/submission/`).
2. Run training in the background:
   ```
   python tools/train.py --config-name $0 $ARGUMENTS[1:]
   ```
   Use `run_in_background: true` on the Bash tool.
3. Tell the user the job is running and that logs go to `*_training.log` in the Hydra output directory.
4. Wait ~30 seconds, then find the latest log file (`ls -t *_training.log | head -1`) and read its last 30 lines to confirm the job started successfully.
5. If errors appear, report them immediately.

## Example

`/train dla34_timm wandb_flag=False training_settings.epochs=5`