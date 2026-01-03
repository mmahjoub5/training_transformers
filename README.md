# training_transformers
Setting up robust pipeline to train LLMs on cloud GPUs

## Quick Start

### Environment Setup

The project uses conda for environment management. The `cloud_run.sh` script handles environment setup automatically.

**First-time setup:**
```bash
./cloud_run.sh
```

**Subsequent runs** (skips env creation/update for faster startup):
```bash
./cloud_run.sh
```

**Force environment update** (when dependencies change):
```bash
FORCE_UPDATE=true ./cloud_run.sh
```

### How It Works

- **First run**: Creates the `llm_env` conda environment from `environment.yml`
- **Subsequent runs**: Skips environment creation/update for faster execution
- **Force update**: Set `FORCE_UPDATE=true` to update dependencies when needed

This optimization saves significant time by avoiding unnecessary environment updates on every run.

## Evaluation Metrics

Behavior-focused evaluation (Socratic style, no final answers, avoid numeric rules-of-thumb) lives in `metrics_eval/`.

Run standalone metrics:
```bash
python scripts/eval_metrics.py --model_path microsoft/Phi-3.5-mini-instruct --eval_json /path/to/eval_data.json --batch_size 4
```

Plug into training with a callback:
```python
from metrics_eval.callback import MetricsEvalCallback

metrics_callback = MetricsEvalCallback(
    eval_json="/path/to/eval_data.json",
    eval_steps=200,
    max_samples=200,
    batch_size=4,
)
trainer.add_callback(metrics_callback)
```
