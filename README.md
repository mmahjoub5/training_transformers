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
