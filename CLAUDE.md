# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM training pipeline for fine-tuning transformer models on cloud GPUs using HuggingFace Transformers, PEFT (LoRA), and TRL (SFTTrainer). Primary focus is training models with Socratic teaching behavior (teaching via guided questions rather than direct answers).

## Commands

### Environment Setup
```bash
./cloud_run.sh                    # First-time setup + training run
FORCE_UPDATE=true ./cloud_run.sh  # Force conda env update when dependencies change
```

### Training
```bash
python -m scripts.lora_train --config configs/<config>.yaml --proc <num_workers>
python -m scripts.lora_train --config configs/phi35qa_lora_socratic.yaml --proc 10 --resume  # Resume from checkpoint
python -m scripts.lora_train --config configs/phi35qa_lora_socratic.yaml --checkpoint_path output/checkpoint-500  # Resume from specific checkpoint
python -m scripts.lora_train --config configs/phi35qa_lora_socratic.yaml --validate-batch  # Validate label masking before training
```

Args:
- `--config`: Path to YAML config file (required)
- `--proc`: Number of parallel workers for dataset preprocessing (default: 1)
- `--resume`: Resume from latest checkpoint in output_dir
- `--checkpoint_path`: Resume from a specific checkpoint path (takes priority over `--resume`)
- `--validate-batch`: Run batch structure validation before training to catch label masking issues

### Evaluation
```bash
python scripts/eval_metrics.py --model_path <model> --eval_json <path> --batch_size 4
```

### Validation
```bash
./validation_run.sh  # Runs phi2_validation_script comparing checkpoint vs baseline
```

### Benchmarking
```bash
# Run training across multiple configs (auto-enables profiling, overrides max_steps)
python -m benchmarks.run_benchmark --configs configs/a.yaml configs/b.yaml --max_steps 50

# Compare experiment manifests side-by-side (produces a markdown table)
python -m benchmarks.compare output/run_a/experiment_manifest.json output/run_b/experiment_manifest.json
```

### Tests
The test environment is the `transformer_llm` conda env. The `KMP_DUPLICATE_LIB_OK=TRUE` flag is required to work around a pre-existing OpenMP conflict in that environment.

```bash
# Run all tests
conda run -n transformer_llm env KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/ -v

# Run a specific test file
conda run -n transformer_llm env KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/test_preprocessing.py -v

# Run tests matching a pattern
conda run -n transformer_llm env KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/ -k "test_name" -v
```

> Note: plain `pytest tests/` or `python -m pytest` without the conda wrapper will fail — the `src` package is only importable inside the `transformer_llm` environment.

## Architecture

### Config System
YAML configs in `configs/` define model, data, tokenizer, training, logging, and LoRA settings. Configs are loaded via `src/core/config.py` and parsed into dataclasses:
- `src/config/model_config.py` - Model name, kind (qa/clm), attention implementation
- `src/config/training_config.py` - Batch size, LR, epochs, precision, gradient settings
- `src/config/lora_config.py` - LoRA rank, alpha, dropout, target modules
- `src/config/logging_config.py` - TensorBoard, save strategy, logging steps

### Data Pipeline
Preprocessors in `src/data/` registered via `PREPROCESSOR_REGISTRY` in `data_utils.py`:
- `SocraticPreprocessor` - Converts junior/senior engineer turns to user/assistant messages with policy-based system prompts
- `ELI5Preprocessor_QA/CLM` - ELI5 dataset formatting
- `SQuADPreprocessor` - SQuAD QA formatting

Preprocessors return `{"messages": [...]}` for SFTTrainer's chat format. The model loader (`src/models/model_loader.py`) injects a custom chat template with `{% generation %}` tags for `assistant_only_loss=True`.

**Socratic Dataset Format** (JSON):
```json
{
  "turns": [
    {"role": "junior_engineer", "content": "..."},
    {"role": "senior_engineer", "content": "..."}
  ],
  "policy": {
    "withhold_final_answer": true,
    "avoid_numeric_rules_of_thumb": true
  }
}
```
Role mapping: `junior_engineer` → `user`, `senior_engineer` → `assistant`

### Model Loading
`src/models/model_loader.py` supports two model kinds:
- `qa` - AutoModelForQuestionAnswering (span extraction)
- `clm` - AutoModelForCausalLM (decoder-only, used for Socratic training)

Adds custom role tokens (`<|system|>`, `<|user|>`, `<|assistant|>`) and resizes embeddings.

### Metrics Evaluation
`metrics_eval/` contains behavior-focused evaluation (Socratic style):
- `evaluator.py` - Generates outputs and computes metrics (question rate, withhold violation, numeric violation)
- `heuristics.py` - Pattern matching for violations and question counting
- `callback.py` - `MetricsEvalCallback` for integration with Trainer

Key metrics tracked:
- `withhold_violation_rate` - Responses that reveal final answers
- `numeric_violation_rate` - Responses with numeric rules-of-thumb
- `question_rate` - Fraction of responses containing questions
- `avg_questions_per_response` - Socratic question density

### Profiling
Profiling is opt-in via a `profiling:` section in the config. When enabled, `ProfilingCallback` (`src/profiling/callback.py`) hooks into the Trainer lifecycle to collect per-step CUDA/wall-clock timings, GPU memory snapshots, and throughput (samples/sec, tokens/sec). It can also emit Chrome/Perfetto traces via `torch.profiler`.

At training end, `save_experiment_manifest` (`src/utils/experiment.py`) writes `experiment_manifest.json` to `output_dir` — combining the config, HF Trainer metrics, raw profiling data, and computed summary stats (avg/median/P95 step time, peak GPU memory).

### DeepSpeed
DeepSpeed is opt-in via a `deepspeed:` config section. Pre-built ZeRO stage 1/2/3 configs live in `configs/deepspeed/`. When `enabled: true`, `DeepSpeedConfig.resolve_config_path()` maps `zero_stage` to the matching JSON file (or accepts an explicit `config_path`).

## Key Implementation Details

- Training uses `SFTTrainer` from TRL with `assistant_only_loss=True` to compute loss only on assistant responses
- LoRA applied via PEFT for efficient fine-tuning (common targets: q_proj, v_proj, k_proj, o_proj)
- Supports bf16/fp16 precision with automatic fallback on unsupported GPUs (Ampere+ required for bf16)
- Gradient checkpointing enabled for large models
- Left-padding for batched generation
- Custom chat template uses `{% generation %}{% endgeneration %}` tags to mark assistant content for loss masking
- Batch validation on startup catches misconfigured label masking before training begins

## Config File Structure

```yaml
model:
  name: microsoft/Phi-3.5-mini-instruct
  kind: clm                    # "clm" or "qa"
  attn_implementation: sdpa

data:
  name: json
  data_file: data/custom_dataset/final_data.json
  preprocessor: SocraticPreprocessor
  seed: 42

tokenizer:
  max_length: 4096

training:
  batch_size: 4
  lr: 1.0e-4
  epochs: 10
  output_dir: output/run/
  precision: bf16              # "fp32" | "fp16" | "bf16"
  gradient_checkpointing: true
  gradient_accumulation_steps: 8
  eval_strategy: steps         # "epoch" | "steps" | false → "no"
  eval_steps: 150
  max_steps: 12000             # -1 to use epochs instead
  lr_scheduler_type: cosine
  warmup_ratio: 0.03
  early_stopping_patience: null  # omit or null to disable

lora:
  rank: 16
  lora_alpha: 32
  lora_dropout: 0.05
  bias: none
  task_type: CAUSAL_LM
  target_modules: null         # defaults to q_proj, v_proj if null

logging:
  report_to: tensorboard
  logging_dir: ./runs
  logging_steps: 10
  save_steps: 250
  save_strategy: steps

# Optional: enable profiling
profiling:
  enabled: false
  log_every_n_steps: 10
  trace_dir: null              # required for torch.profiler traces
  enable_torch_profiler: false
  profiler_start_step: 20
  profiler_end_step: 40

# Optional: enable DeepSpeed
deepspeed:
  enabled: false
  zero_stage: 2                # 1, 2, or 3 — maps to configs/deepspeed/ds_z{n}.json
  config_path: null            # override with explicit path if needed
```

## Adding a New Preprocessor

1. Create preprocessor class in `src/data/` implementing `__call__(self, examples) -> {"messages": [...]}`
2. Register in `PREPROCESSOR_REGISTRY` in `src/data/data_utils.py`
3. Reference by name in config: `data.preprocessor: YourPreprocessor`
