# Training Transformers — Hardware-Aware Performance Analysis for LoRA Fine-Tuning

[![CI](https://github.com/mmahjoub5/training_transformers/actions/workflows/ci.yml/badge.svg)](https://github.com/mmahjoub5/training_transformers/actions/workflows/ci.yml)

A systems-focused project for measuring and optimizing the performance of LoRA-based LLM fine-tuning on cloud GPUs. Built on top of a Socratic teaching pipeline (Phi-3.5 + SFTTrainer + PEFT), the focus is on **performance instrumentation** — GPU profiling, memory analysis, DeepSpeed integration, and systematic benchmarking — not model accuracy.

## Project Structure

```
src/
  config/                        # Dataclass configs parsed from YAML
    training_config.py
    profiling_config.py          # ProfilingConfig — toggle profiling, torch.profiler, trace dir
    deepspeed_config.py          # DeepSpeedConfig — ZeRO stage selection

  profiling/                     # GPU/CPU performance instrumentation
    timer.py                     # CudaTimer — CUDA event-based timing, CPU fallback
    memory.py                    # GPU memory snapshots (allocated, reserved, peak, free)
    system_info.py               # Hardware/software environment capture
    callback.py                  # ProfilingCallback — TrainerCallback for per-step metrics

  training/                      # Trainer construction
    trainer_factory.py           # build_training_args(), build_callbacks()

  kernels/                       # Extension point for Triton kernel experiments
  utils/
    experiment.py                # ExperimentManifest — save/load/summary

configs/
  deepspeed/                     # DeepSpeed JSON configs (ZeRO 1/2/3)

benchmarks/
  run_benchmark.py               # Run configs with profiling, fixed step count
  compare.py                     # Compare experiment manifests side-by-side
```

## Quick Start

### Environment Setup

```bash
./cloud_run.sh                    # First-time setup + training run
FORCE_UPDATE=true ./cloud_run.sh  # Force conda env update when dependencies change
```

### Training

```bash
python -m scripts.lora_train --config configs/<config>.yaml --proc <num_workers>
```

### Training with Profiling

Add to any YAML config:
```yaml
profiling:
  enabled: true
  log_every_n_steps: 10
  enable_torch_profiler: false    # set true for Chrome trace generation
```

This produces an `experiment_manifest.json` with per-step timings, memory snapshots, throughput, and a computed summary (avg/median/p95 step time, peak memory, samples/sec, tokens/sec).

### Training with DeepSpeed

```yaml
deepspeed:
  enabled: true
  zero_stage: 2                   # 1, 2, or 3
```

### Benchmarking

Run multiple configs and compare:
```bash
python -m benchmarks.run_benchmark \
    --configs configs/phi35_baseline.yaml configs/phi35_ds_z2.yaml \
    --max_steps 50

python -m benchmarks.compare \
    experiments/phi35_baseline/experiment_manifest.json \
    experiments/phi35_ds_z2/experiment_manifest.json
```

Example output:
```
| Metric               | phi35_baseline | phi35_ds_z2 |
| ---                  | ---            | ---         |
| Avg Step Time (ms)   | 847.3          | 912.1       |
| Peak GPU Memory (MB) | 14231          | 9842        |
| Avg Samples/sec      | 38.2           | 35.4        |
| Avg Tokens/sec       | 19264          | 17890       |
| DeepSpeed            | off            | ZeRO-2      |
```

## What's Measured

- **Per-step GPU timing** via CUDA events (no pipeline bubbles — sync only on read)
- **Per-step wall-clock timing** via `time.perf_counter()`
- **GPU memory** — allocated, reserved, peak, free (snapshots every N steps)
- **Throughput** — samples/sec and tokens/sec
- **System info** — GPU name/memory/compute capability, CUDA/cuDNN/driver versions, library versions
- **Optional torch.profiler traces** — bounded window, Chrome-viewable via TensorBoard

## Evaluation Metrics

Behavior-focused evaluation (Socratic style) lives in `metrics_eval/`.

```bash
python scripts/eval_metrics.py --model_path <model> --eval_json <path> --batch_size 4
```

## Tests

```bash
pytest tests/
```

## Config File Schema

YAML configs in `configs/` are parsed into typed dataclasses. The top-level sections (required unless noted):

| Section      | Purpose                                                       | Loaded into                       |
| ------------ | ------------------------------------------------------------- | --------------------------------- |
| `model`      | Model name, kind (`qa`/`clm`), adapter, attention impl        | `ModelConfig`                     |
| `data`       | Dataset name/file, preprocessor name, split/sample limits     | `DataConfig`                      |
| `tokenizer`  | `max_length`                                                  | `TokenizerConfig`                 |
| `training`   | Batch size, LR, epochs, precision, eval/save strategy         | `TrainingConfig`                  |
| `lora`       | (optional) LoRA rank, alpha, dropout, target_modules          | `LoraConfigSpec`                  |
| `logging`    | (optional) Tensorboard, save_steps, logging_steps             | `LoggingConfig`                   |
| `deepspeed`  | (optional) Enable + ZeRO stage                                | `DeepSpeedConfig`                 |
| `profiling`  | (optional) Enable + torch.profiler + log frequency            | `ProfilingConfig`                 |

## Adding a New Preprocessor

Preprocessors transform raw dataset rows into `{"messages": [...]}` for SFTTrainer.

1. Create a class in `src/data/` that implements `__call__(self, example: dict) -> dict`.
2. Make sure it satisfies the `BasePreprocessor` Protocol from `src/data/base_preprocessor.py` (any class with the right `__call__` signature does, automatically).
3. Register it in `PREPROCESSOR_REGISTRY` in `src/data/data_utils.py`.
4. Reference it by name in your YAML config: `data.preprocessor: YourPreprocessor`.

## Adding a New Behavior Metric

Behavior heuristics live in `src/metrics_eval/heuristics.py` and are aggregated in `compute_metrics()`.

1. Add a heuristic function in `src/metrics_eval/heuristics.py` (e.g. `is_my_violation(text: str) -> bool`).
2. Wire it into `compute_metrics()` in `src/metrics_eval/evaluator.py` to emit a new metric key.
3. Add unit tests for the heuristic in `tests/test_heuristics.py`.
4. The new metric is automatically picked up by `MetricsEvalCallback` and logged as `custom/<name>`.

## What This Project Does (Plain English)

A framework for fine-tuning small/medium LLMs (Phi-2, Phi-3.5, SmolLM-135M) on hardware-engineering Socratic-dialogue data. Optimized for cloud GPU experiments: built-in profiling, DeepSpeed support, and behavior-focused (not just loss-focused) evaluation metrics.

