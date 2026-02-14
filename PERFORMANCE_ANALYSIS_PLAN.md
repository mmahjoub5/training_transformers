# Hardware-Aware Performance Analysis Scaffold for LoRA Training

## Context

The existing codebase is a well-structured LoRA fine-tuning pipeline (SFTTrainer + PEFT + TRL) focused on Socratic teaching behavior. It has no performance profiling, no DeepSpeed integration, and no infrastructure for systematic hardware-aware benchmarking. This plan extends the codebase with modular performance instrumentation, DeepSpeed as an alternative backend, experiment tracking, and analysis utilities — all without disrupting existing training functionality.

The goal is **systems performance measurement**, not model accuracy optimization.

---

## New Module Structure

```
src/
  profiling/                         # NEW — performance instrumentation
    __init__.py
    callback.py                      # ProfilingCallback (TrainerCallback)
    timer.py                         # CudaTimer — CUDA event-based timing
    memory.py                        # GPU memory snapshot utilities
    system_info.py                   # Hardware/software environment capture

  config/
    profiling_config.py              # NEW — ProfilingConfig dataclass
    deepspeed_config.py              # NEW — DeepSpeedConfig dataclass

  training/                           # NEW — trainer construction
    __init__.py
    trainer_factory.py               # build_training_args(), build_callbacks()

  kernels/                           # NEW — future Triton kernel experiments (empty for now)
    __init__.py

  utils/
    __init__.py
    experiment.py                    # NEW — ExperimentManifest save/load

configs/
  deepspeed/                         # NEW — DeepSpeed JSON configs
    ds_z1.json                       # ZeRO Stage 1
    ds_z2.json                       # ZeRO Stage 2
    ds_z3.json                       # ZeRO Stage 3

benchmarks/                          # NEW — benchmark runner + comparison
    __init__.py
    run_benchmark.py                 # CLI: run configs with profiling, fixed step count
    compare.py                       # CLI: compare experiment_manifest.json files

experiments/                         # NEW — runtime output directory (gitignored)

tests/
  conftest.py                        # NEW — shared test fixtures
  test_profiling.py                  # NEW — timer, memory, system_info tests
  test_configs.py                    # NEW — profiling/deepspeed config parsing
  test_experiment.py                 # NEW — manifest save/load, summary computation
```

---

## Implementation Steps

| Step | Description | Estimated Time |
|------|-------------|---------------|
| 1 | Config dataclasses (ProfilingConfig, DeepSpeedConfig) | ~15 min |
| 2 | Profiling instrumentation (timer, memory, system_info) | ~30 min |
| 3 | ProfilingCallback | ~25 min |
| 4 | Experiment manifest (save/load/summary) | ~20 min |
| 5 | DeepSpeed JSON configs (Z1, Z2, Z3) | ~10 min |
| 6 | Trainer factory + wire into lora_train.py | ~30 min |
| 7 | Benchmark runner + comparison CLI | ~25 min |
| 8 | Triton extension point (empty scaffold) | ~5 min |
| 9 | Tests (configs, profiling, experiment) | ~25 min |
| 10 | Dependencies, gitignore, cleanup | ~5 min |
| **Total** | | **~3 hours** |

---

### Step 1: Config Dataclasses

**Files:** `src/config/profiling_config.py`, `src/config/deepspeed_config.py`

Follow the exact `from_dict()` pattern used in `src/config/training_config.py:31-58`.

**ProfilingConfig:**
- `enabled: bool = False`
- `log_every_n_steps: int = 10`
- `trace_dir: Optional[str] = None`
- `enable_torch_profiler: bool = False`
- `profiler_start_step: int = 20`
- `profiler_end_step: int = 40`

**DeepSpeedConfig:**
- `enabled: bool = False`
- `config_path: Optional[str] = None`
- `zero_stage: int = 2`
- `resolve_config_path()` method returns path to DS JSON or `None`

Both return disabled defaults when their YAML section is absent, preserving full backward compatibility with existing configs.

---

### Step 2: Profiling Instrumentation

**Files:** `src/profiling/__init__.py`, `timer.py`, `memory.py`, `system_info.py`

**CudaTimer** (`timer.py`):
- Uses `torch.cuda.Event(enable_timing=True)` for GPU-accurate timing
- Falls back to `time.perf_counter()` on CPU
- `start()` / `stop()` / `elapsed_ms()` interface
- Only calls `synchronize()` on `elapsed_ms()` to avoid pipeline bubbles

**take_memory_snapshot()** (`memory.py`):
- Returns `MemorySnapshot` dataclass: allocated_mb, reserved_mb, peak_allocated_mb, peak_reserved_mb, free_mb
- Returns `None` when CUDA unavailable
- `reset_peak_stats()` helper

**capture_system_info()** (`system_info.py`):
- Returns `SystemInfo` dataclass: hostname, platform, python/torch/cuda/cudnn/driver versions, GPU names/memory/compute capability, library versions (transformers, peft, trl, deepspeed)
- Uses `torch.cuda.get_device_properties()` and `nvidia-smi` subprocess
- Graceful degradation on CPU-only systems

---

### Step 3: ProfilingCallback

**File:** `src/profiling/callback.py`

TrainerCallback following the pattern in `metrics_eval/callback.py`:

- **`on_train_begin`**: Record start time, reset peak memory stats, optionally start `torch.profiler`
- **`on_step_begin`**: Start CUDA timer
- **`on_step_end`**: Stop timer; every `log_every_n_steps` steps: synchronize, record timing + memory snapshot, compute throughput (samples/sec, tokens/sec), log to TensorBoard under `perf/` prefix
- **`on_train_end`**: Record total time, close torch.profiler if active
- **`get_results()`**: Returns all collected data for experiment manifest

Optional `torch.profiler` integration:
- Bounded window (`profiler_start_step` to `profiler_end_step`)
- Generates Chrome traces to `trace_dir` (viewable in TensorBoard)
- Off by default to avoid large trace files

---

### Step 4: Experiment Manifest

**File:** `src/utils/experiment.py`

**`save_experiment_manifest(output_dir, config, profiling_results, train_metrics)`**:
- Writes `experiment_manifest.json` with:
  - timestamp
  - system_info (full hardware/software snapshot)
  - full config snapshot
  - train_metrics (from HF Trainer)
  - raw profiling data (step timings, memory snapshots)
  - computed summary: avg/median/p95 step time, peak memory, avg throughput

**`load_experiment_manifest(path)`** for analysis utilities.

**Example manifest structure:**
```json
{
  "timestamp": "2026-02-10T15:30:00",
  "system_info": {
    "hostname": "gpu-instance-1",
    "gpus": [{"name": "NVIDIA A100", "memory_total_mb": 81920, "compute_capability": "8.0"}],
    "cuda_version": "12.1",
    "torch_version": "2.3.0"
  },
  "config": { "...full YAML config dict..." },
  "train_metrics": { "train_loss": 1.23, "train_runtime": 3600 },
  "profiling": {
    "total_training_time_sec": 3600,
    "step_timings": [{"step": 10, "wall_clock_ms": 847, "cuda_elapsed_ms": 832, "samples_per_sec": 38}],
    "memory_snapshots": [{"step": 10, "allocated_mb": 12400, "peak_allocated_mb": 14200}]
  },
  "summary": {
    "avg_step_time_ms": 847.3,
    "median_step_time_ms": 842.0,
    "p95_step_time_ms": 912.1,
    "peak_gpu_memory_mb": 14231,
    "avg_samples_per_sec": 38.2,
    "avg_tokens_per_sec": 19264
  }
}
```

---

### Step 5: DeepSpeed Configs

**Files:** `configs/deepspeed/ds_z1.json`, `ds_z2.json`, `ds_z3.json`

All use `"auto"` for batch size, gradient accumulation, and precision — HF Trainer fills these from TrainingArguments, keeping the YAML config as the single source of truth.

All have `wall_clock_breakdown: true` for built-in DeepSpeed profiling.

**ZeRO Stage 1** — optimizer state partitioning only (lowest overhead)
**ZeRO Stage 2** — optimizer + gradient partitioning (good default for LoRA)
**ZeRO Stage 3** — full parameter partitioning (most memory savings, highest comm overhead)

> **Note:** ZeRO-3 can conflict with PEFT/LoRA. ZeRO-1 and ZeRO-2 work well. ZeRO-3 requires `stage3_gather_16bit_weights_on_model_save: true`.

---

### Step 6: Trainer Factory + Wire into Training Script

**New file:** `src/training/trainer_factory.py`

Extract SFTTrainer construction out of `lora_train.py` into a reusable factory:

- `build_training_args(training_config, logging_config, deepspeed_config, ...)` → returns `SFTConfig`
  - Moves the SFTConfig construction (currently lines 230-280 in lora_train.py) into this function
  - Adds `deepspeed=deepspeed_config.resolve_config_path()` parameter
- `build_callbacks(training_config, profiling_config, config)` → returns callback list
  - Handles EarlyStoppingCallback (existing) + ProfilingCallback (new)

**Modified file:** `scripts/lora_train.py`

1. **After config loading (line 125):** Parse new configs
   ```python
   deepspeed_config = DeepSpeedConfig.from_dict(config)
   profiling_config = ProfilingConfig.from_dict(config)
   ```

2. **Replace SFTConfig + callbacks construction (lines 230-285)** with calls to trainer_factory:
   ```python
   training_args = build_training_args(training_config, logging_config, deepspeed_config, config, use_cuda, use_fp16, use_bf16)
   callbacks = build_callbacks(training_config, profiling_config, config)
   ```

3. **After training completes (line 373):** Save experiment manifest
   ```python
   if profiling_config.enabled:
       save_experiment_manifest(output_dir=training_config.output_dir, config=config,
           profiling_results=profiling_cb.get_results(), train_metrics=metrics)
   ```

`lora_train.py` becomes shorter overall — the SFTConfig block moves to `trainer_factory.py`. When profiling/deepspeed sections are absent from YAML, behavior is identical to current code.

---

### Step 7: Benchmark Runner and Comparison

**Files:** `benchmarks/run_benchmark.py`, `benchmarks/compare.py`

**`run_benchmark.py`**: Takes multiple YAML config paths, runs each as a subprocess training job. Configs should have `profiling.enabled: true` and a small `max_steps`.

```bash
python -m benchmarks.run_benchmark \
    --configs configs/phi35_baseline.yaml configs/phi35_ds_z2.yaml \
    --max_steps 50
```

**`compare.py`**: Loads multiple `experiment_manifest.json` files, prints markdown comparison table.

```bash
python -m benchmarks.compare \
    experiments/phi35_baseline/experiment_manifest.json \
    experiments/phi35_ds_z2/experiment_manifest.json
```

**Example output:**
```
| Metric               | phi35_baseline | phi35_ds_z2 |
| ---                  | ---            | ---         |
| Avg Step Time (ms)   | 847.3          | 912.1       |
| Median Step Time (ms)| 842.0          | 905.4       |
| P95 Step Time (ms)   | 912.1          | 978.3       |
| Peak GPU Memory (MB) | 14231          | 9842        |
| Avg Samples/sec      | 38.2           | 35.4        |
| Avg Tokens/sec       | 19264          | 17890       |
| Total Time (sec)     | 42.4           | 45.6        |
| GPU                  | NVIDIA A100    | NVIDIA A100 |
| DeepSpeed            | off            | ZeRO-2      |
```

---

### Step 8: Triton Extension Point

**File:** `src/kernels/__init__.py`

Empty package providing the namespace for future Triton kernel experiments. The pattern for adding kernels later:
1. Create a `@triton.jit` kernel + Python wrapper
2. Monkey-patch into the model via model_loader or a custom wrapper
3. Measure end-to-end impact via ProfilingCallback
4. Use `torch.profiler` traces for kernel-level breakdown

---

### Step 9: Tests

**Files:** `tests/conftest.py`, `tests/test_configs.py`, `tests/test_profiling.py`, `tests/test_experiment.py`

All tests run on CPU — GPU behavior tested via `unittest.mock.patch` on `torch.cuda.is_available`.

Tests cover:
- Config parsing (enabled, disabled, missing sections)
- Timer wall-clock fallback on CPU
- Memory snapshot returns `None` without CUDA
- System info capture and JSON serialization
- Experiment manifest summary computation (avg, median, p95, peak memory)

---

### Step 10: Dependencies and Gitignore

- Add `deepspeed>=0.14.0` to `requirements.in` (CUDA-only, won't install on macOS)
- Add `experiments/` to `.gitignore`

---

## YAML Config Extension

Existing configs remain unchanged. New sections are **optional**:

```yaml
# Add to any existing config to enable profiling:
profiling:
  enabled: true
  log_every_n_steps: 10
  enable_torch_profiler: false    # set true for Chrome trace generation

# Add to enable DeepSpeed:
deepspeed:
  enabled: true
  zero_stage: 2                   # 1, 2, or 3
  # config_path: configs/deepspeed/custom.json  # optional override
```

---

## Files Modified (Existing)

| File | Change |
|------|--------|
| `scripts/lora_train.py` | Extract SFTConfig + callbacks to trainer_factory; add config parsing, manifest save (~net fewer lines) |
| `requirements.in` | Add `deepspeed>=0.14.0` (optional — CUDA-only, use try/except import) |
| `.gitignore` | Add `experiments/` |

---

## Verification Plan

1. **Unit tests:** `pytest tests/test_configs.py tests/test_profiling.py tests/test_experiment.py`
2. **Backward compatibility:** Run existing config without profiling/deepspeed sections — behavior must be identical:
   ```bash
   python -m scripts.lora_train --config configs/smollm-135m_socratic.yaml
   ```
3. **Profiling smoke test:** Add `profiling: {enabled: true}` to smollm-135m config, run for ~20 steps, verify `experiment_manifest.json` is created with timing data
4. **DeepSpeed smoke test (GPU machine):** Run with `deepspeed: {enabled: true, zero_stage: 2}`, verify training runs and DS wall_clock_breakdown prints
5. **Comparison:** Run baseline + DeepSpeed configs, then:
   ```bash
   python -m benchmarks.compare experiments/baseline/experiment_manifest.json experiments/ds_z2/experiment_manifest.json
   ```

---

## Decisions Made

- **DeepSpeed dependency:** Added to `requirements.in`. CUDA-only — won't install on macOS but works on cloud GPU machines. Code uses try/except import for graceful degradation.
- **Trainer factory refactor:** Yes — extract SFTTrainer construction from `lora_train.py` into `src/training/trainer_factory.py` for cleaner separation of concerns.
