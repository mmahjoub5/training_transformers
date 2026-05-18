from transformers import EarlyStoppingCallback
from trl import SFTConfig

from src.config.deepspeed_config import DeepSpeedConfig
from src.config.logging_config import LoggingConfig
from src.config.profiling_config import ProfilingConfig
from src.config.training_config import TrainingConfig
from src.profiling.callback import ProfilingCallback


def build_training_args(
    training_config: TrainingConfig,
    logging_config: LoggingConfig,
    deepspeed_config: DeepSpeedConfig,
    config: dict,
    use_cuda: bool,
    use_fp16: bool,
    use_bf16: bool,
) -> SFTConfig:
    return SFTConfig(
        output_dir=training_config.output_dir,
        per_device_train_batch_size=training_config.batch_size,
        per_device_eval_batch_size=training_config.batch_size,
        learning_rate=training_config.lr,
        num_train_epochs=training_config.epochs,
        weight_decay=training_config.weight_decay,

        # Logging / saving
        logging_steps=logging_config.logging_steps,
        save_steps=logging_config.save_steps,
        report_to=logging_config.report_to,
        logging_dir=logging_config.logging_dir,
        save_strategy=logging_config.save_strategy,
        save_total_limit=logging_config.save_total_limit,
        load_best_model_at_end=training_config.load_best_model_at_end,
        metric_for_best_model=training_config.metric_for_best_model,
        greater_is_better=training_config.greater_is_better,

        # steps
        max_steps=training_config.max_steps,

        # --- GPU performance knobs ---
        dataloader_pin_memory=use_cuda,
        dataloader_num_workers=training_config.num_workers,

        # Mixed precision (CUDA only)
        fp16=use_fp16,
        bf16=use_bf16,
        tf32=use_bf16,

        # Evaluation / saving strategies
        eval_strategy=str(training_config.eval_strategy),
        do_eval=True,
        eval_steps=training_config.eval_steps,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,

        # Common stability/perf options
        optim=training_config.optim,
        max_grad_norm=training_config.max_grad_norm,
        warmup_ratio=training_config.warmup_ratio,
        lr_scheduler_type=training_config.lr_scheduler_type,
        seed=config["data"].get("seed", 42),

        # If you later go multi-GPU with torchrun/DDP
        ddp_find_unused_parameters=False,

        max_length=config["tokenizer"]["max_length"],
        dataset_text_field="messages",
        packing=False,
        assistant_only_loss=training_config.assistant_only_loss,

        deepspeed=deepspeed_config.resolve_config_path(),
    )


def build_callbacks(
    training_config: TrainingConfig,
    profiling_config: ProfilingConfig,
) -> list:
    callbacks = []
    if training_config.early_stopping_patience is not None:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=training_config.early_stopping_patience
            )
        )
    if profiling_config.enabled:
        callbacks.append(ProfilingCallback(profiling_config))
    return callbacks
