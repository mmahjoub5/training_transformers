import argparse
import logging
import sys
from pathlib import Path

import torch
from peft import LoraConfig
from trl import SFTTrainer

from src.config.deepspeed_config import DeepSpeedConfig
from src.config.logging_config import LoggingConfig
from src.config.lora_config import LoraConfigSpec
from src.config.model_config import ModelConfig
from src.config.profiling_config import ProfilingConfig
from src.config.tokenizer_config import TokenizerConfig
from src.config.training_config import TrainingConfig
from src.core.config import load_config
from src.data.data_utils import PREPROCESSOR_REGISTRY
from src.data.dataset_loader import load_dataset_generic
from src.models.model_loader import load_model
from src.profiling.callback import ProfilingCallback
from src.training.trainer_factory import build_callbacks, build_training_args
from src.utils.experiment import save_experiment_manifest

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train QA model with YAML config")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g. configs/smollm-135m.yaml)",
    )
    parser.add_argument(
        "--proc",
        type=int,
        required=False,
        default=1,
        help="Number of processes for data preprocessing",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest checkpoint in output_dir",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to specific checkpoint to resume from (overrides --resume)",
    )
    parser.add_argument(
        "--validate-batch",
        action="store_true",
        help="Run batch structure validation before training (useful for debugging)",
    )
    return parser.parse_args()

def _cuda_bf16_supported() -> bool:
    # bf16 is typically supported on Ampere (A100/3090) and newer
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8


def find_latest_checkpoint(output_dir: str) -> str | None:
    """Find the latest checkpoint in output_dir, if any."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    checkpoints = list(output_path.glob("checkpoint-*"))
    if not checkpoints:
        return None

    # Sort by step number and return latest
    checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))
    latest = str(checkpoints[-1])
    logger.info(f"Found checkpoint: {latest}")
    return latest


def validate_batch(input_ids, labels, tokenizer):
    """Validate a batch and return stats. Raises ValueError if invalid."""
    total_tokens = len(input_ids)
    target_tokens = sum(1 for lab in labels if lab != -100)

    if total_tokens == 0:
        raise ValueError("Batch has 0 tokens - check your data preprocessing")

    if target_tokens == 0:
        raise ValueError(
            "No target tokens found (all labels are -100). "
            "Check that assistant_only_loss is working correctly and your data has assistant responses."
        )

    if target_tokens >= total_tokens:
        raise ValueError(
            f"All tokens are target tokens ({target_tokens}/{total_tokens}). "
            "This suggests labels are not being masked properly."
        )

    # Check EOS is in target tokens (model must learn to produce EOS)
    eos_id = tokenizer.eos_token_id
    eos_in_targets = any(lab == eos_id for lab in labels if lab != -100)
    if not eos_in_targets:
        raise ValueError(
            f"EOS token (id={eos_id}) not found in target tokens! "
            "The model won't learn to generate EOS. Check chat template."
        )

    return total_tokens, target_tokens


def validate_and_log_batch(trainer, tokenizer, config):
    """Validate a batch and log structure details for debugging."""
    logger.info("Validating batch structure...")
    dl = trainer.get_train_dataloader()
    batch = next(iter(dl))
    logger.info(f"Batch keys: {batch.keys()}")

    input_ids = batch["input_ids"][0].tolist()
    labels = batch["labels"][0].tolist()

    total_tokens, target_tokens = validate_batch(input_ids, labels, tokenizer)

    max_chars = config["tokenizer"]["max_length"]
    full_ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
    lab_ids = labels.tolist() if hasattr(labels, "tolist") else list(labels)

    full_text = tokenizer.decode(full_ids, skip_special_tokens=False)
    logger.info("FULL INPUT (first %d chars):\n%s", max_chars, full_text[:max_chars])

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    masked_ids = [tid if lab != -100 else pad_id for tid, lab in zip(full_ids, lab_ids)]
    masked_text = tokenizer.decode(masked_ids, skip_special_tokens=False)
    logger.info("TARGET VIEW (masked, first %d chars):\n%s", max_chars, masked_text[:max_chars])

    spans = []
    start = None
    for i, lab in enumerate(lab_ids):
        if lab != -100 and start is None:
            start = i
        if (lab == -100 or i == len(lab_ids) - 1) and start is not None:
            end = i if lab == -100 else i + 1
            spans.append((start, end))
            start = None

    for j, (s, e) in enumerate(spans[:3]):
        chunk = tokenizer.decode(full_ids[s:e], skip_special_tokens=False)
        logger.info("TARGET SPAN %d (%d:%d): %s", j, s, e, chunk[:300])

    unk_id = tokenizer.unk_token_id
    logger.info("UNK count full=%d masked=%d", full_ids.count(unk_id), masked_ids.count(unk_id))
    logger.info("Token counts: total=%d, target=%d, masked=%d",
                total_tokens, target_tokens, total_tokens - target_tokens)


def main():
    """Main training function for LoRA fine-tuning of transformer models."""
    args = parse_args()
    config = load_config(args.config)
    model_config = ModelConfig.from_dict(config)
    training_config = TrainingConfig.from_dict(config)
    logging_config = LoggingConfig.from_dict(config)
    tokenizer_config = TokenizerConfig.from_dict(config)
    lora_config = LoraConfigSpec.from_dict(config) if config.get("lora") is not None else None
    deepspeed_config = DeepSpeedConfig.from_dict(config)
    profiling_config = ProfilingConfig.from_dict(config)

    # --- Device / precision setup ---
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f"Device: {device}")
    if use_cuda:
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # precision from config: "fp32" | "fp16" | "bf16"
    precision = training_config.precision.lower()

    # Decide dtype flags for Trainer AMP
    use_bf16 = (precision == "bf16") and _cuda_bf16_supported()
    use_fp16 = (precision == "fp16") and use_cuda and not use_bf16

    if precision == "bf16" and not use_bf16:
        logger.warning("bf16 requested but not supported on this GPU. Falling back to fp16.")
        use_fp16 = use_cuda
        use_bf16 = False

    # Load model/tokenizer (your loader may already handle dtype; AMP is controlled by TrainingArguments)
    tokenizer, model = load_model(
        model_name=model_config.model_name,
        adapter=model_config.adapter,
        kind=model_config.kind,
        precision=precision,
        attn_implementation=model_config.attn_implementation,
        custom_template=model_config.custom_template,
        device_map="auto",
    )
    logger.debug("tokenizer.model_max_length=%s", tokenizer.model_max_length)
    logger.debug("model.config.max_position_embeddings=%s", model.config.max_position_embeddings)

    # model_loader.py is the single source of truth for tokenizer pad-token setup.
    # Ensure model generation_config is aligned with tokenizer pad_token_id here.
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    tokenizer.padding_side = 'left'
    logger.info(f"Pad token: '{tokenizer.pad_token}' (id={tokenizer.pad_token_id})")



    logger.info(f"SDP Kernel: {torch.backends.cuda.sdp_kernel}")
    model.config.use_cache = False

    # Optional: gradient checkpointing to reduce VRAM (slower but useful)
    if training_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # helps avoid warnings / extra memory during training
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    # Initialize Preprocessor
    cls = PREPROCESSOR_REGISTRY[config["data"]["preprocessor"]]
    preprocessor = cls(tokenizer,
                        max_length=config["tokenizer"]["max_length"],
                        truncation=True
                    )

    raw_ds = load_dataset_generic(config["data"])
    raw_ds = raw_ds["train"].train_test_split(test_size=0.2, seed=config["data"].get("seed", 42))
    logger.info(f"Raw dataset splits: {raw_ds}")
    logger.info(f"Raw dataset example: {raw_ds['train'][0]}")

    if config["data"]["preprocessor"] == "ELI5Preprocessor_QA":
        raw_ds = raw_ds.flatten()
    processed_data = raw_ds.map(
        preprocessor,
        batched=False,
        num_proc=args.proc,
        remove_columns=raw_ds["train"].column_names,
    )



    if lora_config is not None:
        lora_cfg = LoraConfig(
            r=lora_config.rank,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
            bias=lora_config.bias,
            task_type=lora_config.task_type,
            target_modules=lora_config.target_modules,  # common for decoder LMs
        )
    else:
        lora_cfg = None

    logger.info(f"Training config: {training_config}")
    logger.info(f"Eval strategy: {training_config.eval_strategy}")

    training_args = build_training_args(
        training_config, logging_config, deepspeed_config,
        tokenizer_config, use_cuda, use_fp16, use_bf16
    )


    callbacks = build_callbacks(training_config, profiling_config)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        peft_config=lora_cfg,
        train_dataset=processed_data["train"],
        eval_dataset=processed_data["test"],
        callbacks=callbacks if callbacks else None,
    )

    if args.validate_batch:
        validate_and_log_batch(trainer, tokenizer, config)

    # Check for resume
    resume_checkpoint = None
    if args.checkpoint_path:
        # Explicit path takes priority
        if not Path(args.checkpoint_path).exists():
            raise ValueError(f"Checkpoint path does not exist: {args.checkpoint_path}")
        resume_checkpoint = args.checkpoint_path
        logger.info(f"Using specified checkpoint: {resume_checkpoint}")
    elif args.resume:
        resume_checkpoint = find_latest_checkpoint(training_config.output_dir)
        if resume_checkpoint:
            logger.info(f"Resuming from latest checkpoint: {resume_checkpoint}")
        else:
            logger.warning("--resume flag set but no checkpoint found. Starting from scratch.")

    # Training with error handling
    try:
        logger.info("Starting training...")
        train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        trainer.save_model()

        if profiling_config.enabled:
            profiling_cb = next(
                (c for c in callbacks if isinstance(c, ProfilingCallback)), None
            )
            if profiling_cb:
                save_experiment_manifest(
                    output_dir=training_config.output_dir,
                    config=config,
                    profiling_results=profiling_cb.get_results(),
                    train_metrics=metrics,
                )

        logger.info("Training completed successfully!")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Saving current state...")
        trainer.save_state()
        trainer.save_model()
        logger.info("State saved. You can resume with --resume flag.")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.info("Attempting to save current state...")
        try:
            trainer.save_state()
            logger.info("State saved. You can resume with --resume flag.")
        except Exception as save_error:
            logger.error(f"Failed to save state: {save_error}")
        raise


if __name__ == "__main__":
    main()
