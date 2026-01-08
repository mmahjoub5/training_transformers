import torch
import sys
import os
import logging
from pathlib import Path

from src.models.model_loader import load_model, quick_info
from src.core.config import load_config
from src.data.dataset_loader import load_dataset_generic
from src.data.preprocess import preprocess_dataset
from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig
from src.config.lora_config import LoraConfigSpec
from src.config.logging_config import LoggingConfig
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from src.metrics.compute_metrics import BehaviorMetricsComputer
from src.data.eli_preprocess import ELI5Preprocessor_QA
from src.data.data_utils import PREPROCESSOR_REGISTRY
import argparse
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from metrics_eval.callback import MetricsEvalCallback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
# arg parse for input config 

parser = argparse.ArgumentParser(
        description="Train QA model with YAML config"
)

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

args = parser.parse_args()
logger = logging.getLogger(__name__)

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

    return total_tokens, target_tokens


def main():
    config = load_config(args.config)
    model_config = ModelConfig.from_dict(config)
    training_config = TrainingConfig.from_dict(config)
    logging_config = LoggingConfig.from_dict(config)
    lora_config = LoraConfigSpec.from_dict(config) if config.get("lora") is not None else None

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
        kind=model_config.kind,
        precision=precision,
        attn_implementation=model_config.attn_implementation,
        device_map = "auto"
    )
    # Set pad token: prefer existing pad; else use eos; else add new
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.warning("Using eos_token as pad_token")
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
            logger.warning("Added new [PAD] token and resized embeddings")

    # Make sure model configs align
    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tokenizer.pad_token_id



    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = 'right'
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

    training_args = SFTConfig(
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

        #steps 
        max_steps=training_config.max_steps,

        # --- GPU performance knobs ---
        dataloader_pin_memory=use_cuda,  # True on GPU, False on CPU
        dataloader_num_workers=training_config.num_workers,

        # Mixed precision (CUDA only)
        fp16=use_fp16,
        bf16=use_bf16,
        tf32=use_bf16,  # tf32 benefits Ampere+ GPUs with bf16 
        # Evaluation / saving strategies
        eval_strategy=str(training_config.eval_strategy),
        do_eval=True,
        eval_steps=training_config.eval_steps, 
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,

        # Common stability/perf options (optional but helpful)
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
        assistant_only_loss=True
    )
    
  
    callbacks = []
    if training_config.early_stopping_patience is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=training_config.early_stopping_patience))

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        peft_config=lora_cfg,
        train_dataset=processed_data["train"],
        eval_dataset=processed_data["test"],
        callbacks=callbacks if callbacks else None,
    )

    # Validate a batch before training
    logger.info("Validating batch structure...")
    dl = trainer.get_train_dataloader()
    batch = next(iter(dl))

    logger.info(f"Batch keys: {batch.keys()}")

    input_ids = batch["input_ids"][0].tolist()
    labels = batch["labels"][0].tolist()

    # Validate batch - will raise ValueError with helpful message if invalid
    total_tokens, target_tokens = validate_batch(input_ids, labels, tokenizer)

    max_chars = config["tokenizer"]["max_length"]  # rename: this is chars, not tokens

    # Make sure we're decoding a python list, not a tensor
    full_ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
    lab_ids  = labels.tolist() if hasattr(labels, "tolist") else list(labels)

    full_text = tokenizer.decode(full_ids, skip_special_tokens=False)
    logger.info("FULL INPUT (first %d chars):\n%s", max_chars, full_text[:max_chars])

    # Masked view: keep positions, replace masked tokens with PAD (or EOS)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    masked_ids = [tid if lab != -100 else pad_id for tid, lab in zip(full_ids, lab_ids)]

    masked_text = tokenizer.decode(masked_ids, skip_special_tokens=False)
    logger.info("TARGET VIEW (masked, first %d chars):\n%s", max_chars, masked_text[:max_chars])

    # Optional: show contiguous target spans (much more interpretable)
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

    # UNK sanity
    unk_id = tokenizer.unk_token_id
    logger.info("UNK count full=%d masked=%d",
                full_ids.count(unk_id),
                masked_ids.count(unk_id))

    logger.info("Token counts: total=%d, target=%d, masked=%d",
                total_tokens, target_tokens, total_tokens - target_tokens)


    # Check for resume
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = find_latest_checkpoint(training_config.output_dir)
        if resume_checkpoint:
            logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
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
