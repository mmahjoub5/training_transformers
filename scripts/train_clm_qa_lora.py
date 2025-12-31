import torch

from src.models.model_loader import load_model, quick_info
from src.core.config import load_config
from src.data.dataset_loader import load_dataset_generic
from src.data.preprocess import preprocess_dataset
from transformers import TrainingArguments, Trainer
from src.metrics.compute_metrics import QAMetricsComputer
from src.data.eli_preprocess import ELI5Preprocessor_QA
import argparse
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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
    help="Path to YAML config file (e.g. configs/smollm-135m.yaml)",
)


args = parser.parse_args()

def _cuda_bf16_supported() -> bool:
    # bf16 is typically supported on Ampere (A100/3090) and newer
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8


if __name__ == "__main__":
    config = load_config(args.config)

    # --- Device / precision setup ---
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")
    if use_cuda:
        print("GPU:", torch.cuda.get_device_name(0))

    # precision from config: "fp32" | "fp16" | "bf16"
    precision = config["training"].get("precision", "fp32").lower()

    # Decide dtype flags for Trainer AMP
    use_bf16 = (precision == "bf16") and _cuda_bf16_supported()
    use_fp16 = (precision == "fp16") and use_cuda and not use_bf16

    if precision == "bf16" and not use_bf16:
        print("Warning: bf16 requested but not supported on this GPU. Falling back to fp16.")
        use_fp16 = use_cuda
        use_bf16 = False

    # Load model/tokenizer (your loader may already handle dtype; AMP is controlled by TrainingArguments)
    tokenizer, model = load_model(
        model_name=config["model"]["name"],
        kind=config["model"]["kind"],
        precision=precision,  # keep your existing contract,
        attn_implementation="sdpa",
        
    )

    # Optional: gradient checkpointing to reduce VRAM (slower but useful)
    if config["training"].get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        # helps avoid warnings / extra memory during training
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    # If your load_model DOES NOT move to GPU, Trainer will do it.
    # But moving explicitly is fine too:
    # model.to(device)

    preprocessor = ELI5Preprocessor_QA(tokenizer, 
                                       max_length=config["tokenizer"]["max_length"],
                                       truncation=True)
    
    raw_ds = load_dataset_generic(config["data"])
    raw_ds = raw_ds["train"].train_test_split(test_size=0.2)
    raw_ds = raw_ds.flatten()
    print("Raw dataset splits:", raw_ds)
    print("Raw dataset example:", raw_ds["train"][3])
    print(raw_ds["train"].column_names)

    tokenized_eli5 = raw_ds.map(
        preprocessor,
        batched=False,
        num_proc=args.proc,
        remove_columns=raw_ds["train"].column_names,
    )
    print(tokenized_eli5.shape)

    if config.get("lora", None) is not None:
        lora_cfg = LoraConfig(
            r= config["lora"].get("rank", 16),
            lora_alpha=config["lora"].get("lora_alpha", 12),
            lora_dropout=config["lora"].get("lora_droput", 0.05),
            bias=config["lora"].get("bias", "bias"),
            task_type=config["lora"].get("task_type", "CAUSAL_LM"),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # common for decoder LMs
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()


    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["training"]["batch_size"],
        learning_rate=float(config["training"]["lr"]),
        num_train_epochs=config["training"]["epochs"],
        weight_decay=config["training"].get("weight_decay", 0.0),

        # Logging / saving
        logging_steps=config["logging"].get("logging_steps", 100),
        save_steps=config["logging"].get("save_steps", 100),
        report_to=config["logging"].get("report_to", []),
        logging_dir=config["logging"].get("logging_dir", "./runs"),

        # --- GPU performance knobs ---
        dataloader_pin_memory=use_cuda,  # True on GPU, False on CPU
        dataloader_num_workers=config["training"].get("num_workers", 4),

        # Mixed precision (CUDA only)
        fp16=use_fp16,
        bf16=use_bf16,

        # Evaluation / saving strategies
        eval_strategy="epoch",
        save_strategy="epoch",

        gradient_accumulation_steps=config["training"].get("gradient_accumulation_steps", 1),

        # Common stability/perf options (optional but helpful)
        optim=config["training"].get("optim", "adamw_torch"),
        max_grad_norm=config["training"].get("max_grad_norm", 1.0),
        warmup_ratio=config["training"].get("warmup_ratio", 0.0),

        # If you later go multi-GPU with torchrun/DDP
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_eli5["train"],
        eval_dataset=tokenized_eli5["test"],
        tokenizer=tokenizer,
        # compute_metrics=metrics_computer,
    )

    trainer.train()
