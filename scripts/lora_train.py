import torch

from src.models.model_loader import load_model, quick_info
from src.core.config import load_config
from src.data.dataset_loader import load_dataset_generic
from src.data.preprocess import preprocess_dataset
from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig
from src.config.lora_config import LoraConfigSpec
from src.config.logging_config import LoggingConfig
from transformers import TrainingArguments, Trainer
from src.metrics.compute_metrics import BehaviorMetricsComputer
from src.data.eli_preprocess import ELI5Preprocessor_QA
from src.data.data_utils import PREPROCESSOR_REGISTRY
import argparse
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from metrics_eval.callback import MetricsEvalCallback
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


def main():
    config = load_config(args.config)
    model_config = ModelConfig.from_dict(config)
    training_config = TrainingConfig.from_dict(config)
    logging_config = LoggingConfig.from_dict(config)
    lora_config = LoraConfigSpec.from_dict(config) if config.get("lora") is not None else None

    # --- Device / precision setup ---
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")
    if use_cuda:
        print("GPU:", torch.cuda.get_device_name(0))

    # precision from config: "fp32" | "fp16" | "bf16"
    precision = training_config.precision.lower()

    # Decide dtype flags for Trainer AMP
    use_bf16 = (precision == "bf16") and _cuda_bf16_supported()
    use_fp16 = (precision == "fp16") and use_cuda and not use_bf16

    if precision == "bf16" and not use_bf16:
        print("Warning: bf16 requested but not supported on this GPU. Falling back to fp16.")
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
    tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = 'right'



    print("++++++++++++++++++++++++++++++++++++++++++++++")
    print(torch.backends.cuda.sdp_kernel)
    print("++++++++++++++++++++++++++++++++++++++++++++++")
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
    raw_ds = raw_ds["train"].train_test_split(test_size=0.2)
    print("Raw dataset splits:", raw_ds)
    print("Raw dataset example:", raw_ds["train"][0])
    
    if config["data"]["preprocessor"] == "ELI5Preprocessor_QA":
        raw_ds = raw_ds.flatten()
    processed_data = raw_ds.map(
        preprocessor,
        batched=False,
        num_proc=args.proc,
        remove_columns=raw_ds["train"].column_names,
    )

    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(processed_data["test"][0])
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")



    if lora_config is not None:
        lora_cfg = LoraConfig(
            r=lora_config.rank,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
            bias=lora_config.bias,
            task_type=lora_config.task_type,
            target_modules=lora_config.target_modules,  # common for decoder LMs
        )
    else :
        lora_cfg = None


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

        #steps 
        max_steps=training_config.max_steps,

        # --- GPU performance knobs ---
        dataloader_pin_memory=use_cuda,  # True on GPU, False on CPU
        dataloader_num_workers=training_config.num_workers,

        # Mixed precision (CUDA only)
        fp16=use_fp16,
        bf16=use_bf16,
        tf32=use_fp16, 
        # Evaluation / saving strategies
        eval_strategy=training_config.eval_strategy,
        do_eval=True,
        eval_steps=training_config.eval_steps, 
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,

        # Common stability/perf options (optional but helpful)
        optim=training_config.optim,
        max_grad_norm=training_config.max_grad_norm,
        warmup_ratio=training_config.warmup_ratio,

        # If you later go multi-GPU with torchrun/DDP
        ddp_find_unused_parameters=False,

        max_length=2048,
        dataset_text_field="text",
        
        packing=False
    )
    
    metrics_callback = MetricsEvalCallback(
        eval_data=processed_data["test"],
        eval_steps=training_args.eval_steps,
        max_samples=200,
        batch_size=4,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        peft_config=lora_cfg, 
        train_dataset=processed_data["train"],
        eval_dataset=processed_data["test"],
    
    )
    trainer.add_callback(metrics_callback)
    
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

if __name__ == "__main__":
    main()
