from src.models.model_loader import load_model, quick_info
from src.core.config import load_config
from src.data.dataset_loader import load_dataset_generic
from src.data.preprocess   import preprocess_dataset
from transformers import TrainingArguments, Trainer
from src.metrics.compute_metrics import QAMetricsComputer
from src.data.eli_preprocess import ELI5Preprocessor_QA




if __name__ == "__main__":
    config = load_config("configs/smollm-135m.yaml")
    tokenizer, model = load_model(
        model_name=config["model"]["name"],
        kind=config["model"]["kind"],
        precision=config["training"].get("precision", "fp32"),
    )
    preprocessor = ELI5Preprocessor_QA(tokenizer)
    raw_ds = load_dataset_generic(config["data"])
    raw_ds = raw_ds["train"].train_test_split(test_size=0.2)
    raw_ds = raw_ds.flatten()
    print("Raw dataset splits:", raw_ds)
    print("Raw dataset example:", raw_ds["train"][3])
    print(raw_ds["train"].column_names)


    tokenized_eli5 = raw_ds.map(
        preprocessor,
        batched=False,
        num_proc=1,
        remove_columns=raw_ds["train"].column_names,
    )
    print(tokenized_eli5.shape)
    #print("Tokenized dataset example:", tokenized_eli5["train"][3])
    
    # Create metrics computer with validation dataset
    # metrics_computer = QAMetricsComputer(
    #     validation_dataset=tokenized_eli5["test"],
    # )


    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["training"]["batch_size"],
        learning_rate=float(config["training"]["lr"]),
        num_train_epochs=config["training"]["epochs"],
        weight_decay=config["training"].get("weight_decay", 0.0),
        logging_steps=config["logging"].get("logging_steps", 100),
        save_steps=config["logging"].get("save_steps", 100),
        report_to=config["logging"].get("report_to", []),
        logging_dir=config["logging"].get("logging_dir", "./runs"),
        dataloader_pin_memory=False,  # for CPU training
        # evaluation
        eval_strategy="epoch",
        save_strategy="epoch",
        gradient_accumulation_steps=config["training"].get("gradient_accumulation_steps", 1)
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_eli5["train"],
        eval_dataset=tokenized_eli5["test"],
        tokenizer=tokenizer,
        #compute_metrics=metrics_computer,
    )

    trainer.train()