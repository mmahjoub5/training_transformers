from src.models.model_loader import load_model, quick_info
from src.core.config import load_config
from src.data.dataset_loader import load_dataset_generic
from src.data.preprocess   import preprocess_dataset
from transformers import TrainingArguments, Trainer
from src.metrics.compute_metrics import QAMetricsComputer

if __name__ == "__main__":
    config = load_config("configs/distilbert.yaml")
    tokenizer, model = load_model(
        model_name=config["model"]["name"],
        kind=config["model"]["kind"],
        precision=config["training"].get("precision", "fp32"),
    )
    raw_ds = load_dataset_generic(config["data"])

    processed_ds = preprocess_dataset(
        raw_ds,
        tokenizer,
        max_length=config["tokenizer"]["max_length"],
    )
    print("MODEL CLASS:", model.__class__)
    print("TRAIN KEYS:", processed_ds["train"][0].keys())
    print("VAL KEYS:", processed_ds["validation"][0].keys())
    # return 
    # print("Dataset preprocessing complete.")

    # Create metrics computer with validation dataset
    metrics_computer = QAMetricsComputer(
        validation_dataset=processed_ds["validation"],
        n_best_size=20,
        max_answer_length=30
    )


    # training_args = TrainingArguments(
    #     output_dir=config["training"]["output_dir"],
    #     per_device_train_batch_size=config["training"]["batch_size"],
    #     per_device_eval_batch_size=config["training"]["batch_size"],
    #     learning_rate=float(config["training"]["lr"]),
    #     num_train_epochs=config["training"]["epochs"],
    #     weight_decay=config["training"].get("weight_decay", 0.0),
    #     logging_steps=config["logging"].get("logging_steps", 100),
    #     save_steps=config["logging"].get("save_steps", 100),
    #     report_to=config["logging"].get("report_to", []),
    #     logging_dir=config["logging"].get("logging_dir", "./runs"),
    #     dataloader_pin_memory=False,  # for CPU training
    #     # evaluation
    #     eval_strategy="epoch",
    #     save_strategy="epoch",
    #     gradient_accumulation_steps=config["training"].get("gradient_accumulation_steps", 1)



    # )

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=processed_ds["train"],
    #     eval_dataset=processed_ds["validation"],
    #     tokenizer=tokenizer,
    #     compute_metrics=metrics_computer,
    # )

    # trainer.train()

    #trainer.save_model("./models/test-squad-trained")    