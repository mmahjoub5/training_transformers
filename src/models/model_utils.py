def setup_model(tokenizer, model):
    """Configure model with tokenizer settings."""
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    return model


def load_and_setup_model(config: ModelConfig):
    """Load model and tokenizer, then configure them."""
    tokenizer, model = load_model(
        model_name=config.model_name,
        precision=config.precision,
        low_cpu_mem_usage=config.low_cpu_mem_usage,
        device_map=config.device_map,
        kind=config.kind,
        attn_implementation=config.attn_implementation
    )
    model = setup_model(tokenizer, model)
    model.eval()
    return tokenizer, model