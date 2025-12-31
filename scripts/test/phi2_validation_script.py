from src.models.model_loader import load_model
from src.io.prompt_io import read_prompts
import argparse
import torch

parser = argparse.ArgumentParser(
        description="Train QA model with YAML config"
)

model_id = "microsoft/phi-2"

parser.add_argument(
    "--json",
    type=str,
    required=True,
    help="Path to JSON file with prompts")
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to JSON file with prompts")



args = parser.parse_args()
if __name__ == "__main__":
    prompts = read_prompts(args.json)
    tokenizer, model = load_model(
        model_name=args.checkpoint, 
        precision="fp16",     # or float32 if needed
        low_cpu_mem_usage=True,
        device_map=None,               # <-- key: no auto device map
        kind = "clm",
        attn_implementation="sdpa")
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id  
    prompt = "Explain why the sky is blue like I'm five."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        print(outputs[0])
        print(tokenizer.decode(outputs[0], skip_special_tokens=False))