import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from metrics_eval.evaluator import EvalConfig, GenerationConfig, run_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run behavior-focused eval metrics.")
    parser.add_argument("--model_path", type=str, required=True, help="HF model id or local path")
    parser.add_argument("--eval_json", type=str, required=True, help="Path to frozen eval JSON/JSONL")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation/eval")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional max eval samples")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--do_sample", action="store_true", help="Force sampling")
    parser.add_argument("--no_sample", action="store_true", help="Force greedy decoding")
    parser.add_argument("--precision", type=str, default=None, choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype = None
    if args.precision == "fp16" and torch.cuda.is_available():
        dtype = torch.float16
    elif args.precision == "bf16" and torch.cuda.is_available():
        dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model.to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has no pad_token and no eos_token; can't set padding.")
        tokenizer.pad_token = tokenizer.eos_token

    do_sample = None
    if args.do_sample:
        do_sample = True
    if args.no_sample:
        do_sample = False

    eval_cfg = EvalConfig(
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        generation=GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=do_sample,
        ),
    )

    metrics = run_eval(
        model=model,
        tokenizer=tokenizer,
        eval_json=args.eval_json,
        cfg=eval_cfg,
        use_wandb=args.use_wandb,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
