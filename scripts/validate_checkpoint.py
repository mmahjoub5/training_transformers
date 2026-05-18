"""
Modular text generation script for comparing trained and baseline models.
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import torch

from src.models.model_loader import load_model
from src.io.prompt_io import read_prompts
from src.config.generation_config import GenerationConfig
from src.config.model_config import ModelConfig




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





def generate_text(
    prompt: str,
    tokenizer,
    model,
    gen_config: GenerationConfig
) -> str:
    """Generate text from a single prompt."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    
    # Move to model device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_config.to_dict())
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_from_prompts(
    prompts: List[str],
    tokenizer,
    model,
    gen_config: GenerationConfig,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """Generate text for multiple prompts."""
    results = []
    
    for i, prompt in enumerate(prompts):
        output = generate_text(prompt, tokenizer, model, gen_config)
        
        if verbose:
            print(f"\n=== Prompt {i} ===")
            print(output)
        
        results.append({
            "id": i,
            "prompt": prompt,
            "output": output,
        })
    
    return results


def save_results(results: List[Dict[str, Any]], output_path: str):
    """Save generation results to JSON file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_path}")


def tokenize_text(text: str) -> Set[str]:
    """Tokenize text into words for F1 calculation."""
    # Simple word tokenization (lowercase and split on whitespace/punctuation)
    import re
    tokens = re.findall(r'\b\w+\b', text.lower())
    return set(tokens)


def calculate_f1(prediction: str, reference: str) -> Dict[str, float]:
    """
    Calculate token-level F1 score between two texts.
    
    Args:
        prediction: Generated text from trained model
        reference: Generated text from baseline model
        
    Returns:
        Dictionary with precision, recall, and f1 score
    """
    pred_tokens = tokenize_text(prediction)
    ref_tokens = tokenize_text(reference)
    
    if len(pred_tokens) == 0 and len(ref_tokens) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    # Calculate overlap
    common = pred_tokens & ref_tokens
    
    precision = len(common) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(common) / len(ref_tokens) if ref_tokens else 0.0
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate text using trained and baseline models"
    )
    
    parser.add_argument(
        "--json",
        type=str,
        required=True,
        help="Path to JSON file with prompts"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="HuggingFaceTB/SmolLM-135M",
        help="Baseline model ID (default: SmolLM-135M)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="generation_results.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling parameter"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Load prompts
    print(f"📄 Loading prompts from {args.json}")
    prompts = read_prompts(args.json)
    print(f"✓ Loaded {len(prompts)} prompts")
    
    # Setup configurations
    model_config = ModelConfig(
        model_name=args.checkpoint, 
        device_map="auto", 
        precision="fp16")
    baseline_config = ModelConfig(
        model_name=args.baseline, 
        device_map="auto", 
        precision="fp16")
    gen_config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=False # Greedy decoding 
    )
    
    # Load models
    print(f"\n🔧 Loading trained model from {args.checkpoint}")
    tokenizer_trained, model_trained = load_and_setup_model(model_config)
    
    print(f"🔧 Loading baseline model: {args.baseline}")
    tokenizer_baseline, model_baseline = load_and_setup_model(baseline_config)
    
    # Generate outputs
    print("\n🤖 Generating with trained model...")
    results_trained = generate_from_prompts(
        prompts,
        tokenizer_trained,
        model_trained,
        gen_config,
        verbose=not args.quiet
    )
    
    print("\n🤖 Generating with baseline model...")
    results_baseline = generate_from_prompts(
        prompts,
        tokenizer_baseline,
        model_baseline,
        gen_config,
        verbose=not args.quiet
    )
    
    # Combine results
    combined_results = {
        "trained_model": args.checkpoint,
        "baseline_model": args.baseline,
        "generation_config": gen_config.__dict__,
        "results": []
    }
    
    f1_scores = []
    
    for r_t, r_b in zip(results_trained, results_baseline):
        # Calculate F1 score between outputs
        f1_metrics = calculate_f1(r_t["output"], r_b["output"])
        f1_scores.append(f1_metrics["f1"])
        
        combined_results["results"].append({
            "id": r_t["id"],
            "prompt": r_t["prompt"],
            "trained_output": r_t["output"],
            "baseline_output": r_b["output"],
            "f1_metrics": f1_metrics,
        })
    
    # Add summary statistics
    if f1_scores:
        combined_results["summary"] = {
            "mean_f1": round(sum(f1_scores) / len(f1_scores), 4),
            "min_f1": round(min(f1_scores), 4),
            "max_f1": round(max(f1_scores), 4),
        }
        
        print(f"\n📊 F1 Score Summary:")
        print(f"   Mean: {combined_results['summary']['mean_f1']:.4f}")
        print(f"   Min:  {combined_results['summary']['min_f1']:.4f}")
        print(f"   Max:  {combined_results['summary']['max_f1']:.4f}")
    
    # Save results
    save_results(combined_results, args.output)


if __name__ == "__main__":
    main()