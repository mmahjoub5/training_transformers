import json
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from metrics_eval.heuristics import (
    avg_interrogative_ratio,
    avg_questions_per_response,
    contains_imperative,
    count_questions,
    count_step_lines,
    is_numeric_violation,
    is_withhold_violation,
    question_rate,
)

SYSTEM_PROMPT = (
    "You are a senior hardware engineer. Teach using Socratic questions. "
    "Do not reveal the final answer. Avoid numeric rules-of-thumb."
)


@dataclass
class GenerationConfig:
    max_new_tokens: int = 200
    temperature: float = 0.2
    top_p: float = 1.0
    do_sample: Optional[bool] = None


@dataclass
class EvalConfig:
    batch_size: int = 4
    max_samples: Optional[int] = None
    seed: int = 42
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    max_length: Optional[int] = None


def _load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        raw = f.read().strip()
    if not raw:
        return []
    if raw[0] == "[":
        return json.loads(raw)
    items = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items


def _map_role(role: str) -> str:
    if role == "junior_engineer":
        return "user"
    if role == "senior_engineer":
        return "assistant"
    return "user"


def split_prompt_target(example: Dict[str, Any]) -> Optional[Tuple[List[Dict[str, str]], str]]:
    messages_field = example.get("messages")
    if isinstance(messages_field, list) and messages_field:
        last_user_idx = None
        for i in range(len(messages_field) - 1):
            if messages_field[i].get("role") == "user" and messages_field[i + 1].get("role") == "assistant":
                last_user_idx = i
        if last_user_idx is None:
            return None
        prompt_messages = messages_field[: last_user_idx + 1]
        target_message = messages_field[last_user_idx + 1]
        return prompt_messages, target_message.get("content", "")

    turns = example.get("turns", [])
    if not turns:
        return None
    last_junior_idx = None
    for i in range(len(turns) - 1):
        if turns[i].get("role") == "junior_engineer" and turns[i + 1].get("role") == "senior_engineer":
            last_junior_idx = i
    if last_junior_idx is None:
        return None
    prompt_turns = turns[: last_junior_idx + 1]
    target_turn = turns[last_junior_idx + 1]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in prompt_turns:
        messages.append({"role": _map_role(turn.get("role", "")), "content": turn.get("content", "")})
    return messages, target_turn.get("content", "")


def build_prompt(tokenizer: PreTrainedTokenizerBase, messages: List[Dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    # Fallback formatting for tokenizers without a chat template.
    rendered = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        rendered.append(f"{role.upper()}:\n{content}")
    return "\n\n".join(rendered).strip() + "\n\nASSISTANT:\n"


def count_tokens(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    if hasattr(tokenizer, "encode"):
        return len(tokenizer.encode(text, add_special_tokens=False))
    return len(text.split())


def _batch_iter(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def generate_outputs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    examples: Sequence[Dict[str, Any]],
    gen_cfg: GenerationConfig,
    batch_size: int,
    max_length: Optional[int],
) -> List[str]:
    prompts = []
    for ex in examples:
        split = split_prompt_target(ex)
        if split is None:
            continue
        messages, _target = split
        prompts.append(build_prompt(tokenizer, messages))

    device = model.device
    outputs: List[str] = []
    do_sample = gen_cfg.do_sample
    if do_sample is None:
        do_sample = gen_cfg.temperature > 0

    tokenizer.padding_side = "left"
    model.eval()
    with torch.no_grad():
        for batch_prompts in _batch_iter(prompts, batch_size):
            tokenizer_kwargs = {
                "return_tensors": "pt",
                "padding": True,
                "truncation": max_length is not None,
            }
            if max_length is not None:
                tokenizer_kwargs["max_length"] = max_length
            enc = tokenizer(list(batch_prompts), **tokenizer_kwargs)
            enc = {k: v.to(device) for k, v in enc.items()}
            input_lengths = enc["attention_mask"].sum(dim=1).tolist()
            gen = model.generate(
                **enc,
                max_new_tokens=gen_cfg.max_new_tokens,
                temperature=gen_cfg.temperature,
                top_p=gen_cfg.top_p,
                do_sample=do_sample,
            )
            for idx, seq in enumerate(gen):
                prompt_len = int(input_lengths[idx])
                gen_ids = seq[prompt_len:]
                text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                outputs.append(text.strip())
    return outputs


def compute_eval_loss(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    examples: Sequence[Dict[str, Any]],
    batch_size: int,
    max_length: Optional[int],
) -> float:
    device = model.device
    model.eval()

    prompt_targets = []
    for ex in examples:
        split = split_prompt_target(ex)
        if split is None:
            continue
        messages, target = split
        prompt = build_prompt(tokenizer, messages)
        if not target:
            continue
        prompt_targets.append((prompt, target))

    if not prompt_targets:
        return float("nan")

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in _batch_iter(prompt_targets, batch_size):
            prompts = [p for p, _t in batch]
            full_texts = [p + t for p, t in batch]
            tokenizer_kwargs = {
                "return_tensors": "pt",
                "padding": True,
                "truncation": max_length is not None,
                "add_special_tokens": False,
            }
            if max_length is not None:
                tokenizer_kwargs["max_length"] = max_length
            enc = tokenizer(full_texts, **tokenizer_kwargs)
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = enc["input_ids"].clone()
            labels.fill_(-100)
            prompt_lens = [
                len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts
            ]
            for i, prompt_len in enumerate(prompt_lens):
                labels[i, prompt_len:] = enc["input_ids"][i, prompt_len:]
            labels[enc["attention_mask"] == 0] = -100

            loss = model(**enc, labels=labels).loss
            token_count = int((labels != -100).sum().item())
            total_loss += loss.item() * token_count
            total_tokens += token_count

    if total_tokens == 0:
        return float("nan")
    return total_loss / total_tokens


def compute_metrics(
    outputs: Sequence[str],
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> Dict[str, float]:
    outputs = list(outputs)
    metrics: Dict[str, float] = {}
    if not outputs:
        return metrics

    metrics["withhold_violation_rate"] = sum(is_withhold_violation(t) for t in outputs) / len(outputs)
    metrics["numeric_violation_rate"] = sum(is_numeric_violation(t) for t in outputs) / len(outputs)
    metrics["question_rate"] = question_rate(outputs)
    metrics["avg_questions_per_response"] = avg_questions_per_response(outputs)
    metrics["avg_interrogative_ratio"] = avg_interrogative_ratio(outputs)

    lengths_chars = [len(t) for t in outputs]
    metrics["avg_response_length_chars"] = sum(lengths_chars) / len(lengths_chars)

    if tokenizer is None:
        lengths_tokens = [len(t.split()) for t in outputs]
    else:
        lengths_tokens = [count_tokens(tokenizer, t) for t in outputs]
    metrics["avg_response_length_tokens"] = sum(lengths_tokens) / len(lengths_tokens)

    sorted_tokens = sorted(lengths_tokens)
    p95_rank = math.ceil(0.95 * len(sorted_tokens)) - 1
    p95_idx = max(0, min(p95_rank, len(sorted_tokens) - 1))
    metrics["p95_response_length_tokens"] = sorted_tokens[p95_idx]

    metrics["step_list_rate"] = sum(count_step_lines(t) >= 3 for t in outputs) / len(outputs)
    metrics["contains_imperative_rate"] = sum(contains_imperative(t) for t in outputs) / len(outputs)

    return metrics


def evaluate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    eval_data: Sequence[Dict[str, Any]],
    cfg: EvalConfig,
) -> Dict[str, float]:
    rng = random.Random(cfg.seed)
    eval_examples = list(eval_data)
    if cfg.max_samples is not None and len(eval_examples) > cfg.max_samples:
        eval_examples = rng.sample(eval_examples, cfg.max_samples)

    outputs = generate_outputs(
        model=model,
        tokenizer=tokenizer,
        examples=eval_examples,
        gen_cfg=cfg.generation,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
    )

    metrics = compute_metrics(outputs, tokenizer=tokenizer)
    eval_loss = compute_eval_loss(
        model=model,
        tokenizer=tokenizer,
        examples=eval_examples,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
    )
    metrics["eval_loss"] = eval_loss

    if math.isnan(eval_loss):
        metrics["perplexity"] = float("nan")
    else:
        safe_loss = min(eval_loss, 20.0)
        metrics["perplexity"] = math.exp(safe_loss)
    return metrics


def log_wandb(metrics: Dict[str, float]) -> None:
    try:
        import wandb  # type: ignore
    except Exception:
        return
    wandb.log(metrics)


def run_eval(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    eval_json: str,
    cfg: EvalConfig,
    use_wandb: bool = False,
) -> Dict[str, float]:
    eval_data = _load_json_or_jsonl(eval_json)
    metrics = evaluate(model=model, tokenizer=tokenizer, eval_data=eval_data, cfg=cfg)
    if use_wandb:
        log_wandb(metrics)
    return metrics


if __name__ == "__main__":
    dummy_outputs = [
        "What do you think happens if you change the bias?",
        "Final: Use 10k ohm and set it to 3.3V.",
    ]

    class _WhitespaceTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
            return list(range(len(text.split())))

    metrics = compute_metrics(dummy_outputs, tokenizer=_WhitespaceTokenizer())
    assert "withhold_violation_rate" in metrics
    print("evaluator.py smoke check passed")
