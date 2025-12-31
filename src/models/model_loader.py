# src/models/model_loader.py
from __future__ import annotations

from typing import Literal, Tuple, Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

ModelKind = Literal["qa", "clm"]  # qa = span-extractive, clm = causal language model


def load_model(
    model_name: str,
    kind: ModelKind,
    precision: Literal["fp32", "fp16", "bf16"] = "fp32",
    trust_remote_code: bool = False,
    **args,
) -> Tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """
    Minimal model loader.

    Args:
      model_name: HF name or local path (e.g., "distilbert-base-uncased")
      kind:
        - "qa": AutoModelForQuestionAnswering (span QA, start/end positions) (BERT type models)
        - "clm": AutoModelForCausalLM (decoder LM, next-token prediction) (GPT type models)
      precision: "fp32" (safe), or fp16/bf16 (GPU)
      trust_remote_code: set True only if you know the repo requires it

    Returns:
      (tokenizer, model)
    """
    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[precision]

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=trust_remote_code)

    if kind == "qa":
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=trust_remote_code
        )
        return tokenizer, model

    if kind == "clm":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=trust_remote_code, **args
        )

        # Many decoder-only tokenizers don't have a pad token; set it to eos for batching.
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer has no pad_token and no eos_token; can't set padding.")
            tokenizer.pad_token = tokenizer.eos_token

        # Keep model config consistent with tokenizer
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        return tokenizer, model

    raise ValueError(f"Unknown kind: {kind}")


def quick_info(tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel) -> str:
    """Tiny helper for printing/logging."""
    p = next(model.parameters())
    return (
        f"tokenizer={tokenizer.__class__.__name__} pad={tokenizer.pad_token_id} eos={tokenizer.eos_token_id}\n"
        f"model={model.__class__.__name__} dtype={p.dtype} device={p.device} params={sum(x.numel() for x in model.parameters()):,}"
    )
