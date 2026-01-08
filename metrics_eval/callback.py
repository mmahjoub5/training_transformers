import random
from typing import Any, Dict, List, Optional, Sequence

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from metrics_eval.evaluator import EvalConfig, GenerationConfig, _load_json_or_jsonl, evaluate
import torch

class MetricsEvalCallback(TrainerCallback):
    def __init__(
        self,
        eval_json: Optional[str] = None,
        eval_data: Optional[Sequence[Dict[str, Any]]] = None,
        eval_steps: int = 500,
        max_samples: int = 200,
        batch_size: int = 4,
        max_length: Optional[int] = None,
        seed: int = 42,
        generation: Optional[GenerationConfig] = None,
    ) -> None:
        self.eval_json = eval_json
        self.eval_data = eval_data
        self.eval_steps = eval_steps
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.max_length = max_length
        self.seed = seed
        self.generation = generation or GenerationConfig()
        self._eval_cache: Optional[List[Dict[str, Any]]] = None
        self._eval_subset: Optional[List[Dict[str, Any]]] = None
        self.trainer = None
        self._last_ran_step = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and self._pending_logs:
            logs.update(self._pending_logs)
            self._pending_logs = None
        return control

    def _get_eval_subset(self) -> List[Dict[str, Any]]:
        if self._eval_cache is None:
            if self.eval_data is not None:
                self._eval_cache = list(self.eval_data)
            elif self.eval_json is not None:
                self._eval_cache = _load_json_or_jsonl(self.eval_json)
            else:
                self._eval_cache = []
        if self._eval_subset is None:
            if self.max_samples is None or len(self._eval_cache) <= self.max_samples:
                self._eval_subset = list(self._eval_cache)
            else:
                rng = random.Random(self.seed)
                self._eval_subset = rng.sample(self._eval_cache, self.max_samples)
        return self._eval_subset


    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print("✅ on_train_begin fired; kwargs keys:", list(kwargs.keys())[:100])
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        # global_step increments on optimizer steps (not microbatches)
        #print("✅ on_train_begin fired; kwargs keys:", list(kwargs.keys())[:20])
        step = state.global_step
        if step <= 0 or (step % self.eval_steps) != 0:
            return control

        # guard: sometimes callbacks can be called twice per step in some setups
        if step == self._last_ran_step:
            return control
        self._last_ran_step = step

        # DDP/FSDP guard: only main process should do expensive eval
        # Prefer state.is_world_process_zero if available; else args.local_rank
        is_zero = getattr(state, "is_world_process_zero", None)
        if callable(is_zero):
            if not state.is_world_process_zero:
                return control
        else:
            if getattr(args, "local_rank", -1) not in (-1, 0):
                return control

        model = kwargs.get("model")
        tokenizer = kwargs.get("processing_class")

        # If tokenizer isn't passed, you can optionally stash it at init time,
        # but in many Trainer setups it *is* passed.
        if model is None or tokenizer is None:
            return control

        eval_subset = self._get_eval_subset()
        cfg = EvalConfig(
            batch_size=self.batch_size,
            max_samples=2,
            seed=self.seed,
            generation=self.generation,
            max_length=self.max_length
        )

        was_training = model.training
        model.eval()
        try:
            with torch.inference_mode():
                metrics = evaluate(model=model, tokenizer=tokenizer, eval_data=eval_subset, cfg=cfg)
        finally:
            if was_training:
                model.train()

        # log metrics in a way Trainer picks up
        # on_log will receive these metrics, and they go to wandb/tensorboard/etc.
        self._pending_logs = {f"custom/{k}": v for k, v in metrics.items()}
        control.should_log = True
        
        
        self._pending_logs = {f"custom/{k}": v for k, v in metrics.items()}
        
        # also print if you want:
        print(f"[custom-eval step={step}] {metrics}")

        return control

if __name__ == "__main__":
    callback = MetricsEvalCallback(eval_json="eval_data.json", eval_steps=10)
    assert callback.eval_steps == 10
    print("callback.py smoke check passed")
