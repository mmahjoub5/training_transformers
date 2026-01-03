import random
from typing import Any, Dict, List, Optional

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from metrics_eval.evaluator import EvalConfig, GenerationConfig, _load_json_or_jsonl, evaluate


class MetricsEvalCallback(TrainerCallback):
    def __init__(
        self,
        eval_json: str,
        eval_steps: int = 500,
        max_samples: int = 200,
        batch_size: int = 4,
        seed: int = 42,
        generation: Optional[GenerationConfig] = None,
    ) -> None:
        self.eval_json = eval_json
        self.eval_steps = eval_steps
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.seed = seed
        self.generation = generation or GenerationConfig()
        self._eval_cache: Optional[List[Dict[str, Any]]] = None
        self._eval_subset: Optional[List[Dict[str, Any]]] = None

    def _get_eval_subset(self) -> List[Dict[str, Any]]:
        if self._eval_cache is None:
            self._eval_cache = _load_json_or_jsonl(self.eval_json)
        if self._eval_subset is None:
            if self.max_samples is None or len(self._eval_cache) <= self.max_samples:
                self._eval_subset = list(self._eval_cache)
            else:
                rng = random.Random(self.seed)
                self._eval_subset = rng.sample(self._eval_cache, self.max_samples)
        return self._eval_subset

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        if state.global_step == 0 or state.global_step % self.eval_steps != 0:
            return control
        trainer = kwargs.get("trainer")
        if trainer is None:
            return control
        model = trainer.model
        tokenizer = trainer.tokenizer
        if tokenizer is None:
            return control

        eval_subset = self._get_eval_subset()
        cfg = EvalConfig(
            batch_size=self.batch_size,
            max_samples=None,
            seed=self.seed,
            generation=self.generation,
        )
        metrics = evaluate(model=model, tokenizer=tokenizer, eval_data=eval_subset, cfg=cfg)
        trainer.log(metrics)
        return control


if __name__ == "__main__":
    callback = MetricsEvalCallback(eval_json="eval_data.json", eval_steps=10)
    assert callback.eval_steps == 10
    print("callback.py smoke check passed")
