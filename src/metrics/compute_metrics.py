
import numpy as np
from sklearn.metrics import accuracy_score
from typing import Optional

from metrics_eval.evaluator import compute_metrics as compute_behavior_metrics
from src.metrics.metric_utils import get_best_valid_answer, compute_token_f1




class QAMetricsComputer:
    """
    Simple compute_metrics for QA with access to validation dataset.
    
    Usage:
        metrics_computer = QAMetricsComputer(validation_dataset)
        
        trainer = Trainer(
            model=model,
            compute_metrics=metrics_computer,
            ...
        )
    """
    
    def __init__(self, validation_dataset, n_best_size=20, max_answer_length=30):
        """
        Args:
            validation_dataset: Tokenized validation dataset with offset_mapping
            n_best_size: Number of top candidates to consider
            max_answer_length: Maximum answer length in tokens
        """
        self.dataset = validation_dataset
        self.n_best_size = n_best_size
        self.max_answer_length = max_answer_length
    
    def __call__(self, eval_pred):
        """Called by Trainer.evaluate()"""
        logits, labels = eval_pred
        
        # Extract logits and labels
        start_logits = np.array(logits[0])  # (batch_size, seq_length)
        end_logits = np.array(logits[1])    # (batch_size, seq_length)
        start_labels = np.array(labels[0])  # (batch_size,)
        end_labels = np.array(labels[1])    # (batch_size,)
        
        batch_size = start_logits.shape[0]
        
        # Get predictions with n_best + offset_mapping
        start_preds = []
        end_preds = []
        f1_scores = []

        for i in range(batch_size):
            offset_mapping = self.dataset[i]["offset_mapping"]
            
            start_pred, end_pred = get_best_valid_answer(
                start_logits[i],
                end_logits[i],
                offset_mapping,
                n_best_size=self.n_best_size,
                max_answer_length=self.max_answer_length
            )
            
            start_preds.append(start_pred)
            end_preds.append(end_pred)
            f1 = compute_token_f1(start_pred, end_pred, start_labels[i], end_labels[i])
            f1_scores.append(f1)

        start_preds = np.array(start_preds)
        end_preds = np.array(end_preds)
        
        # Compute metrics
        start_accuracy = accuracy_score(start_labels, start_preds)
        end_accuracy = accuracy_score(end_labels, end_preds)
        exact_match = np.mean((start_preds == start_labels) & (end_preds == end_labels))
        valid_spans = np.mean(start_preds <= end_preds)
        f1 = np.mean(f1_scores)

        return {
            "start_accuracy": float(start_accuracy),
            "end_accuracy": float(end_accuracy),
            "exact_match": float(exact_match),
            "valid_span_ratio": float(valid_spans),
            "f1": float(f1),
        }


class BehaviorMetricsComputer:
    def __init__(self, tokenizer, max_samples: Optional[int] = None):
        self.tokenizer = tokenizer
        self.max_samples = max_samples

    def __call__(self, eval_pred):
        print("we are here")
        logits, labels = eval_pred
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        if self.max_samples is not None:
            logits = logits[: self.max_samples]
            labels = labels[: self.max_samples]

        # Align like CausalLM loss does: logits[:-1] predicts labels[1:]
        pred_ids = np.argmax(logits, axis=-1)          # (B, T)
        pred_ids = pred_ids[:, :-1]                    # (B, T-1)
        labels_shift = labels[:, 1:]                   # (B, T-1)

        outputs = []
        for pred_row, label_row in zip(pred_ids, labels_shift):
            mask = label_row != -100
            if not np.any(mask):
                continue

            # Decode only the predicted tokens where we had label supervision
            text = self.tokenizer.decode(pred_row[mask], skip_special_tokens=True)
            outputs.append(text.strip())

        return compute_behavior_metrics(outputs, tokenizer=self.tokenizer)
