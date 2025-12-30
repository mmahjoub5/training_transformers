def get_best_valid_answer(start_logits, end_logits, offset_mapping, 
                          n_best_size=20, max_answer_length=30):
    """Find best valid answer span using n_best approach."""
    # Get top n_best start and end indices
    start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
    end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
    
    valid_answers = []
    
    for start_index in start_indexes:
        for end_index in end_indexes:
            # Skip out-of-scope answers (special tokens)
            if (
                start_index >= len(offset_mapping)
                or end_index >= len(offset_mapping)
                or offset_mapping[start_index] is None
                or offset_mapping[end_index] is None
            ):
                continue
            
            # Skip invalid spans
            if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                continue
            
            # Valid answer found
            score = start_logits[start_index] + end_logits[end_index]
            valid_answers.append({
                "start": start_index,
                "end": end_index,
                "score": score
            })
    
    # Return best or fallback to argmax
    if valid_answers:
        best = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        return best["start"], best["end"]
    else:
        return int(np.argmax(start_logits)), int(np.argmax(end_logits))

def compute_token_f1(pred_start, pred_end, true_start, true_end):
    """
    Compute F1 score between predicted and true token spans.
    
    Args:
        pred_start: Predicted start position
        pred_end: Predicted end position
        true_start: True start position
        true_end: True end position
        
    Returns:
        float: F1 score (0.0 to 1.0)
    """
    pred_tokens = set(range(pred_start, pred_end + 1))
    true_tokens = set(range(true_start, true_end + 1))
    
    # If either is empty, return 1.0 if both empty, else 0.0
    if len(pred_tokens) == 0 or len(true_tokens) == 0:
        return float(pred_tokens == true_tokens)
    
    # Calculate overlap
    common = pred_tokens & true_tokens
    if len(common) == 0:
        return 0.0
    
    # Calculate precision and recall
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(true_tokens)
    
    # Calculate F1
    f1 = 2 * precision * recall / (precision + recall)
    return f1
