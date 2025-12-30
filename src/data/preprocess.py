def extract_start_end_positions(tokenizer, enc, offsets, data_item, context_start=0, context_end=None):
    """Extract token positions using official HuggingFace approach."""
    
    start_char = data_item["answers"]["answer_start"][0]
    end_char = start_char + len(data_item["answers"]["text"][0])
    
    # Start position: find first token where offset[0] > start_char, then go back one
    token_start_index = context_start
    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
        token_start_index += 1
    token_start = token_start_index - 1
    
    # End position: find first token where offset[1] >= end_char
    token_end_index = context_start
    while token_end_index <= context_end and offsets[token_end_index][1] < end_char:
        token_end_index += 1
    token_end = token_end_index

    # CRITICAL: Check if answer is actually within the token span
    # If start/end fall outside the found tokens, the answer was truncated
    if (token_start < context_start or token_end > context_end or 
        offsets[token_start][0] > start_char or 
        offsets[token_end][1] < end_char):
        # Answer doesn't align with tokens - return CLS (will be handled by caller)
        return -1, -1
    
    return token_start, token_end


def preprocess_dataset(raw_ds, tokenizer, max_length):
    """
    Preprocess Hugging Face SQuAD-style dataset for extractive QA.

    Returns:
      dict with processed splits, e.g. {"train": [...], "validation": [...]}
      Each item contains input_ids, attention_mask, and (for labeled splits) start_positions/end_positions.
    """
    processed = {}

    def has_answers(item):
        return (
            "answers" in item
            and item["answers"] is not None
            and "text" in item["answers"]
            and "answer_start" in item["answers"]
            and len(item["answers"]["text"]) > 0
            and len(item["answers"]["answer_start"]) > 0
        )
    print("Starting preprocessing...")
    print(raw_ds)
    for split_name in raw_ds.keys():
        processed_items = []
        print(f"Preprocessing split: {split_name}")

        for data_item in raw_ds[split_name]:
            enc = tokenizer(
                data_item["question"],
                data_item["context"],
                truncation="only_second",
                max_length=max_length,
                padding="max_length",
                return_offsets_mapping=True,
            )

            # Always keep model inputs
            out = {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "offset_mapping": enc["offset_mapping"],  # for metrics
                "context": data_item["context"],  # ADD THIS!
            }

            # Only compute labels if this split has answers
            if has_answers(data_item):
                offsets = enc["offset_mapping"]
                seq_ids = enc.sequence_ids()  # None / 0(question) / 1(context)

                cls_index = enc["input_ids"].index(tokenizer.cls_token_id)
                context_start = next(i for i, s in enumerate(seq_ids) if s == 1)
                context_end = max(i for i, s in enumerate(seq_ids) if s == 1)

                token_start, token_end = extract_start_end_positions(
                    tokenizer=tokenizer,
                    enc=enc,
                    offsets=offsets,
                    data_item=data_item,
                    context_start=context_start,
                    context_end=context_end,
                )

                if token_start > context_end or token_end < context_start:
                    out["start_positions"] = cls_index
                    out["end_positions"] = cls_index
                else:
                    out["start_positions"] = token_start
                    out["end_positions"] = token_end

            # (optional) keep id for debugging / eval
            if "id" in data_item:
                out["id"] = data_item["id"]

            processed_items.append(out)

        processed[split_name] = processed_items

    return processed
      

