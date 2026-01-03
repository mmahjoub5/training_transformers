class SQuADPreprocessor:
    """Preprocessor for SQuAD-style extractive QA datasets."""
    
    def __init__(self, tokenizer, max_length=384):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, examples):
        """Process examples for HuggingFace datasets.map()"""
        # Handle both single examples and batches
        is_batched = isinstance(examples["question"], list)
        
        if is_batched:
            return self._process_batch(examples)
        else:
            return self._process_single(examples)
    
    def _process_single(self, example):
        """Process a single example."""
        enc = self.tokenizer(
            example["question"],
            example["context"],
            truncation="only_second",
            max_length=self.max_length,
            padding="max_length",
            return_offsets_mapping=True,
        )
        
        result = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "offset_mapping": enc["offset_mapping"],
            "context": example["context"],
        }
        
        # Add labels if answers exist
        if self._has_answers(example):
            start_pos, end_pos = self._extract_positions(enc, example)
            result["start_positions"] = start_pos
            result["end_positions"] = end_pos
        
        # Keep id if present
        if "id" in example:
            result["id"] = example["id"]
        
        return result
    
    def _process_batch(self, examples):
        """Process a batch of examples."""
        batch_size = len(examples["question"])
        
        # Tokenize all at once
        encodings = self.tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=self.max_length,
            padding="max_length",
            return_offsets_mapping=True,
        )
        
        result = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "offset_mapping": encodings["offset_mapping"],
            "context": examples["context"],
        }
        
        # Process labels for each example
        if self._has_answers(examples, batched=True):
            start_positions = []
            end_positions = []
            
            for i in range(batch_size):
                enc_single = {
                    "input_ids": encodings["input_ids"][i],
                    "offset_mapping": encodings["offset_mapping"][i],
                }
                
                example_single = {
                    "answers": {
                        "text": [examples["answers"]["text"][i][0]] if examples["answers"]["text"][i] else [],
                        "answer_start": [examples["answers"]["answer_start"][i][0]] if examples["answers"]["answer_start"][i] else [],
                    }
                }
                
                start, end = self._extract_positions(enc_single, example_single)
                start_positions.append(start)
                end_positions.append(end)
            
            result["start_positions"] = start_positions
            result["end_positions"] = end_positions
        
        # Keep ids if present
        if "id" in examples:
            result["id"] = examples["id"]
        
        return result
    
    def _extract_positions(self, enc, example):
        """Extract start and end token positions."""
        if not self._has_answers(example):
            cls_index = enc["input_ids"].index(self.tokenizer.cls_token_id)
            return cls_index, cls_index
        
        offsets = enc["offset_mapping"]
        seq_ids = self.tokenizer.sequence_ids(enc["input_ids"]) if hasattr(self.tokenizer, 'sequence_ids') else None
        
        # If tokenizer doesn't have sequence_ids, compute manually
        if seq_ids is None:
            seq_ids = []
            question_end = enc["input_ids"].index(self.tokenizer.sep_token_id)
            for i in range(len(enc["input_ids"])):
                if i < question_end:
                    seq_ids.append(0)
                elif i == question_end:
                    seq_ids.append(None)
                else:
                    seq_ids.append(1)
        
        cls_index = enc["input_ids"].index(self.tokenizer.cls_token_id)
        
        try:
            context_start = next(i for i, s in enumerate(seq_ids) if s == 1)
            context_end = max(i for i, s in enumerate(seq_ids) if s == 1)
        except (StopIteration, ValueError):
            return cls_index, cls_index
        
        start_char = example["answers"]["answer_start"][0]
        end_char = start_char + len(example["answers"]["text"][0])
        
        # Find token positions
        token_start = self._find_start_token(offsets, start_char, context_start)
        token_end = self._find_end_token(offsets, end_char, context_start, context_end)
        
        # Validate positions
        if (token_start < context_start or token_end > context_end or
            token_start == -1 or token_end == -1 or
            offsets[token_start][0] > start_char or
            offsets[token_end][1] < end_char):
            return cls_index, cls_index
        
        return token_start, token_end
    
    def _find_start_token(self, offsets, start_char, context_start):
        """Find first token where offset[0] > start_char, then go back one."""
        token_idx = context_start
        while token_idx < len(offsets) and offsets[token_idx][0] <= start_char:
            token_idx += 1
        return token_idx - 1 if token_idx > context_start else -1
    
    def _find_end_token(self, offsets, end_char, context_start, context_end):
        """Find first token where offset[1] >= end_char."""
        token_idx = context_start
        while token_idx <= context_end and offsets[token_idx][1] < end_char:
            token_idx += 1
        return token_idx if token_idx <= context_end else -1
    
    def _has_answers(self, example, batched=False):
        """Check if example has valid answers."""
        if batched:
            return (
                "answers" in example
                and example["answers"] is not None
                and "text" in example["answers"]
                and "answer_start" in example["answers"]
            )
        else:
            return (
                "answers" in example
                and example["answers"] is not None
                and "text" in example["answers"]
                and "answer_start" in example["answers"]
                and len(example["answers"]["text"]) > 0
                and len(example["answers"]["answer_start"]) > 0
            )


