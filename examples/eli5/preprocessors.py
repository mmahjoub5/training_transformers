class ELI5Preprocessor_CLM:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.block_size = 512

    def __call__(self, examples):
        answer_token = self.tokenizer(examples["answers.text"][0])
        question_token = self.tokenizer(examples["title"])

        input_ids = question_token["input_ids"] + answer_token["input_ids"]
        attention_mask = [1] * len(input_ids)
        total_length = 0
        if len(input_ids) >= self.block_size:
            total_length = (len(input_ids) // self.block_size) * self.block_size
        result = {
            "input_ids": [
                input_ids[i : i + self.block_size]
                for i in range(0, total_length, self.block_size)
            ],
            "attention_mask": [
                attention_mask[i : i + self.block_size]
                for i in range(0, total_length, self.block_size)
            ],
        }
        result["labels"] = result["input_ids"].copy()
        return result


class ELI5Preprocessor_QA:
    def __init__(self, tokenizer, max_length=512, **tokenizer_kwargs):
        self.tokenizer = tokenizer
        self.max_length = max_length
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer_kwargs = tokenizer_kwargs

    def __call__(self, examples):
        q_text = "Question: " + examples["title"] + "\nAnswer:"
        a_text = examples["answers.text"][0]
        answer_token = self.tokenizer(a_text, add_special_tokens=False, max_length=self.max_length, **self.tokenizer_kwargs)
        question_token = self.tokenizer(q_text, add_special_tokens=False, max_length=self.max_length, **self.tokenizer_kwargs)
        input_ids = question_token["input_ids"] + answer_token["input_ids"]

        labels = [-100] * len(question_token["input_ids"]) + answer_token["input_ids"]
        input_ids = input_ids[: self.max_length]
        labels = labels[: self.max_length]
        attention_mask = [1] * len(input_ids)

        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            pad_id = self.tokenizer.pad_token_id
            input_ids += [pad_id] * pad_len
            attention_mask += [0] * pad_len
            labels += [-100] * pad_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
