'''
Preprocess ELI5 dataset for category prediction task.
This implements chunking for causal language models

'''

class ELI5Preprocessor_CLM:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.block_size = 512

    def __call__(self, examples):
        result = {}
        print(examples["answers.text"][0])
        answer_token = self.tokenizer(examples["answers.text"][0])
        question_token = self.tokenizer(examples["title"])

        input_ids = question_token["input_ids"] + answer_token["input_ids"]
        attention_mask = [1] * len(input_ids)
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
      

        # Split by chunks of block_size.
        result["labels"] = result["input_ids"].copy()

        return result

"""
Class for preprocessing ELI5 dataset for QA tasks. 

Adds -100 on label for question tokens. so the loss is only computed on answer tokens.


"""
class ELI5Preprocessor_QA:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # For many decoder-only models, pad_token may be None; EOS-as-pad is common.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, examples):
        q_text = f"Question: {examples['title']}\nAnswer:"
        a_text = examples["answers.text"][0]
        
        answer_token = self.tokenizer(a_text, add_special_tokens=False)
        question_token = self.tokenizer(q_text, add_special_tokens=False)
        input_ids = question_token["input_ids"] + answer_token["input_ids"]
        
        # Split by chunks of block_size.
        labels =  [-100] * len(question_token["input_ids"]) + answer_token["input_ids"]
        # Truncate
        input_ids = input_ids[: self.max_length]
        labels = labels[: self.max_length]
        attention_mask = [1] * len(input_ids)

        # Pad
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