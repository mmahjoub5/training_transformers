"""
Class for preprocessing custom socratic dialogue dataset created by myself


"""


class SocraticPreprocessor:
    def __init__(self, tokenizer, max_length=512,**tokenizer_kwargs):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # For many decoder-only models, pad_token may be None; EOS-as-pad is common.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer_kwargs = tokenizer_kwargs
        self.message = [] 

    def _fallback_chat_format(self, messages):
        # Simple, consistent format when tokenizer lacks a chat template.
        rendered = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            rendered.append(f"{role.upper()}:\n{content}")
        return "\n\n".join(rendered).strip()
    
    def policy_system_prompt_generator(self, policy):
        base = "You are a senior engineer who teaches by asking guided questions. The Hint level hint_level: \n 1 = Socratic clarification \n 2 = Physical intuition \n 3 = System-level reasoning \n"
        
        policies = []
        if policy["withhold_final_answer"]:
            policies.append("withhold the final answer")
        if policy["avoid_numeric_rules_of_thumb"]:
            policies.append("avoid numeric rules of thumb")
        
        if policies:
            policy_text = " and ".join(policies)
            system_prompt = f"{base}\nYour policy that you must follow under all circumstances is to {policy_text}."
        else:
            system_prompt = f"{base}\nProvide clear, direct answers with examples when helpful."
    
        return system_prompt
    def __call__(self, examples):
        self.messages = []
        print(examples.keys())
        self.messages.append({
            "role": "system",
            "content": self.policy_system_prompt_generator(examples["policy"])
        })
        for turn in examples["turns"]:
            if turn["role"] == "junior_engineer":
                self.messages.append({
                    "role": "user",
                    "content": turn["content"]
                })
            elif turn["role"] == "senior_engineer":
                self.messages.append({
                    "role": "assistant",
                    "content": f"[HINT_LEVEL={turn['hint_level']}]\n{turn['content']}"
                })

        if getattr(self.tokenizer, "chat_template", None):
            print(self.messages)
            text = self.tokenizer.apply_chat_template(
                self.messages,
                tokenize=False,
                add_generation_prompt=False,
            )
           
        else:
            print("we are here ")
            text = self._fallback_chat_format(self.messages)

        out = {
            "text": text
        }
        return out