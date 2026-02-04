import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL_ID = "microsoft/Phi-3.5-mini-instruct"   # or your base path
ADAPTER_DIR = "/home/ubuntu/training_transformers/output/eos_fix/phi35/checkpoint-20"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=False, use_fast=True)
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
)

INFER_TEMPLATE = r"""{% for message in messages -%}
{% if message['role'] == 'system' -%}<|system|>
{{ message['content'] }}
{% elif message['role'] == 'user' -%}<|user|>
{{ message['content'] }}
{% elif message['role'] == 'assistant' -%}<|assistant|>
{{ message['content'] }}
{% endif -%}
{{ "\n" -}}
{% endfor -%}
{% if add_generation_prompt -%}<|assistant|>
{% endif -%}""".strip()



# IMPORTANT: resize base to tokenizer length BEFORE loading adapter
base.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model.eval()
model = model.merge_and_unload()
print("chat template")
print(tokenizer.chat_template)

tokenizer.chat_template = INFER_TEMPLATE
SYSTEM_PROMPT = """
You are a strict Socratic tutor. 
Ask at most 1 question per response. and stop after initial question
Do not include sections, headings, or exercises.
Do not say “Your Turn”.
Do not role-play.
After asking the questions, stop.
""".strip()

EXAMPLE_CONVERSATION = """
Example conversation style (follow this pattern):

junior_engineer: I heard spread-spectrum clocks can be tricky to measure. What makes them so different from normal clocks when we use lab equipment?
senior_engineer: Before we get into measurement specifics, what happens to the frequency of a spread-spectrum clock compared to a fixed clock?
junior_engineer: I think it changes a little over time, but I’m not sure how that affects what we see.
senior_engineer: Right, it’s being modulated. If your scope or analyzer assumes a single frequency, what might that modulation do to the captured waveform or spectrum?
junior_engineer: Maybe it makes the signal look unstable or blurry?
senior_engineer: Yes. Now when you measure such a clock, you have to decide whether you care about the instantaneous frequency or the averaged behavior. What type of instrument settings or measurement approach would help you see the modulation pattern clearly rather than treating it as noise?
junior_engineer: Maybe use a spectrum view instead of time domain? Or adjust triggering so it locks onto the changing frequency?
senior_engineer: Good thinking. Instead of forcing a perfect lock, you can choose a mode that tolerates slow frequency drift. What sort of time window or resolution trade-offs might you face when you want both modulation detail and overall stability in the reading?
""".strip()

# Build initial conversation
messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
]
end_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

@torch.no_grad()
def generate_reply(messages, max_new_tokens=200, temperature=0.7, top_p=0.9):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=400,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=end_id,
        pad_token_id=tokenizer.pad_token_id,
        # stop_strings=["<|user|>", "<|system|>", "\n### user", "\n### assistant"],
        # tokenizer=tokenizer,   # <-- MUST be here
    )

    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def main():
    print(f"Loaded: {ADAPTER_DIR}")
    print("Type your question. Commands: /exit, /reset, /params\n")

    # default params
    gen_params = {"max_new_tokens": 10, "temperature": 0.7, "top_p": 0.9}

    while True:
        user_text = input("you> ").strip()
        if not user_text:
            continue

        if user_text.lower() in ["/exit", "exit", "quit", "/quit"]:
            break

        if user_text.lower() == "/reset":
            messages.clear()
            messages.append({"role": "system", "content": SYSTEM_PROMPT + "\n\n" + EXAMPLE_CONVERSATION})
            print("(reset conversation)\n")
            continue

        if user_text.lower().startswith("/params"):
            # Example: /params temperature=0.2 top_p=0.95 max_new_tokens=120
            parts = user_text.split()[1:]
            for p in parts:
                if "=" not in p:
                    continue
                k, v = p.split("=", 1)
                if k in ["temperature", "top_p"]:
                    gen_params[k] = float(v)
                elif k in ["max_new_tokens"]:
                    gen_params[k] = int(v)
            print(f"(params) {gen_params}\n")
            continue

        messages.append({"role": "user", "content": user_text})

        assistant_text = generate_reply(messages, **gen_params)
        print("waitinf for response")
        print(f"phi> {assistant_text}\n")

        messages.append({"role": "assistant", "content": assistant_text})

if __name__ == "__main__":
    main()
