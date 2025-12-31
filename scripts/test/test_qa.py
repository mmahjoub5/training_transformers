import torch
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering
)

CKPT_DIR = "/Users/aminmahjoub/training_transformers/output/distilbert_qa/checkpoint-4375"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer + model directly from checkpoint
tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR)
model = AutoModelForQuestionAnswering.from_pretrained(CKPT_DIR)
model.to(device)
model.eval()

def answer_question(question, context):
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits) + 1

    answer = tokenizer.decode(
        inputs["input_ids"][0][start_idx:end_idx],
        skip_special_tokens=True
    )
    return answer

# ---- Try different inputs here ----
if __name__ == "__main__":
    context = """
    Mitochondria are membrane-bound organelles found in most eukaryotic cells. 
    They are often referred to as the powerhouse of the cell because they generate 
    adenosine triphosphate (ATP) through cellular respiration, which provides energy for cellular processes.
    """

    question = "What is the primary function of mitochondria in a cell?"

    print("Q:", question)
    print("A:", answer_question(question, context))

    context = """
    Compound interest is a financial concept where interest is calculated not only on the initial principal but also on the accumulated interest from previous periods. 
    This means that interest can grow exponentially over time if it is reinvested.
    """

    question = "What is compound interest?"

    print("Q:", question)
    print("A:", answer_question(question, context))


    question = "What does an LTI system obey?"

    context  = "A linear time-invariant system obeys the principles of linearity and time invariance, which allow it to be analyzed using convolution and frequency-domain methods."


    print("Q:", question)
    print("A:", answer_question(question, context))