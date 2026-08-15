from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "Qwen/Qwen3-4B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

def generate(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=512)
    response = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

if __name__ == "__main__":
    while True:
        prompt = input("You: ")
        print("Model:", generate(prompt))