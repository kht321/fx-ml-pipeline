"""Simple test of FinGPT inference to debug output issues."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

print("Loading FinGPT...")

# Load base model
base_model_name = "NousResearch/Llama-2-13b-hf"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
base_model = base_model.to("mps")

# Apply LoRA
model = PeftModel.from_pretrained(base_model, "FinGPT/fingpt-sentiment_llama2-13b_lora")
model.eval()

print("✓ Model loaded\n")

# Simple test
prompt = """Analyze the sentiment of this financial news:

News: The Federal Reserve announced interest rate cuts, boosting market sentiment.

Sentiment (positive/negative/neutral):"""

print("Prompt:")
print(prompt)
print("\n" + "="*80)

inputs = tokenizer(prompt, return_tensors="pt").to("mps")

print("Generating response...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

print("Decoding...")
full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nFull response:")
print(full_response)

print("\n" + "="*80)
print("Model response only:")
model_response = full_response[len(prompt):].strip()
print(model_response)

print("\n" + "="*80)
print("Character codes:")
print([ord(c) for c in model_response[:50]])
