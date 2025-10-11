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

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("Using CUDA with float16 weights")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
else:
    if torch.backends.mps.is_available():
        print("MPS available but disabled to avoid generation issues; using CPU.")
    print("Loading base model on CPU (this may take several minutes)")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map={"": "cpu"},
    )

# Apply LoRA
peft_kwargs = {}
if device == "cuda":
    peft_kwargs["device_map"] = {"": "cuda"}
else:
    peft_kwargs["device_map"] = {"": "cpu"}

model = PeftModel.from_pretrained(
    base_model,
    "FinGPT/fingpt-sentiment_llama2-13b_lora",
    **peft_kwargs,
)
model.eval()
if device == "cuda":
    model = model.to("cuda")

print("✓ Model loaded\n")

# Simple test
prompt = """Analyze the sentiment of this financial news:

News: The Federal Reserve announced interest rate cuts, boosting market sentiment.

Sentiment (positive/negative/neutral):"""

print("Prompt:")
print(prompt)
print("\n" + "="*80)

inputs = tokenizer(prompt, return_tensors="pt")
if device == "cuda":
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

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
