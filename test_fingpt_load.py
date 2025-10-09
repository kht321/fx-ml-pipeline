"""Test loading FinGPT model to diagnose issues."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")

model_name = "FinGPT/fingpt-sentiment_llama2-13b_lora"
print(f"\nAttempting to load: {model_name}")
print(f"Start time: {time.strftime('%H:%M:%S')}")

try:
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"   ✓ Tokenizer loaded")

    print("\n2. Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    print(f"   ✓ Model loaded")
    print(f"   Device: {model.device}")
    print(f"   Dtype: {model.dtype}")

    print(f"\nEnd time: {time.strftime('%H:%M:%S')}")
    print("\n✓ SUCCESS: FinGPT model loaded successfully!")

except Exception as e:
    print(f"\n✗ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
