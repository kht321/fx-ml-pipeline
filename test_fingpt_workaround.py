"""Test FinGPT loading with workaround for Python 3.13 sentencepiece issue."""
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

print(f"PyTorch version: {torch.__version__}")
print(f"Python version: {__import__('sys').version}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")

# Try different FinGPT models
models_to_try = [
    "oliverwang15/FinGPT_v32_Llama2_Sentiment_Instruction_LoRA_FT",
    "FinGPT/fingpt-sentiment_llama2-13b_lora",
]

for model_name in models_to_try:
    print(f"\n{'='*80}")
    print(f"Attempting to load: {model_name}")
    print(f"{'='*80}")

    try:
        print("\n1. Loading tokenizer with use_fast=False (bypass fast tokenizer)...")
        start = time.time()

        # Try to bypass the fast tokenizer which has the Python 3.13 bug
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,  # Use slow tokenizer to avoid sentencepiece bug
            trust_remote_code=True
        )

        load_time = time.time() - start
        print(f"   ✓ Tokenizer loaded in {load_time:.2f}s")

        print("\n2. Loading model (this may take a while for 13B model)...")
        start = time.time()

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        load_time = time.time() - start
        print(f"   ✓ Model loaded in {load_time:.2f}s")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

        print("\n3. Testing inference...")
        test_text = "The Federal Reserve announced interest rate cuts, boosting market sentiment."

        inputs = tokenizer(test_text, return_tensors="pt")
        start = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        inference_time = time.time() - start
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"   ✓ Inference completed in {inference_time:.2f}s")
        print(f"   Input: {test_text}")
        print(f"   Output: {response[:200]}...")

        print(f"\n✓ SUCCESS with {model_name}!")
        break  # Stop if successful

    except Exception as e:
        print(f"\n✗ FAILED with {model_name}")
        print(f"   Error: {type(e).__name__}: {str(e)[:200]}")
        continue
else:
    print("\n" + "="*80)
    print("All models failed. Trying alternative approach...")
    print("="*80)
