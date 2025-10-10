"""Test FinGPT with deep patch at transformers level for Python 3.13."""
import os
import sys
from pathlib import Path
import torch
import time

print("Applying deep patches for Python 3.13 compatibility...")

# Patch 1: Fix sentencepiece at the C extension level
import sentencepiece._sentencepiece as sp_ext

original_load_from_file = sp_ext.SentencePieceProcessor_LoadFromFile

def patched_load_from_file(processor, model_file):
    """Convert Path to str before calling C++ extension."""
    if isinstance(model_file, Path):
        model_file = str(model_file)
    elif hasattr(model_file, '__fspath__'):
        model_file = os.fspath(model_file)
    return original_load_from_file(processor, model_file)

sp_ext.SentencePieceProcessor_LoadFromFile = patched_load_from_file
print("  ✓ Patched sentencepiece C++ extension")

# Patch 2: Fix at transformers tokenization level
from transformers.models.llama import tokenization_llama

original_get_spm_processor = tokenization_llama.LlamaTokenizer.get_spm_processor

def patched_get_spm_processor(self, from_slow=False):
    """Ensure vocab_file is string."""
    # Convert vocab_file to string if it's a Path
    if hasattr(self, 'vocab_file'):
        if isinstance(self.vocab_file, Path):
            self.vocab_file = str(self.vocab_file)
        elif hasattr(self.vocab_file, '__fspath__'):
            self.vocab_file = os.fspath(self.vocab_file)
    return original_get_spm_processor(self, from_slow)

tokenization_llama.LlamaTokenizer.get_spm_processor = patched_get_spm_processor
print("  ✓ Patched LlamaTokenizer")

# Now import remaining transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

print(f"\nPyTorch version: {torch.__version__}")
print(f"Python version: {sys.version}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
device = "cuda" if torch.cuda.is_available() else "cpu"
if device != "cuda" and torch.backends.mps.is_available():
    print("NOTE: MPS detected but disabled to avoid generation issues. Using CPU fallback.")

model_name = "FinGPT/fingpt-sentiment_llama2-13b_lora"
print(f"\n{'='*80}")
print(f"Loading FinGPT: {model_name}")
print(f"{'='*80}")

try:
    print("\n1. Loading tokenizer...")
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    load_time = time.time() - start
    print(f"   ✓ Tokenizer loaded in {load_time:.2f}s")

    print("\n2. Loading model (13B parameters - may take 5-10 minutes)...")
    start = time.time()

    # Determine best device and settings
    if device == "cuda":
        print("   Using CUDA with 8-bit quantization")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map="auto",
            trust_remote_code=True
        )
        device = "cuda"
    else:
        print("   WARNING: Using CPU - this will be VERY slow!")
        if torch.backends.mps.is_available():
            print("   (MPS disabled intentionally due to unstable generation behaviour)")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map={"": "cpu"}
        )
        device = "cpu"

    load_time = time.time() - start
    params = sum(p.numel() for p in model.parameters()) / 1e9

    print(f"   ✓ Model loaded in {load_time:.2f}s ({load_time/60:.1f} minutes)")
    print(f"   Parameters: {params:.2f}B")
    print(f"   Device: {device}")

    print("\n3. Testing inference on financial text...")
    test_texts = [
        "The Federal Reserve announced interest rate cuts, boosting market sentiment.",
        "Stock market crashed amid recession fears and bank failures.",
        "GDP growth remains steady despite global economic headwinds."
    ]

    total_inference_time = 0
    for i, text in enumerate(test_texts, 1):
        print(f"\n   Test {i}: {text[:60]}...")

        inputs = tokenizer(text, return_tensors="pt")

        # Move inputs to same device as model
        if device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        start = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id
            )

        inference_time = time.time() - start
        total_inference_time += inference_time

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"   Time: {inference_time:.2f}s")
        print(f"   Output: {response[:150]}...")

    avg_inference_time = total_inference_time / len(test_texts)

    print(f"\n{'='*80}")
    print(f"✓ SUCCESS! FinGPT is now working on Python 3.13")
    print(f"{'='*80}")
    print(f"\nPerformance Summary:")
    print(f"  Model load time:     {load_time:.2f}s ({load_time/60:.1f} min)")
    print(f"  Avg inference time:  {avg_inference_time:.2f}s per article")
    print(f"  Device:              {device}")
    print(f"  Model size:          {params:.2f}B parameters")
    print(f"\nComparison to FinBERT:")
    print(f"  FinBERT load:        ~1s (cached)")
    print(f"  FinBERT inference:   ~0.023s per article")
    print(f"  FinGPT is {avg_inference_time/0.023:.0f}x slower than FinBERT")

except Exception as e:
    print(f"\n✗ FAILED")
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
