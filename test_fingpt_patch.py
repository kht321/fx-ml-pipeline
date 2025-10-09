"""Test FinGPT with direct patch for Python 3.13 sentencepiece bug."""
import os
import sys
from pathlib import Path
import torch
import time

# CRITICAL: Monkey-patch before importing transformers
# The bug is in sentencepiece expecting str but getting Path
import sentencepiece

original_load = sentencepiece.SentencePieceProcessor.Load

def patched_load(self, model_file):
    """Patched Load that converts Path to str."""
    if isinstance(model_file, Path):
        model_file = str(model_file)
    return original_load(self, model_file)

sentencepiece.SentencePieceProcessor.Load = patched_load

# Also patch LoadFromFile
original_load_from_file = sentencepiece.SentencePieceProcessor.LoadFromFile

def patched_load_from_file(self, arg):
    """Patched LoadFromFile that converts Path to str."""
    if isinstance(arg, Path):
        arg = str(arg)
    return original_load_from_file(self, arg)

sentencepiece.SentencePieceProcessor.LoadFromFile = patched_load_from_file

print("✓ Applied sentencepiece monkey-patch for Python 3.13")

# Now import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

print(f"\nPyTorch version: {torch.__version__}")
print(f"Python version: {sys.version}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")

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

    print("\n2. Loading model...")
    print("   NOTE: 13B model will take several minutes and ~30GB RAM")
    start = time.time()

    # Use 8-bit quantization if on CUDA, otherwise full precision
    load_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    if torch.cuda.is_available():
        load_kwargs["load_in_8bit"] = True
        load_kwargs["device_map"] = "auto"
    elif torch.backends.mps.is_available():
        # MPS (Apple Silicon) - use float16
        load_kwargs["torch_dtype"] = torch.float16
        print("   Using MPS (Apple Silicon) acceleration with float16")
    else:
        # CPU only - this will be slow and memory intensive
        load_kwargs["torch_dtype"] = torch.float32
        print("   WARNING: Loading on CPU - this will be very slow!")

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    # Move to MPS if available (for non-CUDA systems)
    if torch.backends.mps.is_available() and not torch.cuda.is_available():
        model = model.to("mps")
        print("   Model moved to MPS device")

    load_time = time.time() - start
    params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"   ✓ Model loaded in {load_time:.2f}s")
    print(f"   Parameters: {params:.2f}B")
    print(f"   Device: {next(model.parameters()).device}")

    print("\n3. Testing inference...")
    test_text = "The Federal Reserve announced interest rate cuts, boosting market sentiment and supporting economic growth."

    print(f"   Input: {test_text}")

    inputs = tokenizer(test_text, return_tensors="pt")

    # Move inputs to same device as model
    if torch.backends.mps.is_available() and not torch.cuda.is_available():
        inputs = {k: v.to("mps") for k, v in inputs.items()}

    start = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=150,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    inference_time = time.time() - start
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"\n   ✓ Inference completed in {inference_time:.2f}s")
    print(f"   Output: {response}")

    print(f"\n{'='*80}")
    print(f"✓ SUCCESS! FinGPT is now working on Python 3.13")
    print(f"{'='*80}")
    print(f"\nSummary:")
    print(f"  - Model load time: {load_time:.2f}s")
    print(f"  - Inference time: {inference_time:.2f}s")
    print(f"  - Device: {next(model.parameters()).device}")
    print(f"  - Model size: {params:.2f}B parameters")

except Exception as e:
    print(f"\n✗ FAILED")
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
