"""Test loading FinBERT model as an alternative to FinGPT."""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import time

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")

model_name = "ProsusAI/finbert"
print(f"\nAttempting to load: {model_name}")
print(f"Start time: {time.strftime('%H:%M:%S')}")

try:
    print("\n1. Loading FinBERT model and tokenizer...")
    start = time.time()

    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model_name,
        device=0 if torch.cuda.is_available() else -1
    )

    load_time = time.time() - start
    print(f"   ✓ Model loaded in {load_time:.2f} seconds")

    print("\n2. Testing inference...")
    test_text = "The Federal Reserve announced interest rate cuts, boosting market sentiment."

    start = time.time()
    result = sentiment_pipeline(test_text)
    inference_time = time.time() - start

    print(f"   ✓ Inference completed in {inference_time:.2f} seconds")
    print(f"   Result: {result}")

    print(f"\nEnd time: {time.strftime('%H:%M:%S')}")
    print("\n✓ SUCCESS: FinBERT model works!")
    print(f"Total time: {load_time + inference_time:.2f} seconds")

except Exception as e:
    print(f"\n✗ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
