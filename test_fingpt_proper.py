"""Properly load FinGPT as a LoRA adapter on top of LLaMA-2 base model."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import time

print("="*80)
print("LOADING FINGPT SENTIMENT ANALYZER")
print("="*80)

print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")

# Step 1: Load base LLaMA-2 13B model
base_model_name = "NousResearch/Llama-2-13b-hf"
lora_adapter_name = "FinGPT/fingpt-sentiment_llama2-13b_lora"

print(f"\n{'='*80}")
print(f"Step 1: Loading base LLaMA-2 model")
print(f"Model: {base_model_name}")
print(f"{'='*80}")

start = time.time()

try:
    # Load tokenizer from base model (use slow tokenizer to avoid tiktoken issues)
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tok_time = time.time() - start
    print(f"   [OK] Tokenizer loaded in {tok_time:.2f}s")

    # Load base model
    print("\n2. Loading base model (13B parameters - this will take 5-10 minutes)...")
    print("   This is a large model that requires ~30GB RAM")
    start_model = time.time()

    # Determine device and load model - Force CPU to avoid MPS generation issues
    if torch.cuda.is_available():
        print("   Using CUDA with 8-bit quantization")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            load_in_8bit=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
        device = "cuda"
    else:
        print("   Using CPU (MPS disabled due to generation issues)")
        print("   WARNING: This will use significant RAM and be slower")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
            low_cpu_mem_usage=True
        )
        device = "cpu"

    model_time = time.time() - start_model
    params = sum(p.numel() for p in base_model.parameters()) / 1e9

    print(f"   [OK] Base model loaded in {model_time:.2f}s ({model_time/60:.1f} min)")
    print(f"   Parameters: {params:.2f}B")
    print(f"   Device: {device}")

    # Step 2: Apply FinGPT LoRA adapter
    print(f"\n{'='*80}")
    print(f"Step 2: Applying FinGPT LoRA adapter")
    print(f"Adapter: {lora_adapter_name}")
    print(f"{'='*80}\n")

    start_lora = time.time()

    model = PeftModel.from_pretrained(base_model, lora_adapter_name)
    model.eval()  # Set to evaluation mode

    lora_time = time.time() - start_lora
    print(f"   [OK] LoRA adapter applied in {lora_time:.2f}s")

    # Step 3: Test inference
    print(f"\n{'='*80}")
    print(f"Step 3: Testing FinGPT Sentiment Analysis")
    print(f"{'='*80}\n")

    test_articles = [
        "The Federal Reserve announced interest rate cuts, boosting market sentiment and economic growth prospects.",
        "Stock market crashed amid recession fears and major bank failures across Europe.",
        "GDP growth remains steady at 2.5% despite global economic headwinds and trade tensions."
    ]

    total_inference_time = 0

    for i, article in enumerate(test_articles, 1):
        print(f"Article {i}: {article[:70]}...")

        # Create prompt for sentiment analysis
        prompt = f"""Analyze the sentiment of this financial news for trading signals.

News: {article}

Sentiment (positive/negative/neutral):"""

        inputs = tokenizer(prompt, return_tensors="pt")

        # Move to device if needed (CPU doesn't need explicit move)
        if device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        start = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                num_return_sequences=1,
                temperature=0.3,  # Lower temperature for more consistent sentiment
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )

        inference_time = time.time() - start
        total_inference_time += inference_time

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the model's response (after the prompt)
        sentiment_response = response[len(prompt):].strip()

        print(f"  Time: {inference_time:.2f}s")
        print(f"  Sentiment: {sentiment_response[:100]}")
        print()

    avg_time = total_inference_time / len(test_articles)

    # Final summary
    print(f"{'='*80}")
    print(f"SUCCESS! FinGPT IS NOW WORKING")
    print(f"{'='*80}")
    print(f"\nPerformance Summary:")
    print(f"  Tokenizer load:      {tok_time:.2f}s")
    print(f"  Base model load:     {model_time:.2f}s ({model_time/60:.1f} min)")
    print(f"  LoRA adapter load:   {lora_time:.2f}s")
    print(f"  Total setup time:    {tok_time + model_time + lora_time:.2f}s")
    print(f"  Avg inference time:  {avg_time:.2f}s per article")
    print(f"  Device:              {device}")
    print(f"  Model size:          {params:.2f}B parameters")
    print(f"\nComparison to FinBERT:")
    print(f"  FinBERT load:        ~1s (cached)")
    print(f"  FinBERT inference:   ~0.023s per article")
    print(f"  FinGPT is ~{avg_time/0.023:.0f}x slower than FinBERT")
    print(f"  But FinGPT provides richer, more nuanced analysis")

except Exception as e:
    print(f"\nFAILED")
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
