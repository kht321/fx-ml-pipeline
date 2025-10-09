"""Diagnose FinGPT tokenizer issue by inspecting the model files."""
from huggingface_hub import hf_hub_download, list_repo_files
from pathlib import Path
import json

model_name = "FinGPT/fingpt-sentiment_llama2-13b_lora"

print(f"Inspecting {model_name}")
print("="*80)

try:
    # List all files in the repo
    print("\n1. Files in repository:")
    files = list(list_repo_files(model_name))
    for f in sorted(files):
        print(f"   - {f}")

    # Download and inspect config
    print("\n2. Downloading config.json...")
    config_path = hf_hub_download(model_name, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"   Model type: {config.get('model_type', 'N/A')}")
    print(f"   Architecture: {config.get('architectures', 'N/A')}")

    # Check for tokenizer files
    print("\n3. Tokenizer files:")
    tokenizer_files = [f for f in files if 'tokenizer' in f.lower() or 'vocab' in f.lower() or '.model' in f]
    for f in tokenizer_files:
        print(f"   - {f}")

    # Download tokenizer_config.json if exists
    if 'tokenizer_config.json' in files:
        print("\n4. Tokenizer configuration:")
        tok_config_path = hf_hub_download(model_name, "tokenizer_config.json")
        with open(tok_config_path, 'r') as f:
            tok_config = json.load(f)

        print(f"   Tokenizer class: {tok_config.get('tokenizer_class', 'N/A')}")
        print(f"   Model max length: {tok_config.get('model_max_length', 'N/A')}")

    # Check if this is a LoRA model
    if 'adapter_config.json' in files:
        print("\n5. LoRA adapter detected:")
        adapter_path = hf_hub_download(model_name, "adapter_config.json")
        with open(adapter_path, 'r') as f:
            adapter_config = json.load(f)

        print(f"   Base model: {adapter_config.get('base_model_name_or_path', 'N/A')}")
        print(f"   PEFT type: {adapter_config.get('peft_type', 'N/A')}")

        base_model = adapter_config.get('base_model_name_or_path')
        if base_model:
            print(f"\n   NOTE: This is a LoRA adapter, not a full model!")
            print(f"   You need to load the base model first: {base_model}")
            print(f"   Then apply this LoRA adapter on top.")

except Exception as e:
    print(f"\nError: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
