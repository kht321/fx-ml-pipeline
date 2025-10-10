"""Test loading FinGPT model using the updated processor with CPU."""
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

try:
    from fingpt_processor import FinGPTProcessor, create_processor

    print("=" * 60)
    print("Testing FinGPT Processor with CPU configuration")
    print("=" * 60)

    # Test 1: Create processor with CPU explicitly
    print("\n[Test 1] Creating FinGPT processor with device='cpu'...")
    processor = create_processor(
        use_fingpt=True,
        device="cpu",
        use_8bit=False
    )

    print(f"Processor type: {type(processor).__name__}")

    if isinstance(processor, FinGPTProcessor):
        print(f"Device: {processor.device}")
        print(f"Model name: {processor.model_name}")
        print("\nSuccess! FinGPT processor loaded on CPU")

        # Test 2: Quick inference test
        print("\n[Test 2] Testing inference with sample news...")
        test_text = "Singapore's GDP grew 3.5% in Q4, exceeding expectations."
        result = processor.analyze_sgd_news(test_text, "Singapore GDP Growth")

        print(f"\nAnalysis Results:")
        print(f"  Sentiment Score: {result.sentiment_score:.2f}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  SGD Signal: {result.sgd_directional_signal:.2f}")
        print(f"  Policy Implications: {result.policy_implications}")

        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
    else:
        print(f"\nNote: Fell back to {type(processor).__name__}")
        print("This is expected if FinGPT model is not available")

except Exception as e:
    print(f"\nERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
