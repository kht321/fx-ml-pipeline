#!/usr/bin/env python3
"""Test script to demonstrate market-contextualized FinGPT analysis.

This script shows how the enhanced news pipeline incorporates real-time market
conditions into FinGPT sentiment analysis for more sophisticated trading signals.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fingpt_processor import create_processor

def test_market_context_analysis():
    """Test FinGPT analysis with and without market context."""

    # Sample news article
    headline = "MAS hints at gradual tightening as inflation persists"
    article = """
    Monetary Authority of Singapore officials signalled a potential tightening bias
    after core inflation remained above 3%. Analysts say sustained wage pressures
    could keep SGD firm against USD in the medium term.
    """

    # Sample market contexts representing different scenarios
    high_vol_context = {
        'mid': 1.2950,
        'ret_5': 0.015,  # 1.5% recent gain
        'vol_20': 0.025,  # 2.5% volatility
        'high_vol_regime': True,
        'spread_pct': 0.0008,
        'zscore_20': 1.5,  # Price above average
        'session': 'Asian'
    }

    low_vol_context = {
        'mid': 1.2920,
        'ret_5': -0.002,  # -0.2% recent decline
        'vol_20': 0.008,  # 0.8% volatility
        'high_vol_regime': False,
        'spread_pct': 0.0005,
        'zscore_20': -0.3,  # Price below average
        'session': 'London'
    }

    print("=" * 60)
    print("MARKET-CONTEXTUALIZED FINGPT ANALYSIS TEST")
    print("=" * 60)

    print(f"\nNEWS ARTICLE:")
    print(f"Headline: {headline}")
    print(f"Body: {article.strip()}")

    try:
        # Create FinGPT processor (will fallback to lexicon if FinGPT unavailable)
        processor = create_processor(use_fingpt=True)

        print(f"\nProcessor type: {type(processor).__name__}")

        # Test 1: No market context (baseline)
        print(f"\n{'=' * 40}")
        print("TEST 1: NO MARKET CONTEXT (Baseline)")
        print(f"{'=' * 40}")

        analysis_baseline = processor.analyze_sgd_news(article, headline)
        print_analysis_results(analysis_baseline)

        # Test 2: High volatility context
        print(f"\n{'=' * 40}")
        print("TEST 2: HIGH VOLATILITY CONTEXT")
        print(f"{'=' * 40}")
        print(f"Market State: USD/SGD {high_vol_context['mid']:.4f} (+{high_vol_context['ret_5']:.1%})")
        print(f"Volatility: {high_vol_context['vol_20']:.1%} (HIGH regime)")
        print(f"Session: {high_vol_context['session']}")

        analysis_high_vol = processor.analyze_sgd_news(article, headline, high_vol_context)
        print_analysis_results(analysis_high_vol)

        # Test 3: Low volatility context
        print(f"\n{'=' * 40}")
        print("TEST 3: LOW VOLATILITY CONTEXT")
        print(f"{'=' * 40}")
        print(f"Market State: USD/SGD {low_vol_context['mid']:.4f} ({low_vol_context['ret_5']:.1%})")
        print(f"Volatility: {low_vol_context['vol_20']:.1%} (Normal regime)")
        print(f"Session: {low_vol_context['session']}")

        analysis_low_vol = processor.analyze_sgd_news(article, headline, low_vol_context)
        print_analysis_results(analysis_low_vol)

        # Comparison
        print(f"\n{'=' * 60}")
        print("COMPARISON SUMMARY")
        print(f"{'=' * 60}")

        scenarios = [
            ("Baseline (No Context)", analysis_baseline),
            ("High Vol Context", analysis_high_vol),
            ("Low Vol Context", analysis_low_vol)
        ]

        print(f"{'Scenario':<20} {'Sentiment':<10} {'Confidence':<10} {'SGD Signal':<12} {'Coherence':<12} {'Adj. Strength':<12}")
        print("-" * 80)

        for name, analysis in scenarios:
            coherence = getattr(analysis, 'market_coherence', 'N/A')
            adj_strength = getattr(analysis, 'signal_strength_adjusted', analysis.confidence)

            print(f"{name:<20} {analysis.sentiment_score:>8.2f} {analysis.confidence:>9.2f} "
                  f"{analysis.sgd_directional_signal:>10.2f} {coherence:>11} {adj_strength:>11.2f}")

        print(f"\nKey Insight: Market context should influence signal strength and coherence analysis!")

    except Exception as e:
        print(f"Error during analysis: {e}")
        print("This might be expected if FinGPT model is not available.")
        print("The system will fallback to lexicon-based analysis in production.")


def print_analysis_results(analysis):
    """Print formatted analysis results."""
    print(f"\nResults:")
    print(f"  Sentiment Score: {analysis.sentiment_score:.3f}")
    print(f"  Confidence: {analysis.confidence:.3f}")
    print(f"  SGD Directional: {analysis.sgd_directional_signal:.3f}")
    print(f"  Policy Tone: {analysis.policy_implications}")
    print(f"  Time Horizon: {analysis.time_horizon}")

    # New market-aware fields
    if hasattr(analysis, 'market_coherence'):
        print(f"  Market Coherence: {analysis.market_coherence}")
        print(f"  Adjusted Strength: {analysis.signal_strength_adjusted:.3f}")

    if analysis.key_factors:
        print(f"  Key Factors: {'; '.join(analysis.key_factors[:3])}")  # Show first 3

    # Show snippet of raw response
    if analysis.raw_response:
        snippet = analysis.raw_response[:150] + "..." if len(analysis.raw_response) > 150 else analysis.raw_response
        print(f"  Response Snippet: {snippet}")


if __name__ == "__main__":
    test_market_context_analysis()