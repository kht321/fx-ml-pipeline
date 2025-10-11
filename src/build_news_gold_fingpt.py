"""Transform News Silver layer to Gold using FinGPT with market and historical context.

This script processes news articles from Silver layer and generates trading signals
using FinGPT with:
1. Current market data context (prices, volatility, trends)
2. Historical news context (past articles and their signals)
3. Multi-factor analysis for SGD trading decisions
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--news-bronze",
        type=Path,
        default=Path("data/bronze/news/financial_news_2025.ndjson"),
        help="Bronze news articles (for full content)",
    )
    parser.add_argument(
        "--market-silver",
        type=Path,
        default=Path("data/market/silver/technical_features/sgd_vs_majors.csv"),
        help="Market technical features for context",
    )
    parser.add_argument(
        "--output-gold",
        type=Path,
        default=Path("data/news/gold/fingpt_signals/trading_signals.csv"),
        help="Output Gold layer trading signals",
    )
    parser.add_argument(
        "--base-model",
        default="NousResearch/Llama-2-13b-hf",
        help="Base LLaMA model",
    )
    parser.add_argument(
        "--lora-adapter",
        default="FinGPT/fingpt-sentiment_llama2-13b_lora",
        help="FinGPT LoRA adapter",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=5,
        help="Max articles to process (for testing)",
    )
    parser.add_argument(
        "--historical-window",
        type=int,
        default=24,
        help="Hours of historical context to include",
    )
    return parser.parse_args()


def log(message: str):
    """Log message to stderr."""
    print(f"[build_news_gold_fingpt] {message}", file=sys.stderr)
    sys.stderr.flush()


def load_fingpt_model(base_model_name: str, lora_adapter_name: str):
    """Load FinGPT model with base LLaMA + LoRA adapter."""
    log(f"Loading FinGPT model...")
    log(f"  Base: {base_model_name}")
    log(f"  Adapter: {lora_adapter_name}")
    log(f"  NOTE: This will take 5-10 minutes and use ~30GB RAM")

    # Load tokenizer
    log("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    log("  Loading base model (13B parameters)...")
    device = None

    if torch.cuda.is_available():
        log("    Using CUDA with 8-bit quantization")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            load_in_8bit=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
        device = "cuda"
    else:
        log("    Using CPU execution (MPS disabled due to generation instability)")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map={"": "cpu"}
        )
        device = "cpu"

    # Apply LoRA adapter
    log("  Applying FinGPT LoRA adapter...")
    peft_kwargs = {"device_map": {"": device}}
    model = PeftModel.from_pretrained(base_model, lora_adapter_name, **peft_kwargs)
    model.eval()

    log(f"  ✓ FinGPT loaded on {device}")

    return tokenizer, model, device


def get_market_context(market_df: pd.DataFrame, timestamp: datetime) -> Dict:
    """Get market context near the given timestamp."""
    if market_df.empty:
        return {}

    # Find closest market data point
    market_df['time'] = pd.to_datetime(market_df['time'], utc=True)
    # Ensure timestamp is also timezone-aware
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=pd.Timestamp.now(tz='UTC').tzinfo)
    market_df['time_diff'] = abs(market_df['time'] - timestamp)
    closest_idx = market_df['time_diff'].idxmin()
    row = market_df.loc[closest_idx]

    context = {
        'timestamp': row['time'],
        'mid_price': row.get('mid', 0),
        'ret_1h': row.get('ret_1', 0) * 100,  # Convert to %
        'ret_5h': row.get('ret_5', 0) * 100,
        'volatility': row.get('roll_vol_20', 0) * 100,
        'high_vol_regime': row.get('high_vol_regime', False),
        'spread_pct': row.get('spread_pct', 0) * 100,
        'zscore': row.get('zscore_20', 0),
        'momentum_5h': row.get('momentum_5', 0),
        'ewma_signal': row.get('ewma_signal', 0),
    }

    return context


def get_historical_context(news_df: pd.DataFrame, current_time: datetime, window_hours: int = 24) -> List[Dict]:
    """Get historical news articles within the window."""
    news_df['published'] = pd.to_datetime(news_df['published'], utc=True)
    # Ensure current_time is timezone-aware
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=pd.Timestamp.now(tz='UTC').tzinfo)

    # Filter to historical window
    cutoff = current_time - timedelta(hours=window_hours)
    historical = news_df[
        (news_df['published'] >= cutoff) &
        (news_df['published'] < current_time)
    ].sort_values('published')

    results = []
    for _, row in historical.iterrows():
        results.append({
            'time': row['published'],
            'title': row.get('title', ''),
            'summary': row.get('summary', '')[:200],  # Truncate for context
        })

    return results


def build_fingpt_prompt(
    article: Dict,
    market_context: Dict,
    historical_context: List[Dict]
) -> str:
    """Build comprehensive prompt for FinGPT with market and historical context."""

    prompt = f"""You are a financial analyst specializing in Singapore Dollar (SGD) foreign exchange trading signals.

TASK: Analyze the following news article and generate a trading signal for USD/SGD, considering current market conditions and recent news context.

=== CURRENT NEWS ARTICLE ===
Title: {article['title']}
Published: {article['published']}
Source: {article['source']}

Content: {article['content']}

=== CURRENT MARKET CONDITIONS (USD/SGD) ===
Timestamp: {market_context.get('timestamp', 'N/A')}
Mid Price: {market_context.get('mid_price', 0):.4f}
1-Hour Return: {market_context.get('ret_1h', 0):+.2f}%
5-Hour Return: {market_context.get('ret_5h', 0):+.2f}%
Volatility (20-period): {market_context.get('volatility', 0):.2f}%
Volatility Regime: {'HIGH' if market_context.get('high_vol_regime') else 'NORMAL'}
Spread: {market_context.get('spread_pct', 0):.3f}%
Price Z-Score: {market_context.get('zscore', 0):+.2f}
5-Hour Momentum: {market_context.get('momentum_5h', 0):+.6f}
EWMA Signal: {market_context.get('ewma_signal', 0):+.6f}

=== RECENT NEWS CONTEXT (Last {len(historical_context)} articles) ==="""

    if historical_context:
        for i, hist in enumerate(historical_context[-5:], 1):  # Last 5 articles
            prompt += f"""
{i}. [{hist['time'].strftime('%Y-%m-%d %H:%M')}] {hist['title']}
   {hist['summary']}"""
    else:
        prompt += "\n(No recent historical context available)"

    prompt += """

=== ANALYSIS REQUIRED ===
Provide a structured analysis in the following format:

SENTIMENT: [bullish/bearish/neutral]
CONFIDENCE: [0.0-1.0]
SGD_DIRECTION: [strengthen/weaken/neutral]
SGD_STRENGTH: [0.0-1.0]
MARKET_COHERENCE: [aligned/divergent/neutral]
TIMEFRAME: [immediate/short_term/medium_term/long_term]
VOLATILITY_IMPACT: [increase/decrease/neutral]
KEY_FACTORS: [list 2-3 key factors driving the signal]
RISK_FACTORS: [list 2-3 risk factors or uncertainties]
TRADING_SIGNAL: [buy_sgd/sell_sgd/hold]
POSITION_SIZE: [small/medium/large]
REASONING: [2-3 sentence explanation of the signal considering market context]

Analysis:"""

    return prompt


def parse_fingpt_response(response: str) -> Dict:
    """Parse FinGPT's structured response into a dictionary."""
    import re

    result = {
        'sentiment': 'neutral',
        'confidence': 0.5,
        'sgd_direction': 'neutral',
        'sgd_strength': 0.5,
        'market_coherence': 'neutral',
        'timeframe': 'unknown',
        'volatility_impact': 'neutral',
        'key_factors': '',
        'risk_factors': '',
        'trading_signal': 'hold',
        'position_size': 'small',
        'reasoning': '',
        'raw_response': response[:500]  # Store truncated raw response
    }

    # Extract fields using regex
    patterns = {
        'sentiment': r'SENTIMENT:\s*(\w+)',
        'confidence': r'CONFIDENCE:\s*([0-9.]+)',
        'sgd_direction': r'SGD_DIRECTION:\s*(\w+)',
        'sgd_strength': r'SGD_STRENGTH:\s*([0-9.]+)',
        'market_coherence': r'MARKET_COHERENCE:\s*(\w+)',
        'timeframe': r'TIMEFRAME:\s*(\w+)',
        'volatility_impact': r'VOLATILITY_IMPACT:\s*(\w+)',
        'trading_signal': r'TRADING_SIGNAL:\s*(\w+)',
        'position_size': r'POSITION_SIZE:\s*(\w+)',
    }

    for field, pattern in patterns.items():
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            value = match.group(1).lower()
            if field in ['confidence', 'sgd_strength']:
                try:
                    result[field] = float(value)
                except ValueError:
                    pass
            else:
                result[field] = value

    # Extract multi-word fields
    factors_match = re.search(r'KEY_FACTORS:\s*(.+?)(?=RISK_FACTORS:|TRADING_SIGNAL:|$)', response, re.DOTALL | re.IGNORECASE)
    if factors_match:
        result['key_factors'] = factors_match.group(1).strip()[:200]

    risk_match = re.search(r'RISK_FACTORS:\s*(.+?)(?=TRADING_SIGNAL:|REASONING:|$)', response, re.DOTALL | re.IGNORECASE)
    if risk_match:
        result['risk_factors'] = risk_match.group(1).strip()[:200]

    reasoning_match = re.search(r'REASONING:\s*(.+?)$', response, re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        result['reasoning'] = reasoning_match.group(1).strip()[:300]

    return result


def process_article_with_fingpt(
    article: Dict,
    market_context: Dict,
    historical_context: List[Dict],
    tokenizer,
    model,
    device: str
) -> Dict:
    """Process a single article with FinGPT."""
    import time

    # Build prompt
    prompt = build_fingpt_prompt(article, market_context, historical_context)

    log(f"  Processing: {article['title'][:60]}...")
    log(f"    Market context: Mid={market_context.get('mid_price', 0):.4f}, Vol={market_context.get('volatility', 0):.2f}%")
    log(f"    Historical context: {len(historical_context)} articles")

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

    # Move to device
    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Generate
    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.3,  # Lower for more consistent structured output
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    inference_time = time.time() - start_time

    # Decode
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the model's response (after the prompt)
    model_response = full_response[len(prompt):].strip()

    log(f"    ✓ Completed in {inference_time:.2f}s")

    # Parse response
    parsed = parse_fingpt_response(model_response)
    parsed['inference_time'] = inference_time
    parsed['article_title'] = article['title']
    parsed['article_published'] = article['published']
    parsed['article_source'] = article['source']

    return parsed


def main():
    """Main processing function."""
    args = parse_args()

    log("Starting News Silver → Gold processing with FinGPT")
    log(f"  News bronze: {args.news_bronze}")
    log(f"  Market silver: {args.market_silver}")
    log(f"  Output gold: {args.output_gold}")

    # Load news articles
    log("Loading news articles...")
    news_articles = []
    with open(args.news_bronze, 'r') as f:
        for line in f:
            news_articles.append(json.loads(line.strip()))

    log(f"  Loaded {len(news_articles)} articles")

    # Limit for testing
    if args.max_articles:
        news_articles = news_articles[:args.max_articles]
        log(f"  Limited to {len(news_articles)} articles for testing")

    # Convert to DataFrame
    news_df = pd.DataFrame(news_articles)
    news_df['published'] = pd.to_datetime(news_df['published'], utc=True)

    # Load market data
    log("Loading market data...")
    market_df = pd.read_csv(args.market_silver)
    log(f"  Loaded {len(market_df)} market observations")

    # Load FinGPT model
    tokenizer, model, device = load_fingpt_model(args.base_model, args.lora_adapter)

    # Process each article
    log(f"Processing {len(news_articles)} articles with FinGPT...")
    results = []

    for i, article in enumerate(news_articles, 1):
        log(f"\nArticle {i}/{len(news_articles)}")

        # Get market context
        article_time = pd.to_datetime(article['published'])
        market_context = get_market_context(market_df, article_time)

        # Get historical context
        historical_context = get_historical_context(news_df, article_time, args.historical_window)

        # Process with FinGPT
        result = process_article_with_fingpt(
            article,
            market_context,
            historical_context,
            tokenizer,
            model,
            device
        )

        results.append(result)

    # Save results
    log("\nSaving Gold layer trading signals...")
    results_df = pd.DataFrame(results)

    args.output_gold.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output_gold, index=False)

    log(f"  ✓ Saved {len(results_df)} trading signals to {args.output_gold}")

    # Summary statistics
    log("\n" + "="*80)
    log("PROCESSING SUMMARY")
    log("="*80)
    log(f"Total articles processed: {len(results)}")
    log(f"Average inference time: {results_df['inference_time'].mean():.2f}s")
    log(f"Total processing time: {results_df['inference_time'].sum():.2f}s")
    log(f"\nTrading Signals:")
    log(f"  Buy SGD: {(results_df['trading_signal'] == 'buy_sgd').sum()}")
    log(f"  Sell SGD: {(results_df['trading_signal'] == 'sell_sgd').sum()}")
    log(f"  Hold: {(results_df['trading_signal'] == 'hold').sum()}")
    log(f"\nSentiment Distribution:")
    for sentiment in ['bullish', 'bearish', 'neutral']:
        count = (results_df['sentiment'] == sentiment).sum()
        log(f"  {sentiment.capitalize()}: {count}")
    log(f"\nAverage Confidence: {results_df['confidence'].mean():.2f}")
    log(f"Average SGD Strength: {results_df['sgd_strength'].mean():.2f}")


if __name__ == "__main__":
    main()
