"""Test news processing with FinBERT and measure timing."""
import json
import time
from pathlib import Path
import pandas as pd
from transformers import pipeline
import torch

print("="*80)
print("NEWS PROCESSING WITH FINBERT - TIMING TEST")
print("="*80)

# Load news data
news_file = Path("data/bronze/news/financial_news_2025.ndjson")
articles = []

print(f"\n1. Loading news articles from {news_file}...")
with open(news_file, 'r') as f:
    for line in f:
        articles.append(json.loads(line.strip()))

print(f"   ✓ Loaded {len(articles)} articles")

# Load FinBERT
print(f"\n2. Loading FinBERT model...")
model_name = "ProsusAI/finbert"
load_start = time.time()

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=model_name,
    device=0 if torch.cuda.is_available() else -1
)

load_time = time.time() - load_start
print(f"   ✓ Model loaded in {load_time:.2f} seconds")
print(f"   Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

# Process articles
print(f"\n3. Processing {len(articles)} articles...")
print(f"   {'Article':<5} {'Time (s)':<10} {'Label':<12} {'Score':<8} {'Title'}")
print(f"   {'-'*5} {'-'*10} {'-'*12} {'-'*8} {'-'*50}")

results = []
total_inference_time = 0

for i, article in enumerate(articles, 1):
    # Combine title and content
    text = f"{article['title']} {article['content']}"

    # Truncate to avoid token limits (FinBERT has 512 token limit)
    text = text[:2000]

    # Time the inference
    start = time.time()
    result = sentiment_pipeline(text)[0]
    inference_time = time.time() - start
    total_inference_time += inference_time

    # Store results
    results.append({
        'article_num': i,
        'title': article['title'][:50],
        'label': result['label'],
        'score': result['score'],
        'inference_time': inference_time,
        'text_length': len(text),
        'source': article.get('source', 'unknown')
    })

    print(f"   {i:<5} {inference_time:<10.3f} {result['label']:<12} {result['score']:<8.3f} {article['title'][:50]}")

print(f"\n{'='*80}")
print(f"TIMING SUMMARY")
print(f"{'='*80}")
print(f"Total articles:          {len(articles)}")
print(f"Model load time:         {load_time:.2f} seconds")
print(f"Total inference time:    {total_inference_time:.2f} seconds")
print(f"Average per article:     {total_inference_time/len(articles):.3f} seconds")
print(f"Min per article:         {min(r['inference_time'] for r in results):.3f} seconds")
print(f"Max per article:         {max(r['inference_time'] for r in results):.3f} seconds")
print(f"Total processing time:   {load_time + total_inference_time:.2f} seconds")

# Create features DataFrame
print(f"\n{'='*80}")
print(f"GENERATED FEATURES")
print(f"{'='*80}")

df = pd.DataFrame(results)
print(df[['article_num', 'label', 'score', 'inference_time', 'text_length']].to_string(index=False))

# Save results
output_file = Path("data/news/silver/finbert_timing_test.csv")
output_file.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_file, index=False)
print(f"\n✓ Results saved to {output_file}")

# Sentiment distribution
print(f"\n{'='*80}")
print(f"SENTIMENT DISTRIBUTION")
print(f"{'='*80}")
print(df['label'].value_counts().to_string())
print(f"\nAverage confidence score: {df['score'].mean():.3f}")
