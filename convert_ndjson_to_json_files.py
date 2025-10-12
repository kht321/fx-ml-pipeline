#!/usr/bin/env python3
"""Convert NDJSON news files to individual JSON files for pipeline processing."""

import json
import sys
from pathlib import Path
from datetime import datetime

def convert_ndjson_to_json_files(ndjson_path: Path, output_dir: Path):
    """Convert an NDJSON file into individual JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    articles_converted = 0

    with open(ndjson_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                article = json.loads(line.strip())

                # Create a unique filename based on published_at or index
                try:
                    pub_dt = datetime.fromisoformat(article['published_at'].replace('Z', '+00:00'))
                    timestamp = pub_dt.strftime('%Y%m%d_%H%M%S')
                except:
                    timestamp = f"{line_num:04d}"

                # Generate a simple story_id
                story_id = f"sp500_{timestamp}_{article.get('source', 'unknown')}"

                # Convert to expected format
                article_formatted = {
                    'story_id': story_id,
                    'headline': article.get('title', 'No Title'),
                    'body': article.get('description', ''),
                    'published_at': article.get('published_at'),
                    'source': article.get('source', 'unknown'),
                    'url': article.get('url', ''),
                    'category': article.get('category', 'general'),
                    'scraped_at': article.get('scraped_at')
                }

                # Write individual JSON file
                output_file = output_dir / f"{story_id}.json"
                with open(output_file, 'w') as out_f:
                    json.dump(article_formatted, out_f, indent=2)

                articles_converted += 1

            except Exception as e:
                print(f"Error processing line {line_num}: {e}", file=sys.stderr)
                continue

    return articles_converted

if __name__ == "__main__":
    # Find all NDJSON files in the news bronze directory
    bronze_dir = Path("data/news/bronze/raw_articles")

    if not bronze_dir.exists():
        print(f"Error: Directory {bronze_dir} does not exist")
        sys.exit(1)

    ndjson_files = list(bronze_dir.glob("*.ndjson"))

    if not ndjson_files:
        print(f"No NDJSON files found in {bronze_dir}")
        sys.exit(1)

    print(f"Found {len(ndjson_files)} NDJSON file(s)")

    total_converted = 0
    for ndjson_file in ndjson_files:
        print(f"\nConverting {ndjson_file.name}...")
        count = convert_ndjson_to_json_files(ndjson_file, bronze_dir)
        print(f"  ✓ Converted {count} articles")
        total_converted += count

    print(f"\n✅ Total: {total_converted} articles converted to individual JSON files")
    print(f"📁 Location: {bronze_dir}")
