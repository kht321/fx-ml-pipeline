"""
News feature engineering pipelines.

Provides two workflows:
1. Convert raw historical news JSON articles into OpenAI embeddings that can be
   stored downstream (e.g., feature store or vector database).
2. Generate rich, structured sentiment and event features via an LLM with
   Pydantic validation and automatic retry logic.

Prerequisites:
    export OPENAI_API_KEY="sk-..."
    export GOOGLE_API_KEY="..."  (required when using provider='gemini')

Example usage:
    python -m src_clean.data_pipelines.silver.news_feature_pipelines \
        --mode embeddings \
        --input-dir data_clean/bronze/news/historical_5year \
        --output-path data_clean/silver/news/news_embeddings.parquet

    python -m src_clean.data_pipelines.silver.news_feature_pipelines \
        --mode llm_features \
        --input-dir data_clean/gold/news/news_with_features.parquet \
        --output-path data_clean/silver/news/news_llm_features.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from dotenv import load_dotenv
from datetime import datetime, timezone

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise EnvironmentError("GOOGLE_API_KEY not found in environment or .env")

# Optionally set explicitly (not usually necessary once the env var is loaded)
os.environ["GOOGLE_API_KEY"] = google_api_key

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def load_articles(input_dir: Path) -> Iterable[Dict[str, Any]]:
    """Yield JSON articles from the provided directory."""
    for path in sorted(input_dir.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                article = json.load(f)
            article["__source_path__"] = str(path)
            yield article
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to read %s: %s", path, exc)


def combine_article_text(article: Dict[str, Any]) -> str:
    """Create a rich text block that mixes headline, summary, and body."""
    headline = article.get("headline") or ""
    body = article.get("body") or ""
    url = article.get("url") or ""
    parts = [
        f"Headline: {headline}".strip(),
        f"URL: {url}".strip(),
        "Body:",
        body.strip(),
    ]
    return "\n\n".join(part for part in parts if part)


def exponential_backoff_sleep(base_delay: float, attempt: int, max_delay: float) -> None:
    """Sleep with exponential backoff capped at max_delay."""
    sleep_for = min(base_delay * (2 ** attempt), max_delay)
    logger.debug("Retrying after %.2f seconds (attempt %d)", sleep_for, attempt + 1)
    time.sleep(sleep_for)


# --------------------------------------------------------------------------- #
# LLM structured feature pipeline
# --------------------------------------------------------------------------- #


class NewsFeatureSchema(BaseModel):
    # Sentiment & Tone
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    sentiment_magnitude: float = Field(..., ge=0.0, le=1.0)
    subjectivity: float = Field(..., ge=0.0, le=1.0)
    uncertainty_score: float = Field(..., ge=0.0, le=1.0)
    confidence_in_trend: float = Field(..., ge=0.0, le=1.0)
    sentiment_score_vs_expectation: float = Field(..., ge=-1.0, le=1.0)
    sentiment_trend: Optional[str]
    sentiment_trend_strength: float = Field(..., ge=0.0, le=1.0)
    surprise_indicator: float = Field(..., ge=-1.0, le=1.0)
    emotion_intensity_score: float = Field(..., ge=0.0, le=1.0)
    volatility_flag: bool
    market_reaction_flag: bool
    reaction_magnitude_score: float = Field(..., ge=0.0, le=1.0)
    volatility_pressure_score: float = Field(..., ge=0.0, le=1.0)
    emotion_keywords: List[str]

    # Predictive Price-Impact
    predicted_price_impact_direction: float = Field(..., ge=-1.0, le=1.0)
    predicted_price_impact_probability: float = Field(..., ge=0.0, le=1.0)
    predicted_price_impact_magnitude: float = Field(..., ge=0.0, le=1.0)
    impact_duration_category: Optional[str]
    impact_duration_score: float = Field(..., ge=0.0, le=1.0)
    catalyst_type: Optional[str]
    catalyst_strength_score: float = Field(..., ge=0.0, le=1.0)

    # Topic & Market Signals
    named_entities: List[str]
    entity_sentiment: Dict[str, float]
    is_macro_related: float = Field(..., ge=0.0, le=10.0)
    macro_sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    relevance_to_stock: float = Field(..., ge=0.0, le=1.0)
    industry_sector: Optional[str]
    sector_sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    cross_sector_correlation_score: float = Field(..., ge=0.0, le=1.0)
    policy_impact_flag: bool
    policy_bias_score: float = Field(..., ge=-1.0, le=1.0)
    data_dependency_flag: bool
    data_tone_score: float = Field(..., ge=-1.0, le=1.0)
    geography_mentions: List[str]
    geopolitical_tension_score: float = Field(..., ge=0.0, le=1.0)
    cross_asset_impact_score: float = Field(..., ge=0.0, le=1.0)

    # Historical-Context
    relative_to_historical_sentiment: float = Field(..., ge=-1.0, le=1.0)
    deviation_from_historical_volatility: float = Field(..., ge=0.0, le=1.0)
    narrative_continuity_score: float = Field(..., ge=0.0, le=1.0)
    cumulative_context_bias_score: float = Field(..., ge=-1.0, le=1.0)
    historical_similarity_examples: List[str]
    recency_alignment_score: float = Field(..., ge=0.0, le=1.0)

    # Event & Market Dynamics
    event_type: Optional[str]
    event_impact_level: float = Field(..., ge=0.0, le=1.0)
    event_novelty_score: float = Field(..., ge=0.0, le=1.0)
    volatility_trigger_keywords: List[str]
    liquidity_impact_flag: bool
    liquidity_stress_score: float = Field(..., ge=0.0, le=1.0)
    safe_haven_mention_flag: bool
    risk_appetite_score: float = Field(..., ge=0.0, le=1.0)
    time_horizon: Optional[str]
    time_horizon_numeric: float = Field(..., ge=0.0, le=1.0)

    # Summary & Context
    context_history_summary: str
    short_summary: str
    key_new_information: List[str]
    overall_market_sentiment: Optional[str]
    overall_market_sentiment_score: float = Field(..., ge=-1.0, le=1.0)

    # Meta / Quality
    model_confidence_score: float = Field(..., ge=0.0, le=1.0)
    language_detected: Optional[str]



@dataclass
class NewsLLMFeaturePipeline:
    """
    Use an LLM to extract structured features as defined by NewsFeatureSchema.

    Parameters
    ----------
    manifest_path:
        Path to a manifest (parquet/csv) containing news metadata, including
        JSON file paths and any quantitative features aligned by timestamp.
    output_path:
        Destination for the validated features (parquet, feather, or csv).
    model:
        Chat model name (e.g., 'gpt-4.1-mini').
    context_history:
        Optional text summary fed into the prompt for historical context.
    """

    manifest_path: Path
    output_path: Path
    model: str = "gemini-2.5-flash"
    context_history: str = "No prior context provided."
    max_prompt_tokens: int = 10000
    max_retries: int = 4
    base_retry_delay: float = 1.0
    max_retry_delay: float = 20.0

    def __post_init__(self) -> None:
        self.metadata_columns = {
            "article_id",
            "headline",
            "body",
            "url",
            "source",
            "published_at",
            "collected_at",
            "language",
            "sp500_relevant",
            "collection_method",
            "content_fetched",
            "file_path",
            "timestamp",
            "file_size_bytes",
        }

        self.project_root = Path(__file__).resolve().parents[3]

        self.quant_feature_hints: Dict[str, str] = {
            "open": "Opening price for the interval.",
            "high": "Highest traded price during the interval.",
            "low": "Lowest traded price during the interval.",
            "close": "Closing price for the interval.",
            "vwap": "Volume-weighted average price.",
            "volume": "Reported trade volume for the interval.",
            "volume_ma20": "20-period moving average of volume.",
            "volume_ma50": "50-period moving average of volume.",
            "volume_ratio": "Volume relative to a moving average baseline.",
            "volume_velocity": "Rate of change in volume levels.",
            "volume_acceleration": "Acceleration in volume levels.",
            "volume_zscore": "Z-score of volume versus its rolling mean.",
            "ema_7": "7-period exponential moving average.",
            "ema_14": "14-period exponential moving average.",
            "ema_21": "21-period exponential moving average.",
            "sma_7": "7-period simple moving average.",
            "sma_14": "14-period simple moving average.",
            "sma_21": "21-period simple moving average.",
            "sma_50": "50-period simple moving average.",
            "bb_upper": "Upper Bollinger Band.",
            "bb_middle": "Middle Bollinger Band (moving average).",
            "bb_lower": "Lower Bollinger Band.",
            "bb_width": "Relative width between upper and lower Bollinger Bands.",
            "bb_position": "Position of price within the Bollinger Band envelope.",
            "rsi_14": "14-period relative strength index.",
            "rsi_20": "20-period relative strength index.",
            "adx_14": "14-period average directional index.",
            "macd": "MACD value (12/26 EMA spread).",
            "macd_signal": "MACD signal line.",
            "macd_histogram": "Difference between MACD and signal line.",
            "atr_14": "Average true range over 14 periods.",
            "momentum_5": "5-period price momentum.",
            "momentum_10": "10-period price momentum.",
            "momentum_20": "20-period price momentum.",
            "roc_5": "5-period rate of change.",
            "roc_10": "10-period rate of change.",
            "return_1": "1-period return.",
            "return_5": "5-period return.",
            "return_10": "10-period return.",
            "price_impact": "Estimated price impact metric.",
            "price_impact_ma20": "20-period moving average of price impact.",
            "order_flow_imbalance": "Order flow imbalance indicator.",
            "hl_range": "High-low range for the interval.",
            "hl_range_pct": "High-low range as a fraction of price.",
            "hl_range_ma20": "20-period moving average of the high-low range.",
            "realized_range": "Realised range volatility estimate.",
            "realized_range_ma": "Moving average of the realised range.",
            "ewma_vol": "Exponentially weighted volatility estimate.",
            "hist_vol_20": "20-period historical volatility.",
            "hist_vol_50": "50-period historical volatility.",
            "volatility_20": "20-period realised volatility.",
            "volatility_50": "50-period realised volatility.",
            "parkinson_vol": "Parkinson volatility estimator.",
            "gk_vol": "Garman–Klass volatility estimator.",
            "yz_vol": "Yang–Zhang volatility estimator.",
            "rs_vol": "Rogers–Satchell volatility estimator.",
            "vol_of_vol": "Volatility of volatility indicator.",
            "illiquidity": "Amihud illiquidity measure.",
            "illiquidity_ma20": "20-period moving average of illiquidity.",
            "spread_pct": "Estimated bid/ask spread percentage.",
            "spread_proxy": "Absolute spread proxy.",
            "close_vwap_ratio": "Ratio of close price to VWAP.",
            "vol_regime_high": "High volatility regime flag (1/0).",
            "vol_regime_low": "Low volatility regime flag (1/0).",
            "event_timestamp": "Timestamp aligned with the quantitative snapshot.",
            "time": "Canonical timestamp for the record.",
            "instrument": "Instrument or index identifier.",
        }

        self.system_prompt = (
            "You are an AI model that analyzes financial news articles to extract "
            "structured sentiment, tone, and topic-related features for a downstream "
            "S&P500 price forecasting model to predict the price in 30 minutes time. "
            "Return exactly one JSON object that follows the provided schema."
        )
        self.response_schema = self._build_response_schema()
        self.schema_text = json.dumps(self.response_schema, indent=2, sort_keys=True)
        self.field_guide_text = self._build_field_guide_text()

        # Only Gemini provider is supported
        self._setup_gemini_client()

    def _setup_gemini_client(self) -> None:
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency management
            raise ImportError(
                "google-generativeai package is required when provider='gemini'. "
                "Install via `pip install google-generativeai`."
            ) from exc

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY environment variable must be set for Gemini usage."
            )

        genai.configure(api_key=api_key)
        self._genai = genai
        self._gemini_model = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=self.system_prompt,
        )

    def _call_gemini(self, prompt: str) -> str:
        """Use Gemini structured-output mode with Pydantic validation."""
        genai = self._genai
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json",
            #response_schema=NewsFeatureSchema,   # enforce structured output
        )
        response = self._gemini_model.generate_content(
            contents=prompt,
            generation_config=generation_config,
        )
        # Gemini returns parsed JSON text matching the schema
        content = getattr(response, "text", None)
        if not content:
            raise RuntimeError("Gemini returned empty content.")
        return content


    def _build_response_schema(self) -> Dict[str, Any]:
        """Generate a Gemini-compatible JSON schema from NewsFeatureSchema."""
        schema = NewsFeatureSchema.model_json_schema()
        return schema

    def _build_field_guide_text(self) -> str:
        """Assemble a concise field guide for prompt injection."""
        lines: List[str] = []
        lines.append(
            "SENTIMENT & TONE\n"
            "- sentiment_score (-1 to +1): Overall tone polarity.\n"
            "- sentiment_magnitude (0-1): Intensity regardless of direction.\n"
            "- subjectivity (0-1): Factual (0) vs opinionated (1).\n"
            "- uncertainty_score (0-1): Presence of speculative or vague terms.\n"
            "- confidence_in_trend (0-1): Certainty of directional view.\n"
            "- sentiment_score_vs_expectation (-1 to +1): Tone vs prior context expectation.\n"
            "- sentiment_trend: Improving, worsening, or stable.\n"
            "- sentiment_trend_strength (0-1): Magnitude of sentiment change.\n"
            "- surprise_indicator (-1 to +1): Unexpectedness vs market expectation.\n"
            "- emotion_intensity_score (0-1): Aggregate emotional energy.\n"
            "- volatility_flag: True if mentions large moves.\n"
            "- market_reaction_flag: True if actual reactions reported.\n"
            "- reaction_magnitude_score (0-1): Implied percent move scale.\n"
            "- volatility_pressure_score (0-1): Market tension intensity.\n"
            "- emotion_keywords: List of emotion or tone terms found.\n\n"
            "PREDICTIVE PRICE-IMPACT\n"
            "- predicted_price_impact_direction (-1 drop to +1 rise): Expected price move direction.\n"
            "- predicted_price_impact_probability (0-1): Confidence in predicted direction.\n"
            "- predicted_price_impact_magnitude (0-1): Expected move size.\n"
            "- impact_duration_category: 'intraday', 'multi-day', 'multi-week', 'long-term'.\n"
            "- impact_duration_score (0-1): Persistence strength (short to long).\n"
            "- catalyst_type: Primary cause (earnings, macro, sentiment, etc.).\n"
            "- catalyst_strength_score (0-1): Strength of catalyst effect.\n\n"
            "TOPIC & MARKET SIGNALS\n"
            "- named_entities: List of companies, people, institutions.\n"
            "- entity_sentiment: Dict of entity to tone (-1 to +1).\n"
            "- is_macro_related (0-10): Degree of macroeconomic relevance.\n"
            "- macro_sentiment_score (-1 to +1): Tone toward macro conditions.\n"
            "- relevance_to_stock (0-1): Whether this information helps forecast the index or equities.\n"
            "- industry_sector: Main sector affected.\n"
            "- sector_sentiment_score (-1 to +1): Sector tone.\n"
            "- cross_sector_correlation_score (0-1): Strength of co-movement implication.\n"
            "- policy_impact_flag: True if policy or regulatory event involved.\n"
            "- policy_bias_score (-1 hawkish to +1 dovish): Policy stance tone.\n"
            "- data_dependency_flag: True if data or reports cited.\n"
            "- data_tone_score (-1 weak to +1 strong data tone).\n"
            "- geography_mentions: List of mentioned countries or regions.\n"
            "- geopolitical_tension_score (0-1): Level of geopolitical stress.\n"
            "- cross_asset_impact_score (0-1): Spillover to other asset classes.\n\n"
            "HISTORICAL CONTEXT\n"
            "- relative_to_historical_sentiment (-1 to +1): More positive/negative than history.\n"
            "- deviation_from_historical_volatility (0-1): Degree of abnormal volatility.\n"
            "- narrative_continuity_score (0-1): Continuation vs disruption of trend.\n"
            "- cumulative_context_bias_score (-1 bearish bias to +1 bullish bias).\n"
            "- historical_similarity_examples: List of similar past cases.\n"
            "- recency_alignment_score (0-1): Alignment with recent dominant themes.\n\n"
            "EVENT & MARKET DYNAMICS\n"
            "- event_type: Nature of event (earnings, policy, merger, etc.).\n"
            "- event_impact_level (0-1): Size of event's effect.\n"
            "- event_novelty_score (0-1): How new or unexpected this event is.\n"
            "- volatility_trigger_keywords: Words implying major moves.\n"
            "- liquidity_impact_flag: True if volume/liquidity mentioned.\n"
            "- liquidity_stress_score (0-1): Degree of liquidity strain.\n"
            "- safe_haven_mention_flag: True if gold, USD, treasuries, etc. cited.\n"
            "- risk_appetite_score (0-1): Risk-on vs risk-off tone.\n"
            "- time_horizon: Qualitative window (short, medium, long).\n"
            "- time_horizon_numeric (0-1): Quantitative encoding (short to long).\n\n"
            "SUMMARY & CONTEXT\n"
            "- context_history_summary: Update prior context history with new relevant information from this article that summarizes the market sentiment, expectations, and information related to S&P 500 for future price forecasting.\n"
            "- short_summary: 1-2 sentence article summary.\n"
            "- key_new_information: 1-5 new facts or developments.\n"
            "- overall_market_sentiment: Bullish / Neutral / Bearish.\n"
            "- overall_market_sentiment_score (-1 to +1): Numeric mapping of tone.\n\n"
            "META / QUALITY\n"
            "- model_confidence_score (0-1): Reliability of generated outputs.\n"
            "- language_detected: Auto-detected article language.\n"
        )
        return "\n".join(lines)

    def _load_manifest(self) -> pd.DataFrame:
        """Load manifest table containing article metadata and features."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        ext = self.manifest_path.suffix.lower()
        if ext in {".parquet", ".pq"}:
            return pd.read_parquet(self.manifest_path)
        if ext in {".csv", ".txt"}:
            return pd.read_csv(self.manifest_path)
        raise ValueError(f"Unsupported manifest extension '{ext}'.")

    def _load_article_from_row(self, row: pd.Series) -> Dict[str, Any]:
        """Read the original article JSON referenced by a manifest row."""
        file_path = row.get("file_path")
        if not file_path:
            raise ValueError("Manifest row missing 'file_path'.")
        path_obj = Path(file_path)
        if not path_obj.is_absolute():
            path_obj = (self.project_root/ path_obj).resolve()
        if not path_obj.exists():
            raise FileNotFoundError(f"Article JSON not found: {path_obj}")
        with path_obj.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _extract_quant_features(
        self,
        row: pd.Series,
    ) -> Dict[str, Any]:
        """Collect quantitative feature columns for prompt conditioning."""
        features: Dict[str, Any] = {}
        quant_features = ['ret_10', 'momentum_20', 'volume_ma_20', 'sma_50',
 'volatility_50', 'volume_ma50']
        for col, value in row.items():
            if col not in quant_features:
                continue
            if pd.isna(value):
                continue
            if isinstance(value, (pd.Timestamp, datetime)):
                features[col] = value.isoformat()
            elif isinstance(value, (float, int, str, bool)):
                features[col] = value
            else:
                try:
                    features[col] = float(value)
                except Exception:
                    features[col] = str(value)
        return features
    
    @staticmethod
    def _round_sigfigs(value: float, sig: int = 4):
        """Round numeric values to a fixed number of significant figures."""
        if not isinstance(value, (float, int)):
            return value
        value = float(value)
        if value == 0.0:
            return 0.0
        return float(f"{value:.{sig}g}")
    
    def _combine_article_text(self, article: Dict[str, Any]) -> str:
        """Create a rich text block that mixes headline, summary, and body."""
        headline = article.get("headline") or ""
        body = article.get("body") or ""
        url = article.get("url") or ""
        parts = [
            f"Headline: {headline}".strip(),
            f"URL: {url}".strip(),
            "Body:",
            body.strip(),
        ]
        return "\n\n".join(part for part in parts if part)

    def run(self) -> Path:
        """Generate LLM features in batches and persist to the gold parquet table."""
        # create prompt directory
        prompts_dir = self.output_path.parent / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        job_snapshot = datetime.now(timezone.utc) 

        manifest_df = self._load_manifest().sort_values("published_at")
        total_rows = len(manifest_df)
        print(f"[run] Loaded manifest with {total_rows} rows")
        if total_rows == 0:
            print("[run] Manifest is empty; nothing to process.")
            return self.output_path

        # Load existing gold table if present
        if self.output_path.exists():
            try:
                combined_df = pd.read_parquet(self.output_path)
            except Exception as exc:
                print(f"[run] Failed to read existing parquet {self.output_path}: {exc}")
                combined_df = pd.DataFrame()
        else:
            combined_df = pd.DataFrame()

        processed_ids: Set[str] = (
            set(combined_df["article_id"].dropna().astype(str))
            if not combined_df.empty 
            and "article_id" in combined_df.columns
            else set()
        )
        print(f"[run] Existing processed rows: {len(processed_ids)}")

        # Seed context from the last processed article
        current_context = self.context_history
        if (
            not combined_df.empty
            and "context_history_summary" in combined_df.columns
            and isinstance(combined_df.iloc[-1].get("context_history_summary"), str)
            and combined_df.iloc[-1]["context_history_summary"].strip()
        ):
            current_context = combined_df.iloc[-1]["context_history_summary"]

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        batch_size = 5
        batch_rows: List[Dict[str, Any]] = []
        articles_processed = 0
        start_time = time.perf_counter()

        for _, row in manifest_df.iterrows():
            article_id = row.get("article_id")
            published_at = row.get("published_at")

            if article_id is None:
                print(f"[run] Skipping row {_}: missing article_id")
                continue
            if str(article_id) in processed_ids:
                prior = combined_df.loc[combined_df["article_id"].astype(str) == str(article_id),"context_history_summary"]
                first_ctx = prior.iloc[0] if not prior.empty else None
                if isinstance(first_ctx, str) and first_ctx.strip():
                    current_context = first_ctx
                    print(f'Skipping {article_id}. Published at {published_at}')
                    continue

            article_start = time.perf_counter()
            self.context_history = current_context

            feature_row, article, prompt = self._analyze_article(row)
            feature_row.update(
                {
                    "article_id": row.get("article_id") or article.get("article_id"),
                    "published_at": published_at or article.get("published_at"),
                    "source": row.get("source") or article.get("source"),
                    "headline": row.get("headline") or article.get("headline"),
                    "url": row.get("url") or article.get("url"),
                    "run_snapshot_time": job_snapshot.isoformat(),
                    "LLMmodel": self.model,
                }
            )

            batch_rows.append(feature_row)
            processed_ids.add(str(article_id))
            articles_processed += 1

            elapsed_article = time.perf_counter() - article_start
            total_elapsed = time.perf_counter() - start_time
            print(
                f"[run] Article {articles_processed}/{total_rows} | "
                f"article_id={article_id} | published_at={published_at} | "
                f"article_time={elapsed_article:.2f}s | total_time={total_elapsed:.2f}s"
            )

            updated_context = feature_row.get("context_history_summary")
            if isinstance(updated_context, str) and updated_context.strip():
                current_context = updated_context

            # save prompts for validation and record keeping
            prompt_payload = {
                "article_id": article_id,
                "run_snapshot_time": job_snapshot.isoformat(),
                "prompt": prompt,
                "metadata": {
                    "headline": row.get("headline"),
                    "published_at": row.get("published_at").isoformat(),
                },
            }
            prompt_path = prompts_dir / f"{article_id}_{job_snapshot.isoformat().replace(':', '-')}.json"
            with prompt_path.open("w", encoding="utf-8") as fh:
                json.dump(prompt_payload, fh, ensure_ascii=False, indent=2)

            if len(batch_rows) >= batch_size:
                print(f"[run] Committing batch of {len(batch_rows)} rows to {self.output_path}")
                batch_df = pd.DataFrame(batch_rows)
                if combined_df.empty:
                    combined_df = batch_df
                else:
                    combined_df = pd.concat([combined_df, batch_df], ignore_index=True)
                    if "article_id" in combined_df.columns:
                        combined_df = combined_df.sort_values("published_at", kind="mergesort")
                        combined_df = combined_df.drop_duplicates(
                            subset="article_id", keep="last"
                        )
                combined_df.to_parquet(self.output_path, index=False)
                batch_rows.clear()

        # Flush any remaining rows
        if batch_rows:
            print(f"[run] Committing final batch of {len(batch_rows)} rows to {self.output_path}")
            batch_df = pd.DataFrame(batch_rows)
            if combined_df.empty:
                combined_df = batch_df
            else:
                combined_df = pd.concat([combined_df, batch_df], ignore_index=True)
                if "article_id" in combined_df.columns:
                    combined_df = combined_df.sort_values("published_at", kind="mergesort")
                    combined_df = combined_df.drop_duplicates(subset="article_id", keep="last")
            combined_df.to_parquet(self.output_path, index=False)

        self.context_history = current_context
        print(
            f"[run] Completed processing ({articles_processed} new articles). "
            f"Final row count: {len(combined_df)}"
        )
        return self.output_path




    def _analyze_article(self, row: pd.Series) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Call the LLM and validate the structured output with retry feedback."""
        error_messages: List[str] = []
        last_error: Optional[Exception] = None
        article_id = row.get("article_id")
        last_article: Optional[Dict[str, Any]] = None
        last_content: Optional[Any] = None
        last_parsed: Optional[Dict[str, Any]] = None

        for attempt in range(self.max_retries):
            prompt, article = self._build_prompt(row=row, error_messages=error_messages)
            last_article = article
            try:
                content = self._call_gemini(prompt)
                last_content = content
                parsed = NewsFeatureSchema.model_validate_json(content)
                result = parsed.model_dump()
                last_parsed = result
                return result, article, prompt
            except ValidationError as exc:
                message = (
                    f"Pydantic validation failed on attempt {attempt + 1}/{self.max_retries}: {exc}"
                )
                logger.warning(
                    "Pydantic validation failed (attempt %d/%d) for %s: %s",
                    attempt + 1,
                    self.max_retries,
                    article_id,
                    exc,
                )
                error_messages.append(message)
                last_error = exc
            except Exception as exc:
                message = (
                    f"LLM request error on attempt {attempt + 1}/{self.max_retries}: {exc}"
                )
                logger.warning(
                    "LLM request failed (attempt %d/%d) for %s: %s",
                    attempt + 1,
                    self.max_retries,
                    article_id,
                    exc,
                )
                error_messages.append(message)
                last_error = exc

            exponential_backoff_sleep(self.base_retry_delay, attempt, self.max_retry_delay)

        logger.warning(
            "Returning last attempted result for %s after %d failed attempts: %s",
            article_id,
            self.max_retries,
            last_error,
        )

        fallback: Dict[str, Any] = last_parsed or {}
        if not fallback and last_content is not None:
            try:
                if isinstance(last_content, str):
                    fallback = json.loads(last_content)
                elif isinstance(last_content, dict):
                    fallback = last_content
            except Exception:
                fallback = {}

        return fallback, last_article or {}, prompt

    def _build_prompt(
        self,
        row: pd.Series,
        error_messages: Optional[List[str]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Prepare the LLM prompt (article text + quantitative features) from a manifest row."""
        article = self._load_article_from_row(row)
        article_text = self._combine_article_text(article)
        quantitative_features = self._extract_quant_features(row)
        language = article.get('language') or row.get('language') or 'unknown'

        rounded_features: Dict[str, Any] = {}
        for key in sorted(quantitative_features.keys()):
            value = quantitative_features[key]
            if isinstance(value, bool):
                rounded = value
            elif isinstance(value, (int,)):
                rounded = value
            elif isinstance(value, float):
                rounded = self._round_sigfigs(value, sig=4)
            else:
                rounded = value
            rounded_features[key] = rounded

        features_block = (
            json.dumps(rounded_features, indent=2, sort_keys=True)
            if rounded_features
            else 'No quantitative features available.'
        )

        quant_hint_lines: List[str] = []
        for key in sorted(rounded_features.keys()):
            hint = self.quant_feature_hints.get(
                key,
                "Quantitative indicator aligned with this article's timestamp.",
            )
            quant_hint_lines.append(f"- {key}: {hint}")
        if not quant_hint_lines:
            quant_hint_lines.append("- (no quantitative features provided)")
        quant_hints_text = "\n".join(quant_hint_lines)

        retry_section = ''
        if error_messages:
            latest_errors = '\n'.join(f'- {msg}' for msg in error_messages[-3:])
            retry_section = (
                'Previous attempt issues (fix all of these exactly before responding):\n'
                f"{latest_errors}\n\n"
            )

        prompt = (
            'You are an AI model that analyzes financial news articles to extract structured, '
            'quantitative, and predictive market features for S&P500 index price forecasting.\n\n'
            'You must output a **single valid JSON object** following the exact schema below: \n'
            f"{self.schema_text}\n\n"
            f"Detected language: {language}\n\n"
            'General rules:\n'
            '- Output only one JSON object (no prose).\n'
            '- Follow schema field names exactly.\n'
            '- Use null for missing strings, 0.0 for floats, false for booleans, [] for lists.\n'
            '- Keep numeric values within their valid ranges.\n\n'
            f"{retry_section}"
            'Quantitative features aligned to this timestamp:\n'
            f"{features_block}\n\n"
            'Quantitative feature hints:\n'
            f"{quant_hints_text}\n\n"
            'Field guide (key expectations for each field):\n'
            f"{self.field_guide_text}\n\n"
            'Current news article:\n<<<ARTICLE>>>\n'
            f"{article_text}\n<<<END ARTICLE>>>\n\n"
            'Previous context history summary:\n<<<CONTEXT_HISTORY>>>\n'
            f"{self.context_history}\n<<<END CONTEXT_HISTORY>>>\n"
        )

        return prompt, article


# --------------------------------------------------------------------------- #
# CLI Entrypoint
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="News feature pipelines.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default='', required=True)
    parser.add_argument("--model", type=str, default="gemini-2.5-flash-lite", help="Override model name.")
    parser.add_argument(
        "--provider",
        type=str,
        default='gemini',
        choices=['gemini'],
        help="LLM provider for structured feature extraction.",
    )
    parser.add_argument(
        "--context-history",
        type=str,
        default="No prior context provided.",
        help="Context summary for LLM feature extraction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provider = args.provider.lower()
    if provider != "gemini":
        raise ValueError("Only provider='gemini' is supported.")
    model_name = args.model or "gemini-2.5-flash-lite"
    pipeline = NewsLLMFeaturePipeline(
        manifest_path=args.input_dir,
        output_path=args.output_path,
        model=model_name,
        context_history=args.context_history,
    )
    pipeline.run()


if __name__ == "__main__":
    main()

"""
 python -m src_clean.data_pipelines.silver.news_feature_pipelines `
    --input-dir data_clean/silver/news/news_manifest_cleaned_20251022_122221.parquet `
    --output-path data_clean/gold/news/LLM_news.parquet `
    --model gemini-2.0-flash-lite

"""