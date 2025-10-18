"""FinGPT-powered news sentiment analysis for the FX ML pipeline.

This module integrates the open-source FinGPT model to enhance news processing
from Silver to Gold layer in the news medallion pipeline. It provides sophisticated
financial sentiment analysis specifically tuned for SGD trading signals.
"""

import logging
import re
import torch
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    import spacy
    SPACY_AVAILABLE = True
    nlp = spacy.load("en_core_web_sm")  # Load spaCy model for preprocessing
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available. Install with: pip install spacy && python -m spacy download en_core_web_sm")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available. Install with: pip install transformers torch")


@dataclass
class FinGPTAnalysis:
    """Structured output from FinGPT analysis."""
    sentiment_score: float  # -1 (bearish) to 1 (bullish)
    confidence: float  # 0 to 1
    sgd_directional_signal: float  # -1 (SGD weak) to 1 (SGD strong)
    policy_implications: str  # "hawkish", "dovish", "neutral"
    time_horizon: str  # "immediate", "short_term", "medium_term"
    key_factors: List[str]  # Main drivers extracted
    market_coherence: str  # "aligned", "divergent", "neutral" - news vs market state
    signal_strength_adjusted: float  # Sentiment adjusted for market context
    topic: str = "general"  # Added for tabular features
    urgency: str = "low"  # Added for tabular features
    raw_response: str  # Full model output for debugging


class FinGPTProcessor:
    """FinGPT-powered financial news analysis for SGD trading signals."""

    def __init__(self,
                 model_name: str = "FinGPT/fingpt-sentiment_llama2-7b_lora",
                 device: Optional[str] = None,
                 use_8bit: bool = True):
        """Initialize the FinGPT processor.

        Parameters
        ----------
        model_name : str
            Hugging Face model identifier for FinGPT
        device : str, optional
            Device to run inference on. Auto-detected if None.
        use_8bit : bool
            Whether to use 8-bit quantization to reduce memory usage
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers required. Install with: pip install transformers torch")

        self.model_name = model_name
        # Force CPU to avoid MPS issues with generation tasks
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Disable MPS even if available
        if self.device == "mps":
            logging.warning("MPS device requested but forcing CPU to avoid generation issues")
            self.device = "cpu"
        self.use_8bit = use_8bit

        self._load_model()

    def _load_model(self):
        """Load the FinGPT model and tokenizer."""
        logging.info(f"Loading FinGPT model: {self.model_name}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Configure model loading options
            if self.device == "cuda":
                model_kwargs = {
                    "torch_dtype": torch.float16,
                    "device_map": "auto",
                }
                if self.use_8bit:
                    model_kwargs["load_in_8bit"] = True
            else:
                # CPU configuration - use float32 and explicit device mapping
                model_kwargs = {
                    "torch_dtype": torch.float32,
                    "device_map": {"": "cpu"},
                }

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )

            # Create pipeline for easier inference
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1,  # Force CPU for pipeline
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )

            logging.info(f"FinGPT loaded successfully on {self.device}")

        except Exception as e:
            logging.error(f"Failed to load FinGPT model: {e}")
            raise

    def preprocess_text(self, text: str) -> str:
        """Clean article text using spaCy and normalize formatting (lowercase, punctuation)."""
        if not SPACY_AVAILABLE:
            raise ImportError("spaCy required for preprocessing. Install with: pip install spacy && python -m spacy download en_core_web_sm")
        
        doc = nlp(text)
        # Clean: remove stop words, lemmatize, keep alpha tokens
        cleaned = " ".join([token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha])
        # Normalize: remove punctuation, extra spaces, lowercase
        cleaned = re.sub(r'[^\w\s]', '', cleaned).strip().lower()
        return cleaned

    def save_cleaned_text(self, cleaned_text: str, output_path: str = "data/silver/cleaned_text.json"):
        """Save cleaned text as intermediate output."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({"cleaned_text": cleaned_text}, f)
        logging.info(f"Cleaned text saved to {output_path}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def analyze_sgd_news(self, news_text: str, headline: str = "", market_context: Dict = None) -> FinGPTAnalysis:
        """Analyze financial news for SGD trading signals using FinGPT with market context, including preprocessing and storage.

        Parameters
        ----------
        news_text : str
            The main article text to analyze
        headline : str, optional
            Article headline for additional context
        market_context : dict, optional
            Current market state (technical indicators, volatility, etc.)

        Returns
        -------
        FinGPTAnalysis
            Structured analysis results with market-aware insights
        """
        # Text Preprocessing: Clean and normalize
        cleaned_text = self.preprocess_text(news_text)
        self.save_cleaned_text(cleaned_text)

        # LLM Feature Extraction: Design prompt (using FinGPT, not OpenAI, as per your code), call on cleaned text
        prompt = self._build_market_aware_prompt(cleaned_text, headline, market_context)

        try:
            # Generate analysis using FinGPT
            response = self.pipeline(
                prompt,
                max_length=1024,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

            raw_output = response[0]['generated_text']

            # Extract the model's response (after the prompt)
            model_response = raw_output[len(prompt):].strip()

            # Parse into tabular features (sentiment, topic, urgency, etc.)
            analysis = self._parse_response(model_response)

            # Store enriched output in /data/silver/
            self._store_enriched_output(analysis)

            return analysis

        except Exception as e:
            logging.error(f"FinGPT analysis failed after retries: {e}")
            # Return neutral analysis on failure
            return FinGPTAnalysis(
                sentiment_score=0.0,
                confidence=0.0,
                sgd_directional_signal=0.0,
                policy_implications="neutral",
                time_horizon="unknown",
                key_factors=[],
                market_coherence="neutral",
                signal_strength_adjusted=0.0,
                topic="general",
                urgency="low",
                raw_response=f"Error: {str(e)}"
            )

    def _build_market_aware_prompt(self, news_text: str, headline: str = "", market_context: Dict = None) -> str:
        """Build a market-aware prompt for SGD trading signal analysis, including topic and urgency extraction."""

        context = f"Headline: {headline}\n" if headline else ""
        context += f"Article: {news_text}"

        # Add market context if available
        market_section = ""
        if market_context:
            market_section = f"""
CURRENT MARKET STATE:
- USD/SGD Mid Price: {market_context.get('mid', 'N/A'):.4f}
- Recent 5-tick Return: {market_context.get('ret_5', 0):.2%}
- 20-period Volatility: {market_context.get('vol_20', 0):.2%}
- Volatility Regime: {'High' if market_context.get('high_vol_regime', False) else 'Normal'}
- Spread (% of mid): {market_context.get('spread_pct', 0):.3%}
- Price Z-Score (20-period): {market_context.get('zscore_20', 0):.2f}
- Trading Session: {market_context.get('session', 'Unknown')}
"""

        prompt = f"""You are a financial analyst specializing in Singapore Dollar (SGD) trading.
Analyze the following news for SGD trading signals, considering BOTH the news content AND current market conditions.

{context}
{market_section}

Consider these key questions:
1. How does this news relate to the current market state?
2. Has the market already priced in this information?
3. Does the news sentiment align with or diverge from current price action?
4. What is the expected impact given current volatility and market regime?

Provide analysis in this exact format:
SENTIMENT: [bullish/bearish/neutral]
CONFIDENCE: [0.0-1.0]
SGD_SIGNAL: [bullish/bearish/neutral]
POLICY: [hawkish/dovish/neutral]
TIMEFRAME: [immediate/short_term/medium_term]
MARKET_COHERENCE: [aligned/divergent/neutral]
ADJUSTED_STRENGTH: [0.0-1.0]
TOPIC: [general/economy/policy/trade/etc.]  # Added for tabular features
URGENCY: [low/medium/high]  # Added for tabular features
FACTORS: [key market drivers, separated by semicolons]

Analysis:"""

        return prompt

    def _build_sgd_prompt(self, news_text: str, headline: str = "") -> str:
        """Build a focused prompt for SGD trading signal analysis (legacy method)."""
        # Fallback to market-aware prompt without market context
        return self._build_market_aware_prompt(news_text, headline, None)

    def _parse_response(self, response: str) -> FinGPTAnalysis:
        """Parse the structured FinGPT response into analysis object, including topic and urgency."""

        # Default values
        sentiment_score = 0.0
        confidence = 0.0
        sgd_directional_signal = 0.0
        policy_implications = "neutral"
        time_horizon = "unknown"
        key_factors = []
        market_coherence = "neutral"
        signal_strength_adjusted = 0.0
        topic = "general"
        urgency = "low"

        try:
            # Extract structured fields using regex
            sentiment_match = re.search(r'SENTIMENT:\s*(\w+)', response, re.IGNORECASE)
            if sentiment_match:
                sentiment = sentiment_match.group(1).lower()
                sentiment_score = self._sentiment_to_score(sentiment)

            confidence_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', response, re.IGNORECASE)
            if confidence_match:
                confidence = float(confidence_match.group(1))
                confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]

            sgd_match = re.search(r'SGD_SIGNAL:\s*(\w+)', response, re.IGNORECASE)
            if sgd_match:
                sgd_signal = sgd_match.group(1).lower()
                sgd_directional_signal = self._sentiment_to_score(sgd_signal)

            policy_match = re.search(r'POLICY:\s*(\w+)', response, re.IGNORECASE)
            if policy_match:
                policy_implications = policy_match.group(1).lower()

            timeframe_match = re.search(r'TIMEFRAME:\s*(\w+)', response, re.IGNORECASE)
            if timeframe_match:
                time_horizon = timeframe_match.group(1).lower()

            # New market-aware fields
            coherence_match = re.search(r'MARKET_COHERENCE:\s*(\w+)', response, re.IGNORECASE)
            if coherence_match:
                market_coherence = coherence_match.group(1).lower()

            adjusted_strength_match = re.search(r'ADJUSTED_STRENGTH:\s*([0-9.]+)', response, re.IGNORECASE)
            if adjusted_strength_match:
                signal_strength_adjusted = float(adjusted_strength_match.group(1))
                signal_strength_adjusted = max(0.0, min(1.0, signal_strength_adjusted))
            else:
                # Fallback: use absolute sentiment as adjusted strength
                signal_strength_adjusted = abs(sentiment_score)

            # Added for tabular features
            topic_match = re.search(r'TOPIC:\s*(\w+)', response, re.IGNORECASE)
            if topic_match:
                topic = topic_match.group(1).lower()

            urgency_match = re.search(r'URGENCY:\s*(\w+)', response, re.IGNORECASE)
            if urgency_match:
                urgency = urgency_match.group(1).lower()

            factors_match = re.search(r'FACTORS:\s*(.+)', response, re.IGNORECASE)
            if factors_match:
                factors_text = factors_match.group(1).strip()
                key_factors = [f.strip() for f in factors_text.split(';') if f.strip()]

        except Exception as e:
            logging.warning(f"Failed to parse FinGPT response: {e}")

        return FinGPTAnalysis(
            sentiment_score=sentiment_score,
            confidence=confidence,
            sgd_directional_signal=sgd_directional_signal,
            policy_implications=policy_implications,
            time_horizon=time_horizon,
            key_factors=key_factors,
            market_coherence=market_coherence,
            signal_strength_adjusted=signal_strength_adjusted,
            topic=topic,
            urgency=urgency,
            raw_response=response
        )

    def _store_enriched_output(self, analysis: FinGPTAnalysis, output_path: str = "data/silver/enriched_features.parquet"):
        """Store enriched output in tabular format (Parquet) in /data/silver/."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([analysis.__dict__])
        df.to_parquet(output_path, index=False)
        logging.info(f"Enriched features stored to {output_path}")

    def _sentiment_to_score(self, sentiment: str) -> float:
        """Convert sentiment text to numeric score."""
        sentiment = sentiment.lower().strip()

        if sentiment in ['bullish', 'positive', 'bull']:
            return 1.0
        elif sentiment in ['bearish', 'negative', 'bear']:
            return -1.0
        else:
            return 0.0


class LexiconFallback:
    """Fallback processor using simple lexicon-based analysis when FinGPT fails."""

    POSITIVE_WORDS = {
        'growth', 'gain', 'improve', 'strong', 'bullish', 'increase',
        'optimistic', 'upgrade', 'boost', 'surge', 'rally'
    }

    NEGATIVE_WORDS = {
        'fall', 'risk', 'slowdown', 'bearish', 'decline', 'downgrade',
        'weak', 'loss', 'crash', 'plunge', 'drop'
    }

    SGD_KEYWORDS = {
        'sgd', 'singapore dollar', 'monetary authority', 'mas', 'singapore'
    }

    @classmethod
    def analyze(cls, text: str) -> FinGPTAnalysis:
        """Perform simple lexicon-based analysis as fallback."""
        text_lower = text.lower()
        words = set(text_lower.split())

        positive_count = len(words & cls.POSITIVE_WORDS)
        negative_count = len(words & cls.NEGATIVE_WORDS)
        sgd_mentions = any(keyword in text_lower for keyword in cls.SGD_KEYWORDS)

        # Simple sentiment calculation
        if positive_count > negative_count:
            sentiment_score = 0.5
            sgd_signal = 0.5 if sgd_mentions else 0.2
        elif negative_count > positive_count:
            sentiment_score = -0.5
            sgd_signal = -0.5 if sgd_mentions else -0.2
        else:
            sentiment_score = 0.0
            sgd_signal = 0.0

        return FinGPTAnalysis(
            sentiment_score=sentiment_score,
            confidence=0.3,  # Low confidence for lexicon
            sgd_directional_signal=sgd_signal,
            policy_implications="neutral",
            time_horizon="unknown",
            key_factors=list(words & (cls.POSITIVE_WORDS | cls.NEGATIVE_WORDS)),
            market_coherence="neutral",  # No market analysis in fallback
            signal_strength_adjusted=abs(sentiment_score),
            topic="general",
            urgency="low",
            raw_response="Lexicon-based fallback analysis"
        )


def create_processor(use_fingpt: bool = True, **kwargs) -> object:
    """Factory function to create appropriate news processor.

    Parameters
    ----------
    use_fingpt : bool
        Whether to use FinGPT or fall back to lexicon analysis
    **kwargs
        Additional arguments for FinGPTProcessor

    Returns
    -------
    processor
        Either FinGPTProcessor or LexiconFallback instance
    """
    if use_fingpt and TRANSFORMERS_AVAILABLE:
        try:
            return FinGPTProcessor(**kwargs)
        except Exception as e:
            logging.warning(f"Failed to initialize FinGPT, falling back to lexicon: {e}")
            return LexiconFallback()
    else:
        logging.info("Using lexicon-based fallback processor")
        return LexiconFallback()