# FX ML Pipeline Architecture

## Overview

This project implements a **dual medallion architecture** for SGD FX prediction, separating Market Data and News processing into independent pipelines with FinGPT-enhanced sentiment analysis.

## Architecture Principles

### **1. Separation of Concerns**
- **Market Pipeline**: High-frequency technical analysis
- **News Pipeline**: Event-driven sentiment analysis
- **Combined Layer**: Multi-modal fusion

### **2. Medallion Pattern**
- **Bronze**: Raw data ingestion
- **Silver**: Feature engineering
- **Gold**: Model-ready datasets
- **Combined**: Multi-modal training

### **3. Independent Scaling**
- Market: Continuous processing (every tick)
- News: Event-driven processing (when articles arrive)
- Combined: Scheduled retraining (hourly/daily)

## Data Flow

```
🏦 Market Pipeline:
OANDA Stream → Bronze (Ticks) → Silver (Technical) → Gold (Features)

📰 News Pipeline:
Articles → Bronze (Raw) → Silver (FinGPT) → Gold (Signals)

🤖 Combined:
Market Gold + News Gold → Training → Models → Predictions
```

## Key Components

### **Market Data Processing**
1. **build_market_features.py**: Bronze → Silver transformation
   - Technical indicators (returns, volatility, z-scores)
   - Microstructure features (spreads, liquidity)
   - Risk metrics (volatility regimes)

2. **build_market_gold.py**: Silver → Gold consolidation
   - Feature merging and validation
   - Cross-instrument relationships
   - Temporal features

### **News Processing with FinGPT**
1. **fingpt_processor.py**: FinGPT integration
   - Financial domain sentiment analysis
   - SGD-specific directional signals
   - Policy implications (hawkish/dovish)

2. **build_news_features.py**: Bronze → Silver transformation
   - FinGPT sentiment extraction
   - Entity recognition
   - Topic classification

3. **build_news_gold.py**: Silver → Gold aggregation
   - Temporal signal aggregation
   - Quality scoring
   - Time decay weighting

### **Combined Modeling**
1. **train_combined_model.py**: Multi-modal training
   - As-of join for temporal alignment
   - Multiple model types (Logistic, RF, GBM)
   - Cross-validation and metrics

### **Orchestration**
1. **orchestrate_pipelines.py**: Pipeline coordination
   - Health monitoring
   - Continuous operation
   - Error handling

## Data Architecture

### **Storage Strategy**
- Gold layers remain **separate** (not merged)
- Temporary merging only during combined training
- Independent optimization per pipeline

### **Directory Structure**
```
data/
├── market/bronze/  → market/silver/  → market/gold/
├── news/bronze/    → news/silver/    → news/gold/
└── combined/       (temporary merges + final models)
```

## Model Architecture

### **Three Approaches**
1. **Market-only**: Technical analysis only
2. **News-only**: Sentiment analysis only
3. **Combined**: Multi-modal fusion

### **Performance Expectations**
- Market-only: 65-70% accuracy, <100ms latency
- News-only: 60-65% accuracy, <2s latency
- Combined: 75-80% accuracy, <5s latency

## Technology Stack

### **Core Dependencies**
- **Data**: pandas, numpy
- **ML**: scikit-learn, joblib
- **NLP**: transformers, torch (FinGPT)
- **API**: oandapyV20, requests
- **Config**: pyyaml, python-dotenv

### **FinGPT Integration**
- Model: `FinGPT/fingpt-sentiment_llama2-7b_lora`
- Hardware: 16GB+ RAM, 8GB+ VRAM recommended
- Fallback: Lexicon-based sentiment if FinGPT fails

## Operational Considerations

### **Deployment**
- Market pipeline: Real-time processing
- News pipeline: Batch processing
- Combined models: Scheduled retraining

### **Monitoring**
- Data freshness tracking
- Processing success rates
- Model performance drift
- Pipeline health checks

### **Scaling**
- Market: Horizontal scaling for tick processing
- News: Vertical scaling for FinGPT inference
- Combined: Model serving infrastructure

## Development Workflow

### **Adding Features**
1. Update appropriate medallion pipeline
2. Modify corresponding config YAML
3. Test independently before combining

### **Model Development**
1. Develop on individual Gold datasets
2. Test market-only and news-only performance
3. Combine for multi-modal training

### **Testing Strategy**
- Unit tests per pipeline component
- Integration tests for full workflows
- Performance tests for model accuracy

This architecture provides a robust, scalable foundation for sophisticated FX prediction models.