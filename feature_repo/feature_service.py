from feast import FeatureService
from feature_repo.market_features import market_features_view
from feature_repo.news_signals import news_signals_view

# Combine market and news FeatureViews for training/serving
combined_service = FeatureService(
    name="fx_combined_service",
    features=[
        market_features_view,
        news_signals_view,
    ],
)

