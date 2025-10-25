from feast import FeatureStore

def get_store() -> FeatureStore:
    return FeatureStore(repo_path="feature_repo")

