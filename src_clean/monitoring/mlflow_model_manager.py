"""
Enhanced MLflow model management with versioning, staging, and promotion workflows.

Features:
- Model registration with automatic versioning
- Stage promotion (None → Staging → Production)
- Model aliasing (champion, challenger)
- Cross-experiment comparison
- Model transition logging
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion

logger = logging.getLogger(__name__)


class MLflowModelManager:
    """
    Advanced MLflow model lifecycle management.

    Capabilities:
    - Register models with automatic versioning
    - Promote models through stages (Staging → Production)
    - Set model aliases for easy reference
    - Compare models across experiments
    - Track model transitions
    """

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5001",
        registry_model_name: str = "sp500_best_model"
    ):
        """
        Initialize MLflow model manager.

        Args:
            tracking_uri: MLflow tracking server URI
            registry_model_name: Name for registered model in MLflow Model Registry
        """
        self.tracking_uri = tracking_uri
        self.registry_model_name = registry_model_name

        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri)

        logger.info(f"MLflow Model Manager initialized")
        logger.info(f"  Tracking URI: {tracking_uri}")
        logger.info(f"  Registry Model: {registry_model_name}")

    def register_model(
        self,
        run_id: str,
        model_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ) -> ModelVersion:
        """
        Register a model from an MLflow run.

        Args:
            run_id: MLflow run ID
            model_name: Model name (defaults to registry_model_name)
            tags: Tags to add to model version
            description: Model version description

        Returns:
            Registered ModelVersion
        """
        model_name = model_name or self.registry_model_name

        logger.info(f"Registering model from run {run_id}")

        # Register model
        model_uri = f"runs:/{run_id}/model"
        model_version = mlflow.register_model(model_uri, model_name)

        logger.info(f"Model registered: {model_name} version {model_version.version}")

        # Add tags
        if tags:
            for key, value in tags.items():
                self.client.set_model_version_tag(
                    name=model_name,
                    version=model_version.version,
                    key=key,
                    value=str(value)
                )
            logger.info(f"Added {len(tags)} tags to model version")

        # Update description
        if description:
            self.client.update_model_version(
                name=model_name,
                version=model_version.version,
                description=description
            )
            logger.info(f"Updated model description")

        return model_version

    def promote_to_staging(
        self,
        model_name: Optional[str] = None,
        version: Optional[str] = None,
        archive_existing: bool = True
    ) -> ModelVersion:
        """
        Promote a model version to Staging stage.

        Args:
            model_name: Model name (defaults to registry_model_name)
            version: Model version (defaults to latest)
            archive_existing: Whether to archive existing Staging models

        Returns:
            Updated ModelVersion
        """
        model_name = model_name or self.registry_model_name

        # Get version
        if version is None:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            if not versions:
                raise ValueError(f"No versions found for model {model_name}")
            version = max([int(v.version) for v in versions])

        logger.info(f"Promoting {model_name} version {version} to Staging")

        # Archive existing Staging models
        if archive_existing:
            staging_versions = self.client.get_latest_versions(model_name, stages=["Staging"])
            for sv in staging_versions:
                logger.info(f"Archiving existing Staging version {sv.version}")
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=sv.version,
                    stage="Archived"
                )

        # Promote to Staging
        model_version = self.client.transition_model_version_stage(
            name=model_name,
            version=str(version),
            stage="Staging"
        )

        logger.info(f"✅ Model promoted to Staging: {model_name} v{version}")

        return model_version

    def promote_to_production(
        self,
        model_name: Optional[str] = None,
        version: Optional[str] = None,
        archive_existing: bool = True,
        set_as_champion: bool = True
    ) -> ModelVersion:
        """
        Promote a model version to Production stage.

        Args:
            model_name: Model name (defaults to registry_model_name)
            version: Model version (defaults to latest Staging)
            archive_existing: Whether to archive existing Production models
            set_as_champion: Whether to set "champion" alias

        Returns:
            Updated ModelVersion
        """
        model_name = model_name or self.registry_model_name

        # Get version (default to latest Staging)
        if version is None:
            staging_versions = self.client.get_latest_versions(model_name, stages=["Staging"])
            if not staging_versions:
                raise ValueError(f"No Staging versions found for model {model_name}")
            version = staging_versions[0].version

        logger.info(f"Promoting {model_name} version {version} to Production")

        # Archive existing Production models
        if archive_existing:
            prod_versions = self.client.get_latest_versions(model_name, stages=["Production"])
            for pv in prod_versions:
                logger.info(f"Archiving existing Production version {pv.version}")
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=pv.version,
                    stage="Archived"
                )

        # Promote to Production
        model_version = self.client.transition_model_version_stage(
            name=model_name,
            version=str(version),
            stage="Production"
        )

        # Set champion alias
        if set_as_champion:
            self.set_model_alias(
                model_name=model_name,
                version=str(version),
                alias="champion"
            )

        logger.info(f"✅ Model promoted to Production: {model_name} v{version}")

        return model_version

    def set_model_alias(
        self,
        alias: str,
        model_name: Optional[str] = None,
        version: Optional[str] = None
    ):
        """
        Set an alias for a model version.

        Common aliases: "champion", "challenger", "shadow"

        Args:
            alias: Alias name
            model_name: Model name (defaults to registry_model_name)
            version: Model version (defaults to latest Production)
        """
        model_name = model_name or self.registry_model_name

        # Get version
        if version is None:
            prod_versions = self.client.get_latest_versions(model_name, stages=["Production"])
            if not prod_versions:
                raise ValueError(f"No Production versions found for model {model_name}")
            version = prod_versions[0].version

        logger.info(f"Setting alias '{alias}' for {model_name} version {version}")

        self.client.set_registered_model_alias(
            name=model_name,
            alias=alias,
            version=str(version)
        )

        logger.info(f"✅ Alias '{alias}' set for {model_name} v{version}")

    def get_model_by_alias(self, alias: str, model_name: Optional[str] = None) -> ModelVersion:
        """
        Get model version by alias.

        Args:
            alias: Alias name (e.g., "champion", "challenger")
            model_name: Model name (defaults to registry_model_name)

        Returns:
            ModelVersion with the specified alias
        """
        model_name = model_name or self.registry_model_name

        model_version = self.client.get_model_version_by_alias(model_name, alias)
        logger.info(f"Retrieved model by alias '{alias}': {model_name} v{model_version.version}")

        return model_version

    def compare_models(
        self,
        version1: str,
        version2: str,
        model_name: Optional[str] = None
    ) -> Dict:
        """
        Compare two model versions.

        Args:
            version1: First model version
            version2: Second model version
            model_name: Model name (defaults to registry_model_name)

        Returns:
            Dictionary with comparison results
        """
        model_name = model_name or self.registry_model_name

        logger.info(f"Comparing {model_name} versions {version1} vs {version2}")

        # Get model versions
        mv1 = self.client.get_model_version(model_name, version1)
        mv2 = self.client.get_model_version(model_name, version2)

        # Get run metrics
        run1 = self.client.get_run(mv1.run_id)
        run2 = self.client.get_run(mv2.run_id)

        comparison = {
            'model_name': model_name,
            'version1': {
                'version': version1,
                'stage': mv1.current_stage,
                'run_id': mv1.run_id,
                'metrics': run1.data.metrics,
                'created_at': mv1.creation_timestamp
            },
            'version2': {
                'version': version2,
                'stage': mv2.current_stage,
                'run_id': mv2.run_id,
                'metrics': run2.data.metrics,
                'created_at': mv2.creation_timestamp
            }
        }

        # Calculate metric differences
        metric_diffs = {}
        for metric_name in run1.data.metrics.keys():
            if metric_name in run2.data.metrics:
                v1_val = run1.data.metrics[metric_name]
                v2_val = run2.data.metrics[metric_name]
                diff = v2_val - v1_val
                pct_change = (diff / v1_val * 100) if v1_val != 0 else 0
                metric_diffs[metric_name] = {
                    'v1': v1_val,
                    'v2': v2_val,
                    'diff': diff,
                    'pct_change': pct_change
                }

        comparison['metric_differences'] = metric_diffs

        logger.info(f"Comparison complete: {len(metric_diffs)} metrics compared")

        return comparison

    def get_latest_versions_summary(self, model_name: Optional[str] = None) -> Dict:
        """
        Get summary of latest model versions by stage.

        Args:
            model_name: Model name (defaults to registry_model_name)

        Returns:
            Dictionary with latest versions by stage
        """
        model_name = model_name or self.registry_model_name

        summary = {}

        for stage in ["None", "Staging", "Production", "Archived"]:
            versions = self.client.get_latest_versions(model_name, stages=[stage])
            if versions:
                v = versions[0]
                run = self.client.get_run(v.run_id)
                summary[stage] = {
                    'version': v.version,
                    'run_id': v.run_id,
                    'metrics': run.data.metrics,
                    'created_at': v.creation_timestamp,
                    'current_stage': v.current_stage
                }

        logger.info(f"Retrieved summary for {model_name}: {len(summary)} stages")

        return summary

    def list_all_versions(self, model_name: Optional[str] = None) -> List[Dict]:
        """
        List all versions of a registered model.

        Args:
            model_name: Model name (defaults to registry_model_name)

        Returns:
            List of dictionaries with version info
        """
        model_name = model_name or self.registry_model_name

        versions = self.client.search_model_versions(f"name='{model_name}'")

        version_list = []
        for v in sorted(versions, key=lambda x: int(x.version), reverse=True):
            version_list.append({
                'version': v.version,
                'stage': v.current_stage,
                'run_id': v.run_id,
                'created_at': v.creation_timestamp,
                'tags': v.tags
            })

        logger.info(f"Found {len(version_list)} versions for {model_name}")

        return version_list


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MLflow Model Manager CLI")
    parser.add_argument('--tracking-uri', default='http://localhost:5001', help='MLflow tracking URI')
    parser.add_argument('--model-name', default='sp500_best_model', help='Registered model name')
    parser.add_argument('--action', required=True,
                       choices=['list', 'promote-staging', 'promote-prod', 'compare', 'summary'],
                       help='Action to perform')
    parser.add_argument('--version', help='Model version')
    parser.add_argument('--version2', help='Second model version (for compare)')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    manager = MLflowModelManager(
        tracking_uri=args.tracking_uri,
        registry_model_name=args.model_name
    )

    if args.action == 'list':
        versions = manager.list_all_versions()
        print("\n" + "=" * 80)
        print(f"Model Versions for {args.model_name}")
        print("=" * 80)
        for v in versions:
            print(f"Version {v['version']}: Stage={v['stage']}, Run={v['run_id'][:8]}")
        print("=" * 80)

    elif args.action == 'promote-staging':
        mv = manager.promote_to_staging(version=args.version)
        print(f"✅ Promoted version {mv.version} to Staging")

    elif args.action == 'promote-prod':
        mv = manager.promote_to_production(version=args.version)
        print(f"✅ Promoted version {mv.version} to Production")

    elif args.action == 'compare':
        if not args.version or not args.version2:
            print("❌ Error: --version and --version2 required for compare")
        else:
            comp = manager.compare_models(args.version, args.version2)
            print("\n" + "=" * 80)
            print(f"Model Comparison: v{args.version} vs v{args.version2}")
            print("=" * 80)
            for metric, data in comp['metric_differences'].items():
                print(f"{metric}:")
                print(f"  v{args.version}: {data['v1']:.4f}")
                print(f"  v{args.version2}: {data['v2']:.4f}")
                print(f"  Change: {data['diff']:.4f} ({data['pct_change']:.2f}%)")
            print("=" * 80)

    elif args.action == 'summary':
        summary = manager.get_latest_versions_summary()
        print("\n" + "=" * 80)
        print(f"Model Summary: {args.model_name}")
        print("=" * 80)
        for stage, info in summary.items():
            print(f"\n{stage}:")
            print(f"  Version: {info['version']}")
            print(f"  Run ID: {info['run_id'][:8]}")
            print(f"  Metrics: {info['metrics']}")
        print("=" * 80)
