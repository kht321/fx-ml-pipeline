"""
Comprehensive Evidently-based drift detection with configurable thresholds and email alerts.

Detects:
- Data drift (feature distribution changes)
- Prediction drift (model output distribution changes)
- Target drift (label distribution changes)
- Missing value drift
- Model performance degradation
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime

import pandas as pd
import numpy as np

# ✓ Evidently 0.7.0 imports
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently.metrics import (
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    RegressionQualityMetric,
    RegressionPredictedVsActualScatter,
    RegressionErrorPlot
)
from evidently import ColumnMapping

from src_clean.monitoring.email_alerter import EmailAlerter

logger = logging.getLogger(__name__)


class EvidentlyDriftDetector:
    """
    Comprehensive drift detection system using Evidently AI.

    Configurable thresholds for:
    - Data drift: KS test threshold (default: 0.1 = 10% drift)
    - Performance degradation: RMSE increase threshold (default: 0.2 = 20% worse)
    - Missing values: Percentage threshold (default: 0.05 = 5% missing)
    """

    def __init__(
        self,
        reference_data_path: Path,
        current_data_path: Path,
        predictions_path: Optional[Path] = None,
        output_dir: Path = Path('data_clean/monitoring/reports'),
        data_drift_threshold: float = 0.1,
        performance_degradation_threshold: float = 0.2,
        missing_values_threshold: float = 0.05,
        email_alerter: Optional[EmailAlerter] = None,
        alert_recipients: Optional[List[str]] = None,
        greeting: str = "Team"
    ):
        """
        Initialize drift detector.

        Args:
            reference_data_path: Path to reference/baseline dataset (training data)
            current_data_path: Path to current dataset (production/OOT data)
            predictions_path: Path to model predictions (optional)
            output_dir: Directory to save reports
            data_drift_threshold: KS test p-value threshold for data drift (0-1)
            performance_degradation_threshold: RMSE increase threshold (0-1, e.g., 0.2 = 20%)
            missing_values_threshold: Missing values percentage threshold (0-1)
            email_alerter: EmailAlerter instance for sending alerts
            alert_recipients: List of email addresses to send alerts
            greeting: Greeting name for emails (e.g., "Boss", "Team")
        """
        self.reference_data_path = Path(reference_data_path)
        self.current_data_path = Path(current_data_path)
        self.predictions_path = Path(predictions_path) if predictions_path else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.data_drift_threshold = data_drift_threshold
        self.performance_degradation_threshold = performance_degradation_threshold
        self.missing_values_threshold = missing_values_threshold

        self.email_alerter = email_alerter
        self.alert_recipients = alert_recipients or []
        self.greeting = greeting

        logger.info(f"DriftDetector initialized with thresholds:")
        logger.info(f"  - Data drift: {data_drift_threshold:.2%}")
        logger.info(f"  - Performance degradation: {performance_degradation_threshold:.2%}")
        logger.info(f"  - Missing values: {missing_values_threshold:.2%}")

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load reference and current datasets.

        Returns:
            Tuple of (reference_df, current_df)
        """
        logger.info(f"Loading reference data from {self.reference_data_path}")
        if str(self.reference_data_path).endswith('.parquet'):
            reference_df = pd.read_parquet(self.reference_data_path)
        else:
            reference_df = pd.read_csv(self.reference_data_path)

        logger.info(f"Loading current data from {self.current_data_path}")
        if str(self.current_data_path).endswith('.parquet'):
            current_df = pd.read_parquet(self.current_data_path)
        else:
            current_df = pd.read_csv(self.current_data_path)

        logger.info(f"Reference data: {len(reference_df)} rows, {len(reference_df.columns)} columns")
        logger.info(f"Current data: {len(current_df)} rows, {len(current_df.columns)} columns")

        return reference_df, current_df

    def prepare_column_mapping(
        self,
        df: pd.DataFrame,
        target_col: str = 'target_return',
        prediction_col: Optional[str] = 'prediction',
        datetime_col: Optional[str] = 'time'
    ) -> ColumnMapping:
        """
        Prepare Evidently column mapping.

        Args:
            df: DataFrame to analyze
            target_col: Target column name
            prediction_col: Prediction column name (if available)
            datetime_col: Datetime column name (if available)

        Returns:
            ColumnMapping object
        """
        column_mapping = ColumnMapping()

        # Set target
        if target_col in df.columns:
            column_mapping.target = target_col
            logger.info(f"Target column: {target_col}")
        else:
            logger.warning(f"Target column '{target_col}' not found")

        # Set prediction
        if prediction_col and prediction_col in df.columns:
            column_mapping.prediction = prediction_col
            logger.info(f"Prediction column: {prediction_col}")

        # Set datetime
        if datetime_col and datetime_col in df.columns:
            column_mapping.datetime = datetime_col
            logger.info(f"Datetime column: {datetime_col}")

        # Identify numerical features (exclude target, prediction, datetime, id columns)
        exclude_cols = {target_col, prediction_col, datetime_col, 'time', 'user_id', 'id', 'date'}
        numerical_features = [
            col for col in df.select_dtypes(include=[np.number]).columns
            if col not in exclude_cols
        ]

        column_mapping.numerical_features = numerical_features
        logger.info(f"Numerical features: {len(numerical_features)}")

        return column_mapping

    def detect_data_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        column_mapping: ColumnMapping
    ) -> Dict:
        """
        Detect data drift using Evidently DataDriftPreset.

        Args:
            reference_df: Reference dataset
            current_df: Current dataset
            column_mapping: Column mapping

        Returns:
            Dictionary with drift metrics
        """
        logger.info("Detecting data drift...")

        report = Report(metrics=[
            DataDriftPreset(stattest='ks', stattest_threshold=self.data_drift_threshold),
            DatasetDriftMetric(),
            DatasetMissingValuesMetric()
        ])

        report.run(
            reference_data=reference_df,
            current_data=current_df,
            column_mapping=column_mapping
        )

        # Save report
        report_path = self.output_dir / f'data_drift_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
        report.save_html(str(report_path))
        logger.info(f"Data drift report saved: {report_path}")

        # Extract metrics
        result_dict = report.as_dict()
        dataset_drift = result_dict['metrics'][1]['result']

        drift_summary = {
            'dataset_drift_detected': dataset_drift.get('dataset_drift', False),
            'drift_share': dataset_drift.get('drift_share', 0.0),
            'number_of_drifted_columns': dataset_drift.get('number_of_drifted_columns', 0),
            'report_path': str(report_path)
        }

        logger.info(f"Dataset drift: {drift_summary['dataset_drift_detected']}")
        logger.info(f"Drift share: {drift_summary['drift_share']:.2%}")
        logger.info(f"Drifted columns: {drift_summary['number_of_drifted_columns']}")

        return drift_summary

    def detect_performance_degradation(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        column_mapping: ColumnMapping
    ) -> Dict:
        """
        Detect model performance degradation.

        Args:
            reference_df: Reference dataset with target and predictions
            current_df: Current dataset with target and predictions
            column_mapping: Column mapping

        Returns:
            Dictionary with performance metrics
        """
        logger.info("Detecting performance degradation...")

        if not column_mapping.prediction or column_mapping.prediction not in current_df.columns:
            logger.warning("Prediction column not available. Skipping performance check.")
            return {'performance_check_available': False}

        report = Report(metrics=[
            RegressionPreset(),
            RegressionQualityMetric(),
            RegressionPredictedVsActualScatter(),
            RegressionErrorPlot()
        ])

        report.run(
            reference_data=reference_df,
            current_data=current_df,
            column_mapping=column_mapping
        )

        # Save report
        report_path = self.output_dir / f'performance_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
        report.save_html(str(report_path))
        logger.info(f"Performance report saved: {report_path}")

        # Extract metrics
        result_dict = report.as_dict()

        # Find RegressionQualityMetric result
        quality_metric = None
        for metric in result_dict['metrics']:
            if metric['metric'] == 'RegressionQualityMetric':
                quality_metric = metric['result']
                break

        if not quality_metric:
            logger.warning("Could not extract regression quality metrics")
            return {'performance_check_available': False}

        ref_rmse = quality_metric.get('reference', {}).get('rmse', 0)
        curr_rmse = quality_metric.get('current', {}).get('rmse', 0)

        # Calculate RMSE increase
        if ref_rmse > 0:
            rmse_increase = (curr_rmse - ref_rmse) / ref_rmse
        else:
            rmse_increase = 0

        degradation_detected = rmse_increase > self.performance_degradation_threshold

        performance_summary = {
            'performance_check_available': True,
            'reference_rmse': ref_rmse,
            'current_rmse': curr_rmse,
            'rmse_increase': rmse_increase,
            'degradation_detected': degradation_detected,
            'degradation_threshold': self.performance_degradation_threshold,
            'report_path': str(report_path)
        }

        logger.info(f"Reference RMSE: {ref_rmse:.4f}")
        logger.info(f"Current RMSE: {curr_rmse:.4f}")
        logger.info(f"RMSE increase: {rmse_increase:.2%}")
        logger.info(f"Degradation detected: {degradation_detected}")

        return performance_summary

    def check_missing_values(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame
    ) -> Dict:
        """
        Check for missing value drift.

        Args:
            reference_df: Reference dataset
            current_df: Current dataset

        Returns:
            Dictionary with missing values metrics
        """
        logger.info("Checking missing values...")

        ref_missing = reference_df.isnull().sum().sum() / (reference_df.shape[0] * reference_df.shape[1])
        curr_missing = current_df.isnull().sum().sum() / (current_df.shape[0] * current_df.shape[1])

        missing_increase = curr_missing - ref_missing
        drift_detected = curr_missing > self.missing_values_threshold

        missing_summary = {
            'reference_missing_pct': ref_missing,
            'current_missing_pct': curr_missing,
            'missing_increase': missing_increase,
            'drift_detected': drift_detected,
            'threshold': self.missing_values_threshold
        }

        logger.info(f"Reference missing: {ref_missing:.2%}")
        logger.info(f"Current missing: {curr_missing:.2%}")
        logger.info(f"Missing drift detected: {drift_detected}")

        return missing_summary

    def run_drift_detection(self) -> Dict:
        """
        Run complete drift detection pipeline.

        Returns:
            Dictionary with all drift metrics and alerts
        """
        logger.info("=" * 80)
        logger.info("Starting comprehensive drift detection")
        logger.info("=" * 80)

        # Load data
        reference_df, current_df = self.load_data()

        # Prepare column mapping
        column_mapping = self.prepare_column_mapping(current_df)

        # Run all drift checks
        results = {
            'timestamp': datetime.now().isoformat(),
            'reference_data': str(self.reference_data_path),
            'current_data': str(self.current_data_path),
            'thresholds': {
                'data_drift': self.data_drift_threshold,
                'performance_degradation': self.performance_degradation_threshold,
                'missing_values': self.missing_values_threshold
            }
        }

        # 1. Data drift detection
        data_drift = self.detect_data_drift(reference_df, current_df, column_mapping)
        results['data_drift'] = data_drift

        # 2. Performance degradation
        performance = self.detect_performance_degradation(reference_df, current_df, column_mapping)
        results['performance'] = performance

        # 3. Missing values check
        missing_values = self.check_missing_values(reference_df, current_df)
        results['missing_values'] = missing_values

        # Determine if any drift detected
        drift_detected = (
            data_drift.get('dataset_drift_detected', False) or
            performance.get('degradation_detected', False) or
            missing_values.get('drift_detected', False)
        )

        results['overall_drift_detected'] = drift_detected

        # Save results JSON
        results_path = self.output_dir / f'drift_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Drift results saved: {results_path}")

        # Send email alert if drift detected
        if drift_detected and self.email_alerter and self.alert_recipients:
            logger.info("Drift detected - sending email alert...")
            self._send_drift_alert(results)
        elif drift_detected:
            logger.warning("Drift detected but email alerter not configured")
        else:
            logger.info("✅ No drift detected - system healthy")

        logger.info("=" * 80)
        logger.info("Drift detection completed")
        logger.info("=" * 80)

        return results

    def _send_drift_alert(self, results: Dict):
        """
        Send drift alert email.

        Args:
            results: Drift detection results dictionary
        """
        # Build drift summary for email
        drift_summary = {}

        # Data drift
        if results['data_drift'].get('dataset_drift_detected'):
            drift_summary['Data Drift'] = {
                'value': f"{results['data_drift']['drift_share']:.2%} of columns",
                'threshold': f"{self.data_drift_threshold:.2%}",
                'drift_detected': True,
                'status': 'DRIFT DETECTED'
            }

        # Performance degradation
        if results['performance'].get('performance_check_available'):
            perf = results['performance']
            if perf.get('degradation_detected'):
                drift_summary['Performance Degradation'] = {
                    'value': f"RMSE increased by {perf['rmse_increase']:.2%}",
                    'threshold': f"{self.performance_degradation_threshold:.2%}",
                    'drift_detected': True,
                    'status': 'DEGRADED'
                }

        # Missing values
        if results['missing_values'].get('drift_detected'):
            mv = results['missing_values']
            drift_summary['Missing Values'] = {
                'value': f"{mv['current_missing_pct']:.2%}",
                'threshold': f"{self.missing_values_threshold:.2%}",
                'drift_detected': True,
                'status': 'THRESHOLD EXCEEDED'
            }

        # Get report paths
        report_paths = []
        if 'report_path' in results['data_drift']:
            report_paths.append(Path(results['data_drift']['report_path']))
        if 'report_path' in results['performance']:
            report_paths.append(Path(results['performance']['report_path']))

        # Send email
        if drift_summary:
            success = self.email_alerter.send_drift_alert(
                to_emails=self.alert_recipients,
                drift_summary=drift_summary,
                report_path=report_paths[0] if report_paths else None,
                greeting=self.greeting
            )

            if success:
                logger.info(f"✅ Drift alert sent to {self.alert_recipients}")
            else:
                logger.error(f"❌ Failed to send drift alert")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Evidently drift detection")
    parser.add_argument('--reference-data', required=True, help='Path to reference dataset')
    parser.add_argument('--current-data', required=True, help='Path to current dataset')
    parser.add_argument('--predictions', help='Path to predictions file')
    parser.add_argument('--output-dir', default='data_clean/monitoring/reports', help='Output directory')
    parser.add_argument('--data-drift-threshold', type=float, default=0.1, help='Data drift threshold')
    parser.add_argument('--performance-threshold', type=float, default=0.2, help='Performance degradation threshold')
    parser.add_argument('--missing-threshold', type=float, default=0.05, help='Missing values threshold')
    parser.add_argument('--alert-email', help='Email address for alerts')
    parser.add_argument('--greeting', default='Team', help='Greeting name for emails')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize email alerter if email provided
    email_alerter = None
    alert_recipients = []
    if args.alert_email:
        email_alerter = EmailAlerter()
        alert_recipients = [args.alert_email]

    # Initialize detector
    detector = EvidentlyDriftDetector(
        reference_data_path=Path(args.reference_data),
        current_data_path=Path(args.current_data),
        predictions_path=Path(args.predictions) if args.predictions else None,
        output_dir=Path(args.output_dir),
        data_drift_threshold=args.data_drift_threshold,
        performance_degradation_threshold=args.performance_threshold,
        missing_values_threshold=args.missing_threshold,
        email_alerter=email_alerter,
        alert_recipients=alert_recipients,
        greeting=args.greeting
    )

    # Run detection
    results = detector.run_drift_detection()

    # Print summary
    print("\n" + "=" * 80)
    print("DRIFT DETECTION SUMMARY")
    print("=" * 80)
    print(f"Overall Drift Detected: {results['overall_drift_detected']}")
    print(f"Data Drift: {results['data_drift'].get('dataset_drift_detected', False)}")
    print(f"Performance Degradation: {results['performance'].get('degradation_detected', False)}")
    print(f"Missing Values Drift: {results['missing_values'].get('drift_detected', False)}")
    print("=" * 80)
