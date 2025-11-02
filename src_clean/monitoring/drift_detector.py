"""Drift detection and email alerting system."""

import json
import logging
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Configuration
PREDICTIONS_LOG_FILE = Path("data_clean/predictions/prediction_log.jsonl")
DRIFT_CONFIG_FILE = Path("config/drift_thresholds.json")
ALERT_LOG_FILE = Path("data_clean/monitoring/alerts.jsonl")

# Default thresholds
DEFAULT_THRESHOLDS = {
    "sentiment_drift_threshold": 0.5,  # Absolute change in avg sentiment
    "prediction_drift_threshold": 0.3,  # Change in prediction distribution
    "confidence_drop_threshold": 0.2,   # Drop in model confidence
    "window_size": 20,                  # Number of recent predictions to compare
    "cooldown_minutes": 30,             # Minimum time between alerts for same drift type
}


class DriftDetector:
    """Detects model and feature drift from prediction logs."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize drift detector.

        Args:
            config_path: Path to drift configuration file
        """
        self.config = self._load_config(config_path)
        self.alert_log_file = ALERT_LOG_FILE
        self.alert_log_file.parent.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: Optional[Path] = None) -> Dict:
        """Load drift detection configuration."""
        if config_path and config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        return DEFAULT_THRESHOLDS.copy()

    def load_recent_predictions(self, limit: int = 100) -> List[Dict]:
        """Load recent predictions from log file.

        Args:
            limit: Maximum number of predictions to load

        Returns:
            List of prediction dictionaries
        """
        if not PREDICTIONS_LOG_FILE.exists():
            return []

        predictions = []
        with open(PREDICTIONS_LOG_FILE) as f:
            for line in f:
                if line.strip():
                    predictions.append(json.loads(line))

        # Return most recent predictions
        return predictions[-limit:] if len(predictions) > limit else predictions

    def detect_sentiment_drift(self, predictions: List[Dict]) -> Optional[Dict]:
        """Detect drift in news sentiment feature.

        Args:
            predictions: List of recent predictions

        Returns:
            Drift alert dictionary if drift detected, None otherwise
        """
        window_size = self.config["window_size"]
        threshold = self.config["sentiment_drift_threshold"]

        if len(predictions) < window_size * 2:
            logger.info("Not enough predictions for sentiment drift detection")
            return None

        # Compare recent window vs baseline window
        baseline = predictions[-window_size * 2:-window_size]
        recent = predictions[-window_size:]

        # Extract sentiment values
        baseline_sentiments = [
            p.get("avg_sentiment", [0])[0] if isinstance(p.get("avg_sentiment"), list)
            else p.get("avg_sentiment", 0)
            for p in baseline
        ]
        recent_sentiments = [
            p.get("avg_sentiment", [0])[0] if isinstance(p.get("avg_sentiment"), list)
            else p.get("avg_sentiment", 0)
            for p in recent
        ]

        baseline_mean = np.mean(baseline_sentiments)
        recent_mean = np.mean(recent_sentiments)
        drift = abs(recent_mean - baseline_mean)

        if drift > threshold:
            return {
                "type": "sentiment_drift",
                "severity": "high" if drift > threshold * 2 else "medium",
                "drift_value": float(drift),
                "baseline_mean": float(baseline_mean),
                "recent_mean": float(recent_mean),
                "threshold": threshold,
                "message": f"Sentiment drift detected: {drift:.3f} (baseline: {baseline_mean:.3f}, recent: {recent_mean:.3f})",
                "timestamp": datetime.now().isoformat(),
            }

        return None

    def detect_prediction_drift(self, predictions: List[Dict]) -> Optional[Dict]:
        """Detect drift in model predictions.

        Args:
            predictions: List of recent predictions

        Returns:
            Drift alert dictionary if drift detected, None otherwise
        """
        window_size = self.config["window_size"]
        threshold = self.config["prediction_drift_threshold"]

        if len(predictions) < window_size * 2:
            return None

        baseline = predictions[-window_size * 2:-window_size]
        recent = predictions[-window_size:]

        # Calculate proportion of bullish predictions
        baseline_bullish = sum(1 for p in baseline if p.get("prediction") == "bullish") / len(baseline)
        recent_bullish = sum(1 for p in recent if p.get("prediction") == "bullish") / len(recent)

        drift = abs(recent_bullish - baseline_bullish)

        if drift > threshold:
            return {
                "type": "prediction_drift",
                "severity": "high" if drift > threshold * 1.5 else "medium",
                "drift_value": float(drift),
                "baseline_bullish_rate": float(baseline_bullish),
                "recent_bullish_rate": float(recent_bullish),
                "threshold": threshold,
                "message": f"Prediction distribution drift detected: {drift:.3f} (baseline bullish: {baseline_bullish:.2%}, recent: {recent_bullish:.2%})",
                "timestamp": datetime.now().isoformat(),
            }

        return None

    def detect_confidence_drop(self, predictions: List[Dict]) -> Optional[Dict]:
        """Detect significant drop in model confidence.

        Args:
            predictions: List of recent predictions

        Returns:
            Drift alert dictionary if detected, None otherwise
        """
        window_size = self.config["window_size"]
        threshold = self.config["confidence_drop_threshold"]

        if len(predictions) < window_size * 2:
            return None

        baseline = predictions[-window_size * 2:-window_size]
        recent = predictions[-window_size:]

        baseline_confidence = np.mean([p.get("confidence", 0) for p in baseline])
        recent_confidence = np.mean([p.get("confidence", 0) for p in recent])

        drop = baseline_confidence - recent_confidence

        if drop > threshold:
            return {
                "type": "confidence_drop",
                "severity": "high" if drop > threshold * 1.5 else "medium",
                "drift_value": float(drop),
                "baseline_confidence": float(baseline_confidence),
                "recent_confidence": float(recent_confidence),
                "threshold": threshold,
                "message": f"Model confidence drop detected: {drop:.3f} (baseline: {baseline_confidence:.3f}, recent: {recent_confidence:.3f})",
                "timestamp": datetime.now().isoformat(),
            }

        return None

    def check_all_drifts(self) -> List[Dict]:
        """Check for all types of drift.

        Returns:
            List of detected drift alerts
        """
        predictions = self.load_recent_predictions()

        if not predictions:
            logger.info("No predictions available for drift detection")
            return []

        alerts = []

        # Check sentiment drift
        sentiment_alert = self.detect_sentiment_drift(predictions)
        if sentiment_alert and not self._is_in_cooldown(sentiment_alert["type"]):
            alerts.append(sentiment_alert)

        # Check prediction drift
        prediction_alert = self.detect_prediction_drift(predictions)
        if prediction_alert and not self._is_in_cooldown(prediction_alert["type"]):
            alerts.append(prediction_alert)

        # Check confidence drop
        confidence_alert = self.detect_confidence_drop(predictions)
        if confidence_alert and not self._is_in_cooldown(confidence_alert["type"]):
            alerts.append(confidence_alert)

        return alerts

    def _is_in_cooldown(self, alert_type: str) -> bool:
        """Check if alert type is in cooldown period.

        Args:
            alert_type: Type of alert to check

        Returns:
            True if in cooldown, False otherwise
        """
        if not self.alert_log_file.exists():
            return False

        cooldown_minutes = self.config["cooldown_minutes"]
        cutoff_time = datetime.now() - timedelta(minutes=cooldown_minutes)

        with open(self.alert_log_file) as f:
            for line in f:
                if line.strip():
                    alert = json.loads(line)
                    if alert.get("type") == alert_type:
                        alert_time = datetime.fromisoformat(alert["timestamp"])
                        if alert_time > cutoff_time:
                            logger.info(f"Alert type {alert_type} is in cooldown")
                            return True

        return False

    def log_alert(self, alert: Dict) -> None:
        """Log alert to file.

        Args:
            alert: Alert dictionary to log
        """
        with open(self.alert_log_file, 'a') as f:
            f.write(json.dumps(alert) + '\n')
        logger.info(f"Logged alert: {alert['type']}")


class EmailAlerter:
    """Sends email alerts for detected drift."""

    def __init__(
        self,
        smtp_server: str = "smtp.gmail.com",
        smtp_port: int = 587,
        sender_email: Optional[str] = None,
        sender_password: Optional[str] = None,
        recipient_emails: Optional[List[str]] = None,
    ):
        """Initialize email alerter.

        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP server port
            sender_email: Email address to send from
            sender_password: Email password or app password
            recipient_emails: List of recipient email addresses
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.recipient_emails = recipient_emails or []
        self.enabled = bool(sender_email and sender_password and recipient_emails)

        if not self.enabled:
            logger.warning("Email alerting disabled - missing credentials or recipients")

    def send_drift_alert(self, alert: Dict) -> bool:
        """Send email alert for detected drift.

        Args:
            alert: Alert dictionary

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            logger.info(f"Email disabled, would have sent: {alert['message']}")
            return False

        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = ', '.join(self.recipient_emails)
            msg['Subject'] = f"🚨 Model Drift Alert: {alert['type'].replace('_', ' ').title()}"

            # Create email body
            body = f"""
Model Drift Alert - S&P 500 ML Pipeline
{'=' * 60}

Alert Type: {alert['type'].replace('_', ' ').title()}
Severity: {alert['severity'].upper()}
Timestamp: {alert['timestamp']}

{alert['message']}

Details:
--------
Drift Value: {alert['drift_value']:.4f}
Threshold: {alert['threshold']:.4f}

Baseline Metrics:
{self._format_baseline_metrics(alert)}

Recent Metrics:
{self._format_recent_metrics(alert)}

Action Required:
----------------
1. Review recent predictions in monitoring dashboard
2. Check for unusual news patterns or market conditions
3. Consider retraining the model if drift persists
4. Verify data pipeline integrity

Dashboard: http://localhost:8501
Monitoring: http://localhost:8050
Predictions Log: data_clean/predictions/prediction_log.jsonl

This is an automated alert from the ML monitoring system.
"""

            msg.attach(MIMEText(body, 'plain'))

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)

            logger.info(f"Sent drift alert email to {', '.join(self.recipient_emails)}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False

    def _format_baseline_metrics(self, alert: Dict) -> str:
        """Format baseline metrics for email."""
        if alert['type'] == 'sentiment_drift':
            return f"  Average Sentiment: {alert['baseline_mean']:.4f}"
        elif alert['type'] == 'prediction_drift':
            return f"  Bullish Rate: {alert['baseline_bullish_rate']:.2%}"
        elif alert['type'] == 'confidence_drop':
            return f"  Average Confidence: {alert['baseline_confidence']:.4f}"
        return ""

    def _format_recent_metrics(self, alert: Dict) -> str:
        """Format recent metrics for email."""
        if alert['type'] == 'sentiment_drift':
            return f"  Average Sentiment: {alert['recent_mean']:.4f}"
        elif alert['type'] == 'prediction_drift':
            return f"  Bullish Rate: {alert['recent_bullish_rate']:.2%}"
        elif alert['type'] == 'confidence_drop':
            return f"  Average Confidence: {alert['recent_confidence']:.4f}"
        return ""


def monitor_and_alert(
    email_config: Optional[Dict] = None,
    dry_run: bool = False,
    config_path: Optional[Path] = None,
) -> List[Dict]:
    """Check for drift and send alerts if detected.

    Args:
        email_config: Email configuration dictionary
        dry_run: If True, don't send emails (just log)
        config_path: Path to drift configuration file

    Returns:
        List of alerts triggered
    """
    detector = DriftDetector(config_path=config_path)
    alerts = detector.check_all_drifts()

    if not alerts:
        logger.info("No drift detected")
        return []

    # Log all alerts
    for alert in alerts:
        detector.log_alert(alert)
        logger.warning(f"DRIFT DETECTED: {alert['message']}")

    # Send email alerts
    if not dry_run and email_config:
        alerter = EmailAlerter(**email_config)
        for alert in alerts:
            alerter.send_drift_alert(alert)

    return alerts


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Run drift detection
    alerts = monitor_and_alert(dry_run=True)

    if alerts:
        print(f"\n{'=' * 60}")
        print(f"DRIFT ALERTS TRIGGERED: {len(alerts)}")
        print(f"{'=' * 60}")
        for alert in alerts:
            print(f"\n{alert['type'].upper()}: {alert['message']}")
    else:
        print("No drift detected")
