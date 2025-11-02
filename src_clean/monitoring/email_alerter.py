"""
Email alerting system for pipeline and model monitoring.
Sends notifications for drift detection, pipeline failures, and status updates.
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from typing import Optional, List
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class EmailAlerter:
    """
    Email alerting system for ML pipeline monitoring.

    Supports:
    - SMTP configuration via environment variables
    - HTML formatted emails
    - File attachments
    - Multiple recipients
    """

    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: Optional[int] = None,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        from_email: Optional[str] = None
    ):
        """
        Initialize email alerter with SMTP configuration.

        Args:
            smtp_host: SMTP server hostname (defaults to env SMTP_HOST or smtp.gmail.com)
            smtp_port: SMTP port (defaults to env SMTP_PORT or 587)
            smtp_user: SMTP username (defaults to env SMTP_USER)
            smtp_password: SMTP password (defaults to env SMTP_PASSWORD)
            from_email: From email address (defaults to env FROM_EMAIL or smtp_user)
        """
        self.smtp_host = smtp_host or os.getenv('SMTP_HOST', 'smtp.gmail.com')
        self.smtp_port = int(smtp_port or os.getenv('SMTP_PORT', '587'))
        self.smtp_user = smtp_user or os.getenv('SMTP_USER')
        self.smtp_password = smtp_password or os.getenv('SMTP_PASSWORD')
        self.from_email = from_email or os.getenv('FROM_EMAIL', self.smtp_user)

        if not self.smtp_user or not self.smtp_password:
            logger.warning("SMTP credentials not configured. Email alerts will be disabled.")
            self.enabled = False
        else:
            self.enabled = True
            logger.info(f"Email alerter configured: {self.smtp_host}:{self.smtp_port}")

    def send_email(
        self,
        to_emails: List[str],
        subject: str,
        body_html: str,
        body_text: Optional[str] = None,
        attachments: Optional[List[Path]] = None
    ) -> bool:
        """
        Send an email with optional attachments.

        Args:
            to_emails: List of recipient email addresses
            subject: Email subject
            body_html: HTML email body
            body_text: Plain text email body (optional, fallback for HTML)
            attachments: List of file paths to attach

        Returns:
            True if email sent successfully, False otherwise
        """
        if not self.enabled:
            logger.warning("Email alerter not enabled. Skipping email.")
            return False

        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.from_email
            msg['To'] = ', '.join(to_emails)
            msg['Date'] = datetime.now().strftime('%a, %d %b %Y %H:%M:%S %z')

            # Add text body (fallback)
            if body_text:
                msg.attach(MIMEText(body_text, 'plain'))

            # Add HTML body
            msg.attach(MIMEText(body_html, 'html'))

            # Add attachments
            if attachments:
                for attachment_path in attachments:
                    if attachment_path.exists():
                        with open(attachment_path, 'rb') as f:
                            part = MIMEApplication(f.read(), Name=attachment_path.name)
                        part['Content-Disposition'] = f'attachment; filename="{attachment_path.name}"'
                        msg.attach(part)
                    else:
                        logger.warning(f"Attachment not found: {attachment_path}")

            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)

            logger.info(f"Email sent successfully to {to_emails}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def send_drift_alert(
        self,
        to_emails: List[str],
        drift_summary: dict,
        report_path: Optional[Path] = None,
        greeting: str = "Team"
    ) -> bool:
        """
        Send drift detection alert email.

        Args:
            to_emails: List of recipient emails
            drift_summary: Dictionary with drift metrics
            report_path: Path to Evidently HTML report
            greeting: Name/title to address recipient (e.g., "Boss", "Team")

        Returns:
            True if sent successfully
        """
        subject = "🚨 Model Drift Detected - FX ML Pipeline"

        # Build HTML body
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #d32f2f; }}
                h2 {{ color: #1976d2; }}
                .metric {{ margin: 10px 0; padding: 10px; background-color: #f5f5f5; border-left: 4px solid #ff9800; }}
                .critical {{ border-left-color: #d32f2f; background-color: #ffebee; }}
                .warning {{ border-left-color: #ff9800; }}
                .ok {{ border-left-color: #4caf50; background-color: #e8f5e9; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #1976d2; color: white; }}
                .footer {{ margin-top: 30px; padding-top: 20px; border-top: 2px solid #ddd; color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <h1>🚨 Model Drift Alert</h1>
            <p>Dear {greeting},</p>
            <p>Our monitoring system has detected drift in the FX ML Pipeline model. Please review the details below:</p>

            <h2>Drift Summary</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Threshold</th>
                    <th>Status</th>
                </tr>
        """

        # Add drift metrics
        for metric_name, metric_data in drift_summary.items():
            status_class = 'critical' if metric_data.get('drift_detected') else 'ok'
            status_icon = '❌' if metric_data.get('drift_detected') else '✅'

            html += f"""
                <tr class="{status_class}">
                    <td><strong>{metric_name}</strong></td>
                    <td>{metric_data.get('value', 'N/A')}</td>
                    <td>{metric_data.get('threshold', 'N/A')}</td>
                    <td>{status_icon} {metric_data.get('status', 'Unknown')}</td>
                </tr>
            """

        html += """
            </table>

            <h2>Recommended Actions</h2>
            <ul>
                <li>Review the attached Evidently report for detailed analysis</li>
                <li>Check recent data quality and feature distributions</li>
                <li>Consider retraining the model with recent data</li>
                <li>Investigate potential data pipeline issues</li>
            </ul>

            <p>The detailed Evidently report is attached to this email.</p>

            <div class="footer">
                <p>This is an automated alert from the FX ML Pipeline Monitoring System.</p>
                <p>Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </body>
        </html>
        """

        attachments = [report_path] if report_path and report_path.exists() else None

        return self.send_email(
            to_emails=to_emails,
            subject=subject,
            body_html=html,
            attachments=attachments
        )

    def send_pipeline_status(
        self,
        to_emails: List[str],
        status: str,
        summary: str,
        details: dict,
        greeting: str = "Team"
    ) -> bool:
        """
        Send pipeline execution status email.

        Args:
            to_emails: List of recipient emails
            status: Overall status (SUCCESS, FAILURE, WARNING)
            summary: Brief summary message
            details: Dictionary with detailed information
            greeting: Name/title to address recipient

        Returns:
            True if sent successfully
        """
        status_icon = {
            'SUCCESS': '✅',
            'FAILURE': '❌',
            'WARNING': '⚠️'
        }.get(status, '📊')

        status_color = {
            'SUCCESS': '#4caf50',
            'FAILURE': '#d32f2f',
            'WARNING': '#ff9800'
        }.get(status, '#1976d2')

        subject = f"{status_icon} FX ML Pipeline - {status}"

        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: {status_color}; }}
                h2 {{ color: #1976d2; }}
                .summary {{ padding: 15px; background-color: #f5f5f5; border-left: 4px solid {status_color}; margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #1976d2; color: white; }}
                .footer {{ margin-top: 30px; padding-top: 20px; border-top: 2px solid #ddd; color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <h1>{status_icon} Pipeline Execution Report</h1>
            <p>Dear {greeting},</p>

            <div class="summary">
                <h2>Summary</h2>
                <p>{summary}</p>
            </div>

            <h2>Execution Details</h2>
            <table>
        """

        for key, value in details.items():
            html += f"""
                <tr>
                    <td><strong>{key}</strong></td>
                    <td>{value}</td>
                </tr>
            """

        html += """
            </table>

            <div class="footer">
                <p>This is an automated report from the FX ML Pipeline.</p>
                <p>Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </body>
        </html>
        """

        return self.send_email(
            to_emails=to_emails,
            subject=subject,
            body_html=html
        )

    def send_test_email(self, to_emails: List[str], greeting: str = "Team") -> bool:
        """
        Send a test email to verify configuration.

        Args:
            to_emails: List of recipient emails
            greeting: Name/title to address recipient

        Returns:
            True if sent successfully
        """
        subject = "🧪 Test Email - FX ML Pipeline Monitoring"

        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #1976d2; }}
                .info {{ padding: 15px; background-color: #e3f2fd; border-left: 4px solid #1976d2; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>🧪 Test Email</h1>
            <p>Dear {greeting},</p>

            <div class="info">
                <p>This is a test email from the FX ML Pipeline Monitoring System.</p>
                <p>If you received this email, the email alerting system is configured correctly.</p>
            </div>

            <p><strong>System Information:</strong></p>
            <ul>
                <li>SMTP Host: {self.smtp_host}</li>
                <li>SMTP Port: {self.smtp_port}</li>
                <li>From Email: {self.from_email}</li>
                <li>Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</li>
            </ul>

            <p>You will receive alerts for:</p>
            <ul>
                <li>Model drift detection</li>
                <li>Pipeline execution failures</li>
                <li>Data quality issues</li>
                <li>Model performance degradation</li>
            </ul>
        </body>
        </html>
        """

        return self.send_email(
            to_emails=to_emails,
            subject=subject,
            body_html=html
        )


if __name__ == "__main__":
    # Test email alerter
    import argparse

    parser = argparse.ArgumentParser(description="Test email alerter")
    parser.add_argument('--to', required=True, help='Recipient email address')
    parser.add_argument('--greeting', default='Team', help='Greeting name')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    alerter = EmailAlerter()
    success = alerter.send_test_email(to_emails=[args.to], greeting=args.greeting)

    if success:
        print(f"✅ Test email sent successfully to {args.to}")
    else:
        print(f"❌ Failed to send test email to {args.to}")
