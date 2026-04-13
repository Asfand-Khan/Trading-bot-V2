"""
utils/email_alerts.py — Email notifications for signals, daily P&L, alerts.
"""

import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone

from config import EMAIL_FROM, EMAIL_TO, EMAIL_APP_PASSWORD, SMTP_SERVER, SMTP_PORT

logger = logging.getLogger("oracle.email")


def _send_email(subject: str, html_body: str):
    if not all([EMAIL_FROM, EMAIL_TO, EMAIL_APP_PASSWORD]):
        logger.warning("Email not configured — skipping send")
        return False
    msg = MIMEMultipart('alternative')
    msg['From'] = EMAIL_FROM
    msg['To'] = EMAIL_TO
    msg['Subject'] = subject
    msg.attach(MIMEText(html_body, 'html'))
    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(EMAIL_FROM, EMAIL_APP_PASSWORD)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        logger.info(f"Email sent: {subject}")
        return True
    except Exception as e:
        logger.error(f"Email failed: {e}")
        return False


def send_signal_email(signal_data: dict):
    try:
        import zoneinfo
        pkt_time = datetime.now(zoneinfo.ZoneInfo("Asia/Karachi")).strftime("%I:%M:%S %p")
    except Exception:
        pkt_time = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

    ml_info = ""
    if signal_data.get('ml_probability'):
        ml_info = f"<p><strong>ML Confidence:</strong> {signal_data['ml_probability']:.1%}</p>"

    regime_info = ""
    if signal_data.get('regime'):
        regime_info = f"<p><strong>Market Regime:</strong> {signal_data['regime']}</p>"

    html = f"""
    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <h2 style="color: {'#22c55e' if signal_data['signal']=='BUY' else '#ef4444'};">
        Wall Street Oracle v5.0: {signal_data['signal']} {signal_data['asset']}
    </h2>
    <p><strong>Time:</strong> {pkt_time}</p>
    <p><strong>Price:</strong> ${signal_data['price']:,.4f}</p>
    <p><strong>Signal:</strong> <b>{signal_data['signal']}</b></p>
    <p><strong>Confidence:</strong> {signal_data['confidence']}%</p>
    {ml_info}
    {regime_info}
    <h3>Why:</h3>
    <p>{signal_data.get('why', '')}</p>
    <h3>Action Plan:</h3>
    <ul>
        <li><b>Entry:</b> {signal_data.get('entry', 'Market')}</li>
        <li><b>Stop Loss:</b> {signal_data.get('stop_loss', 'N/A')}</li>
        <li><b>TP1 (50%):</b> {signal_data.get('tp1', 'N/A')}</li>
        <li><b>TP2 (50%):</b> {signal_data.get('tp2', 'N/A')}</li>
        <li><b>Position Size:</b> {signal_data.get('position_size', 'N/A')}</li>
    </ul>
    <p style="color: #666; font-size: 0.85em;">
        Not financial advice. Autonomous execution on demo-fapi.
    </p>
    </div>
    """
    _send_email(f"Oracle v5.0: {signal_data['signal']} {signal_data['asset']}", html)


def send_daily_report(report: dict):
    html = f"""
    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <h2>Daily P&L Report — {report.get('date', 'Today')}</h2>
    <table style="width: 100%; border-collapse: collapse;">
        <tr><td><b>Equity:</b></td><td>${report.get('equity', 0):,.2f}</td></tr>
        <tr><td><b>Daily P&L:</b></td><td style="color: {'green' if report.get('daily_pnl',0)>=0 else 'red'}">${report.get('daily_pnl', 0):,.2f} ({report.get('daily_pnl_pct', 0):.2f}%)</td></tr>
        <tr><td><b>Open Positions:</b></td><td>{report.get('open_positions', 0)}</td></tr>
        <tr><td><b>Trades Today:</b></td><td>{report.get('trades_today', 0)}</td></tr>
        <tr><td><b>Rolling Sharpe (30d):</b></td><td>{report.get('sharpe_30d', 0):.2f}</td></tr>
        <tr><td><b>Max Drawdown:</b></td><td>{report.get('max_drawdown', 0):.2f}%</td></tr>
        <tr><td><b>Win Rate:</b></td><td>{report.get('win_rate', 0):.1f}%</td></tr>
    </table>
    </div>
    """
    _send_email(f"Oracle Daily Report — {report.get('date', 'Today')}", html)


def send_alert(subject: str, message: str):
    html = f"""
    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <h2 style="color: #f59e0b;">ALERT: {subject}</h2>
    <p>{message}</p>
    <p style="color: #666; font-size: 0.85em;">Wall Street Oracle v5.0 Autonomous System</p>
    </div>
    """
    _send_email(f"ORACLE ALERT: {subject}", html)
