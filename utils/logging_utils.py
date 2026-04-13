"""
utils/logging_utils.py — Structured logging with full audit trail
Every signal (taken or skipped) logs all feature values and reason.
"""

import logging
import csv
import os
from datetime import datetime, timezone

from config import LOG_FILE, TRADE_JOURNAL_FILE


def setup_logging(level=logging.INFO):
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=level, format=fmt)
    # Suppress noisy libs
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('websocket').setLevel(logging.WARNING)


logger = logging.getLogger("oracle")


def _synced_time_str() -> str:
    """Get Binance-synced timestamp string. Falls back to UTC if sync not yet initialized."""
    try:
        from utils.helpers import get_synced_now
        return get_synced_now().strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def log_signal_csv(signal_data: dict):
    fieldnames = [
        'timestamp', 'asset', 'signal', 'price', 'stop_loss', 'tp1', 'tp2',
        'confidence', 'ml_probability', 'regime', 'why', 'status'
    ]
    row = {k: signal_data.get(k, '') for k in fieldnames}
    row['timestamp'] = _synced_time_str()
    row['status'] = signal_data.get('status', 'EXECUTED')
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def log_skip_csv(asset: str, reason: str, features: dict = None):
    fieldnames = [
        'timestamp', 'asset', 'signal', 'price', 'stop_loss', 'tp1', 'tp2',
        'confidence', 'ml_probability', 'regime', 'why', 'status'
    ]
    row = {k: '' for k in fieldnames}
    row['timestamp'] = _synced_time_str()
    row['asset'] = asset
    row['status'] = 'SKIPPED'
    row['why'] = reason
    if features:
        row['confidence'] = features.get('score', '')
        row['ml_probability'] = features.get('ml_prob', '')
        row['regime'] = features.get('regime', '')
        row['price'] = features.get('price', '')
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def log_trade_journal(trade: dict):
    """Tax-ready trade journal with realized PnL."""
    fieldnames = [
        'trade_id', 'timestamp_open', 'timestamp_close', 'asset', 'side',
        'entry_price', 'exit_price', 'quantity', 'leverage', 'pnl_usdt',
        'pnl_pct', 'fees_usdt', 'funding_paid', 'net_pnl', 'duration_minutes',
        'exit_reason'
    ]
    row = {k: trade.get(k, '') for k in fieldnames}
    file_exists = os.path.exists(TRADE_JOURNAL_FILE)
    with open(TRADE_JOURNAL_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def log_decision_audit(asset: str, action: str, reason: str, features: dict):
    """Full decision audit trail — every signal taken or skipped with all feature values."""
    logger.info(
        f"[AUDIT] {asset} | Action={action} | Reason={reason} | "
        f"Features={features}"
    )
