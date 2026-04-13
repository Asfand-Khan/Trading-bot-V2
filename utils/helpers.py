"""
utils/helpers.py — Shared utility functions.
"""

import time
import hmac
import hashlib
import requests
import logging
import threading
from datetime import datetime, timezone

from config import BINANCE_TEST_API, BINANCE_TEST_SECRET, BASE_URL

logger = logging.getLogger("oracle.helpers")

# Global time offset synced with Binance
_time_offset = 0
_time_offset_lock = threading.Lock()


def sync_binance_time():
    """Sync local clock with Binance server time. Call periodically."""
    global _time_offset
    try:
        resp = requests.get(f"{BASE_URL}/fapi/v1/time", timeout=5)
        server_time = resp.json()['serverTime']
        local_time = int(time.time() * 1000)
        with _time_offset_lock:
            _time_offset = server_time - local_time
        logger.info(f"Time synced: offset={_time_offset}ms")
    except Exception as e:
        logger.error(f"Time sync failed: {e}")


def get_synced_timestamp() -> int:
    """Returns Binance-synced epoch milliseconds."""
    with _time_offset_lock:
        return int(time.time() * 1000) + _time_offset


def get_synced_now() -> datetime:
    """Returns a timezone-aware datetime synced to Binance server time.
    Use this everywhere instead of datetime.now() or datetime.utcnow().
    """
    ms = get_synced_timestamp()
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def signed_request(method: str, path: str, params: dict = None) -> dict:
    """Make a signed request to Binance Futures API with synced timestamp."""
    if not BINANCE_TEST_API or not BINANCE_TEST_SECRET:
        return {'code': -1, 'msg': 'API keys not configured'}
    if params is None:
        params = {}
    params['timestamp'] = get_synced_timestamp()
    params.setdefault('recvWindow', 10000)  # 10s window — prevents recvWindow errors
    query = '&'.join([f"{k}={v}" for k, v in params.items()])
    sig = hmac.new(
        BINANCE_TEST_SECRET.encode(), query.encode(), hashlib.sha256
    ).hexdigest()
    url = f"{BASE_URL}{path}?{query}&signature={sig}"
    headers = {"X-MBX-APIKEY": BINANCE_TEST_API}
    try:
        if method == 'GET':
            resp = requests.get(url, headers=headers, timeout=10)
        elif method == 'POST':
            resp = requests.post(url, headers=headers, timeout=10)
        elif method == 'DELETE':
            resp = requests.delete(url, headers=headers, timeout=10)
        else:
            return {'code': -1, 'msg': f'Unsupported method: {method}'}
        return resp.json()
    except Exception as e:
        logger.error(f"Request failed: {method} {path}: {e}")
        return {'code': -1, 'msg': str(e)}


def safe_float(val) -> float:
    if hasattr(val, 'iloc'):
        return float(val.iloc[0])
    elif hasattr(val, 'item'):
        return float(val.item())
    return float(val)


def is_api_error(resp: dict) -> bool:
    return isinstance(resp, dict) and 'code' in resp and resp.get('code', 0) < 0
