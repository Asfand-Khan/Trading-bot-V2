"""
data/binance_rest.py — Binance REST API for historical data.
Replaces all yfinance usage. Uses free Binance klines endpoint + data.binance.vision bulk downloads.
"""

import os
import io
import time
import zipfile
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

from config import BASE_URL, BINANCE_DATA_URL, HISTORICAL_DATA_DIR

logger = logging.getLogger("oracle.rest")

# Rate limiter: Binance allows 1200 req/min for klines
_last_request_time = 0
_request_lock = __import__('threading').Lock()
MIN_REQUEST_INTERVAL = 0.05  # 50ms between requests = 1200/min


def _rate_limit():
    global _last_request_time
    with _request_lock:
        elapsed = time.time() - _last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        _last_request_time = time.time()


def fetch_klines(symbol: str, interval: str = '15m', limit: int = 1500,
                 start_time: int = None, end_time: int = None) -> pd.DataFrame:
    """Fetch klines from Binance Futures REST API. Max 1500 per call."""
    _rate_limit()
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    if start_time:
        params['startTime'] = start_time
    if end_time:
        params['endTime'] = end_time
    try:
        resp = requests.get(f"{BASE_URL}/fapi/v1/klines", params=params, timeout=15)
        data = resp.json()
        if isinstance(data, dict) and 'code' in data:
            logger.error(f"Klines error for {symbol}: {data}")
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=[
            'open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'quote_volume',
                     'taker_buy_base', 'taker_buy_quote']:
            df[col] = df[col].astype(float)
        df['Datetime'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('Datetime', inplace=True)
        df['trades'] = df['trades'].astype(int)
        return df
    except Exception as e:
        logger.error(f"fetch_klines failed: {symbol} {interval}: {e}")
        return pd.DataFrame()


def fetch_klines_range(symbol: str, interval: str, start_date: str,
                       end_date: str = None) -> pd.DataFrame:
    """Fetch historical klines in date range by paginating the API.
    start_date/end_date format: 'YYYY-MM-DD'
    """
    start_ms = int(pd.Timestamp(start_date).timestamp() * 1000)
    if end_date:
        end_ms = int(pd.Timestamp(end_date).timestamp() * 1000)
    else:
        try:
            from utils.helpers import get_synced_timestamp
            end_ms = get_synced_timestamp()
        except Exception:
            end_ms = int(time.time() * 1000)

    all_data = []
    current_start = start_ms

    # Interval to milliseconds mapping for pagination
    interval_ms = {
        '1m': 60000, '3m': 180000, '5m': 300000, '15m': 900000,
        '30m': 1800000, '1h': 3600000, '2h': 7200000, '4h': 14400000,
        '1d': 86400000
    }
    step = interval_ms.get(interval, 900000) * 1500  # 1500 candles per request

    while current_start < end_ms:
        df = fetch_klines(symbol, interval, limit=1500, start_time=current_start, end_time=end_ms)
        if df.empty:
            break
        all_data.append(df)
        last_time = int(df['close_time'].iloc[-1])
        if last_time >= end_ms or len(df) < 1500:
            break
        current_start = last_time + 1

    if not all_data:
        return pd.DataFrame()
    result = pd.concat(all_data)
    result = result[~result.index.duplicated(keep='first')]
    return result.sort_index()


def download_bulk_klines(symbol: str, interval: str = '1m',
                         year: int = 2025, month: int = None) -> pd.DataFrame:
    """Download bulk historical data from data.binance.vision (free).
    Monthly files for specific month, or daily files.
    """
    os.makedirs(HISTORICAL_DATA_DIR, exist_ok=True)

    if month:
        filename = f"{symbol}-{interval}-{year}-{month:02d}.zip"
        url = f"{BINANCE_DATA_URL}/data/futures/um/monthly/klines/{symbol}/{interval}/{filename}"
    else:
        filename = f"{symbol}-{interval}-{year}.zip"
        url = f"{BINANCE_DATA_URL}/data/futures/um/monthly/klines/{symbol}/{interval}/{filename}"

    cache_path = os.path.join(HISTORICAL_DATA_DIR, filename.replace('.zip', '.parquet'))
    if os.path.exists(cache_path):
        logger.info(f"Loading cached: {cache_path}")
        return pd.read_parquet(cache_path)

    logger.info(f"Downloading: {url}")
    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code != 200:
            logger.warning(f"Bulk download failed ({resp.status_code}): {url}")
            return pd.DataFrame()

        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                df = pd.read_csv(f, header=None, names=[
                    'open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])

        for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'quote_volume',
                     'taker_buy_base', 'taker_buy_quote']:
            df[col] = df[col].astype(float)
        df['Datetime'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('Datetime', inplace=True)

        # Cache as parquet
        df.to_parquet(cache_path)
        logger.info(f"Cached {len(df)} rows to {cache_path}")
        return df
    except Exception as e:
        logger.error(f"Bulk download error: {e}")
        return pd.DataFrame()


def fetch_historical_data(symbol: str, interval: str = '1m',
                          days: int = 365) -> pd.DataFrame:
    """Smart historical data fetcher: tries bulk download first, falls back to API pagination."""
    # For long histories (>60 days), try bulk download month by month
    if days > 60 and interval in ('1m', '5m', '15m', '1h'):
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        all_dfs = []

        current = start_date.replace(day=1)
        while current < end_date:
            df = download_bulk_klines(symbol, interval, current.year, current.month)
            if not df.empty:
                all_dfs.append(df)
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        if all_dfs:
            result = pd.concat(all_dfs)
            result = result[~result.index.duplicated(keep='first')]
            # Trim to requested range
            cutoff = end_date - timedelta(days=days)
            result = result[result.index >= pd.Timestamp(cutoff)]
            if not result.empty:
                logger.info(f"Loaded {len(result)} rows for {symbol} {interval} ({days}d)")
                return result.sort_index()

    # Fallback: paginate REST API
    start_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime('%Y-%m-%d')
    return fetch_klines_range(symbol, interval, start_date)


def fetch_funding_rate(symbol: str) -> float:
    """Get current funding rate from Binance."""
    try:
        _rate_limit()
        resp = requests.get(
            f"https://fapi.binance.com/fapi/v1/premiumIndex",
            params={'symbol': symbol}, timeout=5
        )
        data = resp.json()
        return float(data.get('lastFundingRate', 0.0))
    except Exception:
        return 0.0


def fetch_funding_rate_history(symbol: str, days: int = 90) -> pd.DataFrame:
    """Fetch historical funding rates."""
    start_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    all_data = []
    current_start = start_ms

    while True:
        _rate_limit()
        try:
            resp = requests.get(
                f"https://fapi.binance.com/fapi/v1/fundingRate",
                params={'symbol': symbol, 'startTime': current_start, 'limit': 1000},
                timeout=10
            )
            data = resp.json()
            if not data:
                break
            all_data.extend(data)
            if len(data) < 1000:
                break
            current_start = data[-1]['fundingTime'] + 1
        except Exception as e:
            logger.error(f"Funding rate history error: {e}")
            break

    if not all_data:
        return pd.DataFrame()
    df = pd.DataFrame(all_data)
    df['fundingRate'] = df['fundingRate'].astype(float)
    df['Datetime'] = pd.to_datetime(df['fundingTime'], unit='ms')
    df.set_index('Datetime', inplace=True)
    return df


def fetch_exchange_info() -> dict:
    """Get exchange info (symbol precision, filters)."""
    try:
        resp = requests.get(f"{BASE_URL}/fapi/v1/exchangeInfo", timeout=10)
        return resp.json()
    except Exception as e:
        logger.error(f"Exchange info fetch failed: {e}")
        return {}


def get_symbol_precision(symbol: str, exchange_info: dict = None) -> dict:
    """Get quantity and price precision for a symbol."""
    if not exchange_info:
        exchange_info = fetch_exchange_info()
    symbols = exchange_info.get('symbols', [])
    sym_info = next((s for s in symbols if s['symbol'] == symbol), None)
    if sym_info:
        return {
            'quantity_precision': sym_info.get('quantityPrecision', 4),
            'price_precision': sym_info.get('pricePrecision', 2),
            'min_qty': None,
            'min_notional': None
        }
    return {'quantity_precision': 4, 'price_precision': 2, 'min_qty': None, 'min_notional': None}
