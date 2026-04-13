"""
backtester/data_loader.py — Historical data loader for backtesting.
Uses free Binance data: bulk downloads from data.binance.vision + REST API fallback.
"""

import logging
import os
import pandas as pd
from datetime import datetime, timedelta

from data.binance_rest import (
    fetch_historical_data, download_bulk_klines,
    fetch_funding_rate_history
)
from data.data_quality import validate_ohlcv, remove_outliers, calculate_data_quality_score
from config import HISTORICAL_DATA_DIR

logger = logging.getLogger("oracle.bt_data")


def load_backtest_data(symbol: str, interval: str = '15m',
                       days: int = 365) -> pd.DataFrame:
    """
    Load historical data for backtesting.
    Tries bulk download first, then API pagination. Validates quality.
    """
    logger.info(f"Loading {days}d of {interval} data for {symbol}...")

    df = fetch_historical_data(symbol, interval, days)

    if df.empty:
        logger.error(f"No data loaded for {symbol}")
        return pd.DataFrame()

    # Validate and clean
    df = validate_ohlcv(df)
    df = remove_outliers(df)

    quality = calculate_data_quality_score(df, _interval_to_minutes(interval))
    logger.info(f"Loaded {len(df)} bars for {symbol} ({interval}) | "
                f"Quality: {quality:.1f}/100 | "
                f"Range: {df.index[0]} to {df.index[-1]}")

    if quality < 50:
        logger.warning(f"Low data quality ({quality:.1f}) for {symbol} — results may be unreliable")

    return df


def load_multi_symbol_data(symbols: list, interval: str = '15m',
                           days: int = 365) -> dict:
    """Load data for multiple symbols. Returns {symbol: DataFrame}."""
    data = {}
    for symbol in symbols:
        df = load_backtest_data(symbol, interval, days)
        if not df.empty and len(df) > 200:
            data[symbol] = df
        else:
            logger.warning(f"Skipping {symbol}: insufficient data")
    return data


def load_funding_data(symbol: str, days: int = 365) -> pd.Series:
    """Load funding rate history for backtest cost simulation."""
    df = fetch_funding_rate_history(symbol, days)
    if df.empty:
        return pd.Series(dtype=float)
    return df['fundingRate']


def _interval_to_minutes(interval: str) -> int:
    mapping = {
        '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '2h': 120, '4h': 240, '1d': 1440
    }
    return mapping.get(interval, 15)
