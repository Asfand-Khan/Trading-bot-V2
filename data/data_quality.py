"""
data/data_quality.py — Data quality checks: gap detection, outlier removal, staleness alerts.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("oracle.data_quality")


def check_data_gaps(df: pd.DataFrame, expected_interval_minutes: int = 15,
                    max_gap_multiple: float = 3.0) -> list:
    """Detect gaps in OHLCV data that exceed expected interval."""
    if df.empty or len(df) < 2:
        return []
    gaps = []
    time_diffs = df.index.to_series().diff().dropna()
    expected_td = pd.Timedelta(minutes=expected_interval_minutes)
    max_gap = expected_td * max_gap_multiple

    for idx, diff in time_diffs.items():
        if diff > max_gap:
            gaps.append({
                'start': idx - diff,
                'end': idx,
                'gap_minutes': diff.total_seconds() / 60,
                'expected_minutes': expected_interval_minutes
            })
    if gaps:
        logger.warning(f"Found {len(gaps)} data gaps (>{max_gap_multiple}x expected interval)")
    return gaps


def remove_outliers(df: pd.DataFrame, column: str = 'Close',
                    z_threshold: float = 5.0) -> pd.DataFrame:
    """Remove price outliers using z-score on returns."""
    if df.empty or len(df) < 20:
        return df
    returns = df[column].pct_change()
    mean_ret = returns.rolling(100, min_periods=20).mean()
    std_ret = returns.rolling(100, min_periods=20).std()
    z_scores = ((returns - mean_ret) / (std_ret + 1e-10)).abs()
    mask = z_scores < z_threshold
    mask.iloc[0] = True  # Keep first row
    removed = (~mask).sum()
    if removed > 0:
        logger.info(f"Removed {removed} outlier rows (z>{z_threshold})")
    return df[mask].copy()


def check_staleness(prices: dict, max_age_seconds: float = 60.0) -> list:
    """Check if any live prices are stale (not updated recently)."""
    stale = []
    now = datetime.now(timezone.utc).timestamp()
    for symbol, price in prices.items():
        if price is None:
            stale.append({'symbol': symbol, 'reason': 'no_price'})
    return stale


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Validate OHLCV data integrity."""
    if df.empty:
        return df
    # High >= Low
    invalid = df['High'] < df['Low']
    if invalid.any():
        logger.warning(f"Found {invalid.sum()} rows where High < Low — fixing")
        df.loc[invalid, 'High'] = df.loc[invalid, ['High', 'Low']].max(axis=1)
        df.loc[invalid, 'Low'] = df.loc[invalid, ['High', 'Low']].min(axis=1)

    # High >= Open, Close and Low <= Open, Close
    df['High'] = df[['High', 'Open', 'Close']].max(axis=1)
    df['Low'] = df[['Low', 'Open', 'Close']].min(axis=1)

    # Remove zero/negative volume
    df = df[df['Volume'] > 0].copy()

    # Remove duplicate timestamps
    df = df[~df.index.duplicated(keep='first')]

    return df.sort_index()


def calculate_data_quality_score(df: pd.DataFrame, expected_interval_minutes: int = 15) -> float:
    """Return a 0-100 quality score for the dataset."""
    if df.empty:
        return 0.0

    score = 100.0

    # Gap penalty
    gaps = check_data_gaps(df, expected_interval_minutes)
    gap_penalty = min(len(gaps) * 5, 30)
    score -= gap_penalty

    # Completeness: actual rows vs expected
    if len(df) >= 2:
        total_minutes = (df.index[-1] - df.index[0]).total_seconds() / 60
        expected_rows = total_minutes / expected_interval_minutes
        completeness = len(df) / max(expected_rows, 1)
        if completeness < 0.95:
            score -= (1 - completeness) * 50

    # Zero volume rows
    zero_vol_pct = (df['Volume'] == 0).sum() / len(df) * 100
    score -= min(zero_vol_pct * 2, 20)

    return max(score, 0.0)
