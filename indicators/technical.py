"""
indicators/technical.py — All technical indicators in one place.
Pure functions operating on pandas Series/DataFrames. No side effects.
"""

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def rsi_wilders(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26,
         signal: int = 9) -> tuple:
    exp_fast = series.ewm(span=fast, adjust=False).mean()
    exp_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = exp_fast - exp_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(series: pd.Series, period: int = 20,
                    std_dev: float = 2.0) -> tuple:
    mid = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = mid + (std * std_dev)
    lower = mid - (std * std_dev)
    return upper, mid, lower


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> tuple:
    """Returns (adx_series, plus_di_series, minus_di_series)."""
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm_clean = plus_dm.copy()
    minus_dm_clean = minus_dm.copy()

    plus_dm_clean[(plus_dm_clean < 0) | (plus_dm_clean < minus_dm_clean)] = 0.0
    minus_dm_clean[(minus_dm_clean < 0) | (minus_dm_clean < plus_dm_clean)] = 0.0

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    tr_smooth = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_smooth = plus_dm_clean.ewm(alpha=1 / period, adjust=False).mean()
    minus_smooth = minus_dm_clean.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * (plus_smooth / (tr_smooth + 1e-10))
    minus_di = 100 * (minus_smooth / (tr_smooth + 1e-10))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
    adx_val = dx.ewm(alpha=1 / period, adjust=False).mean()

    return adx_val, plus_di, minus_di


def vwap(high: pd.Series, low: pd.Series, close: pd.Series,
         volume: pd.Series, period: int = 20) -> pd.Series:
    typical = (high + low + close) / 3
    return (typical * volume).rolling(period).sum() / (volume.rolling(period).sum() + 1e-10)


def volume_sma(volume: pd.Series, period: int = 10) -> pd.Series:
    return volume.rolling(window=period).mean()


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all standard indicators on an OHLCV DataFrame in-place.
    Expects columns: Open, High, Low, Close, Volume.
    Returns the same DataFrame with indicator columns added.
    """
    df = df.copy()
    df['EMA50'] = ema(df['Close'], 50)
    df['EMA200'] = ema(df['Close'], 200)
    df['RSI'] = rsi_wilders(df['Close'])
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = macd(df['Close'])
    df['BB_Upper'], df['BB_Mid'], df['BB_Lower'] = bollinger_bands(
        df['Close'], std_dev=2.25
    )
    df['ATR'] = atr(df['High'], df['Low'], df['Close'])
    df['ADX'], df['Plus_DI'], df['Minus_DI'] = adx(
        df['High'], df['Low'], df['Close']
    )
    df['VWAP'] = vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    df['Volume_SMA'] = volume_sma(df['Volume'])

    # Volatility measures
    df['Returns'] = df['Close'].pct_change()
    df['Volatility_20'] = df['Returns'].rolling(20).std() * np.sqrt(96)  # annualized for 15m
    df['ATR_Pct'] = df['ATR'] / (df['Close'] + 1e-10) * 100

    return df


def detect_smc_zones(df: pd.DataFrame, lookback: int = 20) -> dict:
    """Smart Money Concepts: liquidity zones + wick validation."""
    recent = df.tail(lookback)
    if len(recent) < 5:
        return {'smc_buy': False, 'smc_sell': False, 'discount_top': 0,
                'premium_bottom': 0}

    recent_low = recent['Low'].min()
    recent_high = recent['High'].max()
    range_size = recent_high - recent_low

    discount_zone_top = recent_low + (range_size * 0.5)
    premium_zone_bottom = recent_high - (range_size * 0.5)

    # Last completed candle wick analysis
    last = recent.iloc[-2] if len(recent) > 1 else recent.iloc[-1]
    body = abs(last['Close'] - last['Open'])
    lower_wick = min(last['Open'], last['Close']) - last['Low']
    upper_wick = last['High'] - max(last['Open'], last['Close'])
    current_price = float(df['Close'].iloc[-1])
    min_body = max(body, current_price * 0.0005)

    has_bullish_wick = lower_wick > (min_body * 1.5)
    has_bearish_wick = upper_wick > (min_body * 1.5)

    smc_buy = (current_price <= discount_zone_top) or has_bullish_wick
    smc_sell = (current_price >= premium_zone_bottom) or has_bearish_wick

    return {
        'smc_buy': smc_buy,
        'smc_sell': smc_sell,
        'discount_top': discount_zone_top,
        'premium_bottom': premium_zone_bottom,
        'bullish_wick': has_bullish_wick,
        'bearish_wick': has_bearish_wick,
    }
