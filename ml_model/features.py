"""
ml_model/features.py — Feature engineering for ML model.
Extracts features from OHLCV + indicators for LightGBM classifier.
"""

import logging
import numpy as np
import pandas as pd

from indicators.technical import (
    compute_all_indicators, ema, rsi_wilders, macd, atr as calc_atr,
    adx as calc_adx, vwap as calc_vwap
)

logger = logging.getLogger("oracle.features")


def engineer_features(df: pd.DataFrame, funding_rates: pd.Series = None) -> pd.DataFrame:
    """
    Build feature matrix from OHLCV data.
    Returns DataFrame with feature columns + 'target' column (1=up, 0=down).
    """
    if len(df) < 250:
        logger.warning(f"Insufficient data for features: {len(df)} rows")
        return pd.DataFrame()

    feat = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    feat = compute_all_indicators(feat)

    # ---------- Price action features ----------
    feat['return_1'] = feat['Close'].pct_change(1)
    feat['return_5'] = feat['Close'].pct_change(5)
    feat['return_10'] = feat['Close'].pct_change(10)
    feat['return_20'] = feat['Close'].pct_change(20)

    # Momentum features
    feat['rsi_slope'] = feat['RSI'].diff(3)
    feat['macd_hist_slope'] = feat['MACD_Hist'].diff(3)
    feat['ema_spread'] = (feat['EMA50'] - feat['EMA200']) / (feat['Close'] + 1e-10)

    # Volatility features
    feat['vol_ratio'] = feat['Volume'] / (feat['Volume_SMA'] + 1e-10)
    feat['atr_pct'] = feat['ATR'] / (feat['Close'] + 1e-10) * 100
    feat['bb_width'] = (feat['BB_Upper'] - feat['BB_Lower']) / (feat['BB_Mid'] + 1e-10)
    feat['bb_position'] = (feat['Close'] - feat['BB_Lower']) / (feat['BB_Upper'] - feat['BB_Lower'] + 1e-10)

    # Candle structure
    feat['body_pct'] = (feat['Close'] - feat['Open']).abs() / (feat['High'] - feat['Low'] + 1e-10)
    feat['upper_wick_pct'] = (feat['High'] - feat[['Close', 'Open']].max(axis=1)) / (feat['High'] - feat['Low'] + 1e-10)
    feat['lower_wick_pct'] = (feat[['Close', 'Open']].min(axis=1) - feat['Low']) / (feat['High'] - feat['Low'] + 1e-10)

    # VWAP distance
    feat['vwap_distance'] = (feat['Close'] - feat['VWAP']) / (feat['ATR'] + 1e-10)

    # Trend features
    feat['above_ema50'] = (feat['Close'] > feat['EMA50']).astype(int)
    feat['above_ema200'] = (feat['Close'] > feat['EMA200']).astype(int)
    feat['ema50_above_200'] = (feat['EMA50'] > feat['EMA200']).astype(int)
    feat['above_vwap'] = (feat['Close'] > feat['VWAP']).astype(int)

    # MACD crossover features
    feat['macd_cross_up'] = ((feat['MACD'] > feat['MACD_Signal']) &
                              (feat['MACD'].shift(1) <= feat['MACD_Signal'].shift(1))).astype(int)
    feat['macd_cross_down'] = ((feat['MACD'] < feat['MACD_Signal']) &
                                (feat['MACD'].shift(1) >= feat['MACD_Signal'].shift(1))).astype(int)

    # ADX features
    feat['adx_trending'] = (feat['ADX'] > 25).astype(int)
    feat['adx_strong'] = (feat['ADX'] > 40).astype(int)
    feat['di_spread'] = feat['Plus_DI'] - feat['Minus_DI']

    # Volume features
    feat['vol_surge'] = (feat['vol_ratio'] > 1.5).astype(int)
    feat['vol_dry'] = (feat['vol_ratio'] < 0.5).astype(int)

    # Order flow proxy: taker buy ratio (if available in data)
    if 'taker_buy_base' in df.columns:
        feat['taker_buy_ratio'] = df['taker_buy_base'].astype(float) / (df['Volume'].astype(float) + 1e-10)
    else:
        feat['taker_buy_ratio'] = 0.5

    # Funding rate features (if provided)
    if funding_rates is not None and len(funding_rates) > 0:
        # Resample funding to match OHLCV frequency
        feat['funding_rate'] = funding_rates.reindex(feat.index, method='ffill').fillna(0)
        feat['funding_momentum'] = feat['funding_rate'].diff(3)
    else:
        feat['funding_rate'] = 0.0
        feat['funding_momentum'] = 0.0

    # ---------- Cross-asset correlation proxy ----------
    # Rolling correlation of returns with itself lagged (autocorrelation)
    feat['autocorr_5'] = feat['return_1'].rolling(20).apply(
        lambda x: x.autocorr(lag=5) if len(x) > 5 else 0, raw=False
    )

    # ---------- Target: 15-min forward return direction ----------
    # 1 = price goes up in next candle, 0 = down/flat
    feat['future_return'] = feat['Close'].shift(-1) / feat['Close'] - 1
    feat['target'] = (feat['future_return'] > 0).astype(int)

    # Clean
    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.dropna()

    return feat


FEATURE_COLUMNS = [
    'return_1', 'return_5', 'return_10', 'return_20',
    'RSI', 'rsi_slope', 'MACD', 'MACD_Signal', 'MACD_Hist', 'macd_hist_slope',
    'ema_spread', 'ADX', 'Plus_DI', 'Minus_DI', 'di_spread',
    'adx_trending', 'adx_strong',
    'ATR', 'atr_pct', 'Volatility_20',
    'bb_width', 'bb_position', 'vwap_distance',
    'vol_ratio', 'vol_surge', 'vol_dry',
    'body_pct', 'upper_wick_pct', 'lower_wick_pct',
    'above_ema50', 'above_ema200', 'ema50_above_200', 'above_vwap',
    'macd_cross_up', 'macd_cross_down',
    'taker_buy_ratio', 'funding_rate', 'funding_momentum', 'autocorr_5',
]


def get_feature_columns() -> list:
    return [c for c in FEATURE_COLUMNS]


def extract_live_features(signal_data: dict) -> dict:
    """Extract features from a live signal_data dict for ML prediction."""
    f = signal_data.get('features', {})
    df = signal_data.get('df_15m')
    if df is None or df.empty:
        return {}

    feat_df = engineer_features(df)
    if feat_df.empty:
        return {}

    last_row = feat_df.iloc[-1]
    result = {}
    for col in get_feature_columns():
        if col in last_row.index:
            result[col] = float(last_row[col])
        else:
            result[col] = 0.0

    return result
