"""
strategy/regime.py — Market regime detection using Gaussian Mixture Model.
Classifies market into 4 regimes: Trending, Ranging, High-Volatility, Low-Liquidity.
No paid dependencies — uses sklearn's GaussianMixture (free, open-source).
"""

import logging
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from config import (REGIME_TRENDING, REGIME_RANGING, REGIME_HIGH_VOLATILITY,
                    REGIME_LOW_LIQUIDITY, REGIME_NAMES)

logger = logging.getLogger("oracle.regime")


class RegimeDetector:
    def __init__(self, n_regimes: int = 4, lookback: int = 500):
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.model = None
        self.scaler = StandardScaler()
        self._regime_map = {}  # Maps GMM label -> our regime constant

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract regime-relevant features from OHLCV data."""
        if len(df) < 50:
            return pd.DataFrame()

        features = pd.DataFrame(index=df.index)

        # 1. Volatility: rolling std of returns
        ret = df['Close'].pct_change()
        features['volatility_20'] = ret.rolling(20).std()
        features['volatility_5'] = ret.rolling(5).std()

        # 2. Trend strength: slope of EMA50 (normalized)
        ema50 = df['Close'].ewm(span=50, adjust=False).mean()
        features['trend_slope'] = ema50.pct_change(10)

        # 3. ADX if available
        if 'ADX' in df.columns:
            features['adx'] = df['ADX']
        else:
            from indicators.technical import adx as calc_adx
            adx_val, _, _ = calc_adx(df['High'], df['Low'], df['Close'])
            features['adx'] = adx_val

        # 4. Volume relative to average
        vol_sma = df['Volume'].rolling(20).mean()
        features['volume_ratio'] = df['Volume'] / (vol_sma + 1e-10)

        # 5. Range (High-Low) / Close as % — spread indicator
        features['range_pct'] = (df['High'] - df['Low']) / (df['Close'] + 1e-10)

        # 6. Absolute return over 20 bars (directional move magnitude)
        features['abs_return_20'] = ret.rolling(20).sum().abs()

        features = features.dropna()
        return features

    def fit(self, df: pd.DataFrame):
        """Fit the GMM on historical data."""
        features_df = self._extract_features(df)
        if len(features_df) < 100:
            logger.warning("Insufficient data for regime detection fit")
            return False

        X = features_df.values
        X_scaled = self.scaler.fit_transform(X)

        self.model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            n_init=5,
            max_iter=200,
            random_state=42
        )
        self.model.fit(X_scaled)

        # Map GMM labels to our regime constants based on cluster centers
        labels = self.model.predict(X_scaled)
        self._map_regimes(features_df, labels)

        logger.info(f"Regime detector fitted on {len(X)} samples, "
                     f"{self.n_regimes} regimes identified")
        return True

    def _map_regimes(self, features_df: pd.DataFrame, labels: np.ndarray):
        """Map GMM cluster labels to meaningful regime names based on feature statistics."""
        features_df = features_df.copy()
        features_df['label'] = labels
        cluster_stats = features_df.groupby('label').mean()

        if len(cluster_stats) < self.n_regimes:
            # Fallback: simple 1-to-1 mapping
            for i in range(self.n_regimes):
                self._regime_map[i] = i
            return

        # Sort clusters by volatility to assign regimes
        vol_col = 'volatility_20' if 'volatility_20' in cluster_stats.columns else cluster_stats.columns[0]
        adx_col = 'adx' if 'adx' in cluster_stats.columns else None

        sorted_by_vol = cluster_stats[vol_col].sort_values()
        labels_by_vol = sorted_by_vol.index.tolist()

        # Lowest volatility + lowest volume ratio = Low Liquidity
        # Lowest volatility + decent volume = Ranging
        # High volatility + high ADX = Trending
        # Highest volatility = High Volatility / News

        self._regime_map[labels_by_vol[0]] = REGIME_LOW_LIQUIDITY
        self._regime_map[labels_by_vol[1]] = REGIME_RANGING

        if adx_col and len(labels_by_vol) >= 4:
            # Between the two high-vol clusters, higher ADX = trending
            adx_2 = cluster_stats.loc[labels_by_vol[2], adx_col]
            adx_3 = cluster_stats.loc[labels_by_vol[3], adx_col]
            if adx_2 > adx_3:
                self._regime_map[labels_by_vol[2]] = REGIME_TRENDING
                self._regime_map[labels_by_vol[3]] = REGIME_HIGH_VOLATILITY
            else:
                self._regime_map[labels_by_vol[2]] = REGIME_HIGH_VOLATILITY
                self._regime_map[labels_by_vol[3]] = REGIME_TRENDING
        elif len(labels_by_vol) >= 4:
            self._regime_map[labels_by_vol[2]] = REGIME_TRENDING
            self._regime_map[labels_by_vol[3]] = REGIME_HIGH_VOLATILITY
        else:
            # Handle fewer clusters gracefully
            for i, lbl in enumerate(labels_by_vol):
                if lbl not in self._regime_map:
                    self._regime_map[lbl] = i % 4

    def predict(self, df: pd.DataFrame) -> int:
        """Predict current regime from recent data."""
        if self.model is None:
            return REGIME_RANGING  # Default safe fallback

        features_df = self._extract_features(df)
        if features_df.empty:
            return REGIME_RANGING

        # Use last row
        X = features_df.values[-1:].reshape(1, -1)
        try:
            X_scaled = self.scaler.transform(X)
            label = self.model.predict(X_scaled)[0]
            return self._regime_map.get(label, REGIME_RANGING)
        except Exception as e:
            logger.warning(f"Regime prediction failed: {e}")
            return REGIME_RANGING

    def get_regime_name(self, regime_id: int) -> str:
        return REGIME_NAMES.get(regime_id, "UNKNOWN")

    def get_regime_probabilities(self, df: pd.DataFrame) -> dict:
        """Get probability distribution across all regimes."""
        if self.model is None:
            return {name: 0.25 for name in REGIME_NAMES.values()}

        features_df = self._extract_features(df)
        if features_df.empty:
            return {name: 0.25 for name in REGIME_NAMES.values()}

        X = features_df.values[-1:].reshape(1, -1)
        try:
            X_scaled = self.scaler.transform(X)
            probs = self.model.predict_proba(X_scaled)[0]
            result = {}
            for gmm_label, prob in enumerate(probs):
                regime_id = self._regime_map.get(gmm_label, gmm_label)
                regime_name = REGIME_NAMES.get(regime_id, f"REGIME_{regime_id}")
                result[regime_name] = float(prob)
            return result
        except Exception:
            return {name: 0.25 for name in REGIME_NAMES.values()}
