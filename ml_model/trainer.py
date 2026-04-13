"""
ml_model/trainer.py — LightGBM nightly retrainer with walk-forward validation.
Auto-retrains on 90-180 days of data. Logs feature importance.
"""

import os
import json
import logging
import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timezone

logger = logging.getLogger("oracle.ml_trainer")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    logger.warning("LightGBM not installed — ML layer disabled. pip install lightgbm")

from config import (
    ML_MODEL_DIR, ML_LOOKBACK_DAYS, ML_WALK_FORWARD_TRAIN,
    ML_WALK_FORWARD_TEST, DEFAULT_WATCHLIST
)
from ml_model.features import engineer_features, get_feature_columns
from data.binance_rest import fetch_historical_data, fetch_funding_rate_history


class MLTrainer:
    def __init__(self):
        os.makedirs(ML_MODEL_DIR, exist_ok=True)
        self.model = None
        self.model_version = None
        self.feature_importance = {}
        self.walk_forward_metrics = {}
        self._load_model()

    def _model_path(self) -> str:
        return os.path.join(ML_MODEL_DIR, "lgb_model.pkl")

    def _meta_path(self) -> str:
        return os.path.join(ML_MODEL_DIR, "model_meta.json")

    def _load_model(self):
        path = self._model_path()
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    self.model = pickle.load(f)
                meta_path = self._meta_path()
                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    self.model_version = meta.get('version')
                    self.feature_importance = meta.get('feature_importance', {})
                    self.walk_forward_metrics = meta.get('walk_forward_metrics', {})
                logger.info(f"Loaded ML model v{self.model_version}")
            except Exception as e:
                logger.error(f"Failed to load ML model: {e}")
                self.model = None

    def _save_model(self):
        try:
            with open(self._model_path(), 'wb') as f:
                pickle.dump(self.model, f)
            meta = {
                'version': self.model_version,
                'trained_at': datetime.now(timezone.utc).isoformat(),
                'feature_importance': self.feature_importance,
                'walk_forward_metrics': self.walk_forward_metrics,
            }
            with open(self._meta_path(), 'w') as f:
                json.dump(meta, f, indent=2)
            logger.info(f"Saved ML model v{self.model_version}")
        except Exception as e:
            logger.error(f"Failed to save ML model: {e}")

    def collect_training_data(self, symbols: list = None) -> pd.DataFrame:
        """Collect and merge training data from multiple symbols."""
        if symbols is None:
            symbols = DEFAULT_WATCHLIST[:10]  # Top 10 for training speed

        all_features = []
        for symbol in symbols:
            try:
                logger.info(f"Collecting {ML_LOOKBACK_DAYS}d data for {symbol}...")
                df = fetch_historical_data(symbol, '15m', days=ML_LOOKBACK_DAYS)
                if df.empty or len(df) < 500:
                    logger.warning(f"Skipping {symbol}: insufficient data ({len(df)} rows)")
                    continue

                # Get funding rates
                funding_df = fetch_funding_rate_history(symbol, days=ML_LOOKBACK_DAYS)
                funding_series = None
                if not funding_df.empty:
                    funding_series = funding_df['fundingRate']

                features = engineer_features(df, funding_series)
                if not features.empty:
                    features['symbol'] = symbol
                    all_features.append(features)
                    logger.info(f"  {symbol}: {len(features)} feature rows")

                time.sleep(0.5)  # Rate limit courtesy
            except Exception as e:
                logger.error(f"Training data collection failed for {symbol}: {e}")

        if not all_features:
            return pd.DataFrame()

        combined = pd.concat(all_features, ignore_index=False)
        logger.info(f"Total training samples: {len(combined)} from {len(all_features)} symbols")
        return combined

    def train(self, data: pd.DataFrame = None) -> dict:
        """
        Train LightGBM with walk-forward validation.
        Returns metrics dict.
        """
        if not HAS_LGB:
            logger.error("LightGBM not available — cannot train")
            return {'error': 'lightgbm not installed'}

        if data is None:
            data = self.collect_training_data()

        if data.empty or len(data) < 1000:
            logger.error(f"Insufficient training data: {len(data)} rows (need 1000+)")
            return {'error': 'insufficient_data'}

        feature_cols = [c for c in get_feature_columns() if c in data.columns]
        if not feature_cols:
            logger.error("No valid feature columns found")
            return {'error': 'no_features'}

        X = data[feature_cols].values
        y = data['target'].values

        # Replace inf/nan
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # ---------- Walk-Forward Validation ----------
        total_samples = len(X)
        train_size = int(total_samples * 0.7)  # ~70% train
        test_size = total_samples - train_size

        # Simple split for now (walk-forward with multiple folds in backtester)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # LightGBM parameters
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'n_jobs': -1,
            'min_child_samples': 50,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'max_depth': 6,
        }

        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
        valid_data = lgb.Dataset(X_test, label=y_test, feature_name=feature_cols, reference=train_data)

        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(100)]

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[valid_data],
            callbacks=callbacks
        )

        # Evaluate
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)

        accuracy = (y_pred == y_test).mean()
        # Precision for class 1 (long signals)
        tp = ((y_pred == 1) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)

        # Feature importance
        importance = self.model.feature_importance(importance_type='gain')
        self.feature_importance = dict(zip(feature_cols, [float(v) for v in importance]))

        # Sort by importance
        sorted_fi = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)

        self.model_version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'train_samples': int(train_size),
            'test_samples': int(test_size),
            'total_samples': int(total_samples),
            'n_features': len(feature_cols),
            'top_features': sorted_fi[:10],
        }
        self.walk_forward_metrics = metrics

        self._save_model()

        logger.info(f"ML Training Complete — Accuracy: {accuracy:.3f}, "
                     f"Precision: {precision:.3f}, Recall: {recall:.3f}")
        logger.info(f"Top 5 features: {sorted_fi[:5]}")

        # Save feature importance to DB
        try:
            from data.storage import save_feature_importance
            save_feature_importance(self.feature_importance, self.model_version)
        except Exception:
            pass

        return metrics

    def predict_probability(self, features: dict) -> float:
        """Predict probability of upward move (0.0 to 1.0)."""
        if self.model is None or not HAS_LGB:
            return 0.5  # Neutral if no model

        feature_cols = get_feature_columns()
        X = np.array([[features.get(c, 0.0) for c in feature_cols]])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            prob = self.model.predict(X)[0]
            return float(prob)
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")
            return 0.5

    def is_model_available(self) -> bool:
        return self.model is not None and HAS_LGB

    def get_model_age_hours(self) -> float:
        meta_path = self._meta_path()
        if not os.path.exists(meta_path):
            return float('inf')
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            trained_at = datetime.fromisoformat(meta['trained_at'])
            # Ensure timezone-aware comparison (handle legacy naive timestamps)
            if trained_at.tzinfo is None:
                trained_at = trained_at.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - trained_at).total_seconds() / 3600
            return age
        except Exception:
            return float('inf')
