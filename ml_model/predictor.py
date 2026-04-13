"""
ml_model/predictor.py — Live ML prediction wrapper.
Combines rule-based confidence with ML probability.
"""

import logging
from config import ML_MIN_PROBABILITY, ENABLE_ML_LAYER
from ml_model.trainer import MLTrainer
from ml_model.features import extract_live_features

logger = logging.getLogger("oracle.predictor")

# Singleton trainer instance
_trainer = None


def get_trainer() -> MLTrainer:
    global _trainer
    if _trainer is None:
        _trainer = MLTrainer()
    return _trainer


def ml_filter(signal_data: dict) -> dict:
    """
    Apply ML filter to a rule-based signal.
    Adds 'ml_probability' and 'ml_approved' fields to signal_data.
    Returns modified signal_data.
    """
    if not ENABLE_ML_LAYER:
        signal_data['ml_probability'] = None
        signal_data['ml_approved'] = True
        return signal_data

    trainer = get_trainer()
    if not trainer.is_model_available():
        logger.info("ML model not yet trained — passing signal through unfiltered")
        signal_data['ml_probability'] = None
        signal_data['ml_approved'] = True
        return signal_data

    # Extract features
    features = extract_live_features(signal_data)
    if not features:
        signal_data['ml_probability'] = None
        signal_data['ml_approved'] = True
        return signal_data

    # Get probability
    prob = trainer.predict_probability(features)
    signal = signal_data['signal']

    # For SELL signals, we want low probability of up move
    if signal == "SELL":
        effective_prob = 1.0 - prob  # Probability of down move
    else:
        effective_prob = prob  # Probability of up move

    signal_data['ml_probability'] = effective_prob
    signal_data['ml_approved'] = effective_prob >= ML_MIN_PROBABILITY

    if signal_data['ml_approved']:
        logger.info(f"ML APPROVED: {signal_data['asset']} {signal} "
                     f"(prob={effective_prob:.1%} >= {ML_MIN_PROBABILITY:.0%})")
    else:
        logger.info(f"ML REJECTED: {signal_data['asset']} {signal} "
                     f"(prob={effective_prob:.1%} < {ML_MIN_PROBABILITY:.0%})")

    return signal_data


def retrain_if_needed():
    """Check if model needs retraining (>24h old) and retrain."""
    trainer = get_trainer()
    age = trainer.get_model_age_hours()

    if age > 24:
        logger.info(f"ML model is {age:.1f}h old — triggering nightly retrain")
        metrics = trainer.train()
        return metrics
    else:
        logger.info(f"ML model is {age:.1f}h old — no retrain needed")
        return None
