"""Prediction module.

Provides `predict_single` for one-off predictions and `predict_batch` for bulk scoring.
Used by predict.py CLI script.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent / "models" / "churn_model.pkl"


def load_model(model_path: Path = MODEL_PATH) -> Dict[str, Any]:
    """Load the saved model bundle.

    Returns:
        Dict with 'model', 'feature_names', 'model_name'.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run main.py first to train the model."
        )
    bundle = joblib.load(model_path)
    logger.info("Loaded model: %s", bundle.get("model_name", "unknown"))
    return bundle


def predict_single(customer: Dict[str, Any], model_path: Path = MODEL_PATH) -> Dict[str, Any]:
    """Predict churn for a single customer dict.

    Args:
        customer: Dict of feature values (raw, unnormalised).
        model_path: Path to the saved model pickle.

    Returns:
        Dict with 'churn_probability', 'prediction', 'risk_label'.
    """
    bundle = load_model(model_path)
    model = bundle["model"]
    feature_names: list = bundle["feature_names"]

    # Build a 1-row DataFrame aligned to expected features
    row = {f: customer.get(f, 0) for f in feature_names}
    X = np.array([[row[f] for f in feature_names]], dtype=float)

    prob = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") else 0.5
    label = int(prob >= 0.5)

    if prob >= 0.75:
        risk_label = "Very High Risk — Likely to Churn"
    elif prob >= 0.50:
        risk_label = "High Risk — Likely to Churn"
    elif prob >= 0.30:
        risk_label = "Moderate Risk"
    else:
        risk_label = "Low Risk — Likely to Stay"

    return {
        "churn_probability_pct": round(prob * 100, 1),
        "prediction": label,
        "prediction_label": "Likely to Churn" if label else "Likely to Stay",
        "risk_label": risk_label,
        "model_used": bundle.get("model_name", "unknown"),
    }


def predict_batch(df: pd.DataFrame, model_path: Path = MODEL_PATH) -> pd.DataFrame:
    """Predict churn probability for every row in df.

    Args:
        df: Feature-engineered DataFrame (same schema as training).
        model_path: Path to saved model pickle.

    Returns:
        Original df with added 'predicted_churn_prob' and 'predicted_churn' columns.
    """
    from .model_training import prepare_data  # local import to avoid circular

    bundle = load_model(model_path)
    model = bundle["model"]
    X, _, _ = prepare_data(df)

    probs = (
        model.predict_proba(X)[:, 1]
        if hasattr(model, "predict_proba")
        else np.zeros(len(df))
    )
    preds = (probs >= 0.5).astype(int)

    out = df.copy()
    out["predicted_churn_prob"] = (probs * 100).round(1)
    out["predicted_churn"] = preds
    return out
