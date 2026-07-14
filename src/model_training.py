"""Machine Learning model training module.

Trains and evaluates four classifiers:
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - Gradient Boosting

Selects the best model by ROC-AUC and saves it to models/churn_model.pkl.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent / "models" / "churn_model.pkl"
REPORTS_PATH = Path(__file__).parent.parent / "output" / "reports"

FEATURE_COLS = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "CLTV",
    "Churn Score",
    "contract_length_months",
    "service_count",
    "has_internet",
    "has_phone",
    "high_value_flag",
    "long_term_flag",
    "digital_customer",
    # Encoded categoricals
    "gender_enc",
    "senior_enc",
    "partner_enc",
    "dependents_enc",
    "internet_enc",
    "contract_enc",
    "payment_enc",
    "segment_enc",
]

TARGET_COL = "Churn Value"


def _encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode categorical columns needed for ML.

    Args:
        df: Feature-engineered DataFrame.

    Returns:
        DataFrame with added encoded columns.
    """
    df = df.copy()
    binary_map = {"Yes": 1, "No": 0}

    df["gender_enc"] = (df["Gender"].str.lower() == "male").astype(int)
    df["senior_enc"] = df["Senior Citizen"].map(binary_map).fillna(0).astype(int)
    df["partner_enc"] = df["Partner"].map(binary_map).fillna(0).astype(int)
    df["dependents_enc"] = df["Dependents"].map(binary_map).fillna(0).astype(int)

    # Ordinal contract encoding
    contract_ord = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    df["contract_enc"] = df["Contract"].map(contract_ord).fillna(0).astype(int)

    # Internet service
    internet_map = {"No": 0, "DSL": 1, "Fiber optic": 2}
    df["internet_enc"] = df["Internet Service"].map(internet_map).fillna(0).astype(int)

    # Payment method
    le_payment = LabelEncoder()
    df["payment_enc"] = le_payment.fit_transform(df["Payment Method"].fillna("Unknown"))

    # Customer segment
    seg_map = {"Budget": 0, "Standard": 1, "Premium": 2}
    df["segment_enc"] = df["customer_segment"].map(seg_map).fillna(1).astype(int)

    return df


def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
    """Encode features and return X, y arrays.

    Args:
        df: Feature-engineered DataFrame.

    Returns:
        (X, y, feature_names) tuple.
    """
    df_enc = _encode_features(df)

    available = [c for c in FEATURE_COLS if c in df_enc.columns]
    missing = set(FEATURE_COLS) - set(available)
    if missing:
        logger.warning("Missing feature columns: %s", missing)

    X = df_enc[available].fillna(0).values
    y = df_enc[TARGET_COL].values
    return X, y, available


def _evaluate(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    name: str,
) -> Dict[str, Any]:
    """Compute all classification metrics for a fitted model.

    Args:
        model: Fitted sklearn estimator.
        X_test: Test features.
        y_test: True labels.
        name: Model name string.

    Returns:
        Dict of metrics.
    """
    y_pred = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(X_test)
    )

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)

    return {
        "name": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "fpr": fpr,
        "tpr": tpr,
        "prec_curve": prec_curve,
        "rec_curve": rec_curve,
        "y_proba": y_proba,
        "y_pred": y_pred,
    }


def train_models(df: pd.DataFrame) -> Dict[str, Any]:
    """Train all four classifiers and return evaluation results.

    Args:
        df: Feature-engineered DataFrame.

    Returns:
        Dict with keys:
            - 'metrics': list of metric dicts per model
            - 'best_model': fitted best estimator
            - 'best_name': name of best model
            - 'feature_names': list of feature column names
            - 'X_test', 'y_test': held-out test arrays
            - 'feature_importance': DataFrame
    """
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORTS_PATH.mkdir(parents=True, exist_ok=True)

    X, y, feature_names = prepare_data(df)
    logger.info(
        "Training data: %d samples, %d features, %.1f%% positive",
        len(y), X.shape[1], y.mean() * 100,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define models
    candidates = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42,
            class_weight="balanced", n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            random_state=42, subsample=0.8,
        ),
    }

    metrics_list = []
    fitted_models: Dict[str, Any] = {}

    for name, model in candidates.items():
        logger.info("Training %s ...", name)
        model.fit(X_train, y_train)
        fitted_models[name] = model
        m = _evaluate(model, X_test, y_test, name)
        metrics_list.append(m)
        logger.info(
            "%s → Acc=%.4f Prec=%.4f Rec=%.4f F1=%.4f AUC=%.4f",
            name, m["accuracy"], m["precision"], m["recall"], m["f1"], m["roc_auc"],
        )

    # Select best by ROC-AUC
    best = max(metrics_list, key=lambda x: x["roc_auc"])
    best_model = fitted_models[best["name"]]
    logger.info("Best model: %s (AUC=%.4f)", best["name"], best["roc_auc"])

    # Save best model
    joblib.dump(
        {
            "model": best_model,
            "feature_names": feature_names,
            "model_name": best["name"],
        },
        MODEL_PATH,
    )
    logger.info("Model saved to %s", MODEL_PATH)

    # Feature importance
    fi_df = _get_feature_importance(best_model, best["name"], feature_names)

    # Save metrics CSV
    metrics_df = pd.DataFrame([
        {k: v for k, v in m.items()
         if k not in ("confusion_matrix", "classification_report",
                      "fpr", "tpr", "prec_curve", "rec_curve", "y_proba", "y_pred")}
        for m in metrics_list
    ])
    metrics_df.to_csv(REPORTS_PATH / "model_metrics.csv", index=False)

    # Save feature importance CSV
    fi_df.to_csv(REPORTS_PATH / "feature_importance.csv", index=False)

    return {
        "metrics": metrics_list,
        "best_model": best_model,
        "best_name": best["name"],
        "feature_names": feature_names,
        "X_test": X_test,
        "y_test": y_test,
        "feature_importance": fi_df,
        "fitted_models": fitted_models,
    }


def _get_feature_importance(model: Any, name: str, feature_names: list) -> pd.DataFrame:
    """Extract feature importances from a fitted model."""
    importances = None

    # For Pipeline (e.g. Logistic Regression)
    actual_model = model
    if hasattr(model, "named_steps"):
        actual_model = model.named_steps.get("clf", model)

    if hasattr(actual_model, "feature_importances_"):
        importances = actual_model.feature_importances_
    elif hasattr(actual_model, "coef_"):
        importances = np.abs(actual_model.coef_[0])

    if importances is None:
        return pd.DataFrame({"feature": feature_names, "importance": [0.0] * len(feature_names)})

    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False).reset_index(drop=True)
    fi_df["importance"] = fi_df["importance"].round(6)
    return fi_df
