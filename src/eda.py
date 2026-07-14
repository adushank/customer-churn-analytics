"""Exploratory Data Analysis module.

Generates summary statistics and EDA DataFrames used by the visualizations module.
"""

import logging
from typing import Dict, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def run_eda(df: pd.DataFrame) -> Dict[str, Any]:
    """Execute comprehensive EDA on the cleaned, feature-engineered DataFrame.

    Args:
        df: Feature-engineered DataFrame.

    Returns:
        Dictionary of EDA results (DataFrames, series, scalars).
    """
    results: Dict[str, Any] = {}

    logger.info("Running EDA on %d rows × %d columns", *df.shape)

    # ── Basic stats ───────────────────────────────────────────────────────────
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    results["describe"] = df[num_cols].describe().T.round(3)

    # ── Churn summary ─────────────────────────────────────────────────────────
    results["churn_counts"] = df["Churn Label"].value_counts()
    results["churn_rate"] = round(df["Churn Value"].mean() * 100, 2)

    # ── Correlation matrix (numeric cols) ─────────────────────────────────────
    corr_cols = [
        "Tenure Months", "Monthly Charges", "Total Charges", "CLTV",
        "Churn Score", "service_count", "contract_length_months",
        "risk_score", "high_value_flag", "long_term_flag", "Churn Value",
    ]
    corr_cols = [c for c in corr_cols if c in df.columns]
    results["correlation"] = df[corr_cols].corr().round(3)

    # ── Contract distribution ─────────────────────────────────────────────────
    results["contract_dist"] = df["Contract"].value_counts()

    # ── Payment method distribution ───────────────────────────────────────────
    results["payment_dist"] = df["Payment Method"].value_counts()

    # ── Internet service distribution ─────────────────────────────────────────
    results["internet_dist"] = df["Internet Service"].value_counts()

    # ── Gender distribution ───────────────────────────────────────────────────
    results["gender_dist"] = df["Gender"].value_counts()

    # ── Senior citizen breakdown ──────────────────────────────────────────────
    results["senior_dist"] = df["Senior Citizen"].value_counts()

    # ── Customer segment ──────────────────────────────────────────────────────
    results["segment_dist"] = df["customer_segment"].value_counts()

    # ── Tenure group distribution ─────────────────────────────────────────────
    results["tenure_group_dist"] = df["tenure_group"].value_counts()

    # ── Churn by contract ─────────────────────────────────────────────────────
    results["churn_by_contract"] = (
        df.groupby("Contract")["Churn Value"].mean().mul(100).round(2).rename("churn_rate_pct")
    )

    # ── Churn by payment method ───────────────────────────────────────────────
    results["churn_by_payment"] = (
        df.groupby("Payment Method")["Churn Value"].mean().mul(100).round(2)
    )

    # ── Churn by internet service ──────────────────────────────────────────────
    results["churn_by_internet"] = (
        df.groupby("Internet Service")["Churn Value"].mean().mul(100).round(2)
    )

    # ── Churn by tenure group ─────────────────────────────────────────────────
    results["churn_by_tenure"] = (
        df.groupby("tenure_group")["Churn Value"].mean().mul(100).round(2)
    )

    # ── Monthly charges distribution ──────────────────────────────────────────
    results["monthly_charges_series"] = df["Monthly Charges"]

    # ── Revenue by segment ────────────────────────────────────────────────────
    results["revenue_by_segment"] = (
        df.groupby("customer_segment")["Monthly Charges"].sum().round(2)
    )

    # ── Churn reason top 10 ───────────────────────────────────────────────────
    churned = df[df["Churn Value"] == 1]
    results["churn_reasons"] = churned["Churn Reason"].value_counts().head(10)

    # ── Service count distribution ────────────────────────────────────────────
    results["service_count_dist"] = df["service_count"].value_counts().sort_index()

    # ── Risk score by churn ───────────────────────────────────────────────────
    results["risk_churned"] = df[df["Churn Value"] == 1]["risk_score"]
    results["risk_retained"] = df[df["Churn Value"] == 0]["risk_score"]

    logger.info("EDA complete: %d result sets generated", len(results))
    return results
