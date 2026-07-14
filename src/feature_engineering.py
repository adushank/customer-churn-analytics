"""Feature engineering module.

Generates derived columns used in analytics and ML:
    - customer_tenure_group
    - avg_monthly_spend (alias Monthly Charges)
    - customer_lifetime_value_calc
    - total_charges_calc
    - avg_revenue_per_month
    - risk_score
    - customer_segment
    - high_value_flag
    - long_term_flag
    - contract_length_months
    - service_count
    - has_internet
    - has_phone
"""

import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

CONTRACT_LENGTHS = {
    "Month-to-month": 1,
    "One year": 12,
    "Two year": 24,
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered feature columns to the DataFrame.

    Args:
        df: Cleaned DataFrame.

    Returns:
        DataFrame with additional feature columns.
    """
    df = df.copy()

    # ── 1. Contract Length (months) ────────────────────────────────────────────
    df["contract_length_months"] = df["Contract"].map(CONTRACT_LENGTHS).fillna(1).astype(int)

    # ── 2. Tenure Group ───────────────────────────────────────────────────────
    df["tenure_group"] = pd.cut(
        df["Tenure Months"],
        bins=[0, 12, 24, 48, 72, np.inf],
        labels=["0-12m", "13-24m", "25-48m", "49-72m", "72m+"],
        right=True,
    ).astype(str)

    # ── 3. Average Monthly Spend ──────────────────────────────────────────────
    df["avg_monthly_spend"] = df["Monthly Charges"]

    # ── 4. Total Charges Calculated ───────────────────────────────────────────
    df["total_charges_calc"] = (
        df["Monthly Charges"] * df["Tenure Months"].clip(lower=1)
    ).round(2)

    # ── 5. Average Revenue Per Month (using recorded Total Charges) ────────────
    df["avg_revenue_per_month"] = (
        df["Total Charges"] / df["Tenure Months"].clip(lower=1)
    ).round(2)

    # ── 6. Customer Lifetime Value (calculated) ───────────────────────────────
    # CLV = Monthly Charges × avg tenure remaining (contract length - tenure used)
    df["clv_calc"] = (
        df["Monthly Charges"] * df["contract_length_months"].clip(lower=1)
    ).round(2)

    # ── 7. Service Count ──────────────────────────────────────────────────────
    service_cols = [
        "Phone Service",
        "Multiple Lines",
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
    ]
    # Each "Yes" counts as 1 service
    for col in service_cols:
        df[f"_svc_{col}"] = (df[col].str.lower() == "yes").astype(int)

    df["service_count"] = df[[f"_svc_{c}" for c in service_cols]].sum(axis=1)
    df.drop(columns=[f"_svc_{c}" for c in service_cols], inplace=True)

    # ── 8. Has Internet / Phone ───────────────────────────────────────────────
    df["has_internet"] = (df["Internet Service"].str.lower() != "no").astype(int)
    df["has_phone"] = (df["Phone Service"].str.lower() == "yes").astype(int)

    # ── 9. Risk Score (normalised 0-100) ─────────────────────────────────────
    # Use the existing Churn Score column directly (it's already 0-100)
    df["risk_score"] = df["Churn Score"].clip(0, 100)

    # ── 10. Customer Segment ─────────────────────────────────────────────────
    # Based on Monthly Charges quartiles
    q33 = df["Monthly Charges"].quantile(0.33)
    q66 = df["Monthly Charges"].quantile(0.66)

    df["customer_segment"] = pd.cut(
        df["Monthly Charges"],
        bins=[-np.inf, q33, q66, np.inf],
        labels=["Budget", "Standard", "Premium"],
    ).astype(str)

    # ── 11. High-Value Customer Flag ─────────────────────────────────────────
    cltv_threshold = df["CLTV"].quantile(0.75)
    df["high_value_flag"] = (df["CLTV"] >= cltv_threshold).astype(int)

    # ── 12. Long-Term Customer Flag ──────────────────────────────────────────
    df["long_term_flag"] = (df["Tenure Months"] >= 24).astype(int)

    # ── 13. Age Group (Senior Citizen is Yes/No text) ─────────────────────────
    df["age_group"] = df["Senior Citizen"].map(
        {"Yes": "Senior", "No": "Non-Senior"}
    ).fillna("Non-Senior")

    # ── 14. Paperless + Electronic → digital customer ─────────────────────────
    df["digital_customer"] = (
        (df["Paperless Billing"].str.lower() == "yes")
        & (df["Payment Method"].str.contains("electronic", case=False, na=False))
    ).astype(int)

    logger.info("Feature engineering complete: %d features added", 14)
    return df
