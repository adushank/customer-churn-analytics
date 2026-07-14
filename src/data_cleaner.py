"""Data cleaning and validation module."""

import logging
from typing import Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Valid domain values
VALID_CONTRACTS = {"Month-to-month", "One year", "Two year"}
VALID_PAYMENT_METHODS = {
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
}
VALID_INTERNET_SERVICES = {"DSL", "Fiber optic", "No"}
VALID_GENDERS = {"Male", "Female"}


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Clean and validate the raw DataFrame.

    Steps:
        1. Remove duplicate customers.
        2. Fix / cast column types.
        3. Handle missing values.
        4. Remove invalid ages (Senior Citizen is binary Yes/No; infer age from tenure).
        5. Remove negative Monthly Charges.
        6. Standardise Contract types.
        7. Standardise Payment Methods.
        8. Validate Churn columns.

    Args:
        df: Raw DataFrame from data_loader.

    Returns:
        Tuple of (cleaned DataFrame, quality report dict).
    """
    report: dict = {}
    original_len = len(df)
    df = df.copy()

    # ── 1. Duplicate rows ──────────────────────────────────────────────────────
    dup_count = df.duplicated(subset=["CustomerID"]).sum()
    df = df.drop_duplicates(subset=["CustomerID"]).reset_index(drop=True)
    report["duplicates_removed"] = int(dup_count)
    logger.info("Removed %d duplicate customers", dup_count)

    # ── 2. Column type fixes ───────────────────────────────────────────────────
    # Total Charges can arrive as str " " for brand-new customers (tenure=0)
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")

    # Fill missing Total Charges with Monthly Charges * Tenure (at least 1 month)
    mask_tc = df["Total Charges"].isna()
    df.loc[mask_tc, "Total Charges"] = (
        df.loc[mask_tc, "Monthly Charges"] * df.loc[mask_tc, "Tenure Months"].clip(lower=1)
    )
    report["total_charges_imputed"] = int(mask_tc.sum())

    # Ensure numeric columns
    for col in ["Monthly Charges", "Total Charges", "CLTV", "Churn Score", "Tenure Months"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── 3. Missing values ──────────────────────────────────────────────────────
    missing_before = df.isnull().sum()

    # Churn Reason: missing means customer did not churn → fill with "Not Churned"
    df["Churn Reason"] = df["Churn Reason"].fillna("Not Churned")

    # For numeric cols, fill remaining NaN with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        n_miss = df[col].isna().sum()
        if n_miss > 0:
            df[col].fillna(df[col].median(), inplace=True)
            logger.warning("Imputed %d NaN in '%s' with median", n_miss, col)

    # For categorical cols, fill with "Unknown"
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        n_miss = df[col].isna().sum()
        if n_miss > 0:
            df[col].fillna("Unknown", inplace=True)
            logger.warning("Imputed %d NaN in '%s' with 'Unknown'", n_miss, col)

    report["missing_values_fixed"] = int(missing_before.sum())

    # ── 4. Negative Monthly Charges ────────────────────────────────────────────
    neg_mask = df["Monthly Charges"] < 0
    report["negative_monthly_charges"] = int(neg_mask.sum())
    if neg_mask.any():
        df.loc[neg_mask, "Monthly Charges"] = df["Monthly Charges"].median()
        logger.warning("Replaced %d negative Monthly Charges with median", neg_mask.sum())

    # Also fix negative Total Charges
    neg_tc = df["Total Charges"] < 0
    if neg_tc.any():
        df.loc[neg_tc, "Total Charges"] = (
            df.loc[neg_tc, "Monthly Charges"] * df.loc[neg_tc, "Tenure Months"].clip(lower=1)
        )

    # ── 5. Invalid Contract types ──────────────────────────────────────────────
    invalid_contract = ~df["Contract"].isin(VALID_CONTRACTS)
    report["invalid_contracts"] = int(invalid_contract.sum())
    if invalid_contract.any():
        df.loc[invalid_contract, "Contract"] = "Month-to-month"
        logger.warning("Fixed %d invalid Contract values", invalid_contract.sum())

    # ── 6. Invalid Payment Methods ─────────────────────────────────────────────
    invalid_pm = ~df["Payment Method"].isin(VALID_PAYMENT_METHODS)
    report["invalid_payment_methods"] = int(invalid_pm.sum())
    if invalid_pm.any():
        df.loc[invalid_pm, "Payment Method"] = "Unknown"
        logger.warning("Fixed %d invalid Payment Method values", invalid_pm.sum())

    # ── 7. Churn columns consistency ───────────────────────────────────────────
    # Ensure Churn Value aligns with Churn Label
    df["Churn Value"] = df["Churn Label"].str.strip().str.lower().map({"yes": 1, "no": 0})
    df["Churn Value"] = df["Churn Value"].fillna(0).astype(int)

    # ── 8. Strip whitespace from string columns ────────────────────────────────
    str_cols = df.select_dtypes(include=["object"]).columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()

    # ── 9. Ensure Tenure ≥ 0 ──────────────────────────────────────────────────
    df["Tenure Months"] = df["Tenure Months"].clip(lower=0)

    report["rows_after_cleaning"] = len(df)
    report["rows_removed_total"] = original_len - len(df)

    logger.info(
        "Cleaning complete: %d rows retained (removed %d)",
        len(df),
        original_len - len(df),
    )
    return df, report


def get_data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy data-quality summary DataFrame."""
    rows = []
    for col in df.columns:
        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
                "null_pct": round(df[col].isnull().mean() * 100, 2),
                "unique_count": int(df[col].nunique()),
            }
        )
    return pd.DataFrame(rows)
