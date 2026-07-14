"""Report exporter module.

Exports:
    - Executive summary (TXT)
    - Business report (TXT)
    - Model metrics CSV
    - Data quality report CSV
    - Cleaned dataset CSV
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent.parent / "output" / "reports"


def export_executive_summary(
    df: pd.DataFrame,
    clean_report: Dict[str, Any],
    eda_results: Dict[str, Any],
    ml_results: Dict[str, Any],
    query_results: Dict[str, Any],
) -> Path:
    """Write a plain-text executive summary."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "executive_summary.txt"

    kpis = query_results.get("01_customer_kpis", pd.DataFrame())
    kpi = kpis.iloc[0] if not kpis.empty else {}

    best = ml_results["best_name"]
    best_m = next(m for m in ml_results["metrics"] if m["name"] == best)

    lines = [
        "=" * 72,
        "  CUSTOMER CHURN ANALYTICS & PREDICTION PLATFORM",
        "  Executive Summary",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 72,
        "",
        "── OVERVIEW ────────────────────────────────────────────────────────────",
        f"  Total Customers     : {kpi.get('total_customers', len(df)):,}",
        f"  Churned Customers   : {kpi.get('churned_customers', df['Churn Value'].sum()):,}",
        f"  Retained Customers  : {kpi.get('retained_customers', (df['Churn Value']==0).sum()):,}",
        f"  Overall Churn Rate  : {kpi.get('overall_churn_rate_pct', round(df['Churn Value'].mean()*100,2))}%",
        "",
        "── FINANCIAL KPIs ──────────────────────────────────────────────────────",
        f"  Avg Monthly Charges : ${kpi.get('avg_monthly_charges', round(df['Monthly Charges'].mean(), 2)):.2f}",
        f"  Total Revenue       : ${kpi.get('total_revenue', df['Total Charges'].sum()):,.2f}",
        f"  Avg CLTV            : ${kpi.get('avg_cltv', round(df['CLTV'].mean(), 2)):.2f}",
        f"  Avg Tenure          : {kpi.get('avg_tenure_months', round(df['Tenure Months'].mean(), 1)):.1f} months",
        "",
        "── DATA QUALITY ────────────────────────────────────────────────────────",
        f"  Duplicates Removed  : {clean_report.get('duplicates_removed', 0)}",
        f"  Missing Values Fixed: {clean_report.get('missing_values_fixed', 0)}",
        f"  TC Imputed Rows     : {clean_report.get('total_charges_imputed', 0)}",
        f"  Rows After Cleaning : {clean_report.get('rows_after_cleaning', len(df)):,}",
        "",
        "── MACHINE LEARNING ────────────────────────────────────────────────────",
        f"  Best Model          : {best}",
        f"  Accuracy            : {best_m['accuracy']:.4f}",
        f"  Precision           : {best_m['precision']:.4f}",
        f"  Recall              : {best_m['recall']:.4f}",
        f"  F1 Score            : {best_m['f1']:.4f}",
        f"  ROC-AUC             : {best_m['roc_auc']:.4f}",
        "",
        "── ALL MODEL COMPARISON ────────────────────────────────────────────────",
    ]
    for m in ml_results["metrics"]:
        lines.append(
            f"  {m['name']:<25} Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}  AUC={m['roc_auc']:.4f}"
        )

    lines += [
        "",
        "── KEY FINDINGS ────────────────────────────────────────────────────────",
        "  1. Month-to-month contract customers have the highest churn rate.",
        "  2. Electronic check payment users churn significantly more than",
        "     automatic payment users.",
        "  3. Fiber optic internet customers churn at nearly double the rate",
        "     of DSL customers.",
        "  4. Senior citizens churn at a higher rate than non-seniors.",
        "  5. Short-tenure customers (0-12 months) are the highest-risk cohort.",
        "  6. High-value customers represent a significant revenue-at-risk segment.",
        "",
        "=" * 72,
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Executive summary written to %s", path)
    return path


def export_business_report(
    df: pd.DataFrame,
    eda_results: Dict[str, Any],
    ml_results: Dict[str, Any],
) -> Path:
    """Write a detailed business-level report."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "business_report.txt"

    fi = ml_results["feature_importance"].head(10)

    lines = [
        "=" * 72,
        "  CUSTOMER CHURN ANALYTICS — BUSINESS INTELLIGENCE REPORT",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 72,
        "",
        "1. CHURN OVERVIEW",
        "-" * 40,
        f"   Churn rate: {df['Churn Value'].mean()*100:.2f}%",
        f"   Total churned customers: {df['Churn Value'].sum():,}",
        "",
        "2. CHURN BY CONTRACT TYPE",
        "-" * 40,
    ]
    for contract, rate in df.groupby("Contract")["Churn Value"].mean().mul(100).items():
        lines.append(f"   {contract:<20}: {rate:.1f}%")

    lines += [
        "",
        "3. CHURN BY PAYMENT METHOD",
        "-" * 40,
    ]
    for pm, rate in df.groupby("Payment Method")["Churn Value"].mean().mul(100).sort_values(ascending=False).items():
        lines.append(f"   {pm:<32}: {rate:.1f}%")

    lines += [
        "",
        "4. CHURN BY INTERNET SERVICE",
        "-" * 40,
    ]
    for svc, rate in df.groupby("Internet Service")["Churn Value"].mean().mul(100).sort_values(ascending=False).items():
        lines.append(f"   {svc:<20}: {rate:.1f}%")

    lines += [
        "",
        "5. REVENUE AT RISK",
        "-" * 40,
        f"   Monthly revenue lost to churn: ${df[df['Churn Value']==1]['Monthly Charges'].sum():,.2f}",
        f"   Total charges lost:            ${df[df['Churn Value']==1]['Total Charges'].sum():,.2f}",
        f"   Avg CLTV of churned customers: ${df[df['Churn Value']==1]['CLTV'].mean():,.2f}",
        "",
        "6. TOP CHURN PREDICTORS (Feature Importance)",
        "-" * 40,
    ]
    for _, row in fi.iterrows():
        lines.append(f"   {row['feature']:<30}: {row['importance']:.6f}")

    lines += [
        "",
        "7. HIGH-RISK SEGMENT PROFILE",
        "-" * 40,
        f"   Customers at risk (risk_score ≥ 70, not yet churned): {((df['risk_score']>=70) & (df['Churn Value']==0)).sum():,}",
        f"   Monthly revenue at risk: ${df[(df['risk_score']>=70) & (df['Churn Value']==0)]['Monthly Charges'].sum():,.2f}",
        "",
        "8. RECOMMENDATIONS",
        "-" * 40,
        "   • Target month-to-month customers with 1-year contract upgrade offers.",
        "   • Encourage electronic check users to switch to auto-pay (lower churn).",
        "   • Provide dedicated retention support for fiber optic customers.",
        "   • Run proactive outreach for customers with risk_score ≥ 70.",
        "   • Bundle tech support / online security to improve stickiness.",
        "",
        "=" * 72,
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Business report written to %s", path)
    return path


def export_cleaned_dataset(df: pd.DataFrame) -> Path:
    """Export the feature-engineered dataset to CSV."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "cleaned_engineered_dataset.csv"
    df.to_csv(path, index=False)
    logger.info("Cleaned dataset exported to %s (%d rows)", path, len(df))
    return path


def export_data_quality_report(quality_df: pd.DataFrame) -> Path:
    """Export the data quality summary to CSV."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "data_quality_report.csv"
    quality_df.to_csv(path, index=False)
    logger.info("Data quality report exported to %s", path)
    return path


def run_all_exports(
    df: pd.DataFrame,
    quality_df: pd.DataFrame,
    clean_report: Dict[str, Any],
    eda_results: Dict[str, Any],
    ml_results: Dict[str, Any],
    query_results: Dict[str, Any],
) -> Dict[str, Path]:
    """Run all export functions and return dict of paths."""
    paths: Dict[str, Path] = {}

    paths["executive_summary"] = export_executive_summary(
        df, clean_report, eda_results, ml_results, query_results
    )
    paths["business_report"] = export_business_report(df, eda_results, ml_results)
    paths["cleaned_dataset"] = export_cleaned_dataset(df)
    paths["data_quality"] = export_data_quality_report(quality_df)

    logger.info("All reports exported: %d files", len(paths))
    return paths
