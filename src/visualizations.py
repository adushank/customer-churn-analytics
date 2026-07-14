"""Visualizations module.

Generates 20+ charts to output/plots/:
    1.  churn_distribution (pie)
    2.  monthly_charges_distribution (histogram)
    3.  tenure_distribution (histogram)
    4.  churn_by_contract (bar)
    5.  churn_by_payment_method (bar)
    6.  churn_by_internet_service (bar)
    7.  monthly_charges_boxplot (boxplot by churn)
    8.  tenure_boxplot (boxplot by churn)
    9.  correlation_heatmap
    10. customer_segment_distribution (pie)
    11. revenue_by_segment (bar)
    12. churn_by_gender (bar)
    13. churn_by_senior_citizen (bar)
    14. service_count_distribution (bar)
    15. risk_score_distribution (histogram, churned vs retained)
    16. churn_reasons (horizontal bar, top 10)
    17. roc_curve (all models)
    18. feature_importance (horizontal bar)
    19. confusion_matrix_heatmap (best model)
    20. precision_recall_curve (all models)
    21. cltv_by_contract (boxplot)
    22. churn_by_tenure_group (bar)
    23. revenue_concentration (area / Pareto)
"""

import logging
from pathlib import Path
from typing import Dict, Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

PLOTS_DIR = Path(__file__).parent.parent / "output" / "plots"
PALETTE = sns.color_palette("tab10")
CHURN_COLORS = {"Yes": "#e74c3c", "No": "#2ecc71"}

plt.rcParams.update({
    "figure.dpi": 120,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 14,
    "axes.labelsize": 11,
})


def _save(fig: plt.Figure, name: str) -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    logger.info("Saved chart: %s", path.name)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# 1. Churn Distribution Pie
# ──────────────────────────────────────────────────────────────────────────────
def plot_churn_distribution(df: pd.DataFrame) -> Path:
    counts = df["Churn Label"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        counts,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=[CHURN_COLORS.get(l, "#95a5a6") for l in counts.index],
        startangle=140,
        pctdistance=0.8,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    ax.set_title("Customer Churn Distribution", fontweight="bold")
    return _save(fig, "01_churn_distribution")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Monthly Charges Distribution
# ──────────────────────────────────────────────────────────────────────────────
def plot_monthly_charges_distribution(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, color in CHURN_COLORS.items():
        subset = df[df["Churn Label"] == label]["Monthly Charges"]
        ax.hist(subset, bins=40, alpha=0.6, color=color, label=f"Churn = {label}", edgecolor="white")
    ax.set_xlabel("Monthly Charges ($)")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Monthly Charges Distribution by Churn Status", fontweight="bold")
    ax.legend()
    return _save(fig, "02_monthly_charges_distribution")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Tenure Distribution
# ──────────────────────────────────────────────────────────────────────────────
def plot_tenure_distribution(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, color in CHURN_COLORS.items():
        subset = df[df["Churn Label"] == label]["Tenure Months"]
        ax.hist(subset, bins=40, alpha=0.6, color=color, label=f"Churn = {label}", edgecolor="white")
    ax.set_xlabel("Tenure (Months)")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Customer Tenure Distribution by Churn Status", fontweight="bold")
    ax.legend()
    return _save(fig, "03_tenure_distribution")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Churn Rate by Contract
# ──────────────────────────────────────────────────────────────────────────────
def plot_churn_by_contract(df: pd.DataFrame) -> Path:
    rates = df.groupby("Contract")["Churn Value"].mean().mul(100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(rates.index, rates.values, color=PALETTE[:len(rates)], edgecolor="white")
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{bar.get_height():.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    ax.set_ylabel("Churn Rate (%)")
    ax.set_title("Churn Rate by Contract Type", fontweight="bold")
    ax.set_ylim(0, rates.max() * 1.25)
    return _save(fig, "04_churn_by_contract")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Churn Rate by Payment Method
# ──────────────────────────────────────────────────────────────────────────────
def plot_churn_by_payment_method(df: pd.DataFrame) -> Path:
    rates = df.groupby("Payment Method")["Churn Value"].mean().mul(100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(rates.index, rates.values, color=PALETTE[:len(rates)], edgecolor="white")
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{bar.get_height():.1f}%",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_ylabel("Churn Rate (%)")
    ax.set_title("Churn Rate by Payment Method", fontweight="bold")
    ax.set_ylim(0, rates.max() * 1.25)
    plt.xticks(rotation=15, ha="right")
    return _save(fig, "05_churn_by_payment_method")


# ──────────────────────────────────────────────────────────────────────────────
# 6. Churn Rate by Internet Service
# ──────────────────────────────────────────────────────────────────────────────
def plot_churn_by_internet_service(df: pd.DataFrame) -> Path:
    rates = df.groupby("Internet Service")["Churn Value"].mean().mul(100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(rates.index, rates.values, color=["#e74c3c", "#3498db", "#2ecc71"], edgecolor="white")
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{bar.get_height():.1f}%",
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_ylabel("Churn Rate (%)")
    ax.set_title("Churn Rate by Internet Service", fontweight="bold")
    ax.set_ylim(0, rates.max() * 1.25)
    return _save(fig, "06_churn_by_internet_service")


# ──────────────────────────────────────────────────────────────────────────────
# 7. Monthly Charges Boxplot (by churn)
# ──────────────────────────────────────────────────────────────────────────────
def plot_monthly_charges_boxplot(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [
        df[df["Churn Label"] == "Yes"]["Monthly Charges"].dropna(),
        df[df["Churn Label"] == "No"]["Monthly Charges"].dropna(),
    ]
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops={"color": "white", "linewidth": 2})
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Churned", "Retained"])
    for patch, color in zip(bp["boxes"], ["#e74c3c", "#2ecc71"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Monthly Charges ($)")
    ax.set_title("Monthly Charges Distribution — Churned vs Retained", fontweight="bold")
    return _save(fig, "07_monthly_charges_boxplot")


# ──────────────────────────────────────────────────────────────────────────────
# 8. Tenure Boxplot (by churn)
# ──────────────────────────────────────────────────────────────────────────────
def plot_tenure_boxplot(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [
        df[df["Churn Label"] == "Yes"]["Tenure Months"].dropna(),
        df[df["Churn Label"] == "No"]["Tenure Months"].dropna(),
    ]
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops={"color": "white", "linewidth": 2})
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Churned", "Retained"])
    for patch, color in zip(bp["boxes"], ["#e74c3c", "#2ecc71"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Tenure (Months)")
    ax.set_title("Customer Tenure — Churned vs Retained", fontweight="bold")
    return _save(fig, "08_tenure_boxplot")


# ──────────────────────────────────────────────────────────────────────────────
# 9. Correlation Heatmap
# ──────────────────────────────────────────────────────────────────────────────
def plot_correlation_heatmap(eda_results: Dict[str, Any]) -> Path:
    corr = eda_results["correlation"]
    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, linewidths=0.5, ax=ax, square=True,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Matrix", fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right")
    return _save(fig, "09_correlation_heatmap")


# ──────────────────────────────────────────────────────────────────────────────
# 10. Customer Segment Distribution (pie)
# ──────────────────────────────────────────────────────────────────────────────
def plot_customer_segment_pie(df: pd.DataFrame) -> Path:
    counts = df["customer_segment"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        counts,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=["#3498db", "#f39c12", "#e74c3c"],
        startangle=120,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    ax.set_title("Customer Segment Distribution", fontweight="bold")
    return _save(fig, "10_customer_segment_pie")


# ──────────────────────────────────────────────────────────────────────────────
# 11. Revenue by Segment (bar)
# ──────────────────────────────────────────────────────────────────────────────
def plot_revenue_by_segment(df: pd.DataFrame) -> Path:
    rev = df.groupby("customer_segment")["Monthly Charges"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(rev.index, rev.values, color=["#e74c3c", "#f39c12", "#3498db"], edgecolor="white")
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + rev.max() * 0.01,
            f"${bar.get_height():,.0f}",
            ha="center", va="bottom", fontsize=9,
        )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_ylabel("Total Monthly Revenue ($)")
    ax.set_title("Monthly Revenue by Customer Segment", fontweight="bold")
    return _save(fig, "11_revenue_by_segment")


# ──────────────────────────────────────────────────────────────────────────────
# 12. Churn by Gender
# ──────────────────────────────────────────────────────────────────────────────
def plot_churn_by_gender(df: pd.DataFrame) -> Path:
    ct = df.groupby(["Gender", "Churn Label"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 5))
    ct.plot(kind="bar", ax=ax, color=[CHURN_COLORS["No"], CHURN_COLORS["Yes"]],
            edgecolor="white", rot=0)
    ax.set_ylabel("Number of Customers")
    ax.set_title("Churn Count by Gender", fontweight="bold")
    ax.legend(title="Churn")
    return _save(fig, "12_churn_by_gender")


# ──────────────────────────────────────────────────────────────────────────────
# 13. Churn by Senior Citizen
# ──────────────────────────────────────────────────────────────────────────────
def plot_churn_by_senior(df: pd.DataFrame) -> Path:
    rates = df.groupby("Senior Citizen")["Churn Value"].mean().mul(100)
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(["Non-Senior (No)", "Senior (Yes)"], rates.values,
                  color=["#3498db", "#e74c3c"], edgecolor="white")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Churn Rate (%)")
    ax.set_title("Churn Rate: Senior vs Non-Senior Citizens", fontweight="bold")
    return _save(fig, "13_churn_by_senior_citizen")


# ──────────────────────────────────────────────────────────────────────────────
# 14. Service Count Distribution
# ──────────────────────────────────────────────────────────────────────────────
def plot_service_count_distribution(df: pd.DataFrame) -> Path:
    grp = df.groupby("service_count").agg(
        total=("Churn Value", "count"),
        churned=("Churn Value", "sum"),
    ).reset_index()
    grp["retained"] = grp["total"] - grp["churned"]

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.35
    x = np.arange(len(grp))
    ax.bar(x - width / 2, grp["churned"], width, label="Churned", color="#e74c3c", edgecolor="white")
    ax.bar(x + width / 2, grp["retained"], width, label="Retained", color="#2ecc71", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(grp["service_count"].astype(str))
    ax.set_xlabel("Number of Services Subscribed")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Churn vs Retention by Number of Services", fontweight="bold")
    ax.legend()
    return _save(fig, "14_service_count_distribution")


# ──────────────────────────────────────────────────────────────────────────────
# 15. Risk Score Distribution (churned vs retained)
# ──────────────────────────────────────────────────────────────────────────────
def plot_risk_score_distribution(eda_results: Dict[str, Any]) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(eda_results["risk_churned"], bins=40, alpha=0.6, color="#e74c3c",
            label="Churned", edgecolor="white")
    ax.hist(eda_results["risk_retained"], bins=40, alpha=0.6, color="#2ecc71",
            label="Retained", edgecolor="white")
    ax.set_xlabel("Risk Score (Churn Score)")
    ax.set_ylabel("Count")
    ax.set_title("Risk Score Distribution — Churned vs Retained", fontweight="bold")
    ax.legend()
    return _save(fig, "15_risk_score_distribution")


# ──────────────────────────────────────────────────────────────────────────────
# 16. Top 10 Churn Reasons
# ──────────────────────────────────────────────────────────────────────────────
def plot_churn_reasons(eda_results: Dict[str, Any]) -> Path:
    reasons = eda_results["churn_reasons"]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Reds_r(np.linspace(0.3, 0.9, len(reasons)))
    bars = ax.barh(reasons.index, reasons.values, color=colors, edgecolor="white")
    for bar in bars:
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                str(int(bar.get_width())), va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Customers")
    ax.set_title("Top 10 Churn Reasons", fontweight="bold")
    return _save(fig, "16_churn_reasons")


# ──────────────────────────────────────────────────────────────────────────────
# 17. ROC Curves (all models)
# ──────────────────────────────────────────────────────────────────────────────
def plot_roc_curves(ml_results: Dict[str, Any]) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    for i, m in enumerate(ml_results["metrics"]):
        ax.plot(m["fpr"], m["tpr"], lw=2, color=colors[i],
                label=f"{m['name']} (AUC={m['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    return _save(fig, "17_roc_curves")


# ──────────────────────────────────────────────────────────────────────────────
# 18. Feature Importance
# ──────────────────────────────────────────────────────────────────────────────
def plot_feature_importance(ml_results: Dict[str, Any]) -> Path:
    fi = ml_results["feature_importance"].head(15)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(fi)))
    bars = ax.barh(fi["feature"], fi["importance"], color=colors, edgecolor="white")
    for bar in bars:
        ax.text(bar.get_width() + fi["importance"].max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.4f}", va="center", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top Feature Importances — {ml_results['best_name']}", fontweight="bold")
    return _save(fig, "18_feature_importance")


# ──────────────────────────────────────────────────────────────────────────────
# 19. Confusion Matrix Heatmap (best model)
# ──────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(ml_results: Dict[str, Any]) -> Path:
    best_name = ml_results["best_name"]
    best_m = next(m for m in ml_results["metrics"] if m["name"] == best_name)
    cm = best_m["confusion_matrix"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Predicted No", "Predicted Yes"],
        yticklabels=["Actual No", "Actual Yes"],
        linewidths=1, linecolor="white",
    )
    ax.set_title(f"Confusion Matrix — {best_name}", fontweight="bold")
    return _save(fig, "19_confusion_matrix")


# ──────────────────────────────────────────────────────────────────────────────
# 20. Precision-Recall Curves
# ──────────────────────────────────────────────────────────────────────────────
def plot_precision_recall_curves(ml_results: Dict[str, Any]) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    for i, m in enumerate(ml_results["metrics"]):
        f1 = m["f1"]
        ax.plot(m["rec_curve"], m["prec_curve"], lw=2, color=colors[i],
                label=f"{m['name']} (F1={f1:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — All Models", fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    return _save(fig, "20_precision_recall_curves")


# ──────────────────────────────────────────────────────────────────────────────
# 21. CLTV Distribution by Contract
# ──────────────────────────────────────────────────────────────────────────────
def plot_cltv_by_contract(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    contracts = df["Contract"].unique()
    data = [df[df["Contract"] == c]["CLTV"].dropna() for c in contracts]
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops={"color": "white", "linewidth": 2})
    ax.set_xticks(list(range(1, len(contracts) + 1)))
    ax.set_xticklabels(contracts)
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Customer Lifetime Value ($)")
    ax.set_title("CLTV Distribution by Contract Type", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    return _save(fig, "21_cltv_by_contract")


# ──────────────────────────────────────────────────────────────────────────────
# 22. Churn by Tenure Group (stacked bar)
# ──────────────────────────────────────────────────────────────────────────────
def plot_churn_by_tenure_group(df: pd.DataFrame) -> Path:
    order = ["0-12m", "13-24m", "25-48m", "49-72m", "72m+"]
    grp = (
        df.groupby(["tenure_group", "Churn Label"])
        .size()
        .unstack(fill_value=0)
        .reindex(order)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    grp.plot(kind="bar", stacked=True, ax=ax,
             color=[CHURN_COLORS["No"], CHURN_COLORS["Yes"]], edgecolor="white", rot=0)
    ax.set_xlabel("Tenure Group")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Churn Distribution by Tenure Group", fontweight="bold")
    ax.legend(title="Churn")
    return _save(fig, "22_churn_by_tenure_group")


# ──────────────────────────────────────────────────────────────────────────────
# 23. Revenue Concentration (Pareto-style)
# ──────────────────────────────────────────────────────────────────────────────
def plot_revenue_concentration(df: pd.DataFrame) -> Path:
    df_sorted = df.sort_values("Total Charges", ascending=False).reset_index(drop=True)
    df_sorted["cumulative_pct"] = df_sorted["Total Charges"].cumsum() / df_sorted["Total Charges"].sum() * 100
    df_sorted["customer_pct"] = (df_sorted.index + 1) / len(df_sorted) * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(df_sorted["customer_pct"], df_sorted["cumulative_pct"],
                    alpha=0.3, color="#3498db")
    ax.plot(df_sorted["customer_pct"], df_sorted["cumulative_pct"],
            color="#2c3e50", lw=2)
    ax.axvline(20, color="#e74c3c", linestyle="--", lw=1.5, label="Top 20% customers")
    top20_rev = df_sorted[df_sorted["customer_pct"] <= 20]["cumulative_pct"].iloc[-1]
    ax.axhline(top20_rev, color="#e74c3c", linestyle="--", lw=1.5,
               label=f"→ {top20_rev:.1f}% of revenue")
    ax.set_xlabel("Cumulative % of Customers")
    ax.set_ylabel("Cumulative % of Revenue")
    ax.set_title("Revenue Concentration Curve (Pareto)", fontweight="bold")
    ax.legend()
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    return _save(fig, "23_revenue_concentration")


# ──────────────────────────────────────────────────────────────────────────────
# Master runner
# ──────────────────────────────────────────────────────────────────────────────
def generate_all_visualizations(
    df: pd.DataFrame,
    eda_results: Dict[str, Any],
    ml_results: Dict[str, Any],
) -> list:
    """Run all 23 visualization functions and return list of saved paths."""
    generated = []
    plots = [
        lambda: plot_churn_distribution(df),
        lambda: plot_monthly_charges_distribution(df),
        lambda: plot_tenure_distribution(df),
        lambda: plot_churn_by_contract(df),
        lambda: plot_churn_by_payment_method(df),
        lambda: plot_churn_by_internet_service(df),
        lambda: plot_monthly_charges_boxplot(df),
        lambda: plot_tenure_boxplot(df),
        lambda: plot_correlation_heatmap(eda_results),
        lambda: plot_customer_segment_pie(df),
        lambda: plot_revenue_by_segment(df),
        lambda: plot_churn_by_gender(df),
        lambda: plot_churn_by_senior(df),
        lambda: plot_service_count_distribution(df),
        lambda: plot_risk_score_distribution(eda_results),
        lambda: plot_churn_reasons(eda_results),
        lambda: plot_roc_curves(ml_results),
        lambda: plot_feature_importance(ml_results),
        lambda: plot_confusion_matrix(ml_results),
        lambda: plot_precision_recall_curves(ml_results),
        lambda: plot_cltv_by_contract(df),
        lambda: plot_churn_by_tenure_group(df),
        lambda: plot_revenue_concentration(df),
    ]

    for fn in plots:
        try:
            path = fn()
            generated.append(path)
        except Exception as exc:
            logger.error("Chart generation failed: %s", exc)

    logger.info("Generated %d / %d charts", len(generated), len(plots))
    return generated
