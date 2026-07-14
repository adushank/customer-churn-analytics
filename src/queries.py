"""30 Business SQL Analytics Queries module.

Each query is executed against the SQLite database and results are saved to CSV.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)

CSV_OUT = Path(__file__).parent.parent / "output" / "csv"


QUERIES: Dict[str, str] = {

    # ── 1. Customer KPIs ──────────────────────────────────────────────────────
    "01_customer_kpis": """
        SELECT
            COUNT(*)                                               AS total_customers,
            SUM(churn_value)                                       AS churned_customers,
            COUNT(*) - SUM(churn_value)                           AS retained_customers,
            ROUND(AVG(churn_value) * 100, 2)                      AS overall_churn_rate_pct,
            ROUND(AVG(monthly_charges), 2)                        AS avg_monthly_charges,
            ROUND(SUM(total_charges), 2)                          AS total_revenue,
            ROUND(AVG(cltv), 2)                                   AS avg_cltv,
            ROUND(AVG(tenure_months), 1)                          AS avg_tenure_months
        FROM customers;
    """,

    # ── 2. Churn Rate by Contract Type ───────────────────────────────────────
    "02_churn_by_contract": """
        SELECT
            contract,
            COUNT(*)                                     AS total,
            SUM(churn_value)                             AS churned,
            ROUND(AVG(churn_value) * 100, 2)            AS churn_rate_pct,
            ROUND(AVG(monthly_charges), 2)              AS avg_monthly_charges
        FROM customers
        GROUP BY contract
        ORDER BY churn_rate_pct DESC;
    """,

    # ── 3. Churn Rate by Payment Method ──────────────────────────────────────
    "03_churn_by_payment_method": """
        SELECT
            payment_method,
            COUNT(*)                                     AS total,
            SUM(churn_value)                             AS churned,
            ROUND(AVG(churn_value) * 100, 2)            AS churn_rate_pct
        FROM customers
        GROUP BY payment_method
        ORDER BY churn_rate_pct DESC;
    """,

    # ── 4. Churn Rate by Internet Service ─────────────────────────────────────
    "04_churn_by_internet_service": """
        SELECT
            internet_service,
            COUNT(*)                                     AS total,
            SUM(churn_value)                             AS churned,
            ROUND(AVG(churn_value) * 100, 2)            AS churn_rate_pct,
            ROUND(AVG(monthly_charges), 2)              AS avg_monthly_charges
        FROM customers
        GROUP BY internet_service
        ORDER BY churn_rate_pct DESC;
    """,

    # ── 5. Churn Rate by Gender ───────────────────────────────────────────────
    "05_churn_by_gender": """
        SELECT
            gender,
            COUNT(*)                                     AS total,
            SUM(churn_value)                             AS churned,
            ROUND(AVG(churn_value) * 100, 2)            AS churn_rate_pct
        FROM customers
        GROUP BY gender
        ORDER BY churn_rate_pct DESC;
    """,

    # ── 6. Churn Rate by Senior Citizen ──────────────────────────────────────
    "06_churn_by_senior_citizen": """
        SELECT
            senior_citizen,
            COUNT(*)                                     AS total,
            SUM(churn_value)                             AS churned,
            ROUND(AVG(churn_value) * 100, 2)            AS churn_rate_pct,
            ROUND(AVG(monthly_charges), 2)              AS avg_monthly_charges
        FROM customers
        GROUP BY senior_citizen
        ORDER BY churn_rate_pct DESC;
    """,

    # ── 7. Revenue Analysis by Customer Segment ───────────────────────────────
    "07_revenue_by_segment": """
        SELECT
            customer_segment,
            COUNT(*)                                     AS customers,
            ROUND(SUM(monthly_charges), 2)              AS total_monthly_revenue,
            ROUND(AVG(monthly_charges), 2)              AS avg_monthly_revenue,
            ROUND(SUM(total_charges), 2)                AS total_lifetime_revenue,
            ROUND(AVG(cltv), 2)                         AS avg_cltv,
            ROUND(AVG(churn_value) * 100, 2)            AS churn_rate_pct
        FROM customers
        GROUP BY customer_segment
        ORDER BY total_monthly_revenue DESC;
    """,

    # ── 8. Revenue Lost to Churn ──────────────────────────────────────────────
    "08_revenue_lost_to_churn": """
        SELECT
            contract,
            customer_segment,
            COUNT(*)                                     AS churned_customers,
            ROUND(SUM(monthly_charges), 2)              AS monthly_revenue_lost,
            ROUND(SUM(total_charges), 2)                AS total_revenue_lost,
            ROUND(AVG(cltv), 2)                         AS avg_cltv_lost
        FROM customers
        WHERE churn_value = 1
        GROUP BY contract, customer_segment
        ORDER BY monthly_revenue_lost DESC;
    """,

    # ── 9. Top 20 Highest-Value Customers ─────────────────────────────────────
    "09_top_customers": """
        SELECT
            customer_id,
            city,
            contract,
            tenure_months,
            monthly_charges,
            total_charges,
            cltv,
            churn_label,
            risk_score
        FROM customers
        ORDER BY cltv DESC
        LIMIT 20;
    """,

    # ── 10. High-Risk Customers (likely to churn, not yet churned) ─────────────
    "10_high_risk_customers": """
        SELECT
            customer_id,
            city,
            contract,
            tenure_months,
            monthly_charges,
            cltv,
            risk_score,
            churn_reason
        FROM customers
        WHERE churn_value = 0 AND risk_score >= 70
        ORDER BY risk_score DESC
        LIMIT 50;
    """,

    # ── 11. Churn Reason Breakdown ───────────────────────────────────────────
    "11_churn_reason_breakdown": """
        SELECT
            churn_reason,
            COUNT(*)                                     AS occurrences,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS share_pct
        FROM customers
        WHERE churn_value = 1
        GROUP BY churn_reason
        ORDER BY occurrences DESC;
    """,

    # ── 12. Tenure Distribution ──────────────────────────────────────────────
    "12_tenure_distribution": """
        SELECT
            tenure_group,
            COUNT(*)                                     AS total,
            SUM(churn_value)                             AS churned,
            ROUND(AVG(churn_value) * 100, 2)            AS churn_rate_pct,
            ROUND(AVG(monthly_charges), 2)              AS avg_monthly_charges
        FROM customers
        GROUP BY tenure_group
        ORDER BY MIN(tenure_months);
    """,

    # ── 13. City-Level Churn Analysis (top 15 cities) ─────────────────────────
    "13_churn_by_city": """
        SELECT
            city,
            COUNT(*)                                     AS total_customers,
            SUM(churn_value)                             AS churned,
            ROUND(AVG(churn_value) * 100, 2)            AS churn_rate_pct,
            ROUND(SUM(monthly_charges), 2)              AS total_monthly_revenue
        FROM customers
        GROUP BY city
        HAVING COUNT(*) >= 10
        ORDER BY churn_rate_pct DESC
        LIMIT 15;
    """,

    # ── 14. Service Adoption vs Churn ─────────────────────────────────────────
    "14_service_count_vs_churn": """
        SELECT
            service_count,
            COUNT(*)                                     AS total,
            SUM(churn_value)                             AS churned,
            ROUND(AVG(churn_value) * 100, 2)            AS churn_rate_pct,
            ROUND(AVG(monthly_charges), 2)              AS avg_monthly_charges
        FROM customers
        GROUP BY service_count
        ORDER BY service_count;
    """,

    # ── 15. Partner & Dependents Impact ──────────────────────────────────────
    "15_partner_dependents_churn": """
        SELECT
            partner,
            dependents,
            COUNT(*)                                     AS total,
            SUM(churn_value)                             AS churned,
            ROUND(AVG(churn_value) * 100, 2)            AS churn_rate_pct
        FROM customers
        GROUP BY partner, dependents
        ORDER BY churn_rate_pct DESC;
    """,

    # ── 16. Monthly Revenue by Tenure Group (Running Total via Window) ─────────
    "16_running_revenue_by_tenure": """
        SELECT
            tenure_group,
            ROUND(SUM(monthly_charges), 2)              AS segment_revenue,
            ROUND(SUM(SUM(monthly_charges)) OVER (
                ORDER BY MIN(tenure_months)
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ), 2)                                        AS running_total_revenue
        FROM customers
        GROUP BY tenure_group
        ORDER BY MIN(tenure_months);
    """,

    # ── 17. RANK: Customers by CLTV within Segment ───────────────────────────
    "17_cltv_rank_within_segment": """
        SELECT
            customer_id,
            customer_segment,
            cltv,
            monthly_charges,
            churn_label,
            RANK()       OVER (PARTITION BY customer_segment ORDER BY cltv DESC) AS rank_in_segment,
            DENSE_RANK() OVER (PARTITION BY customer_segment ORDER BY cltv DESC) AS dense_rank,
            ROW_NUMBER() OVER (PARTITION BY customer_segment ORDER BY cltv DESC) AS row_num
        FROM customers
        ORDER BY customer_segment, rank_in_segment
        LIMIT 60;
    """,

    # ── 18. Moving Average of Monthly Charges by Tenure ───────────────────────
    "18_moving_avg_charges": """
        WITH tenure_agg AS (
            SELECT
                tenure_months,
                ROUND(AVG(monthly_charges), 2) AS avg_charges,
                COUNT(*)                        AS customer_count
            FROM customers
            GROUP BY tenure_months
        )
        SELECT
            tenure_months,
            avg_charges,
            customer_count,
            ROUND(AVG(avg_charges) OVER (
                ORDER BY tenure_months
                ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING
            ), 2) AS moving_avg_5m
        FROM tenure_agg
        ORDER BY tenure_months;
    """,

    # ── 19. LAG / LEAD: Churn Score Progression by Tenure ────────────────────
    "19_lag_lead_churn_score": """
        WITH tenure_score AS (
            SELECT
                tenure_months,
                ROUND(AVG(risk_score), 2)  AS avg_risk_score,
                ROUND(AVG(churn_value) * 100, 2) AS churn_rate_pct
            FROM customers
            GROUP BY tenure_months
        )
        SELECT
            tenure_months,
            avg_risk_score,
            churn_rate_pct,
            LAG(avg_risk_score)  OVER (ORDER BY tenure_months) AS prev_risk_score,
            LEAD(avg_risk_score) OVER (ORDER BY tenure_months) AS next_risk_score,
            avg_risk_score - LAG(avg_risk_score) OVER (ORDER BY tenure_months) AS risk_delta
        FROM tenure_score
        ORDER BY tenure_months;
    """,

    # ── 20. CASE: Customer Risk Tier ──────────────────────────────────────────
    "20_customer_risk_tiers": """
        SELECT
            CASE
                WHEN risk_score >= 80 THEN 'Critical'
                WHEN risk_score >= 60 THEN 'High'
                WHEN risk_score >= 40 THEN 'Medium'
                ELSE 'Low'
            END                                          AS risk_tier,
            COUNT(*)                                     AS customer_count,
            SUM(churn_value)                             AS actual_churned,
            ROUND(AVG(churn_value) * 100, 2)            AS churn_rate_pct,
            ROUND(AVG(monthly_charges), 2)              AS avg_monthly_charges,
            ROUND(SUM(monthly_charges), 2)              AS total_at_risk_revenue
        FROM customers
        GROUP BY risk_tier
        ORDER BY MIN(risk_score) DESC;
    """,

    # ── 21. CTE: High-Value Churned Customers ─────────────────────────────────
    "21_high_value_churned_cte": """
        WITH high_value AS (
            SELECT * FROM customers WHERE high_value_flag = 1
        ),
        churned_hv AS (
            SELECT * FROM high_value WHERE churn_value = 1
        )
        SELECT
            contract,
            payment_method,
            COUNT(*)                                     AS churned_high_value,
            ROUND(AVG(monthly_charges), 2)              AS avg_monthly_charges,
            ROUND(SUM(monthly_charges), 2)              AS revenue_lost,
            ROUND(AVG(cltv), 2)                         AS avg_cltv
        FROM churned_hv
        GROUP BY contract, payment_method
        ORDER BY revenue_lost DESC;
    """,

    # ── 22. Subquery: Customers Above Average CLTV ────────────────────────────
    "22_above_avg_cltv": """
        SELECT
            customer_id,
            city,
            contract,
            cltv,
            monthly_charges,
            churn_label,
            risk_score
        FROM customers
        WHERE cltv > (SELECT AVG(cltv) FROM customers)
        ORDER BY cltv DESC
        LIMIT 30;
    """,

    # ── 23. JOIN Self: Compare Churned vs Retained per Segment ───────────────
    "23_churned_vs_retained_segment": """
        SELECT
            c.customer_segment,
            c.contract,
            COUNT(CASE WHEN c.churn_value = 1 THEN 1 END)  AS churned,
            COUNT(CASE WHEN c.churn_value = 0 THEN 1 END)  AS retained,
            ROUND(AVG(CASE WHEN c.churn_value = 1 THEN c.monthly_charges END), 2) AS avg_charges_churned,
            ROUND(AVG(CASE WHEN c.churn_value = 0 THEN c.monthly_charges END), 2) AS avg_charges_retained
        FROM customers c
        GROUP BY c.customer_segment, c.contract
        ORDER BY c.customer_segment, c.contract;
    """,

    # ── 24. Digital Customers Analysis ────────────────────────────────────────
    "24_digital_customers": """
        SELECT
            digital_customer,
            COUNT(*)                                     AS total,
            SUM(churn_value)                             AS churned,
            ROUND(AVG(churn_value) * 100, 2)            AS churn_rate_pct,
            ROUND(AVG(monthly_charges), 2)              AS avg_monthly_charges
        FROM customers
        GROUP BY digital_customer;
    """,

    # ── 25. Long-Term Customers ───────────────────────────────────────────────
    "25_long_term_customers": """
        SELECT
            long_term_flag,
            COUNT(*)                                     AS total,
            SUM(churn_value)                             AS churned,
            ROUND(AVG(churn_value) * 100, 2)            AS churn_rate_pct,
            ROUND(AVG(cltv), 2)                         AS avg_cltv,
            ROUND(SUM(monthly_charges), 2)              AS total_monthly_revenue
        FROM customers
        GROUP BY long_term_flag;
    """,

    # ── 26. Quarterly Churn Trend (using tenure as proxy) ─────────────────────
    "26_quarterly_churn_trend": """
        SELECT
            CASE
                WHEN tenure_months BETWEEN 1  AND 3  THEN 'Q1 (1-3m)'
                WHEN tenure_months BETWEEN 4  AND 6  THEN 'Q2 (4-6m)'
                WHEN tenure_months BETWEEN 7  AND 9  THEN 'Q3 (7-9m)'
                WHEN tenure_months BETWEEN 10 AND 12 THEN 'Q4 (10-12m)'
                WHEN tenure_months BETWEEN 13 AND 24 THEN 'Year 2'
                WHEN tenure_months BETWEEN 25 AND 48 THEN 'Years 3-4'
                ELSE 'Year 5+'
            END                                          AS tenure_quarter,
            MIN(tenure_months)                           AS min_tenure,
            COUNT(*)                                     AS total,
            SUM(churn_value)                             AS churned,
            ROUND(AVG(churn_value) * 100, 2)            AS churn_rate_pct,
            ROUND(AVG(monthly_charges), 2)              AS avg_monthly_charges
        FROM customers
        GROUP BY tenure_quarter
        ORDER BY min_tenure;
    """,

    # ── 27. Year-over-Year Retention Proxy ────────────────────────────────────
    "27_yoy_retention": """
        WITH yearly AS (
            SELECT
                CASE
                    WHEN tenure_months <= 12 THEN 'Year 1'
                    WHEN tenure_months <= 24 THEN 'Year 2'
                    WHEN tenure_months <= 36 THEN 'Year 3'
                    WHEN tenure_months <= 48 THEN 'Year 4'
                    ELSE 'Year 5+'
                END                          AS cohort_year,
                MIN(tenure_months)           AS sort_key,
                COUNT(*)                     AS total,
                SUM(churn_value)             AS churned,
                ROUND(AVG(churn_value)*100,2)AS churn_rate_pct,
                ROUND(AVG(cltv), 2)          AS avg_cltv
            FROM customers
            GROUP BY cohort_year
        )
        SELECT
            cohort_year,
            total,
            churned,
            churn_rate_pct,
            100 - churn_rate_pct             AS retention_rate_pct,
            avg_cltv,
            SUM(total) OVER (ORDER BY sort_key) AS cumulative_customers
        FROM yearly
        ORDER BY sort_key;
    """,

    # ── 28. Revenue Concentration (Pareto) ────────────────────────────────────
    "28_revenue_concentration": """
        WITH ranked AS (
            SELECT
                customer_id,
                total_charges,
                NTILE(10) OVER (ORDER BY total_charges DESC) AS revenue_decile
            FROM customers
        )
        SELECT
            revenue_decile,
            COUNT(*)                                         AS customers,
            ROUND(SUM(total_charges), 2)                    AS total_revenue,
            ROUND(SUM(total_charges) * 100.0 /
                  SUM(SUM(total_charges)) OVER (), 2)       AS pct_of_total_revenue,
            ROUND(SUM(SUM(total_charges)) OVER (
                ORDER BY revenue_decile
            ), 2)                                           AS cumulative_revenue
        FROM ranked
        GROUP BY revenue_decile
        ORDER BY revenue_decile;
    """,

    # ── 29. Internet + Phone Service Combinations ─────────────────────────────
    "29_service_combo_analysis": """
        SELECT
            has_internet,
            has_phone,
            internet_service,
            COUNT(*)                                     AS total,
            SUM(churn_value)                             AS churned,
            ROUND(AVG(churn_value) * 100, 2)            AS churn_rate_pct,
            ROUND(AVG(monthly_charges), 2)              AS avg_monthly_charges,
            ROUND(SUM(monthly_charges), 2)              AS total_monthly_revenue
        FROM customers
        GROUP BY has_internet, has_phone, internet_service
        ORDER BY total_monthly_revenue DESC;
    """,

    # ── 30. Executive Churn Summary Dashboard ────────────────────────────────
    "30_executive_summary": """
        SELECT
            'Total Customers'               AS metric, CAST(COUNT(*) AS TEXT) AS value
        FROM customers
        UNION ALL
        SELECT 'Churned Customers',          CAST(SUM(churn_value) AS TEXT)
        FROM customers
        UNION ALL
        SELECT 'Churn Rate (%)',             CAST(ROUND(AVG(churn_value)*100,2) AS TEXT)
        FROM customers
        UNION ALL
        SELECT 'Avg Monthly Charges ($)',    CAST(ROUND(AVG(monthly_charges),2) AS TEXT)
        FROM customers
        UNION ALL
        SELECT 'Total Revenue ($)',          CAST(ROUND(SUM(total_charges),2) AS TEXT)
        FROM customers
        UNION ALL
        SELECT 'Avg CLTV ($)',              CAST(ROUND(AVG(cltv),2) AS TEXT)
        FROM customers
        UNION ALL
        SELECT 'Avg Tenure (months)',        CAST(ROUND(AVG(tenure_months),1) AS TEXT)
        FROM customers
        UNION ALL
        SELECT 'High-Value Customers',       CAST(SUM(high_value_flag) AS TEXT)
        FROM customers
        UNION ALL
        SELECT 'High-Risk (not yet churned)',CAST(COUNT(*) AS TEXT)
        FROM customers WHERE churn_value=0 AND risk_score>=70
        UNION ALL
        SELECT 'Revenue Lost to Churn ($)', CAST(ROUND(SUM(monthly_charges),2) AS TEXT)
        FROM customers WHERE churn_value=1;
    """,
}


def run_all_queries(
    conn: sqlite3.Connection,
    output_dir: Path = CSV_OUT,
) -> Dict[str, pd.DataFrame]:
    """Execute all 30 queries and export results to CSV.

    Args:
        conn: Active SQLite connection.
        output_dir: Directory for CSV exports.

    Returns:
        Dict mapping query name → result DataFrame.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, pd.DataFrame] = {}

    for name, sql in QUERIES.items():
        try:
            df = pd.read_sql_query(sql, conn)
            results[name] = df
            csv_path = output_dir / f"{name}.csv"
            df.to_csv(csv_path, index=False)
            logger.info("Query %s → %d rows → %s", name, len(df), csv_path.name)
        except Exception as exc:
            logger.error("Query %s failed: %s", name, exc)

    logger.info("Executed %d / %d queries", len(results), len(QUERIES))
    return results
