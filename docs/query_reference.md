# SQL Query Reference — 30 Business Analytics Queries

All queries run against the `customers` table in `output/churn_analytics.db`.
Results are exported to `output/csv/<query_name>.csv`.

| # | Query Name | Description | SQL Features |
|---|---|---|---|
| 01 | `01_customer_kpis` | Overall KPIs: churn rate, revenue, avg tenure | Aggregates |
| 02 | `02_churn_by_contract` | Churn rate per contract type | GROUP BY |
| 03 | `03_churn_by_payment_method` | Churn rate per payment method | GROUP BY |
| 04 | `04_churn_by_internet_service` | Churn and revenue by internet type | GROUP BY |
| 05 | `05_churn_by_gender` | Gender-level churn analysis | GROUP BY |
| 06 | `06_churn_by_senior_citizen` | Senior vs non-senior churn comparison | GROUP BY |
| 07 | `07_revenue_by_segment` | Revenue breakdown by customer segment | GROUP BY |
| 08 | `08_revenue_lost_to_churn` | Monthly and lifetime revenue lost | WHERE + GROUP BY |
| 09 | `09_top_customers` | Top 20 customers by CLTV | ORDER BY + LIMIT |
| 10 | `10_high_risk_customers` | Not-yet-churned customers with risk_score ≥ 70 | WHERE + ORDER BY |
| 11 | `11_churn_reason_breakdown` | Churn reason share (%) | Window `SUM OVER ()` |
| 12 | `12_tenure_distribution` | Churn rate across tenure groups | GROUP BY + ORDER BY |
| 13 | `13_churn_by_city` | Top 15 cities by churn rate | HAVING |
| 14 | `14_service_count_vs_churn` | Churn vs number of services | GROUP BY |
| 15 | `15_partner_dependents_churn` | Impact of partner & dependents on churn | Multi-column GROUP BY |
| 16 | `16_running_revenue_by_tenure` | Running total revenue by tenure group | `SUM OVER (ROWS BETWEEN ... AND ...)` |
| 17 | `17_cltv_rank_within_segment` | RANK, DENSE_RANK, ROW_NUMBER per segment | `RANK() OVER` / `DENSE_RANK()` / `ROW_NUMBER()` |
| 18 | `18_moving_avg_charges` | 5-month moving average of monthly charges | CTE + `AVG OVER (ROWS BETWEEN ...)` |
| 19 | `19_lag_lead_churn_score` | Risk score delta by tenure month | CTE + `LAG` + `LEAD` |
| 20 | `20_customer_risk_tiers` | CASE-based risk tier counts and revenue | `CASE WHEN` |
| 21 | `21_high_value_churned_cte` | High-value churned customers by contract/payment | Multi-CTE |
| 22 | `22_above_avg_cltv` | Customers above average CLTV | Scalar subquery |
| 23 | `23_churned_vs_retained_segment` | Segment × contract cross-tab | Conditional aggregation |
| 24 | `24_digital_customers` | Digital vs traditional customer churn | GROUP BY |
| 25 | `25_long_term_customers` | Long-term vs short-term churn | GROUP BY |
| 26 | `26_quarterly_churn_trend` | Churn rate by tenure quarter cohort | CASE + GROUP BY |
| 27 | `27_yoy_retention` | Year-over-year retention with running total | CTE + `SUM OVER` |
| 28 | `28_revenue_concentration` | Pareto: revenue concentration by decile | `NTILE` + `SUM OVER` |
| 29 | `29_service_combo_analysis` | Internet + phone combo revenue/churn | Multi-column GROUP BY |
| 30 | `30_executive_summary` | Dashboard KPI snapshot (UNION ALL) | `UNION ALL` |

## Database Schema

```sql
CREATE TABLE customers (
    customer_id              TEXT    PRIMARY KEY,
    -- ... 46 columns total (see data_dictionary.md)
);
```

### Views

| View | Description |
|---|---|
| `vw_churned_customers` | All churned customers |
| `vw_retained_customers` | All retained customers |
| `vw_high_risk` | Not-yet-churned with risk_score ≥ 70 |
| `vw_revenue_summary` | Revenue summary by segment × contract |

### Indexes

`idx_churn_value`, `idx_contract`, `idx_segment`, `idx_tenure`,
`idx_monthly_chg`, `idx_city`, `idx_state`, `idx_payment`, `idx_risk_score`
