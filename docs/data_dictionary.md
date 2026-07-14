# Data Dictionary — Customer Churn Analytics

## Source Columns (from `data/customers.csv`)

| Column | Type | Description |
|---|---|---|
| `CustomerID` | str | Unique customer identifier |
| `Count` | int | Row count (always 1) |
| `Country` | str | Customer country |
| `State` | str | US state |
| `City` | str | City name |
| `Zip Code` | int | US ZIP code |
| `Lat Long` | str | Latitude, longitude string |
| `Latitude` | float | Geographic latitude |
| `Longitude` | float | Geographic longitude |
| `Gender` | str | Male / Female |
| `Senior Citizen` | str | Yes / No |
| `Partner` | str | Has a partner: Yes / No |
| `Dependents` | str | Has dependents: Yes / No |
| `Tenure Months` | int | Months as a customer (0–72) |
| `Phone Service` | str | Has phone: Yes / No |
| `Multiple Lines` | str | Multiple phone lines: Yes / No / No phone service |
| `Internet Service` | str | DSL / Fiber optic / No |
| `Online Security` | str | Yes / No / No internet service |
| `Online Backup` | str | Yes / No / No internet service |
| `Device Protection` | str | Yes / No / No internet service |
| `Tech Support` | str | Yes / No / No internet service |
| `Streaming TV` | str | Yes / No / No internet service |
| `Streaming Movies` | str | Yes / No / No internet service |
| `Contract` | str | Month-to-month / One year / Two year |
| `Paperless Billing` | str | Yes / No |
| `Payment Method` | str | Electronic check / Mailed check / Bank transfer (automatic) / Credit card (automatic) |
| `Monthly Charges` | float | Current monthly bill ($) |
| `Total Charges` | float | Total amount charged to date ($) |
| `Churn Label` | str | Yes (churned) / No (retained) |
| `Churn Value` | int | 1 = churned, 0 = retained |
| `Churn Score` | int | Propensity score 0–100 (higher = more likely to churn) |
| `CLTV` | int | Customer Lifetime Value ($) |
| `Churn Reason` | str | Reason for churning (NaN for retained customers) |

## Engineered Features

| Feature | Type | Description |
|---|---|---|
| `contract_length_months` | int | Contract length in months: 1 / 12 / 24 |
| `tenure_group` | str | Tenure bucket: 0-12m / 13-24m / 25-48m / 49-72m / 72m+ |
| `avg_monthly_spend` | float | Alias for Monthly Charges |
| `total_charges_calc` | float | Monthly Charges × Tenure Months |
| `avg_revenue_per_month` | float | Total Charges ÷ Tenure Months |
| `clv_calc` | float | Monthly Charges × contract_length_months |
| `service_count` | int | Number of add-on services (0–8) |
| `has_internet` | int | 1 if Internet Service ≠ No |
| `has_phone` | int | 1 if Phone Service = Yes |
| `risk_score` | float | Normalised churn propensity (= Churn Score) |
| `customer_segment` | str | Budget / Standard / Premium (by Monthly Charges tertiles) |
| `high_value_flag` | int | 1 if CLTV ≥ 75th percentile |
| `long_term_flag` | int | 1 if Tenure Months ≥ 24 |
| `age_group` | str | Senior / Non-Senior |
| `digital_customer` | int | 1 if Paperless Billing = Yes AND electronic check payment |

## ML Feature Encodings

| Encoded Column | Source | Encoding |
|---|---|---|
| `gender_enc` | Gender | 1 = Male, 0 = Female |
| `senior_enc` | Senior Citizen | 1 = Yes, 0 = No |
| `partner_enc` | Partner | 1 = Yes, 0 = No |
| `dependents_enc` | Dependents | 1 = Yes, 0 = No |
| `contract_enc` | Contract | 0 = Month-to-month, 1 = One year, 2 = Two year |
| `internet_enc` | Internet Service | 0 = No, 1 = DSL, 2 = Fiber optic |
| `payment_enc` | Payment Method | Label-encoded 0–3 |
| `segment_enc` | customer_segment | 0 = Budget, 1 = Standard, 2 = Premium |
