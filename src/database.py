"""SQLite database module.

Creates and populates the churn analytics database with:
    - Primary keys
    - Indexes
    - Views
    - Constraints
    - Bulk insert
"""

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "output" / "churn_analytics.db"

DDL_CUSTOMERS = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id              TEXT    PRIMARY KEY,
    country                  TEXT    NOT NULL DEFAULT 'United States',
    state                    TEXT,
    city                     TEXT,
    zip_code                 INTEGER,
    latitude                 REAL,
    longitude                REAL,
    gender                   TEXT    CHECK(gender IN ('Male','Female','Unknown')),
    senior_citizen           TEXT,
    partner                  TEXT,
    dependents               TEXT,
    tenure_months            INTEGER CHECK(tenure_months >= 0),
    phone_service            TEXT,
    multiple_lines           TEXT,
    internet_service         TEXT,
    online_security          TEXT,
    online_backup            TEXT,
    device_protection        TEXT,
    tech_support             TEXT,
    streaming_tv             TEXT,
    streaming_movies         TEXT,
    contract                 TEXT    CHECK(contract IN ('Month-to-month','One year','Two year')),
    paperless_billing        TEXT,
    payment_method           TEXT,
    monthly_charges          REAL    CHECK(monthly_charges >= 0),
    total_charges            REAL    CHECK(total_charges >= 0),
    churn_label              TEXT    CHECK(churn_label IN ('Yes','No')),
    churn_value              INTEGER CHECK(churn_value IN (0,1)),
    churn_score              INTEGER,
    cltv                     INTEGER,
    churn_reason             TEXT,
    -- Engineered features
    contract_length_months   INTEGER,
    tenure_group             TEXT,
    avg_monthly_spend        REAL,
    total_charges_calc       REAL,
    avg_revenue_per_month    REAL,
    clv_calc                 REAL,
    service_count            INTEGER,
    has_internet             INTEGER,
    has_phone                INTEGER,
    risk_score               REAL,
    customer_segment         TEXT,
    high_value_flag          INTEGER,
    long_term_flag           INTEGER,
    age_group                TEXT,
    digital_customer         INTEGER
);
"""

DDL_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_churn_value   ON customers(churn_value);",
    "CREATE INDEX IF NOT EXISTS idx_contract       ON customers(contract);",
    "CREATE INDEX IF NOT EXISTS idx_segment        ON customers(customer_segment);",
    "CREATE INDEX IF NOT EXISTS idx_tenure         ON customers(tenure_months);",
    "CREATE INDEX IF NOT EXISTS idx_monthly_chg    ON customers(monthly_charges);",
    "CREATE INDEX IF NOT EXISTS idx_city           ON customers(city);",
    "CREATE INDEX IF NOT EXISTS idx_state          ON customers(state);",
    "CREATE INDEX IF NOT EXISTS idx_payment        ON customers(payment_method);",
    "CREATE INDEX IF NOT EXISTS idx_risk_score     ON customers(risk_score);",
]

DDL_VIEWS = [
    """
    CREATE VIEW IF NOT EXISTS vw_churned_customers AS
    SELECT * FROM customers WHERE churn_value = 1;
    """,
    """
    CREATE VIEW IF NOT EXISTS vw_retained_customers AS
    SELECT * FROM customers WHERE churn_value = 0;
    """,
    """
    CREATE VIEW IF NOT EXISTS vw_high_risk AS
    SELECT * FROM customers WHERE risk_score >= 70 AND churn_value = 0;
    """,
    """
    CREATE VIEW IF NOT EXISTS vw_revenue_summary AS
    SELECT
        customer_segment,
        contract,
        COUNT(*) AS customer_count,
        ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges,
        ROUND(SUM(total_charges), 2)  AS total_revenue,
        ROUND(AVG(cltv), 2)            AS avg_cltv
    FROM customers
    GROUP BY customer_segment, contract;
    """,
]

# Mapping of DataFrame columns → DB columns
COL_MAP = {
    "CustomerID": "customer_id",
    "Country": "country",
    "State": "state",
    "City": "city",
    "Zip Code": "zip_code",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "Gender": "gender",
    "Senior Citizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "Tenure Months": "tenure_months",
    "Phone Service": "phone_service",
    "Multiple Lines": "multiple_lines",
    "Internet Service": "internet_service",
    "Online Security": "online_security",
    "Online Backup": "online_backup",
    "Device Protection": "device_protection",
    "Tech Support": "tech_support",
    "Streaming TV": "streaming_tv",
    "Streaming Movies": "streaming_movies",
    "Contract": "contract",
    "Paperless Billing": "paperless_billing",
    "Payment Method": "payment_method",
    "Monthly Charges": "monthly_charges",
    "Total Charges": "total_charges",
    "Churn Label": "churn_label",
    "Churn Value": "churn_value",
    "Churn Score": "churn_score",
    "CLTV": "cltv",
    "Churn Reason": "churn_reason",
    # Engineered
    "contract_length_months": "contract_length_months",
    "tenure_group": "tenure_group",
    "avg_monthly_spend": "avg_monthly_spend",
    "total_charges_calc": "total_charges_calc",
    "avg_revenue_per_month": "avg_revenue_per_month",
    "clv_calc": "clv_calc",
    "service_count": "service_count",
    "has_internet": "has_internet",
    "has_phone": "has_phone",
    "risk_score": "risk_score",
    "customer_segment": "customer_segment",
    "high_value_flag": "high_value_flag",
    "long_term_flag": "long_term_flag",
    "age_group": "age_group",
    "digital_customer": "digital_customer",
}


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Return a SQLite connection with WAL mode and foreign keys enabled."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.row_factory = sqlite3.Row
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    """Create tables, indexes, and views."""
    cursor = conn.cursor()
    cursor.execute(DDL_CUSTOMERS)
    for idx in DDL_INDEXES:
        cursor.execute(idx)
    for view in DDL_VIEWS:
        cursor.execute(view)
    conn.commit()
    logger.info("Database schema created / verified")


def bulk_insert(df: pd.DataFrame, conn: sqlite3.Connection, batch_size: int = 1000) -> int:
    """Bulk-insert DataFrame rows into the customers table.

    Args:
        df: Feature-engineered DataFrame.
        conn: Active SQLite connection.
        batch_size: Rows per INSERT batch.

    Returns:
        Number of rows inserted.
    """
    # Select and rename only the columns we need
    available_src = [c for c in COL_MAP if c in df.columns]
    db_df = df[available_src].rename(columns=COL_MAP)

    # Drop rows with null primary key
    db_df = db_df.dropna(subset=["customer_id"])

    cols = list(db_df.columns)
    placeholders = ", ".join(["?"] * len(cols))
    col_names = ", ".join(cols)
    sql = f"INSERT OR REPLACE INTO customers ({col_names}) VALUES ({placeholders})"

    inserted = 0
    cursor = conn.cursor()
    for start in range(0, len(db_df), batch_size):
        batch = db_df.iloc[start : start + batch_size]
        records = [tuple(row) for row in batch.itertuples(index=False, name=None)]
        cursor.executemany(sql, records)
        inserted += len(records)

    conn.commit()
    logger.info("Bulk inserted %d rows", inserted)
    return inserted


def setup_database(df: pd.DataFrame, db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Full database setup: create schema and load data.

    Args:
        df: Feature-engineered DataFrame.
        db_path: Path to the SQLite file.

    Returns:
        Open SQLite connection.
    """
    if db_path.exists():
        db_path.unlink()
        logger.info("Removed existing database for fresh load")

    conn = get_connection(db_path)
    create_schema(conn)
    n = bulk_insert(df, conn)
    logger.info("Database ready at %s with %d rows", db_path, n)
    return conn
