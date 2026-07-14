#!/usr/bin/env python3
"""
Customer Churn Analytics & Prediction Platform
===============================================
Entry point. Run:

    python main.py

Executes the full end-to-end pipeline:
    1. Load data
    2. Clean & validate
    3. Feature engineering
    4. Store in SQLite
    5. Run 30 SQL queries
    6. Exploratory Data Analysis
    7. Train ML models
    8. Generate 23 visualizations
    9. Export reports
"""

import logging
import sys
import time
from pathlib import Path

# ── Bootstrap logging ─────────────────────────────────────────────────────────
LOG_FILE = Path(__file__).parent / "output" / "pipeline.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")


def run_pipeline() -> None:
    """Execute the complete analytics pipeline."""
    start = time.perf_counter()

    logger.info("=" * 60)
    logger.info("  CUSTOMER CHURN ANALYTICS PIPELINE STARTING")
    logger.info("=" * 60)

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    logger.info("[1/9] Loading data ...")
    from src.data_loader import load_data
    raw_df = load_data()
    logger.info("      Loaded %d rows × %d columns", *raw_df.shape)

    # ── Step 2: Clean ─────────────────────────────────────────────────────────
    logger.info("[2/9] Cleaning & validating data ...")
    from src.data_cleaner import clean_data, get_data_quality_report
    df_clean, clean_report = clean_data(raw_df)
    quality_df = get_data_quality_report(df_clean)
    logger.info("      Clean report: %s", clean_report)

    # ── Step 3: Feature Engineering ───────────────────────────────────────────
    logger.info("[3/9] Engineering features ...")
    from src.feature_engineering import engineer_features
    df = engineer_features(df_clean)
    logger.info("      DataFrame shape after engineering: %s", df.shape)

    # ── Step 4: Database ──────────────────────────────────────────────────────
    logger.info("[4/9] Loading SQLite database ...")
    from src.database import setup_database
    conn = setup_database(df)

    # ── Step 5: SQL Queries ───────────────────────────────────────────────────
    logger.info("[5/9] Running 30 SQL analytics queries ...")
    from src.queries import run_all_queries
    query_results = run_all_queries(conn)
    logger.info("      %d queries executed, CSVs saved to output/csv/", len(query_results))

    # ── Step 6: EDA ───────────────────────────────────────────────────────────
    logger.info("[6/9] Running Exploratory Data Analysis ...")
    from src.eda import run_eda
    eda_results = run_eda(df)
    logger.info("      EDA complete: %d result sets", len(eda_results))

    # ── Step 7: Machine Learning ──────────────────────────────────────────────
    logger.info("[7/9] Training Machine Learning models ...")
    from src.model_training import train_models
    ml_results = train_models(df)
    logger.info("      Best model: %s", ml_results["best_name"])

    # ── Step 8: Visualizations ────────────────────────────────────────────────
    logger.info("[8/9] Generating visualizations ...")
    from src.visualizations import generate_all_visualizations
    charts = generate_all_visualizations(df, eda_results, ml_results)
    logger.info("      %d charts saved to output/plots/", len(charts))

    # ── Step 9: Export Reports ────────────────────────────────────────────────
    logger.info("[9/9] Exporting reports ...")
    from src.exporter import run_all_exports
    exported = run_all_exports(df, quality_df, clean_report, eda_results, ml_results, query_results)
    for name, path in exported.items():
        logger.info("      %s → %s", name, path)

    conn.close()

    elapsed = time.perf_counter() - start
    logger.info("")
    logger.info("=" * 60)
    logger.info("  PIPELINE COMPLETE in %.1f seconds", elapsed)
    logger.info("=" * 60)
    logger.info("")
    logger.info("Outputs:")
    logger.info("  output/plots/      — 23 charts")
    logger.info("  output/csv/        — 30 SQL query results")
    logger.info("  output/reports/    — executive summary, business report, CSVs")
    logger.info("  output/churn_analytics.db — SQLite database")
    logger.info("  models/churn_model.pkl    — best trained model")
    logger.info("")
    logger.info("To predict a single customer:   python predict.py")


if __name__ == "__main__":
    run_pipeline()
