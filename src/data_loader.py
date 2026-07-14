"""Data loader module: reads the customer churn CSV and returns a raw DataFrame."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent / "data" / "customers.csv"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load customer churn dataset from CSV.

    Args:
        path: Path to the CSV file.

    Returns:
        Raw DataFrame.

    Raises:
        FileNotFoundError: If the CSV does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    logger.info("Loading dataset from %s", path)
    df = pd.read_csv(path, low_memory=False)
    logger.info("Loaded %d rows × %d columns", df.shape[0], df.shape[1])
    return df
