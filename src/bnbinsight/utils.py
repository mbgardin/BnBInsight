"""
Utility helper functions used across the bnbinsight package.
"""

import os
import pandas as pd


def ensure_directory(path: str) -> None:
    """Create directory (and parents) if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def safe_numeric_conversion(series: pd.Series) -> pd.Series:
    """Convert a Series to numeric, coercing errors to NaN."""
    return pd.to_numeric(series, errors="coerce")


def validate_required_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    """Raise ValueError if any required columns are missing from the DataFrame."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
