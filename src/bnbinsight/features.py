"""
Feature engineering module — create derived columns for analysis.
"""

import re
import numpy as np
import pandas as pd


def parse_amenities_count(df: pd.DataFrame, amenities_col: str = "amenities") -> pd.DataFrame:
    """
    Count the number of amenities from a comma-separated or list-like string.
    Adds an 'amenities_count' column.
    """
    df = df.copy()
    if amenities_col not in df.columns:
        print("Warning: amenities column not found — setting amenities_count = 0.")
        df["amenities_count"] = 0
        return df

    def _count(val):
        if pd.isna(val):
            return 0
        val = str(val)
        # Strip surrounding brackets / braces / quotes
        val = re.sub(r'^[\[\{\"\' ]+|[\]\}\"\' ]+$', "", val)
        if not val:
            return 0
        # Split on commas
        items = [i.strip() for i in val.split(",") if i.strip()]
        return len(items)

    df["amenities_count"] = df[amenities_col].apply(_count)
    return df


def create_log_price(df: pd.DataFrame, price_col: str = "price") -> pd.DataFrame:
    """
    Ensure both price and log_price columns exist.

    - If price exists but log_price doesn't → compute log_price = ln(price).
    - If log_price exists but price doesn't → compute price = exp(log_price).
    - If both exist → no-op.
    - If neither exists → raise ValueError.
    """
    df = df.copy()

    has_price = price_col in df.columns
    has_log = "log_price" in df.columns

    if has_price and not has_log:
        df["log_price"] = np.where(df[price_col] > 0, np.log(df[price_col]), np.nan)
    elif has_log and not has_price:
        df["price"] = np.exp(df["log_price"])
    elif not has_price and not has_log:
        raise ValueError("Need either 'price' or 'log_price' column.")
    # else: both exist, nothing to do

    return df


def create_price_per_bedroom(df: pd.DataFrame) -> pd.DataFrame:
    """Add price_per_bedroom (price / bedrooms).  Bedrooms == 0 → NaN."""
    df = df.copy()
    if "price" in df.columns and "bedrooms" in df.columns:
        df["price_per_bedroom"] = np.where(
            df["bedrooms"] > 0,
            df["price"] / df["bedrooms"],
            np.nan,
        )
    return df


def bin_rating_levels(df: pd.DataFrame, rating_col: str = "rating") -> pd.DataFrame:
    """Bin ratings into Low / Medium / High categorical levels."""
    df = df.copy()
    if rating_col not in df.columns:
        return df
    df["rating_level"] = pd.cut(
        df[rating_col],
        bins=[0, 3.0, 4.0, 5.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )
    return df


def select_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a clean subset with only the columns needed for regression.
    Drops rows with any NaN in the model columns.
    """
    model_cols = ["price", "log_price", "bedrooms", "rating", "amenities_count"]
    available = [c for c in model_cols if c in df.columns]
    return df[available].dropna()
