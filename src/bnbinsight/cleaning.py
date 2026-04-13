"""
Cleaning module — transform raw listing columns into usable numeric/categorical fields.
"""

import pandas as pd
from bnbinsight.utils import safe_numeric_conversion


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase and snake_case all column names."""
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return df


def clean_price_column(df: pd.DataFrame, price_col: str = "price") -> pd.DataFrame:
    """
    Clean a price column: strip $, commas, spaces → convert to float.
    Rows that cannot be parsed become NaN.
    """
    df = df.copy()
    if price_col not in df.columns:
        print(f"Warning: '{price_col}' column not found — skipping price cleaning.")
        return df

    df[price_col] = (
        df[price_col]
        .astype(str)
        .str.replace(r"[\$,\s]", "", regex=True)
    )
    df[price_col] = safe_numeric_conversion(df[price_col])
    return df


def clean_rating_column(df: pd.DataFrame, rating_col: str = "review_scores_rating") -> pd.DataFrame:
    """
    Convert rating to numeric.  If ratings are on a 0-100 scale, rescale to 0-5.
    Out-of-range values are set to NaN.
    """
    df = df.copy()
    # Try common column name variants
    if rating_col not in df.columns:
        for alt in ["rating", "review_scores_rating", "overall_rating"]:
            if alt in df.columns:
                rating_col = alt
                break
        else:
            print("Warning: No rating column found — skipping rating cleaning.")
            return df

    df[rating_col] = safe_numeric_conversion(df[rating_col])

    # Rescale 0-100 → 0-5 if needed
    if df[rating_col].max() > 5:
        df[rating_col] = df[rating_col] / 20.0

    # Clamp to valid range
    df.loc[df[rating_col] < 0, rating_col] = pd.NA
    df.loc[df[rating_col] > 5, rating_col] = pd.NA

    # Rename to standardised name
    if rating_col != "rating":
        df = df.rename(columns={rating_col: "rating"})

    return df


def clean_bedrooms_column(df: pd.DataFrame, bedrooms_col: str = "bedrooms") -> pd.DataFrame:
    """Extract numeric bedroom count; non-numeric → NaN."""
    df = df.copy()
    if bedrooms_col not in df.columns:
        print(f"Warning: '{bedrooms_col}' column not found — skipping.")
        return df
    df[bedrooms_col] = safe_numeric_conversion(df[bedrooms_col])
    return df


def clean_bathrooms_column(df: pd.DataFrame, bathrooms_col: str = "bathrooms") -> pd.DataFrame:
    """Extract numeric bathroom count."""
    df = df.copy()
    if bathrooms_col not in df.columns:
        # Try variant names
        for alt in ["bathrooms_text", "bathrooms"]:
            if alt in df.columns:
                bathrooms_col = alt
                break
        else:
            return df
    # Handle text like "1 bath" or "2 shared baths"
    df[bathrooms_col] = (
        df[bathrooms_col]
        .astype(str)
        .str.extract(r"(\d+\.?\d*)", expand=False)
    )
    df[bathrooms_col] = safe_numeric_conversion(df[bathrooms_col])
    if bathrooms_col != "bathrooms":
        df = df.rename(columns={bathrooms_col: "bathrooms"})
    return df


def clean_beds_column(df: pd.DataFrame, beds_col: str = "beds") -> pd.DataFrame:
    """Extract numeric bed count."""
    df = df.copy()
    if beds_col not in df.columns:
        return df
    df[beds_col] = safe_numeric_conversion(df[beds_col])
    return df


def clean_reviews_column(df: pd.DataFrame, reviews_col: str = "number_of_reviews") -> pd.DataFrame:
    """Ensure review count is numeric."""
    df = df.copy()
    if reviews_col not in df.columns:
        for alt in ["review_count", "reviews", "number_of_reviews"]:
            if alt in df.columns:
                reviews_col = alt
                break
        else:
            return df
    df[reviews_col] = safe_numeric_conversion(df[reviews_col])
    if reviews_col != "review_count":
        df = df.rename(columns={reviews_col: "review_count"})
    return df


def drop_duplicates_and_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows and rows where price is null.

    Handles columns containing lists (e.g. amenities from APIs)
    by converting them to strings before dedup.
    """
    df = df.copy()

    # Convert list-typed columns to strings so they're hashable for dedup
    list_cols = [c for c in df.columns if df[c].apply(type).eq(list).any()]
    for col in list_cols:
        df[col] = df[col].astype(str)

    df = df.drop_duplicates()
    if "price" in df.columns:
        df = df.dropna(subset=["price"])
    return df


def filter_reasonable_prices(
    df: pd.DataFrame,
    min_price: float = 20,
    max_price: float = 2000,
) -> pd.DataFrame:
    """Remove listings with prices outside a sensible range."""
    if "price" not in df.columns:
        return df
    return df[(df["price"] >= min_price) & (df["price"] <= max_price)].copy()
