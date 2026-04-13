#!/usr/bin/env python3
"""
Clean raw data from Kaggle AND AirROI, combine into one dataset.

Usage:
    python scripts/clean_data.py

Place files in data/raw/:
  - kaggle_airbnb.csv   (required)
  - airroi_raw.json     (optional — merged if present)
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bnbinsight.data_collection import load_kaggle_data, load_airroi_json
from bnbinsight.cleaning import (
    clean_price_column,
    clean_rating_column,
    clean_bedrooms_column,
    clean_bathrooms_column,
    clean_beds_column,
    clean_reviews_column,
    drop_duplicates_and_nulls,
    filter_reasonable_prices,
)
from bnbinsight.features import (
    parse_amenities_count,
    create_log_price,
    create_price_per_bedroom,
)
from bnbinsight.utils import ensure_directory

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DEFAULT_CSV = os.path.join(BASE_DIR, "data", "raw", "kaggle_airbnb.csv")
AIRROI_JSON = os.path.join(BASE_DIR, "data", "raw", "airroi_raw.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_listings.csv")


def _standardize_airroi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map AirROI column names to match the Kaggle schema so the two
    datasets can be concatenated.  Missing columns get NaN.
    """
    # AirROI json_normalize produces dotted names like
    # "listing_info.listing_id", "property_details.bedrooms", etc.
    # We map them to our standard column names.
    rename_map = {
        # Identity
        "listing_info.listing_id": "id",
        "listing_info.listing_name": "name",
        "listing_info.listing_type": "property_type",
        "listing_info.room_type": "room_type",
        "listing_info.description": "description",
        # Property
        "property_details.bedrooms": "bedrooms",
        "property_details.beds": "beds",
        "property_details.baths": "bathrooms",
        "property_details.guests": "accommodates",
        "property_details.amenities": "amenities",
        # Location
        "location_info.latitude": "latitude",
        "location_info.longitude": "longitude",
        "location_info.locality": "city",
        "location_info.district": "neighbourhood",
        # Ratings
        "ratings.rating_overall": "rating",
        "ratings.num_reviews": "review_count",
        # Price — use TTM average daily rate as nightly price
        "performance_metrics.ttm_avg_rate": "price",
        # Booking
        "booking_settings.cancellation_policy": "cancellation_policy",
        "booking_settings.instant_book": "instant_bookable",
        "booking_settings.min_nights": "min_nights",
        "pricing_info.cleaning_fee": "cleaning_fee",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


def main():
    # ------------------------------------------------------------------
    # 1. Load Kaggle data
    # ------------------------------------------------------------------
    csv_path = os.environ.get("KAGGLE_CSV", DEFAULT_CSV)
    print(f"Loading Kaggle data from: {csv_path}")
    kaggle_df = load_kaggle_data(csv_path)
    kaggle_df["source"] = "kaggle"
    kaggle_df = create_log_price(kaggle_df)  # derive price from log_price
    print(f"  Kaggle rows: {len(kaggle_df)}")

    # ------------------------------------------------------------------
    # 2. Load AirROI data (if saved offline)
    # ------------------------------------------------------------------
    combined = kaggle_df
    if os.path.exists(AIRROI_JSON):
        print(f"Loading AirROI data from: {AIRROI_JSON}")
        api_df = load_airroi_json(AIRROI_JSON)
        api_df = _standardize_airroi(api_df)
        api_df["source"] = "airroi"
        api_df = create_log_price(api_df)  # derive log_price from price
        print(f"  AirROI rows: {len(api_df)}")

        # Concatenate on shared columns (union of both)
        combined = pd.concat([kaggle_df, api_df], ignore_index=True, sort=False)
        print(f"  Combined rows: {len(combined)}")
    else:
        print("  No AirROI data found — using Kaggle only.")
        print(f"  (Run 'python scripts/fetch_data.py' first to fetch API data)")

    df = combined

    # ------------------------------------------------------------------
    # 4. Cleaning pipeline
    # ------------------------------------------------------------------
    df = clean_price_column(df)
    df = clean_rating_column(df)
    df = clean_bedrooms_column(df)
    df = clean_bathrooms_column(df)
    df = clean_beds_column(df)
    df = clean_reviews_column(df)
    df = drop_duplicates_and_nulls(df)
    df = filter_reasonable_prices(df)
    print(f"  After cleaning: {len(df)} rows")

    # ------------------------------------------------------------------
    # 5. Feature engineering
    # ------------------------------------------------------------------
    df = parse_amenities_count(df)
    df = create_price_per_bedroom(df)

    # ------------------------------------------------------------------
    # 6. Keep only analysis-relevant columns
    # ------------------------------------------------------------------
    keep_cols = [
        "id", "name", "price", "log_price",
        "bedrooms", "beds", "bathrooms", "accommodates",
        "rating", "review_count", "amenities", "amenities_count",
        "room_type", "property_type", "city", "neighbourhood",
        "latitude", "longitude",
        "cleaning_fee", "instant_bookable", "cancellation_policy",
        "source", "price_per_bedroom",
    ]
    available = [c for c in keep_cols if c in df.columns]
    df = df[available]

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    ensure_directory(os.path.dirname(OUTPUT_PATH))
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved cleaned data → {OUTPUT_PATH}")
    print(f"  Final columns: {list(df.columns)}")

    # Quick source breakdown
    if "source" in df.columns:
        print(f"  Source breakdown:")
        for src, count in df["source"].value_counts().items():
            print(f"    {src}: {count}")


if __name__ == "__main__":
    main()
