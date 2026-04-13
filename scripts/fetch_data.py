#!/usr/bin/env python3
"""
Fetch raw data from the AirROI API and save to data/raw/.

Usage:
    python scripts/fetch_data.py

Set the AIRROI_API_KEY environment variable (or create a .env file)
before running.
"""

import os
import sys

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
from bnbinsight.data_collection import fetch_airroi_data

load_dotenv()  # loads .env if present

CITY = os.environ.get("AIRROI_CITY", "Salt Lake City")
MAX_LISTINGS = int(os.environ.get("AIRROI_MAX_LISTINGS", "5000"))
SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "airroi_raw.json")


def main():
    print(f"Fetching AirROI data for: {CITY} (max {MAX_LISTINGS} listings)")
    df = fetch_airroi_data(city=CITY, save_path=SAVE_PATH, page_size=MAX_LISTINGS)
    if df is not None:
        print(f"Fetched {len(df)} listings.")
        print(df.head())
    else:
        print("No data returned. Check your API key and network connection.")


if __name__ == "__main__":
    main()
