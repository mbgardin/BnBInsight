"""
Data collection module — load Kaggle CSV and fetch AirROI API data.
"""
from __future__ import annotations

import os
import json
import pandas as pd
import requests

from bnbinsight.utils import ensure_directory


# ---------------------------------------------------------------------------
# Kaggle data helpers
# ---------------------------------------------------------------------------

def load_kaggle_data(filepath: str) -> pd.DataFrame:
    """
    Load a Kaggle Airbnb CSV into a DataFrame.

    - Validates that the file exists
    - Standardises column names to lowercase snake_case

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Kaggle data file not found: {filepath}")

    df = pd.read_csv(filepath)
    # Standardise column names
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return df


# ---------------------------------------------------------------------------
# AirROI API helpers
# ---------------------------------------------------------------------------

AIRROI_BASE = "https://api.airroi.com"


def fetch_airroi_data(
    city: str,
    api_key: str | None = None,
    save_path: str | None = None,
    page_size: int = 200,
) -> pd.DataFrame | None:
    """
    Fetch short-term-rental listing data from the AirROI API.

    Flow:
      1. GET /markets/search?query=<city>  → get market location fields
      2. POST /listings/search/market       → pull listings using market object

    Parameters
    ----------
    city : str
        City name to search (e.g. "Salt Lake City").
    api_key : str | None
        AirROI API key.  Falls back to the AIRROI_API_KEY env var.
    save_path : str | None
        If provided, save the raw JSON response to this path.
    page_size : int
        Number of listings to request (default 200).

    Returns
    -------
    pd.DataFrame | None
        DataFrame of listing-level data, or None on failure.
    """
    api_key = api_key or os.environ.get("AIRROI_API_KEY")
    if not api_key:
        print("Warning: No AirROI API key provided. Skipping API fetch.")
        return None

    headers = {"x-api-key": api_key, "Content-Type": "application/json"}

    # Step 1 — resolve city name to a market object
    try:
        market_resp = requests.get(
            f"{AIRROI_BASE}/markets/search",
            headers=headers,
            params={"query": city},
            timeout=30,
        )
        market_resp.raise_for_status()
        market_data = market_resp.json()
    except requests.RequestException as exc:
        print(f"AirROI market lookup failed: {exc}")
        return None

    # Extract the first matching market entry
    entries = market_data.get("entries", []) if isinstance(market_data, dict) else market_data
    if not entries:
        print(f"No markets found for '{city}'.")
        return None

    first = entries[0]
    market_obj = {
        "country": first.get("country", ""),
        "region": first.get("region", ""),
        "locality": first.get("locality", ""),
    }
    # Include district only if present
    if first.get("district"):
        market_obj["district"] = first["district"]

    print(f"  Market: {first.get('full_name', city)}")

    # Step 2 — search listings in that market (paginated, max 10 per page)
    all_results = []
    offset = 0
    max_listings = page_size  # total we want

    while offset < max_listings:
        try:
            body = {
                "market": market_obj,
                "pagination": {"page_size": 10, "offset": offset},
            }
            listings_resp = requests.post(
                f"{AIRROI_BASE}/listings/search/market",
                headers=headers,
                json=body,
                timeout=60,
            )
            listings_resp.raise_for_status()
            raw = listings_resp.json()
        except requests.RequestException as exc:
            if hasattr(exc, 'response') and exc.response is not None:
                print(f"AirROI listings search failed: {exc}")
                print(f"  Response body: {exc.response.text[:500]}")
            else:
                print(f"AirROI listings search failed: {exc}")
            break

        page_results = raw.get("results", []) if isinstance(raw, dict) else raw
        if not page_results:
            break

        all_results.extend(page_results)
        total_available = raw.get("pagination", {}).get("total_count", 0)
        offset += len(page_results)
        print(f"  Fetched {len(all_results)}/{total_available} listings...")

        # Stop if we've gotten everything available
        if offset >= total_available:
            break

    if not all_results:
        print("No listing records returned from AirROI.")
        return None

    # Save raw JSON if requested (so we don't have to call API again)
    save_data = {"results": all_results, "market": market_obj, "total_count": len(all_results)}
    if save_path:
        ensure_directory(os.path.dirname(save_path))
        with open(save_path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"Saved raw AirROI JSON → {save_path}")

    print(f"  Total: {len(all_results)} listings")
    return pd.json_normalize(all_results)


def load_airroi_json(filepath: str) -> pd.DataFrame:
    """
    Load a previously-saved AirROI JSON file into a DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the JSON file.

    Returns
    -------
    pd.DataFrame
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"AirROI JSON not found: {filepath}")

    with open(filepath) as f:
        raw = json.load(f)

    records = raw if isinstance(raw, list) else raw.get(
        "results", raw.get("listings", raw.get("data", [raw]))
    )
    return pd.json_normalize(records)


# ---------------------------------------------------------------------------
# Generic save helper
# ---------------------------------------------------------------------------

def save_raw_data(df: pd.DataFrame, filepath: str) -> None:
    """Save a DataFrame to CSV, creating parent directories as needed."""
    ensure_directory(os.path.dirname(filepath))
    df.to_csv(filepath, index=False)
    print(f"Saved raw data → {filepath}")
