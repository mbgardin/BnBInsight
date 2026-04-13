"""Tests for the cleaning module."""

import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bnbinsight.cleaning import (
    clean_price_column,
    clean_bedrooms_column,
    filter_reasonable_prices,
)


def test_clean_price_removes_dollar_and_commas():
    df = pd.DataFrame({"price": ["$1,200.00", "$50", "  $99.99 "]})
    result = clean_price_column(df)
    assert result["price"].tolist() == [1200.0, 50.0, 99.99]


def test_clean_bedrooms_numeric():
    df = pd.DataFrame({"bedrooms": ["3", "2", "abc", None]})
    result = clean_bedrooms_column(df)
    assert result["bedrooms"].iloc[0] == 3.0
    assert pd.isna(result["bedrooms"].iloc[2])


def test_filter_reasonable_prices():
    df = pd.DataFrame({"price": [10, 50, 500, 5000]})
    result = filter_reasonable_prices(df)
    assert len(result) == 2  # 50 and 500


if __name__ == "__main__":
    test_clean_price_removes_dollar_and_commas()
    test_clean_bedrooms_numeric()
    test_filter_reasonable_prices()
    print("All cleaning tests passed!")
