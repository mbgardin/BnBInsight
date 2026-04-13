"""Tests for the features module."""

import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bnbinsight.features import parse_amenities_count, create_log_price


def test_amenities_count_comma_separated():
    df = pd.DataFrame({"amenities": ['{"WiFi", "Kitchen", "Pool"}']})
    result = parse_amenities_count(df)
    assert result["amenities_count"].iloc[0] == 3


def test_amenities_count_missing():
    df = pd.DataFrame({"amenities": [None]})
    result = parse_amenities_count(df)
    assert result["amenities_count"].iloc[0] == 0


def test_log_price():
    import numpy as np
    df = pd.DataFrame({"price": [100.0, 200.0]})
    result = create_log_price(df)
    assert abs(result["log_price"].iloc[0] - np.log(100)) < 1e-6


if __name__ == "__main__":
    test_amenities_count_comma_separated()
    test_amenities_count_missing()
    test_log_price()
    print("All features tests passed!")
