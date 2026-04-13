"""Tests for the analysis module."""

import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bnbinsight.analysis import summary_statistics, run_simple_regression


def _make_sample_df(n=100):
    """Create a small synthetic listing DataFrame for testing."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "price": rng.uniform(50, 500, n),
        "bedrooms": rng.choice([1, 2, 3, 4], n),
        "rating": rng.uniform(3, 5, n),
        "amenities_count": rng.integers(5, 30, n),
        "log_price": np.log(rng.uniform(50, 500, n)),
    })


def test_summary_statistics_columns():
    df = _make_sample_df()
    stats = summary_statistics(df)
    assert "mean" in stats.index
    assert "price" in stats.columns


def test_simple_regression_returns_model():
    df = _make_sample_df()
    model = run_simple_regression(df)
    assert hasattr(model, "rsquared")
    assert hasattr(model, "params")


if __name__ == "__main__":
    test_summary_statistics_columns()
    test_simple_regression_returns_model()
    print("All analysis tests passed!")
