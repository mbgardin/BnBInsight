#!/usr/bin/env python3
"""
Run the full analysis pipeline: summary stats, regression, hypothesis test, plots.

Usage:
    python scripts/run_analysis.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from bnbinsight.analysis import (
    summary_statistics,
    correlation_matrix,
    run_linear_regression,
    run_simple_regression,
    evaluate_model,
    hypothesis_test_summary,
    find_over_underpriced_listings,
)
from bnbinsight.features import select_model_features
from bnbinsight.visualization import (
    plot_price_distribution,
    plot_price_vs_bedrooms,
    plot_price_vs_rating,
    plot_feature_importance_from_coefficients,
    plot_residuals,
)
from bnbinsight.utils import ensure_directory

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_listings.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")


def main():
    # Load
    print(f"Loading processed data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"  Loaded {len(df)} rows\n")

    # Summary
    print("=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    stats = summary_statistics(df)
    print(stats.to_string())
    print()

    # Correlations
    print("=" * 60)
    print("CORRELATION MATRIX")
    print("=" * 60)
    corr = correlation_matrix(df)
    print(corr.to_string())
    print()

    # Regression (primary)
    model_df = select_model_features(df)
    print(f"Model-ready rows: {len(model_df)}\n")

    try:
        model = run_linear_regression(model_df)
        print("=" * 60)
        print("PRIMARY MODEL: log_price ~ bedrooms + rating + amenities_count")
        print("=" * 60)
    except Exception as e:
        print(f"Primary model failed ({e}). Falling back to simple regression.")
        model = run_simple_regression(model_df)
        print("=" * 60)
        print("FALLBACK MODEL: price ~ bedrooms")
        print("=" * 60)

    print(model.summary())
    print()

    # Hypothesis test verdict
    print("=" * 60)
    print("HYPOTHESIS TEST RESULTS")
    print("=" * 60)
    ht = hypothesis_test_summary(model)
    print(f"\n  VERDICT: {ht['overall_verdict'].upper()}")
    print(f"  {ht['overall_explanation']}")
    print(f"\n  R² = {ht['r_squared']:.4f}  |  Adj R² = {ht['adj_r_squared']:.4f}")
    print(f"  N  = {ht['n_observations']:,}")
    print()
    for p in ht["predictors"]:
        sig = "✓ SIG" if p["significant"] else "✗ n.s."
        print(f"  {p['name']:20s}  coef={p['coefficient']:+.4f}  "
              f"p={p['pvalue']:.2e}  [{sig}]")
        print(f"    → {p['interpretation']}")
    print()

    # Over/under-priced
    flagged = find_over_underpriced_listings(model_df, model)
    flagged_path = os.path.join(OUTPUT_DIR, "flagged_listings.csv")
    flagged.to_csv(flagged_path, index=False)
    print(f"Saved flagged listings → {flagged_path}")
    print(f"  Overpriced:  {(flagged['price_flag']=='overpriced').sum()}")
    print(f"  Underpriced: {(flagged['price_flag']=='underpriced').sum()}")
    print()

    # Plots
    ensure_directory(PLOTS_DIR)
    print("Generating plots...")
    plot_price_distribution(df, save_path=os.path.join(PLOTS_DIR, "price_distribution.png"))
    plot_price_vs_bedrooms(df, save_path=os.path.join(PLOTS_DIR, "price_vs_bedrooms.png"))

    if "rating" in df.columns:
        plot_price_vs_rating(df, save_path=os.path.join(PLOTS_DIR, "price_vs_rating.png"))

    plot_feature_importance_from_coefficients(
        model, save_path=os.path.join(PLOTS_DIR, "coefficients.png")
    )
    plot_residuals(model_df, model, save_path=os.path.join(PLOTS_DIR, "residuals.png"))

    print("\nDone! All outputs saved.")


if __name__ == "__main__":
    main()
