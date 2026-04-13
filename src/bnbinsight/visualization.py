"""
Visualization module — clean, publication-ready plots for the analysis.
"""
from __future__ import annotations

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from bnbinsight.utils import ensure_directory

# Global style tweaks
plt.rcParams.update({
    "figure.figsize": (8, 5),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


def _save_or_show(fig, save_path: str | None):
    """Helper: save figure to disk or display it."""
    if save_path:
        ensure_directory(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved → {save_path}")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Distribution
# ---------------------------------------------------------------------------

def plot_price_distribution(df: pd.DataFrame, save_path: str | None = None):
    """Histogram of nightly prices."""
    fig, ax = plt.subplots()
    ax.hist(df["price"].dropna(), bins=50, color="#4c72b0", edgecolor="white")
    ax.set_title("Distribution of Nightly Prices")
    ax.set_xlabel("Price ($)")
    ax.set_ylabel("Count")
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Price vs. features
# ---------------------------------------------------------------------------

def plot_price_vs_bedrooms(df: pd.DataFrame, save_path: str | None = None):
    """Box plot of price by bedroom count."""
    fig, ax = plt.subplots()
    bedroom_vals = sorted(df["bedrooms"].dropna().unique())
    data_groups = [
        df.loc[df["bedrooms"] == b, "price"].dropna()
        for b in bedroom_vals
    ]
    ax.boxplot(data_groups, labels=[str(int(b)) for b in bedroom_vals], patch_artist=True,
               boxprops=dict(facecolor="#4c72b0", alpha=0.7))
    ax.set_title("Price by Number of Bedrooms")
    ax.set_xlabel("Bedrooms")
    ax.set_ylabel("Price ($)")
    _save_or_show(fig, save_path)


def plot_price_vs_rating(df: pd.DataFrame, save_path: str | None = None):
    """Scatter plot of price vs. rating."""
    fig, ax = plt.subplots()
    sample = df.dropna(subset=["price", "rating"])
    if len(sample) > 2000:
        sample = sample.sample(2000, random_state=42)
    ax.scatter(sample["rating"], sample["price"], alpha=0.3, s=10, color="#dd8452")
    ax.set_title("Price vs. Rating")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Price ($)")
    _save_or_show(fig, save_path)


def plot_avg_price_by_neighborhood(
    df: pd.DataFrame, top_n: int = 10, save_path: str | None = None
):
    """Horizontal bar chart of average price for the top-N neighbourhoods."""
    neigh_col = None
    for col in ["neighbourhood", "neighborhood", "neighbourhood_cleansed"]:
        if col in df.columns:
            neigh_col = col
            break
    if neigh_col is None:
        print("No neighbourhood column found — skipping plot.")
        return

    avg = (
        df.groupby(neigh_col)["price"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
    )
    fig, ax = plt.subplots()
    avg.sort_values().plot.barh(ax=ax, color="#55a868")
    ax.set_title(f"Top {top_n} Neighbourhoods by Avg Price")
    ax.set_xlabel("Avg Price ($)")
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Model-related
# ---------------------------------------------------------------------------

def plot_feature_importance_from_coefficients(model, save_path: str | None = None):
    """Bar chart of OLS regression coefficients (excluding intercept)."""
    coefs = model.params.drop("Intercept", errors="ignore")
    fig, ax = plt.subplots()
    coefs.plot.barh(ax=ax, color="#8172b3")
    ax.set_title("OLS Regression Coefficients")
    ax.set_xlabel("Coefficient Value")
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    _save_or_show(fig, save_path)


def plot_residuals(df: pd.DataFrame, model, save_path: str | None = None):
    """Scatter plot of predicted vs. residuals."""
    dep_var = model.model.endog_names
    pred = model.predict(df)
    resid = df[dep_var] - pred

    fig, ax = plt.subplots()
    ax.scatter(pred, resid, alpha=0.3, s=10, color="#c44e52")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_title("Residuals vs. Predicted")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    _save_or_show(fig, save_path)
