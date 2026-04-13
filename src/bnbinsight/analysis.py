"""
Analysis module — descriptive statistics, OLS regression, hypothesis testing,
and residual analysis.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------

def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return count, mean, median, std, min, max for key numeric columns.
    """
    numeric_cols = ["price", "log_price", "bedrooms", "rating", "amenities_count",
                    "bathrooms", "beds", "review_count"]
    available = [c for c in numeric_cols if c in df.columns]
    stats = df[available].agg(["count", "mean", "median", "std", "min", "max"])
    return stats


def correlation_matrix(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """
    Pearson correlation matrix for selected columns.
    Defaults to key analysis columns if none specified.
    """
    if cols is None:
        cols = ["price", "log_price", "bedrooms", "rating", "amenities_count"]
    available = [c for c in cols if c in df.columns]
    return df[available].corr()


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

DEFAULT_FORMULA = "log_price ~ bedrooms + rating + amenities_count"


def run_linear_regression(df: pd.DataFrame, formula: str | None = None):
    """
    Fit an OLS regression using statsmodels.

    Default formula:  log_price ~ bedrooms + rating + amenities_count

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the columns referenced in the formula.
    formula : str | None
        Patsy-style formula string.

    Returns
    -------
    statsmodels RegressionResultsWrapper
    """
    if formula is None:
        formula = DEFAULT_FORMULA

    clean = df.dropna(subset=_formula_vars(formula))
    model = smf.ols(formula, data=clean).fit()
    return model


def run_simple_regression(df: pd.DataFrame):
    """
    Fallback: price ~ bedrooms
    """
    formula = "price ~ bedrooms"
    clean = df.dropna(subset=["price", "bedrooms"])
    model = smf.ols(formula, data=clean).fit()
    return model


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model) -> dict:
    """
    Extract key regression diagnostics.

    Returns
    -------
    dict with keys: r_squared, adj_r_squared, coefficients, p_values, f_pvalue
    """
    return {
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "coefficients": model.params.to_dict(),
        "p_values": model.pvalues.to_dict(),
        "f_pvalue": model.f_pvalue,
    }


# ---------------------------------------------------------------------------
# Hypothesis testing summary (plain-English output)
# ---------------------------------------------------------------------------

def hypothesis_test_summary(model, alpha: float = 0.05) -> dict:
    """
    Produce a layman-friendly summary of the regression hypothesis test.

    Returns
    -------
    dict with keys:
        overall_verdict : str  — "reject" or "fail to reject"
        overall_explanation : str
        f_pvalue : float
        predictors : list[dict]
            Each dict has: name, coefficient, pvalue, significant, interpretation
    """
    dep_var = model.model.endog_names
    is_log = "log" in dep_var.lower()

    predictors = []
    for name in model.params.index:
        if name == "Intercept":
            continue
        coef = model.params[name]
        pval = model.pvalues[name]
        sig = pval < alpha

        # Build plain-English interpretation
        if is_log:
            # log model: coefficient ≈ percentage change
            pct_change = (np.exp(coef) - 1) * 100
            if "bedroom" in name.lower():
                interp = (
                    f"Each additional bedroom is associated with a "
                    f"{abs(pct_change):.1f}% {'increase' if pct_change > 0 else 'decrease'} "
                    f"in nightly price."
                )
            elif "rating" in name.lower():
                interp = (
                    f"A one-point increase in rating is associated with a "
                    f"{abs(pct_change):.1f}% {'increase' if pct_change > 0 else 'decrease'} "
                    f"in nightly price."
                )
            elif "ameniti" in name.lower():
                interp = (
                    f"Each additional amenity is associated with a "
                    f"{abs(pct_change):.1f}% {'increase' if pct_change > 0 else 'decrease'} "
                    f"in nightly price."
                )
            else:
                direction = "increases" if coef > 0 else "decreases"
                interp = (
                    f"A one-unit increase in {name} {direction} "
                    f"nightly price by ~{abs(pct_change):.1f}%."
                )
        else:
            direction = "increases" if coef > 0 else "decreases"
            interp = (
                f"A one-unit increase in {name} {direction} "
                f"price by ${abs(coef):.2f}."
            )

        predictors.append({
            "name": name,
            "coefficient": coef,
            "pvalue": pval,
            "significant": sig,
            "interpretation": interp,
        })

    # Overall model verdict
    f_pval = model.f_pvalue
    reject = f_pval < alpha
    sig_names = [p["name"] for p in predictors if p["significant"]]

    if reject and sig_names:
        verdict = "reject"
        explanation = (
            f"We reject H₀ (F-test p = {f_pval:.2e}). "
            f"{', '.join(sig_names)} significantly predict nightly price "
            f"(all p < {alpha})."
        )
    elif reject:
        verdict = "reject"
        explanation = (
            f"We reject H₀ overall (F-test p = {f_pval:.2e}), "
            f"but no individual predictor reaches significance at α = {alpha}."
        )
    else:
        verdict = "fail to reject"
        explanation = (
            f"We fail to reject H₀ (F-test p = {f_pval:.2e}). "
            f"There is not enough evidence that bedrooms, rating, and amenities "
            f"predict nightly price."
        )

    return {
        "overall_verdict": verdict,
        "overall_explanation": explanation,
        "f_pvalue": f_pval,
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "n_observations": int(model.nobs),
        "predictors": predictors,
    }


def predict_price(model, bedrooms: float, rating: float, amenities_count: float) -> float:
    """
    Predict the nightly price given feature values.

    Handles both log-price and raw-price models.

    Returns
    -------
    float — predicted nightly price in dollars
    """
    new_data = pd.DataFrame({
        "bedrooms": [bedrooms],
        "rating": [rating],
        "amenities_count": [amenities_count],
    })
    pred = model.predict(new_data)[0]

    # If the model predicts log_price, convert back to dollars
    dep_var = model.model.endog_names
    if "log" in dep_var.lower():
        return np.exp(pred)
    return pred


# ---------------------------------------------------------------------------
# Over / under-priced detection
# ---------------------------------------------------------------------------

def find_over_underpriced_listings(df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Add predicted price and residual columns to the data.
    Flag the top 10% as overpriced and bottom 10% as underpriced.

    Works with both log-price and raw-price models.
    """
    df = df.copy()
    dep_var = model.model.endog_names  # e.g. "log_price" or "price"

    # Predict on available rows
    pred = model.predict(df)
    df["predicted"] = pred
    df["residual"] = df[dep_var] - df["predicted"]

    # Flag extremes
    low_q = df["residual"].quantile(0.10)
    high_q = df["residual"].quantile(0.90)
    df["price_flag"] = "normal"
    df.loc[df["residual"] <= low_q, "price_flag"] = "underpriced"
    df.loc[df["residual"] >= high_q, "price_flag"] = "overpriced"

    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _formula_vars(formula: str) -> list[str]:
    """Extract variable names from a patsy formula string."""
    import re
    # Remove "~" and split on "+"
    parts = re.split(r"[~+]", formula)
    return [p.strip() for p in parts if p.strip()]
