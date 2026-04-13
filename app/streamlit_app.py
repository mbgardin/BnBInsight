"""
BnBInsight — Streamlit Dashboard

Interactive dashboard for exploring Airbnb pricing data and testing
the hypothesis that bedrooms, rating, and amenities predict price.

Launch:  streamlit run app/streamlit_app.py
"""

import os
import sys

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from bnbinsight.analysis import (
    summary_statistics,
    run_linear_regression,
    evaluate_model,
    hypothesis_test_summary,
    predict_price,
)
from bnbinsight.features import select_model_features

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="BnBInsight — Airbnb Pricing Analysis",
    page_icon="🏠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS for a polished look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 1.2rem;
        border-radius: 12px;
        color: white;
    }
    [data-testid="stMetric"] label {color: rgba(255,255,255,0.85) !important;}
    [data-testid="stMetric"] [data-testid="stMetricValue"] {color: white !important; font-weight: 700;}

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }

    /* Verdict boxes */
    .verdict-box {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    .verdict-reject {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 5px solid #28a745;
        color: #155724;
    }
    .verdict-fail {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
        border-left: 5px solid #ffc107;
        color: #856404;
    }

    /* Price estimator result */
    .price-result {
        font-size: 3.5rem;
        font-weight: 800;
        color: #667eea;
        text-align: center;
        padding: 1rem;
    }
    .price-label {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .percentile-bar {
        background: linear-gradient(90deg, #28a745 0%, #ffc107 50%, #dc3545 100%);
        height: 12px;
        border-radius: 6px;
        position: relative;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "cleaned_listings.csv")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, low_memory=False)


@st.cache_resource
def fit_model(data):
    """Fit the regression model once and cache it."""
    model_df = select_model_features(data)
    if len(model_df) < 10:
        return None, model_df
    try:
        model = run_linear_regression(model_df)
        return model, model_df
    except Exception:
        return None, model_df


try:
    df = load_data()
except FileNotFoundError:
    st.error("Cleaned dataset not found. Run `python scripts/clean_data.py` first.")
    st.stop()

model, model_df = fit_model(df)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🏠 BnBInsight — Airbnb Pricing Analysis")
st.markdown(
    "> **Hypothesis:** Listings with more bedrooms, higher ratings, "
    "and more amenities have significantly higher nightly prices."
)

# Key metrics row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Listings", f"{len(df):,}")
col2.metric("Avg Nightly Price", f"${df['price'].mean():.0f}")
col3.metric("Median Price", f"${df['price'].median():.0f}")
if "rating" in df.columns:
    col4.metric("Avg Rating", f"{df['rating'].dropna().mean():.2f} ★")

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Explore the Data", "🔬 Hypothesis Test", "🎛️ Price Estimator"])

# ===== TAB 1 — EXPLORE =====
with tab1:
    # Sidebar-like filters inside the tab
    st.subheader("Filter & Explore")
    fcol1, fcol2, fcol3 = st.columns(3)

    with fcol1:
        bed_min = int(df["bedrooms"].dropna().min())
        bed_max = min(int(df["bedrooms"].dropna().max()), 10)
        bed_range = st.slider("Bedrooms", bed_min, bed_max, (bed_min, bed_max))

    with fcol2:
        min_rating = 0.0
        if "rating" in df.columns:
            min_rating = st.slider("Minimum Rating", 0.0, 5.0, 0.0, 0.5)

    with fcol3:
        price_cap = st.slider("Max Price ($)", 50, 2000, 1000, 50)

    # Filter
    filtered = df[
        (df["bedrooms"] >= bed_range[0])
        & (df["bedrooms"] <= bed_range[1])
        & (df["price"] <= price_cap)
    ]
    if "rating" in df.columns:
        filtered = filtered[filtered["rating"].fillna(0) >= min_rating]

    st.caption(f"Showing **{len(filtered):,}** of {len(df):,} listings")

    # Charts
    chart_left, chart_right = st.columns(2)

    with chart_left:
        st.markdown("#### 💰 Price Distribution")
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        ax1.hist(
            filtered["price"].dropna(), bins=60,
            color="#667eea", edgecolor="white", alpha=0.85,
        )
        ax1.axvline(filtered["price"].median(), color="#ff6b6b", linestyle="--",
                    linewidth=2, label=f"Median: ${filtered['price'].median():.0f}")
        ax1.set_xlabel("Nightly Price ($)", fontsize=12)
        ax1.set_ylabel("Number of Listings", fontsize=12)
        ax1.legend(fontsize=11)
        ax1.xaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    with chart_right:
        st.markdown("#### 🛏️ Price by Bedrooms")
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        bedroom_vals = sorted(filtered["bedrooms"].dropna().unique())
        bedroom_vals = [b for b in bedroom_vals if b <= 8]
        groups = [filtered.loc[filtered["bedrooms"] == b, "price"].dropna() for b in bedroom_vals]
        if groups:
            bp = ax2.boxplot(
                groups, labels=[str(int(b)) for b in bedroom_vals],
                patch_artist=True, showfliers=False,
                medianprops=dict(color="#ff6b6b", linewidth=2),
            )
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(groups)))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.8)
        ax2.set_xlabel("Bedrooms", fontsize=12)
        ax2.set_ylabel("Nightly Price ($)", fontsize=12)
        ax2.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # Second row of charts
    chart_left2, chart_right2 = st.columns(2)

    with chart_left2:
        if "rating" in filtered.columns:
            st.markdown("#### ⭐ Price vs Rating")
            fig3, ax3 = plt.subplots(figsize=(7, 4))
            rated = filtered.dropna(subset=["rating", "price"])
            ax3.scatter(rated["rating"], rated["price"], alpha=0.15,
                        s=8, color="#764ba2")
            # Add trend line
            if len(rated) > 10:
                z = np.polyfit(rated["rating"], rated["price"], 1)
                x_line = np.linspace(rated["rating"].min(), rated["rating"].max(), 100)
                ax3.plot(x_line, np.polyval(z, x_line), color="#ff6b6b",
                         linewidth=2, label=f"Trend (slope={z[0]:.1f})")
                ax3.legend(fontsize=11)
            ax3.set_xlabel("Rating (1–5 stars)", fontsize=12)
            ax3.set_ylabel("Nightly Price ($)", fontsize=12)
            ax3.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)

    with chart_right2:
        st.markdown("#### 🧳 Price vs Amenities Count")
        fig4, ax4 = plt.subplots(figsize=(7, 4))
        amenity_data = filtered.dropna(subset=["amenities_count", "price"])
        ax4.scatter(amenity_data["amenities_count"], amenity_data["price"],
                    alpha=0.15, s=8, color="#28a745")
        if len(amenity_data) > 10:
            z = np.polyfit(amenity_data["amenities_count"], amenity_data["price"], 1)
            x_line = np.linspace(amenity_data["amenities_count"].min(),
                                 amenity_data["amenities_count"].max(), 100)
            ax4.plot(x_line, np.polyval(z, x_line), color="#ff6b6b",
                     linewidth=2, label=f"Trend (slope={z[0]:.1f})")
            ax4.legend(fontsize=11)
        ax4.set_xlabel("Number of Amenities", fontsize=12)
        ax4.set_ylabel("Nightly Price ($)", fontsize=12)
        ax4.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

    # Data table
    with st.expander("📋 Summary Statistics"):
        st.dataframe(summary_statistics(filtered))

    with st.expander("🔎 Raw Data Preview"):
        display_cols = ["name", "price", "bedrooms", "rating", "amenities_count",
                        "room_type", "city", "source"]
        avail = [c for c in display_cols if c in filtered.columns]
        st.dataframe(filtered[avail].head(100))


# ===== TAB 2 — HYPOTHESIS TEST =====
with tab2:
    st.subheader("Testing the Hypothesis")
    st.markdown("""
    **H₀ (Null):** Bedrooms, rating, and amenities count have *no relationship*
    with nightly price.

    **H₁ (Alternative):** Bedrooms, rating, and amenities count *positively
    predict* nightly price.

    **Method:** Ordinary Least Squares (OLS) regression on log-transformed price.
    Using log(price) normalizes the right-skewed price distribution and lets us
    interpret coefficients as approximate percentage changes.
    """)

    if model is not None:
        ht = hypothesis_test_summary(model)

        # Verdict box
        css_class = "verdict-reject" if ht["overall_verdict"] == "reject" else "verdict-fail"
        emoji = "✅" if ht["overall_verdict"] == "reject" else "⚠️"
        st.markdown(
            f'<div class="verdict-box {css_class}">'
            f'<strong>{emoji} Verdict:</strong> {ht["overall_explanation"]}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Model fit metrics
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("R²", f"{ht['r_squared']:.4f}")
        mcol2.metric("Adjusted R²", f"{ht['adj_r_squared']:.4f}")
        mcol3.metric("Observations", f"{ht['n_observations']:,}")

        st.divider()

        # Predictors table
        st.markdown("#### Individual Predictor Results")
        pred_rows = []
        for p in ht["predictors"]:
            sig_icon = "✅" if p["significant"] else "❌"
            pval_str = f"{p['pvalue']:.2e}" if p["pvalue"] < 0.001 else f"{p['pvalue']:.4f}"
            pred_rows.append({
                "Predictor": p["name"],
                "Coefficient": f"{p['coefficient']:.4f}",
                "p-value": pval_str,
                "Significant?": sig_icon,
                "What it means": p["interpretation"],
            })
        st.table(pd.DataFrame(pred_rows))

        # Coefficient chart with error bars
        st.markdown("#### Coefficient Magnitudes")
        fig_coef, ax_coef = plt.subplots(figsize=(8, 4))
        names = [p["name"] for p in ht["predictors"]]
        coefs = [p["coefficient"] for p in ht["predictors"]]
        # Get confidence intervals from model
        conf = model.conf_int()
        errors = []
        for name in names:
            low, high = conf.loc[name]
            errors.append((coefs[names.index(name)] - low, high - coefs[names.index(name)]))
        err_low = [e[0] for e in errors]
        err_high = [e[1] for e in errors]

        colors = ["#28a745" if p["significant"] else "#dc3545" for p in ht["predictors"]]
        bars = ax_coef.barh(names, coefs, xerr=[err_low, err_high],
                            color=colors, alpha=0.8, capsize=5, height=0.5)
        ax_coef.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax_coef.set_xlabel("Coefficient (effect on log-price)", fontsize=12)
        ax_coef.set_title("Green = Significant (p < 0.05), Red = Not Significant", fontsize=10, color="gray")
        plt.tight_layout()
        st.pyplot(fig_coef)
        plt.close(fig_coef)

        # Full model summary in expander
        with st.expander("📄 Full Regression Output (statsmodels)"):
            st.text(model.summary().as_text())
    else:
        st.warning("Could not fit regression model. Ensure the dataset has "
                    "bedrooms, rating, and amenities_count columns.")


# ===== TAB 3 — PRICE ESTIMATOR =====
with tab3:
    st.subheader("🎛️ Interactive Price Estimator")
    st.markdown(
        "Use the sliders below to see how different listing features affect "
        "the predicted nightly price. This uses the regression model trained "
        "on real Airbnb data."
    )

    if model is not None:
        est_col1, est_col2 = st.columns([1, 1])

        with est_col1:
            st.markdown("#### Set Your Listing Features")
            est_bedrooms = st.slider("🛏️ Bedrooms", 0, 10, 2, key="est_bed")
            est_rating = st.slider("⭐ Rating", 1.0, 5.0, 4.5, 0.1, key="est_rat")
            est_amenities = st.slider("🧳 Amenities Count", 0, 100, 20, key="est_amen")

        with est_col2:
            # Predict
            predicted = predict_price(model, est_bedrooms, est_rating, est_amenities)

            st.markdown('<p class="price-label">Predicted Nightly Price</p>',
                        unsafe_allow_html=True)
            st.markdown(f'<p class="price-result">${predicted:,.0f}</p>',
                        unsafe_allow_html=True)

            # Percentile in the dataset
            pct = (df["price"] <= predicted).mean() * 100
            st.markdown(f"This would be in the **{pct:.0f}th percentile** "
                        f"of all listings in the dataset.")

            # Visual context
            st.markdown("---")
            st.caption("For context:")
            ctx_col1, ctx_col2, ctx_col3 = st.columns(3)
            ctx_col1.metric("Dataset Min", f"${df['price'].min():.0f}")
            ctx_col2.metric("Dataset Median", f"${df['price'].median():.0f}")
            ctx_col3.metric("Dataset Max", f"${df['price'].max():.0f}")

        st.divider()

        # Sensitivity analysis — how each feature changes price
        st.markdown("#### 📈 Feature Sensitivity")
        st.markdown("See how changing each feature (while holding others constant) affects price.")

        sens_col1, sens_col2, sens_col3 = st.columns(3)

        with sens_col1:
            st.markdown("**Bedrooms →**")
            bed_range_vals = range(0, 8)
            bed_prices = [predict_price(model, b, est_rating, est_amenities) for b in bed_range_vals]
            fig_s1, ax_s1 = plt.subplots(figsize=(4, 3))
            ax_s1.plot(list(bed_range_vals), bed_prices, "o-", color="#667eea",
                       linewidth=2, markersize=6)
            ax_s1.axhline(predicted, color="#ff6b6b", linestyle="--", alpha=0.5)
            ax_s1.set_xlabel("Bedrooms")
            ax_s1.set_ylabel("Predicted Price ($)")
            ax_s1.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
            plt.tight_layout()
            st.pyplot(fig_s1)
            plt.close(fig_s1)

        with sens_col2:
            st.markdown("**Rating →**")
            rat_range_vals = np.arange(1.0, 5.1, 0.5)
            rat_prices = [predict_price(model, est_bedrooms, r, est_amenities)
                          for r in rat_range_vals]
            fig_s2, ax_s2 = plt.subplots(figsize=(4, 3))
            ax_s2.plot(rat_range_vals, rat_prices, "o-", color="#764ba2",
                       linewidth=2, markersize=6)
            ax_s2.axhline(predicted, color="#ff6b6b", linestyle="--", alpha=0.5)
            ax_s2.set_xlabel("Rating (stars)")
            ax_s2.set_ylabel("Predicted Price ($)")
            ax_s2.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
            plt.tight_layout()
            st.pyplot(fig_s2)
            plt.close(fig_s2)

        with sens_col3:
            st.markdown("**Amenities →**")
            amen_range_vals = range(0, 80, 5)
            amen_prices = [predict_price(model, est_bedrooms, est_rating, a)
                           for a in amen_range_vals]
            fig_s3, ax_s3 = plt.subplots(figsize=(4, 3))
            ax_s3.plot(list(amen_range_vals), amen_prices, "o-", color="#28a745",
                       linewidth=2, markersize=6)
            ax_s3.axhline(predicted, color="#ff6b6b", linestyle="--", alpha=0.5)
            ax_s3.set_xlabel("Amenities Count")
            ax_s3.set_ylabel("Predicted Price ($)")
            ax_s3.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
            plt.tight_layout()
            st.pyplot(fig_s3)
            plt.close(fig_s3)
    else:
        st.warning("Model not available. Run `python scripts/clean_data.py` first.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "Built with [BnBInsight](https://github.com/) by Monte Gardiner and Nandintsetseg Batsaikhan · "
    "Data: Kaggle Airbnb + AirROI API · "
    f"Dataset: {len(df):,} listings"
)
