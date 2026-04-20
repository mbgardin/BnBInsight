# BnBInsight — Airbnb Short-Term Rental Pricing Analysis

> **STAT 386 Final Project** — A Python package that demonstrates the full data-science pipeline for short-term rental pricing analysis.

---

## Hypothesis

**Primary hypothesis:** Listings with more bedrooms, higher ratings, and more amenities have significantly higher nightly prices.

| | Statement |
|---|---|
| **H₀** | Bedrooms, rating, and amenities count have no relationship with nightly price. |
| **H₁** | Bedrooms, rating, and amenities count positively predict nightly price. |

**Statistical model:** `log_price ~ bedrooms + rating + amenities_count`

---

## Data Sources

| Source | Purpose |
|---|---|
| **Kaggle Airbnb Listings** | Primary analysis dataset |
| **AirROI API** | Demonstrates original data acquisition pipeline |

---

## Installation

```bash
# Clone the repo
git clone https://github.com/mbgardin/BnBInsight.git
cd BnBInsight

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

---

## Project Structure

```
airbnb-pricing-project/
├── src/bnbinsight/        # Core package modules
│   ├── data_collectionls.py # Load Kaggle CSV & fetch AirROI API
│   ├── cleaning.py        # Clean price, rating, bedrooms, etc.
│   ├── features.py        # Amenity count, log price, feature selection
│   ├── analysis.py        # Summary stats, OLS regression, flagging
│   ├── visualization.py   # Price distribution, scatter, box plots
│   └── utils.py           # Helper utilities
├── scripts/               # Pipeline scripts
│   ├── fetch_data.py      # Fetch AirROI data
│   ├── clean_data.py      # Clean raw → processed
│   └── run_analysis.py    # Run full analysis
├── app/
│   └── streamlit_app.py   # Interactive dashboard
├── data/
│   ├── raw/               # Raw CSV / JSON
│   └── processed/         # Cleaned dataset
├── docs/                  # Quarto documentation
├── tests/                 # Unit tests
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Place your data

Put the Kaggle Airbnb CSV in `data/raw/kaggle_airbnb.csv`.

### 2. (Optional) Fetch API data

```bash
export AIRROI_API_KEY=your_key_here
python scripts/fetch_data.py
```

### 3. Clean the data

```bash
python scripts/clean_data.py
```

### 4. Run the analysis

```bash
python scripts/run_analysis.py
```

### 5. Launch the dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## Run Tests

```bash
python -m pytest tests/
```

---

## Documentation

Full documentation (built with Quarto) is available on GitHub Pages:

**[https://mbgardin.github.io/BnBInsight/](https://mbgardin.github.io/BnBInsight/)**

---

## License

This project is for educational purposes (BYU STAT 386).
