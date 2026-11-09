# AV/IT Infrastructure Manufacturing Brand (India) — Market Validation

Data Analytics (MGB) individual submission: synthetic survey → cleaning → EDA → 4 algorithms → Streamlit dashboard.

## Files
| File | What it is |
|---|---|
| `generate_data.py` | Creates the raw synthetic survey data (3,245 rows with missing values, outliers, messy labels and duplicates; seed 42, so it reproduces exactly) |
| `clean_data.py` | Cleaning and transformation pipeline; writes the cleaned data (3,200 × 69, 0 missing) and a cleaning log |
| `analysis.py` | Classification, regression, K-means clustering and Apriori association rules |
| `app.py` | Streamlit dashboard: 9 tabs, 40+ charts, each with a written insight |

The `data/` folder is generated automatically on the app's first run.

## Run it
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dashboard tabs
1. Overview & verdict
2. Data quality: raw vs clean, plus the cleaning log
3. Descriptive EDA (D1–D14)
4. Diagnostic & correlation (G1–G10)
5. Classification: who will buy
6. Regression: how much they will spend
7. Clustering: buyer personas
8. Association rules: product bundles, with adjustable thresholds
9. Feasibility: industry data and an editable break-even scenario

The data is synthetic and was built for coursework. It is not evidence of real demand.
