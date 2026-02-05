# Week4 - Statistical Analysis and Tests

This repository contains a Streamlit app implementing the Week 4 assignment on statistical hypothesis testing.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py --server.port 8501
```

Open: `http://localhost:8501`

## VPS path mapping

To serve this under `http://YOUR_VM_IP_ADDRESS/week4`, configure your reverse proxy to map `/week4` to the Streamlit service.

## Dataset

Required source (as stated in assignment):

```python
import kagglehub
path = kagglehub.dataset_download("ayeshaimran1619/customer-spending-patterns")
```

The app first uses KaggleHub. If unavailable in environment, it falls back to a local file:

- `customer_spending.csv`

## What the app now includes

- Data check section with:
  - missing values per column,
  - data types,
  - number of unique values,
  - duplicate row count.
- Column-name normalization and robust matching for common customer-spending fields (e.g., Gender, Age, Annual Income, Spending Score).
- Three research questions with H0/H1.
- Assumption checks: Shapiro-Wilk, Levene, and expected-frequency caution for chi-square.
- Statistical outputs: statistic, p-value, df (where applicable).
- Effect sizes: Cohen's d and Cramer's V.
- Confidence interval: 95% CI for mean difference in RQ1.
- At least two visualizations directly linked to tests.
