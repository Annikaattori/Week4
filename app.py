from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Week 4 - Statistical Analysis", layout="wide")

DATASET_ID = "ayeshaimran1619/customer-spending-patterns"


# -----------------------------
# Data loading and preparation
# -----------------------------
def download_dataset_with_kagglehub() -> pd.DataFrame:
    """Load the assignment dataset using kagglehub."""
    import kagglehub  # required by assignment

    dataset_dir = kagglehub.dataset_download(DATASET_ID)
    dataset_path = Path(dataset_dir)
    csv_files = sorted(dataset_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_path}")
    return pd.read_csv(csv_files[0])


def load_data() -> pd.DataFrame:
    """Try kagglehub first, fallback to local file if needed."""
    try:
        df = download_dataset_with_kagglehub()
        st.success("Dataset downloaded via kagglehub.")
        return df
    except Exception as exc:
        st.warning(
            "Could not download from kagglehub in this environment. "
            "Trying local fallback file `customer_spending.csv`."
        )
        fallback = Path("customer_spending.csv")
        if fallback.exists():
            return pd.read_csv(fallback)

        st.error(
            f"Data loading failed. KaggleHub error: {exc}. "
            "Please add `customer_spending.csv` in project root or configure Kaggle access."
        )
        st.stop()


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names for robust matching while preserving readability."""
    normalized = {}
    for col in df.columns:
        clean = " ".join(str(col).strip().replace("_", " ").split())
        normalized[col] = clean
    return df.rename(columns=normalized)


def choose_column(columns: List[str], keywords: List[str], default: Optional[str] = None) -> Optional[str]:
    low_map = {col.lower(): col for col in columns}
    for key in keywords:
        for low_col, original in low_map.items():
            if key in low_col:
                return original
    return default


def infer_recommended_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Infer likely columns from common Kaggle customer spending schema.
    Common names include: Gender, Age, Annual Income, Spending Score, etc.
    """
    cols = list(df.columns)
    return {
        "gender": choose_column(cols, ["gender", "sex"]),
        "age": choose_column(cols, ["age"]),
        "income": choose_column(cols, ["annual income", "income"]),
        "spending": choose_column(cols, ["spending score", "spending", "score"]),
        "category_a": choose_column(cols, ["segment", "membership", "city", "payment", "gender", "category"]),
        "category_b": choose_column(cols, ["category", "payment", "channel", "product", "city", "gender"]),
    }


def coerce_numeric_columns(df: pd.DataFrame, candidates: List[Optional[str]]) -> pd.DataFrame:
    out = df.copy()
    for col in candidates:
        if col and col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def dataset_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for col in df.columns:
        missing = int(df[col].isna().sum())
        missing_pct = (missing / n * 100) if n else 0
        unique_vals = int(df[col].nunique(dropna=True))
        dtype = str(df[col].dtype)
        rows.append(
            {
                "column": col,
                "dtype": dtype,
                "missing_count": missing,
                "missing_%": round(missing_pct, 2),
                "unique_values": unique_vals,
            }
        )
    return pd.DataFrame(rows)


# -----------------------------
# Statistics helpers
# -----------------------------
def cohens_d(x1: np.ndarray, x2: np.ndarray) -> float:
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        return np.nan
    s1, s2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled == 0:
        return np.nan
    return (np.mean(x1) - np.mean(x2)) / pooled


def cramers_v(contingency: pd.DataFrame) -> float:
    chi2, _, _, _ = stats.chi2_contingency(contingency)
    n = contingency.to_numpy().sum()
    r, k = contingency.shape
    if n == 0 or min(r, k) <= 1:
        return np.nan
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))


def ci_mean_difference_welch(x1: np.ndarray, x2: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    """Approximate CI for mean difference using Welch standard error and df."""
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    diff = np.mean(x1) - np.mean(x2)
    v1 = np.var(x1, ddof=1) / len(x1)
    v2 = np.var(x2, ddof=1) / len(x2)
    se = np.sqrt(v1 + v2)
    df = (v1 + v2) ** 2 / ((v1**2) / (len(x1) - 1) + (v2**2) / (len(x2) - 1))
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    return diff - tcrit * se, diff + tcrit * se


def decision_text(p: float, alpha: float = 0.05) -> str:
    return "Reject H₀" if p < alpha else "Fail to reject H₀"


# -----------------------------
# App
# -----------------------------
st.title("Week 4 Assignment: Statistical Analysis and Tests")
df_raw = load_data()
df_raw = normalize_column_names(df_raw)

rec = infer_recommended_columns(df_raw)
df = coerce_numeric_columns(df_raw, [rec["age"], rec["income"], rec["spending"]])

st.header("1) Data & EDA")
info_col1, info_col2, info_col3 = st.columns(3)
info_col1.metric("Rows", f"{df.shape[0]:,}")
info_col2.metric("Columns", df.shape[1])
info_col3.metric("Duplicate rows", int(df.duplicated().sum()))

st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

quality = dataset_quality_report(df)
st.subheader("Data Quality Check")
st.dataframe(quality, use_container_width=True)

missing_total = int(df.isna().sum().sum())
if missing_total > 0:
    st.warning(f"Detected {missing_total} missing values across the dataset. Tests use pairwise row filtering.")
else:
    st.success("No missing values detected.")

num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in df.columns if c not in num_cols]

st.subheader("Summary Statistics (Numerical)")
if num_cols:
    st.dataframe(df[num_cols].describe().T, use_container_width=True)
else:
    st.warning("No numeric columns detected; statistical tests requiring numeric outcomes cannot run.")

if not cat_cols:
    st.error("No categorical columns detected; grouping/association tests cannot run.")
    st.stop()
if not num_cols:
    st.stop()

st.sidebar.header("Variable selection")
def safe_index(options: List[str], value: Optional[str], fallback: int = 0) -> int:
    if value in options:
        return options.index(value)
    return fallback

rq1_group = st.sidebar.selectbox(
    "RQ1 group variable (categorical)",
    cat_cols,
    index=safe_index(cat_cols, rec["gender"]),
)
rq1_outcome = st.sidebar.selectbox(
    "RQ1 outcome variable (numeric)",
    num_cols,
    index=safe_index(num_cols, rec["spending"]),
)

rq2_group = st.sidebar.selectbox(
    "RQ2 group variable (3+ categories)",
    cat_cols,
    index=safe_index(cat_cols, rec["category_a"]),
)
rq2_outcome = st.sidebar.selectbox(
    "RQ2 outcome variable (numeric)",
    num_cols,
    index=safe_index(num_cols, rec["income"]),
)

rq3_a = st.sidebar.selectbox("RQ3 categorical variable A", cat_cols, index=safe_index(cat_cols, rec["category_a"]))
rq3_b = st.sidebar.selectbox(
    "RQ3 categorical variable B",
    cat_cols,
    index=safe_index(cat_cols, rec["category_b"], fallback=min(1, len(cat_cols) - 1)),
)

st.header("2) Research Questions & Hypotheses")
st.markdown(
    f"""
1. **RQ1:** Does mean **{rq1_outcome}** differ between two largest groups in **{rq1_group}**?
   - H₀: μ₁ = μ₂
   - H₁: μ₁ ≠ μ₂

2. **RQ2:** Does mean **{rq2_outcome}** differ across levels of **{rq2_group}**?
   - H₀: μ₁ = μ₂ = ... = μₖ
   - H₁: At least one group mean differs.

3. **RQ3:** Is there an association between **{rq3_a}** and **{rq3_b}**?
   - H₀: The two variables are independent.
   - H₁: The two variables are associated.
"""
)

st.header("3) Assumptions, Test Selection, and Results")

# -------- RQ1 --------
st.subheader("RQ1: Two-group mean comparison")
rq1_df = df[[rq1_group, rq1_outcome]].dropna().copy()
counts = rq1_df[rq1_group].value_counts()
if counts.shape[0] < 2:
    st.warning("Need at least two categories to run RQ1.")
else:
    top2 = counts.index[:2]
    g1 = rq1_df.loc[rq1_df[rq1_group] == top2[0], rq1_outcome].to_numpy()
    g2 = rq1_df.loc[rq1_df[rq1_group] == top2[1], rq1_outcome].to_numpy()

    if len(g1) < 3 or len(g2) < 3:
        st.warning("Each group should have at least 3 observations for robust assumption checks.")
    else:
        sh1 = stats.shapiro(g1 if len(g1) <= 500 else np.random.default_rng(42).choice(g1, 500, replace=False))
        sh2 = stats.shapiro(g2 if len(g2) <= 500 else np.random.default_rng(42).choice(g2, 500, replace=False))
        lev = stats.levene(g1, g2)

        normal_ok = (sh1.pvalue > 0.05) and (sh2.pvalue > 0.05)
        equal_var_ok = lev.pvalue > 0.05

        if normal_ok:
            t_res = stats.ttest_ind(g1, g2, equal_var=equal_var_ok)
            test_name = "Independent t-test"
            stat_value = t_res.statistic
            p_value = t_res.pvalue
            df_txt = "Welch-adjusted" if not equal_var_ok else f"{len(g1)+len(g2)-2}"
        else:
            mw = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            test_name = "Mann-Whitney U (non-parametric fallback)"
            stat_value = mw.statistic
            p_value = mw.pvalue
            df_txt = "N/A"

        d = cohens_d(g1, g2)
        ci_low, ci_high = ci_mean_difference_welch(g1, g2)

        st.write(f"**Chosen test:** {test_name}")
        st.write(f"Shapiro-Wilk p-values: {top2[0]}={sh1.pvalue:.4f}, {top2[1]}={sh2.pvalue:.4f}")
        st.write(f"Levene p-value: {lev.pvalue:.4f}")
        st.write(f"Test statistic: {stat_value:.4f}")
        st.write(f"Degrees of freedom: {df_txt}")
        st.write(f"p-value: {p_value:.4g}")
        st.write(f"Cohen's d: {d:.3f}")
        st.write(f"95% CI for mean difference ({top2[0]} - {top2[1]}): [{ci_low:.3f}, {ci_high:.3f}]")
        st.success(decision_text(p_value))

        fig1, ax1 = plt.subplots(figsize=(8, 4))
        plot_df = rq1_df[rq1_df[rq1_group].isin(top2)].copy()
        sns.boxplot(data=plot_df, x=rq1_group, y=rq1_outcome, ax=ax1)
        sns.stripplot(data=plot_df, x=rq1_group, y=rq1_outcome, color="black", alpha=0.4, ax=ax1)
        ax1.set_title(f"{rq1_outcome} by {rq1_group} (RQ1)")
        st.pyplot(fig1)
        st.caption("Box + strip plot for the two compared groups in RQ1.")

# -------- RQ2 --------
st.subheader("RQ2: Multi-group mean comparison")
rq2_df = df[[rq2_group, rq2_outcome]].dropna().copy()
groups = [g[rq2_outcome].to_numpy() for _, g in rq2_df.groupby(rq2_group) if len(g) >= 3]

if len(groups) < 3:
    st.warning("Need at least 3 groups (with at least 3 rows each) for RQ2.")
else:
    shapiro_ps = []
    rng = np.random.default_rng(42)
    for arr in groups:
        sample = arr if len(arr) <= 500 else rng.choice(arr, 500, replace=False)
        shapiro_ps.append(stats.shapiro(sample).pvalue)

    lev2 = stats.levene(*groups)
    normal_ok2 = all(p > 0.05 for p in shapiro_ps)
    equal_var_ok2 = lev2.pvalue > 0.05

    if normal_ok2 and equal_var_ok2:
        aov = stats.f_oneway(*groups)
        test2 = "One-way ANOVA"
        stat2 = aov.statistic
        p2 = aov.pvalue
    else:
        kw = stats.kruskal(*groups)
        test2 = "Kruskal-Wallis (non-parametric fallback)"
        stat2 = kw.statistic
        p2 = kw.pvalue

    st.write(f"**Chosen test:** {test2}")
    st.write(f"Levene p-value: {lev2.pvalue:.4f}")
    st.write(f"Minimum Shapiro p-value across groups: {min(shapiro_ps):.4f}")
    st.write(f"Test statistic: {stat2:.4f}")
    st.write(f"p-value: {p2:.4g}")
    st.success(decision_text(p2))

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.violinplot(data=rq2_df, x=rq2_group, y=rq2_outcome, inner="box", ax=ax2)
    ax2.set_title(f"{rq2_outcome} distribution by {rq2_group} (RQ2)")
    ax2.tick_params(axis="x", rotation=30)
    st.pyplot(fig2)
    st.caption("Violin/box summaries showing group-wise distributions in RQ2.")

# -------- RQ3 --------
st.subheader("RQ3: Association between categorical variables")
rq3_df = df[[rq3_a, rq3_b]].dropna().copy()
ct = pd.crosstab(rq3_df[rq3_a], rq3_df[rq3_b])

if ct.shape[0] < 2 or ct.shape[1] < 2:
    st.warning("Need at least a 2x2 contingency table for RQ3.")
else:
    chi2, p3, dof3, expected = stats.chi2_contingency(ct)
    cv = cramers_v(ct)
    min_expected = float(np.min(expected))

    if min_expected < 5:
        st.warning(
            f"Some expected counts are < 5 (minimum expected={min_expected:.2f}); "
            "interpret χ² approximation with caution."
        )

    st.write("**Chosen test:** Chi-square test of independence")
    st.write(f"χ² statistic: {chi2:.4f}")
    st.write(f"Degrees of freedom: {dof3}")
    st.write(f"p-value: {p3:.4g}")
    st.write(f"Cramer's V: {cv:.3f}")
    st.success(decision_text(p3))

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ct_prop = ct.div(ct.sum(axis=1), axis=0)
    ct_prop.plot(kind="bar", stacked=True, ax=ax3)
    ax3.set_ylabel("Proportion within group")
    ax3.set_title(f"Proportional stacked bars: {rq3_a} vs {rq3_b} (RQ3)")
    ax3.legend(title=rq3_b, bbox_to_anchor=(1.02, 1), loc="upper left")
    st.pyplot(fig3)
    st.caption("Stacked proportions used to visually support the χ² result.")

st.header("4) Interpretation and Discussion")
st.markdown(
    """
- Each test reports statistic, p-value, and decision on H₀.
- Practical significance is included using effect sizes (Cohen's d and Cramer's V).
- Uncertainty is communicated with a 95% CI for a key mean-difference result (RQ1).
- Assumption checks drive test choice (parametric vs non-parametric).
- Data quality checks (missing values, duplicates, dtypes, unique counts) are shown before inference.
"""
)

st.header("5) Documentation")
st.markdown(
    f"""
**Dataset source:** `{DATASET_ID}` via `kagglehub.dataset_download(...)`.

This app follows the Week 4 workflow:
1. Formulate H₀/H₁ for 3 research questions.
2. Check assumptions (Shapiro-Wilk, Levene, expected cell counts for χ²).
3. Run appropriate tests and robust alternatives when assumptions fail.
4. Report statistics, p-values, effect sizes, and confidence intervals.
5. Visualize and interpret findings in context.
"""
)
