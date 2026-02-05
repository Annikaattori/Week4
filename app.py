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
        "spending": choose_column(cols, ["purchase amount", "spending score", "spending", "score"]),
        "category": choose_column(cols, ["category", "item purchased"]),
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


def welch_df(x1: np.ndarray, x2: np.ndarray) -> float:
    """Welch–Satterthwaite degrees of freedom."""
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    n1, n2 = len(x1), len(x2)
    v1 = np.var(x1, ddof=1)
    v2 = np.var(x2, ddof=1)
    num = (v1 / n1 + v2 / n2) ** 2
    den = (v1**2) / ((n1**2) * (n1 - 1)) + (v2**2) / ((n2**2) * (n2 - 1))
    return num / den


def rank_biserial_from_u(u: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation effect size for Mann–Whitney U."""
    if n1 <= 0 or n2 <= 0:
        return np.nan
    return 1 - (2 * u) / (n1 * n2)


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


def effect_size_label_abs(val: float) -> str:
    """Generic small/medium/large label for absolute effect size."""
    a = abs(val)
    if a < 0.3:
        return "small"
    if a < 0.5:
        return "medium"
    return "large"


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
    num_cols_filtered = [c for c in num_cols if "customer id" not in c.lower()]
    st.dataframe(df[num_cols_filtered].describe().T, use_container_width=True)
else:
    st.warning("No numeric columns detected; statistical tests requiring numeric outcomes cannot run.")

if not cat_cols:
    st.error("No categorical columns detected; grouping/association tests cannot run.")
    st.stop()
if not num_cols:
    st.stop()

# Fixed variables for the three research questions
age_col = rec["age"]
gender_col = rec["gender"]
spending_col = rec["spending"]
category_col = rec["category"]

st.header("2) Research Questions & Hypotheses")
st.markdown(
    """
**Research Question 1: Relationship between Age and Purchase Amount**

**Variables:**
- **Age** (numerical, predictor)
- **Spending Score** (numerical, outcome)

**Hypotheses:**
- **H₀ (Null):** There is no correlation between age and purchase amount (ρ = 0).
- **H₁ (Alternative):** There is a statistically significant correlation between age and purchase amount (ρ ≠ 0).

**Planned Test:** Pearson correlation (if assumptions look reasonable) or Spearman rank correlation (non-parametric alternative).

---

**Research Question 2: Gender Difference in Purchase Amount**

**Variables:**
- **Gender** (categorical: two largest categories)
- **Spending Score** (numerical outcome)

**Hypotheses:**
- **H₀ (Null):** The two gender groups do not differ in mean spending (μ₁ = μ₂).
- **H₁ (Alternative):** The two gender groups differ in mean spending (μ₁ ≠ μ₂).

**Planned Test:** Independent samples t-test (or Welch’s t-test if variances differ) or Mann–Whitney U test (non-parametric alternative).

---

**Research Question 3: Gender Difference in Clothing Purchases**

**Variables:**
- **Gender** (categorical: two largest categories within Clothing)
- **Spending Score** (numerical outcome)

**Hypotheses:**
- **H₀ (Null):** The two gender groups do not differ in mean clothing spending (μ₁ = μ₂).
- **H₁ (Alternative):** The two gender groups differ in mean clothing spending (μ₁ ≠ μ₂).

**Planned Test:** Independent samples t-test (or Welch’s t-test) or Mann–Whitney U test.
"""
)

st.header("3) Assumption Checking & Test Selection")

# -------- RQ1 --------
st.subheader("RQ1: Correlation between Age and Purchase Amount")

if age_col is None or spending_col is None:
    st.warning("Could not identify Age or Spending Score columns.")
else:
    rq1_df = df[[age_col, spending_col]].dropna().copy()

    if rq1_df.shape[0] < 4:
        st.warning("Not enough observations for RQ1 correlation analysis.")
    else:
        st.write("**Assumption Checks (practical):**")
        st.caption("Pearson is best for roughly linear relationships and is sensitive to outliers; Spearman is rank-based and more robust.")

        # Normality tests (Shapiro-Wilk; sample if huge)
        rng = np.random.default_rng(42)
        age_sample = rq1_df[age_col].to_numpy()
        spend_sample = rq1_df[spending_col].to_numpy()
        if len(rq1_df) > 500:
            age_sample = rng.choice(age_sample, 500, replace=False)
            spend_sample = rng.choice(spend_sample, 500, replace=False)

        sh_age = stats.shapiro(age_sample)
        sh_spend = stats.shapiro(spend_sample)

        age_normal = sh_age.pvalue > 0.05
        spend_normal = sh_spend.pvalue > 0.05
        both_normal = age_normal and spend_normal

        st.write(f"- Shapiro-Wilk normality test for Age: p = {sh_age.pvalue:.4f} {'✓' if age_normal else '✗'}")
        st.write(f"- Shapiro-Wilk normality test for Spending Score: p = {sh_spend.pvalue:.4f} {'✓' if spend_normal else '✗'}")

        # Perform correlation test
        if both_normal:
            corr_res = stats.pearsonr(rq1_df[age_col], rq1_df[spending_col])
            test_name_rq1 = "Pearson correlation"
        else:
            corr_res = stats.spearmanr(rq1_df[age_col], rq1_df[spending_col])
            test_name_rq1 = "Spearman rank correlation (non-parametric)"

        corr_coef = float(corr_res[0])
        p_val_rq1 = float(corr_res[1])

        st.write(f"**Chosen test:** {test_name_rq1}")
        st.write(f"- Correlation coefficient (r): {corr_coef:.4f}")
        st.write(f"- p-value: {p_val_rq1:.4g}")
        st.write(f"- Effect size (|r|): {abs(corr_coef):.3f} ({effect_size_label_abs(corr_coef)})")
        st.success(decision_text(p_val_rq1))

        # Visualization for RQ1
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.scatter(rq1_df[age_col], rq1_df[spending_col], alpha=0.5, s=30)

        # Trend line (simple linear fit for visualization)
        z = np.polyfit(rq1_df[age_col], rq1_df[spending_col], 1)
        poly = np.poly1d(z)
        x_sorted = rq1_df[age_col].sort_values()
        ax1.plot(x_sorted, poly(x_sorted), "r-", alpha=0.8, linewidth=2)

        ax1.set_xlabel(age_col, fontsize=11)
        ax1.set_ylabel(spending_col, fontsize=11)
        ax1.set_title(f"RQ1: {age_col} vs {spending_col}\n{test_name_rq1}: r = {corr_coef:.3f}, p = {p_val_rq1:.4g}")
        ax1.grid(alpha=0.3)
        st.pyplot(fig1)
        st.caption("Scatter plot with a fitted trend line (visual aid).")

# -------- RQ2 --------
st.subheader("RQ2: Gender Difference in Purchase Amount")

if gender_col not in df.columns or spending_col not in df.columns:
    st.warning("Could not identify Gender or Spending Score columns.")
else:
    rq2_df = df[[gender_col, spending_col]].dropna().copy()
    counts_rq2 = rq2_df[gender_col].value_counts()

    if counts_rq2.shape[0] < 2:
        st.warning("Need at least two gender categories to run RQ2.")
    else:
        top2_rq2 = counts_rq2.index[:2]
        g1_rq2 = rq2_df.loc[rq2_df[gender_col] == top2_rq2[0], spending_col].to_numpy()
        g2_rq2 = rq2_df.loc[rq2_df[gender_col] == top2_rq2[1], spending_col].to_numpy()

        if len(g1_rq2) < 3 or len(g2_rq2) < 3:
            st.warning("Each group should have at least 3 observations for robust assumption checks.")
        else:
            st.write("**Assumption Checks:**")

            rng = np.random.default_rng(42)
            s1 = g1_rq2 if len(g1_rq2) <= 500 else rng.choice(g1_rq2, 500, replace=False)
            s2 = g2_rq2 if len(g2_rq2) <= 500 else rng.choice(g2_rq2, 500, replace=False)

            sh1_rq2 = stats.shapiro(s1)
            sh2_rq2 = stats.shapiro(s2)
            lev_rq2 = stats.levene(g1_rq2, g2_rq2)

            normal_ok_rq2 = (sh1_rq2.pvalue > 0.05) and (sh2_rq2.pvalue > 0.05)
            equal_var_ok_rq2 = lev_rq2.pvalue > 0.05

            st.write(
                f"- Shapiro-Wilk p-values: {top2_rq2[0]} = {sh1_rq2.pvalue:.4f} {'✓' if sh1_rq2.pvalue > 0.05 else '✗'}, "
                f"{top2_rq2[1]} = {sh2_rq2.pvalue:.4f} {'✓' if sh2_rq2.pvalue > 0.05 else '✗'}"
            )
            st.write(
                f"- Levene's test p-value: {lev_rq2.pvalue:.4f} "
                f"{'✓ (equal variances)' if equal_var_ok_rq2 else '✗ (unequal variances → Welch)'}"
            )

            # Choose test
            if normal_ok_rq2:
                t_res_rq2 = stats.ttest_ind(g1_rq2, g2_rq2, equal_var=equal_var_ok_rq2)
                test2 = "Independent t-test" if equal_var_ok_rq2 else "Welch t-test"
                stat2 = float(t_res_rq2.statistic)
                p2 = float(t_res_rq2.pvalue)
                df2 = (len(g1_rq2) + len(g2_rq2) - 2) if equal_var_ok_rq2 else welch_df(g1_rq2, g2_rq2)

                # Effect size + CI (parametric case)
                d_rq2 = cohens_d(g1_rq2, g2_rq2)
                ci_low_rq2, ci_high_rq2 = ci_mean_difference_welch(g1_rq2, g2_rq2)

                st.write(f"**Chosen test:** {test2}")
                st.write(f"- Test statistic: t({df2:.1f}) = {stat2:.4f}")
                st.write(f"- p-value: {p2:.4g}")
                st.write(f"- Effect size (Cohen's d): {d_rq2:.3f} ({effect_size_label_abs(d_rq2)})")
                st.write(f"- 95% CI for mean difference ({top2_rq2[0]} - {top2_rq2[1]}): [{ci_low_rq2:.3f}, {ci_high_rq2:.3f}]")
                st.success(decision_text(p2))

                title_stat = f"t({df2:.1f}) = {stat2:.3f}"
                effect_text = f"d = {d_rq2:.3f}"
            else:
                mw_rq2 = stats.mannwhitneyu(g1_rq2, g2_rq2, alternative="two-sided")
                test2 = "Mann–Whitney U test (non-parametric)"
                stat2 = float(mw_rq2.statistic)
                p2 = float(mw_rq2.pvalue)

                rbc = rank_biserial_from_u(stat2, len(g1_rq2), len(g2_rq2))

                st.write(f"**Chosen test:** {test2}")
                st.write(f"- Test statistic: U = {stat2:.4f}")
                st.write(f"- p-value: {p2:.4g}")
                st.write(f"- Effect size (rank-biserial r): {rbc:.3f} ({effect_size_label_abs(rbc)})")
                st.caption("For the non-parametric test, a rank-based effect size is reported. Mean-difference CI is provided in the parametric t-test case.")
                st.success(decision_text(p2))

                title_stat = f"U = {stat2:.1f}"
                effect_text = f"r (rank-biserial) = {rbc:.3f}"

            # Visualization for RQ2
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            plot_df_rq2 = rq2_df[rq2_df[gender_col].isin(top2_rq2)].copy()
            sns.boxplot(data=plot_df_rq2, x=gender_col, y=spending_col, ax=ax2, width=0.6)
            sns.stripplot(data=plot_df_rq2, x=gender_col, y=spending_col, color="black", alpha=0.4, ax=ax2, size=4)
            ax2.set_xlabel(gender_col, fontsize=11)
            ax2.set_ylabel(spending_col, fontsize=11)
            ax2.set_title(f"RQ2: {spending_col} by {gender_col}\n{test2}: {title_stat}, p = {p2:.4g}, {effect_text}")
            ax2.set_xticklabels([f"{top2_rq2[0]}\n(n={len(g1_rq2)})", f"{top2_rq2[1]}\n(n={len(g2_rq2)})"])
            ax2.grid(alpha=0.3, axis="y")
            st.pyplot(fig2)
            st.caption("Box plot with individual data points. Each box shows median and IQR; whiskers extend to 1.5×IQR.")

# -------- RQ3 --------
st.subheader("RQ3: Gender Difference in Clothing Spending")

if category_col is None:
    st.warning("Could not find a category/item column for filtering Clothing.")
else:
    # Robust Clothing filter (case-insensitive; substring match)
    rq3_df_filtered = df[df[category_col].astype(str).str.contains("clothing", case=False, na=False)].copy()

    if rq3_df_filtered.shape[0] < 10:
        st.warning(f"Not enough Clothing records ({rq3_df_filtered.shape[0]}) to perform RQ3 analysis reliably.")
    else:
        if gender_col not in rq3_df_filtered.columns or spending_col not in rq3_df_filtered.columns:
            st.warning("Could not identify Gender or Spending Score columns for RQ3.")
        else:
            rq3_df_analysis = rq3_df_filtered[[gender_col, spending_col]].dropna().copy()
            counts_rq3 = rq3_df_analysis[gender_col].value_counts()

            if counts_rq3.shape[0] < 2:
                st.warning("Need at least two gender categories in Clothing purchases for RQ3.")
            else:
                top2_rq3 = counts_rq3.index[:2]
                g1_rq3 = rq3_df_analysis.loc[rq3_df_analysis[gender_col] == top2_rq3[0], spending_col].to_numpy()
                g2_rq3 = rq3_df_analysis.loc[rq3_df_analysis[gender_col] == top2_rq3[1], spending_col].to_numpy()

                if len(g1_rq3) < 3 or len(g2_rq3) < 3:
                    st.warning("Each group should have at least 3 observations for robust assumption checks.")
                else:
                    st.write("**Assumption Checks:**")

                    rng = np.random.default_rng(42)
                    s1 = g1_rq3 if len(g1_rq3) <= 500 else rng.choice(g1_rq3, 500, replace=False)
                    s2 = g2_rq3 if len(g2_rq3) <= 500 else rng.choice(g2_rq3, 500, replace=False)

                    sh1_rq3 = stats.shapiro(s1)
                    sh2_rq3 = stats.shapiro(s2)
                    lev_rq3 = stats.levene(g1_rq3, g2_rq3)

                    normal_ok_rq3 = (sh1_rq3.pvalue > 0.05) and (sh2_rq3.pvalue > 0.05)
                    equal_var_ok_rq3 = lev_rq3.pvalue > 0.05

                    st.write(
                        f"- Shapiro-Wilk p-values: {top2_rq3[0]} = {sh1_rq3.pvalue:.4f} {'✓' if sh1_rq3.pvalue > 0.05 else '✗'}, "
                        f"{top2_rq3[1]} = {sh2_rq3.pvalue:.4f} {'✓' if sh2_rq3.pvalue > 0.05 else '✗'}"
                    )
                    st.write(
                        f"- Levene's test p-value: {lev_rq3.pvalue:.4f} "
                        f"{'✓ (equal variances)' if equal_var_ok_rq3 else '✗ (unequal variances → Welch)'}"
                    )
                    st.write(f"- Sample sizes: {top2_rq3[0]} (n={len(g1_rq3)}), {top2_rq3[1]} (n={len(g2_rq3)})")

                    # Choose test
                    if normal_ok_rq3:
                        t_res_rq3 = stats.ttest_ind(g1_rq3, g2_rq3, equal_var=equal_var_ok_rq3)
                        test3 = "Independent t-test" if equal_var_ok_rq3 else "Welch t-test"
                        stat3 = float(t_res_rq3.statistic)
                        p3 = float(t_res_rq3.pvalue)
                        df3 = (len(g1_rq3) + len(g2_rq3) - 2) if equal_var_ok_rq3 else welch_df(g1_rq3, g2_rq3)

                        d_rq3 = cohens_d(g1_rq3, g2_rq3)
                        ci_low_rq3, ci_high_rq3 = ci_mean_difference_welch(g1_rq3, g2_rq3)

                        st.write(f"**Chosen test:** {test3}")
                        st.write(f"- Test statistic: t({df3:.1f}) = {stat3:.4f}")
                        st.write(f"- p-value: {p3:.4g}")
                        st.write(f"- Effect size (Cohen's d): {d_rq3:.3f} ({effect_size_label_abs(d_rq3)})")
                        st.write(f"- 95% CI for mean difference ({top2_rq3[0]} - {top2_rq3[1]}): [{ci_low_rq3:.3f}, {ci_high_rq3:.3f}]")
                        st.success(decision_text(p3))

                        title_stat = f"t({df3:.1f}) = {stat3:.3f}"
                        effect_text = f"d = {d_rq3:.3f}"
                    else:
                        mw_rq3 = stats.mannwhitneyu(g1_rq3, g2_rq3, alternative="two-sided")
                        test3 = "Mann–Whitney U test (non-parametric)"
                        stat3 = float(mw_rq3.statistic)
                        p3 = float(mw_rq3.pvalue)

                        rbc3 = rank_biserial_from_u(stat3, len(g1_rq3), len(g2_rq3))

                        st.write(f"**Chosen test:** {test3}")
                        st.write(f"- Test statistic: U = {stat3:.4f}")
                        st.write(f"- p-value: {p3:.4g}")
                        st.write(f"- Effect size (rank-biserial r): {rbc3:.3f} ({effect_size_label_abs(rbc3)})")
                        st.caption("For the non-parametric test, a rank-based effect size is reported. Mean-difference CI is provided in the parametric t-test case.")
                        st.success(decision_text(p3))

                        title_stat = f"U = {stat3:.1f}"
                        effect_text = f"r (rank-biserial) = {rbc3:.3f}"

                    # Visualization for RQ3
                    fig3, ax3 = plt.subplots(figsize=(8, 5))
                    plot_df_rq3 = rq3_df_analysis[rq3_df_analysis[gender_col].isin(top2_rq3)].copy()
                    sns.boxplot(data=plot_df_rq3, x=gender_col, y=spending_col, ax=ax3, width=0.6)
                    sns.stripplot(data=plot_df_rq3, x=gender_col, y=spending_col, color="black", alpha=0.4, ax=ax3, size=4)
                    ax3.set_xlabel(gender_col, fontsize=11)
                    ax3.set_ylabel(spending_col, fontsize=11)
                    ax3.set_title(f"RQ3: Clothing {spending_col} by {gender_col}\n{test3}: {title_stat}, p = {p3:.4g}, {effect_text}")
                    ax3.set_xticklabels([f"{top2_rq3[0]}\n(n={len(g1_rq3)})", f"{top2_rq3[1]}\n(n={len(g2_rq3)})"])
                    ax3.grid(alpha=0.3, axis="y")
                    st.pyplot(fig3)
                    st.caption("Box plot with individual data points (Clothing records only).")

st.header("4) Interpretation and Discussion")

st.subheader("RQ1: Relationship between Age and Purchase Amount")
st.markdown(
    """
**Interpretation:**
This analysis tested whether customer age is associated with spending. Correlation quantifies direction and strength.

**Key points:**
- **p < 0.05** suggests evidence of a non-zero association.
- Effect size **|r|** indicates practical strength (rough guide: <0.3 small, 0.3–0.5 medium, >0.5 large).
- Correlation does not imply causation; other variables (e.g., income) may confound the association.
"""
)

st.subheader("RQ2: Gender Difference in Purchase Amount")
st.markdown(
    """
**Interpretation:**
This analysis compared two gender groups' spending to assess whether typical spending differs by gender.

**Key points:**
- **t-test/Welch** is used when normality assumptions are satisfied; **Mann–Whitney U** is applied otherwise.
- **Cohen's d** and a **95% CI for the mean difference** are reported for t-tests.
- **Rank-based effect size** (rank-biserial r) is reported for Mann–Whitney U, which is robust to non-normality.

**Practical note:**
Small p-values can occur with tiny effects in large samples. Interpretation should always include effect size and the confidence interval (when applicable).
"""
)

st.subheader("RQ3: Gender Difference in Clothing Spending")
st.markdown(
    """
**Interpretation:**
This analysis repeated the gender comparison within Clothing records only to reveal category-specific differences.

**Key points:**
- Results may be less conclusive if Clothing sample sizes are smaller, reducing statistical power.
- Interpretation balances statistical evidence (p-value) with practical magnitude (effect size).
"""
)

st.header("5) Summary and Conclusions")
st.markdown(
    """
This analysis addressed three research questions:

1. **RQ1 (Correlation):** Assessed association between age and spending.
2. **RQ2 (Group difference):** Assessed gender differences in overall spending via t-test/Welch or Mann–Whitney U.
3. **RQ3 (Filtered group difference):** Assessed gender differences in Clothing spending specifically.

**Strengths:**
- Assumption checks guide test selection (Shapiro–Wilk normality; Levene variance homogeneity).
- p-values, effect sizes, and confidence intervals are reported (parametric case).
- Multiple visualizations (scatter plot and boxplots) support the statistical findings.

**Limitations:**
- Cross-sectional data design limits causal inference.
- Potential confounders (e.g., income, preferences) are not controlled.
- Multiple comparisons can inflate false-positive risk; correction should be considered for extended analyses.
"""
)

st.header("6) Documentation")
st.markdown(
    f"""
**Dataset source:** `{DATASET_ID}` via `kagglehub.dataset_download(...)`.

**Workflow:**
1. Define null and alternative hypotheses (H₀/H₁) for three research questions.
2. Check statistical assumptions (Shapiro–Wilk normality test; Levene variance homogeneity test).
3. Select appropriate statistical tests (Pearson/Spearman correlation; t-test/Welch t-test; Mann–Whitney U as alternative).
4. Report test statistic, degrees of freedom (when applicable), p-values, effect sizes, and confidence intervals.
5. Visualize findings with corresponding plots and interpret results in context.
"""
)
