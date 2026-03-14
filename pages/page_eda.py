import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from utils.data_loader import MONTH_LABELS, FEATURE_COLS, TARGET_COL

def render(df):
    st.title("Exploratory Data Analysis")
    st.write("Quick data checks and the EDA points that informed preprocessing and modelling.")
    st.divider()

    # Dataset preview
    st.subheader("Dataset preview", anchor="dataset-preview", help="First 10 rows of the dataset with key features. Note that engineered features are excluded here for clarity.")
    raw_cols = ["date", "cloud_cover", "sunshine", "global_radiation", "max_temp", "mean_temp", "min_temp", "precipitation", "pressure", "snow_depth"]
    available = [c for c in raw_cols if c in df.columns]
    st.dataframe(df[available].head(10), use_container_width=True)

    with st.expander("Column descriptions", expanded=False):
        st.markdown("""
        | Column | Unit | Notes |
        |---|---|---|
        | cloud_cover | oktas (0–9) | 0 = clear, 9 = overcast |
        | sunshine | hours/day | Daily sunshine duration |
        | global_radiation | W/m² | Incoming solar radiation |
        | max_temp | °C | Daily max |
        | mean_temp | °C | Target — daily mean |
        | min_temp | °C | Daily min |
        | precipitation | mm | Daily total |
        | pressure | Pa | Converted to hPa in preprocessing |
        | snow_depth | cm | Filled with zero when missing |
        """)

    st.divider()

    # Summary stats
    st.subheader("Summary statistics", anchor="summary-statistics", help="Basic stats for key features. Note that engineered features are excluded here as they are derived from the raw features and would not provide additional insight at this stage.")
    stat_cols = ["max_temp", "mean_temp", "min_temp", "precipitation", "cloud_cover", "sunshine", "global_radiation"]
    stat_cols = [c for c in stat_cols if c in df.columns]
    desc = df[stat_cols].describe().round(2)
    st.dataframe(desc, use_container_width=True)
    st.caption("Temperatures look roughly normal; precipitation is right-skewed with many zeros.")
    st.divider()

    # Missing values
    st.subheader("Missing values", anchor="missing-values", help="Missing value counts and percentages for key features. Snow depth is handled by filling NaN with zero, as it indicates no snow.")
    check_cols = ["cloud_cover", "sunshine", "global_radiation", "max_temp", "mean_temp", "min_temp", "precipitation", "pressure", "snow_depth"]
    check_cols = [c for c in check_cols if c in df.columns]
    missing_counts = df[check_cols].isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"Feature": missing_counts.index, "Missing Count": missing_counts.values, "Missing %": missing_pct.values})
    # assume snow_depth NaN -> 0
    if "snow_depth" in missing_df["Feature"].values:
        idx = missing_df[missing_df["Feature"] == "snow_depth"].index
        missing_df.loc[idx, "Action"] = "Fill with 0"
    missing_df["Action"] = missing_df.get("Action", "Impute with median")
    has_missing = missing_df[missing_df["Missing Count"] > 0]
    no_missing = missing_df[missing_df["Missing Count"] == 0]

    if len(has_missing):
        col_tbl, col_chart = st.columns([2, 1])
        with col_tbl:
            st.dataframe(has_missing.reset_index(drop=True), use_container_width=True, hide_index=True)
            st.markdown(f"**{len(no_missing)}** features have no missing values. **{len(has_missing)}** need imputation.")
        with col_chart:
            fig, ax = plt.subplots(figsize=(4, len(has_missing) * 0.55 + 1.0))
            bar_colours = ["#dc2626" if p > 5 else "#f59e0b" if p > 1 else "#14b8a6" for p in has_missing["Missing %"]]
            ax.barh(has_missing["Feature"], has_missing["Missing %"], color=bar_colours, height=0.5)
            ax.set_xlabel("Missing %")
            ax.axvline(5, color="#dc2626", lw=1, ls="--", alpha=0.6)
            for i, (_, row) in enumerate(has_missing.iterrows()):
                ax.text(row["Missing %"] + 0.05, i, f"{row['Missing %']:.1f}%", va="center", fontsize=8)
            ax.grid(True, axis="x", alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

    st.divider()

    # Preprocessing notes
    st.subheader("Preprocessing & feature engineering (summary)")
    st.markdown("""
    Steps applied in data loading:
    1. Sort by date.
    2. Convert pressure from Pa to hPa.
    3. Fill snow_depth NaN with 0.
    4. Add calendar features and cyclical encodings.
    5. Create lag (lag-1, lag-7) and rolling (7-day mean, 7-day sum) features.
    6. Drop first ~7 rows with NaN after lag/rolling.
    """)

    st.markdown(f"Final feature set: {len(FEATURE_COLS)} columns (raw + engineered).")
    st.divider()

    # Correlation heatmap
    st.subheader("Correlation heatmap", anchor="correlation-heatmap", help="Pearson correlation between raw features and target. Engineered features are excluded to avoid spurious correlations.")
    heatmap_cols = ["mean_temp", "max_temp", "min_temp", "cloud_cover", "sunshine", "global_radiation", "precipitation", "pressure", "snow_depth"]
    heatmap_cols = [c for c in heatmap_cols if c in df.columns]
    corr_df = df[heatmap_cols].dropna().corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="RdYlBu_r", center=0, linewidths=0.4, linecolor="white", annot_kws={"size": 9}, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Pearson correlation — raw features")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.caption("Mean temp correlates strongly with max/min temp and with sunshine; cloud_cover is negatively correlated.")
    st.divider()

    # Distribution of mean temperature
    st.subheader("Distribution of mean temperature")
    temp_vals = df["mean_temp"].dropna()
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.hist(temp_vals, bins=60, alpha=0.72, edgecolor="white", linewidth=0.4)
    ax.axvline(temp_vals.mean(), color="#dc2626", lw=2.0, ls="--", label=f"Mean = {temp_vals.mean():.1f}°C")
    ax.axvline(temp_vals.median(), color="#f59e0b", lw=1.8, ls=":", label=f"Median = {temp_vals.median():.1f}°C")
    ax.set_xlabel("Mean Temperature (°C)")
    ax.set_ylabel("Number of days")
    ax.set_title("Distribution of daily mean temperature")
    ax.legend(fontsize=9, framealpha=0)
    ax.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Mean", f"{temp_vals.mean():.2f}°C")
    col_s2.metric("Std dev", f"{temp_vals.std():.2f}°C")
    col_s3.metric("Range", f"{temp_vals.min():.1f}°C to {temp_vals.max():.1f}°C")

    st.divider()

    # Key findings
    st.subheader("Key findings")
    st.markdown("""
    - Strong lag-1 autocorrelation: yesterday's temperature is the single strongest predictor.
    - Clear seasonal cycle: cyclical encoding is necessary.
    - Sunshine and radiation add signal beyond seasonality.
    - Missing data is limited; snow_depth handled by zero fill.
    """)