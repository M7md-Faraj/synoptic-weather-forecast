import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from utils.data_loader import MONTH_LABELS


def render(df):

    # ---------------------------------------------------------
    # PAGE TITLE
    # ---------------------------------------------------------
    st.title("London Daily Temperature Analysis")
    st.write(
        "Interactive dashboard exploring historical temperature patterns "
        "and evaluating machine learning models for temperature prediction."
    )

    st.divider()

    # ---------------------------------------------------------
    # PROJECT OBJECTIVE
    # ---------------------------------------------------------
    st.subheader("Project Overview")

    st.markdown("""
    This project investigates how different machine learning algorithms perform 
    when predicting **daily mean temperature** using historical climate data.

    The models compared in this study include:

    • Linear Regression  
    • Decision Tree Regressor  
    • Random Forest Regressor  

    Key objectives:

    1. Evaluate whether machine learning models outperform simple baselines.
    2. Compare model generalisation performance.
    3. Identify which input variables contribute most to prediction accuracy.
    """)

    st.divider()

    # ---------------------------------------------------------
    # DATASET SUMMARY
    # ---------------------------------------------------------
    st.subheader("Dataset Information")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Observation Period", "1979 – 2020")
    col3.metric("Original Variables", "9")
    col4.metric("Derived Variables", "18")

    st.markdown(
        "The dataset consists of **daily meteorological observations from Heathrow Airport**, "
        "providing a consistent long-term station record for London."
    )

    st.divider()

    # ---------------------------------------------------------
    # DATA PREVIEW
    # ---------------------------------------------------------
    st.subheader("Sample of Dataset")

    st.dataframe(df.head(), use_container_width=True)

    st.divider()

    # ---------------------------------------------------------
    # ANNUAL TEMPERATURE TREND
    # ---------------------------------------------------------
    st.subheader("Long-Term Temperature Trend (1979–2020)")

    annual = df.groupby("year")["mean_temp"].mean()
    trend = np.polyfit(annual.index, annual.values, 1)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.fill_between(annual.index, annual.values, alpha=0.15)

    ax.plot(
        annual.index,
        annual.values,
        marker="o",
        ms=3.5,
        lw=2,
        label="Annual Mean Temperature",
    )

    ax.plot(
        annual.index,
        np.polyval(trend, annual.index),
        "--",
        color="#dc2626",
        lw=2,
        label=f"Trend: +{trend[0]*10:.2f}°C per decade",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Mean Temperature (°C)")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=9)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.caption(
        f"Estimated warming over the full study period: "
        f"{trend[0]*(2020-1979):.1f}°C."
    )

    st.divider()

    # ---------------------------------------------------------
    # MONTHLY CLIMATE PATTERN
    # ---------------------------------------------------------
    st.subheader("Average Monthly Temperature Pattern")

    monthly = df.groupby("month")["mean_temp"].mean()

    colours = [
        "#3b82f6" if t < 8 else "#f59e0b" if t > 16 else "#14b8a6"
        for t in monthly.values
    ]

    fig2, ax2 = plt.subplots(figsize=(10, 4))

    bars = ax2.bar(
        MONTH_LABELS,
        monthly.values,
        color=colours,
        width=0.65,
        alpha=0.9,
    )

    ax2.set_xlabel("Month")
    ax2.set_ylabel("Mean Temperature (°C)")
    ax2.grid(True, axis="y", alpha=0.4)

    for bar, val in zip(bars, monthly.values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{val:.1f}°",
            ha="center",
            fontsize=8.5,
        )

    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close()

    st.caption(
        "London exhibits strong seasonal variation. "
        "Because of this cyclic pattern, temporal variables such as month "
        "are encoded using cyclical transformations for machine learning models."
    )

    st.divider()

    # ---------------------------------------------------------
    # TEMPERATURE DISTRIBUTION
    # ---------------------------------------------------------
    st.subheader("Temperature Distribution")

    fig3, ax3 = plt.subplots(figsize=(8, 4))

    ax3.hist(df["mean_temp"], bins=40, alpha=0.8)

    ax3.set_xlabel("Temperature (°C)")
    ax3.set_ylabel("Frequency")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    st.caption("Histogram showing the distribution of daily mean temperatures.")