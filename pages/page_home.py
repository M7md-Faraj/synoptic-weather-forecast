import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from utils.data_loader import MONTH_LABELS

def render(df):
    st.title("London Daily Temperature")
    st.write("A compact demo comparing 3 regression algorithms on historical daily data.")
    st.divider()

    st.subheader("Project aim")
    st.markdown("""
    Compare Linear Regression, Decision Tree, and Random Forest for daily mean temperature prediction.
    Key questions:
    1. Can ML beat naive baselines?
    2. Which algorithm generalises best?
    3. Which features matter most?
    """)

    st.divider()
    st.subheader("Dataset summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Records", f"{len(df):,}")
    col2.metric("Date range", "1979 – 2020")
    col3.metric("Raw features", "9")
    col4.metric("Engineered features", "18")

    st.markdown("Data source: historical daily observations at Heathrow (used as a consistent station record).")
    st.divider()

    st.subheader("Graph 1 — Annual mean temperature (1979–2020)")
    annual = df.groupby("year")["mean_temp"].mean()
    trend = np.polyfit(annual.index, annual.values, 1)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.fill_between(annual.index, annual.values, alpha=0.12)
    ax.plot(annual.index, annual.values, lw=2.0, marker="o", ms=3.5, label="Annual mean")
    ax.plot(annual.index, np.polyval(trend, annual.index), "--", color="#dc2626", lw=1.8, label=f"Trend: +{trend[0]*10:.2f}°C / decade")
    ax.set_xlabel("Year"); ax.set_ylabel("Mean Temperature (°C)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.4)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
    st.caption(f"Approximate trend over study period: {trend[0]*(2020-1979):.1f}°C")

    st.divider()
    st.subheader("Graph 2 — Monthly average (climatology)")
    monthly = df.groupby("month")["mean_temp"].mean()
    colours = ["#3b82f6" if t < 8 else "#f59e0b" if t > 16 else "#14b8a6" for t in monthly.values]
    fig2, ax2 = plt.subplots(figsize=(10, 3.8))
    bars = ax2.bar(MONTH_LABELS, monthly.values, color=colours, width=0.65, alpha=0.88)
    ax2.set_xlabel("Month"); ax2.set_ylabel("Mean Temperature (°C)")
    for bar, val in zip(bars, monthly.values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15, f"{val:.1f}°", ha="center", fontsize=8.5)
    ax2.grid(True, axis="y", alpha=0.4)
    plt.tight_layout(); st.pyplot(fig2, use_container_width=True); plt.close()
    st.caption("Seasonal amplitude is large; cyclical encoding is required for temporal features.")