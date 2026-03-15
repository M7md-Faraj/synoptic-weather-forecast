import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from utils.data_loader import MONTH_LABELS
from utils.model_utils import load_results, MODEL_COLORS, MODEL_NAMES, get_feature_importance

def _pipeline_markdown():
    return """
    **Data processing pipeline — high level**

    1. **Load & Inspect** — raw station observations, basic checks (missing values, units).
    2. **Clean** — fill or impute missing values (snow depth → 0, others → median), convert units.
    3. **Engineer** — calendar features, cyclical encodings, lag-1/lag-7, 7-day rolling means and sums.
    4. **Split (chronological)** — 80% train / 20% test (no shuffle) to avoid leakage.
    5. **Train & Validate** — train LR, Decision Tree, Random Forest. Compare against baselines (persistence, climatology).
    6. **Time CV (diagnostic)** — 5-fold time-slices across the training set to check stability of skill.
    7. **Evaluate** — use MAE, RMSE, MAPE%, R² and residual diagnostics.
    8. **Forecast** — build climatological/feature-based forecast input and produce day-by-day predictions.
    9. **Communicate** — make scope, limitations and uncertainty explicit before sharing forecasts.
    """

def render(df):
    st.title("London Daily Temperature Analysis — Home")
    st.write("Overview and pipeline for the project — how the dataset is transformed from raw observations to model predictions.")
    st.divider()

    # Pipeline
    st.subheader("Project pipeline")
    st.markdown(_pipeline_markdown())
    st.divider()

    # Dataset summary metrics
    st.subheader("Dataset snapshot")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Observation Period", f"{df['date'].min().date()} → {df['date'].max().date()}")
    col3.metric("Original Variables", "9")
    col4.metric("Derived Variables", "18")
    st.caption("Data from Heathrow long-term station record used for proof-of-concept forecasting.")

    st.divider()

    # Best model snapshot (from results.json)
    st.subheader("Best model snapshot")
    results = load_results()
    best = results.get("best_model", "Random Forest")
    best_metrics = results.get(best, {})
    st.markdown(f"**Best model (test set):** {best}")
    st.write(f"MAE: {best_metrics.get('MAE', 'N/A')} °C · RMSE: {best_metrics.get('RMSE', 'N/A')} °C · MAPE: {best_metrics.get('MAPE%', 'N/A')}% · R²: {best_metrics.get('R2', 'N/A')}")
    st.caption("Snapshot from the last training run — this helps viewers quickly see which model performed best on the reserved test set.")
    st.divider()

    # Feature engineering summary (linked to EDA)
    st.subheader("Feature engineering — groups and rationale")
    st.markdown("""
    **Why these groups were included**:

    - **Raw atmospheric**: observations (temperature extremes, radiation, pressure, precipitation). These are the immediate physical drivers of daily mean temperature.
    - **Cyclical encodings**: month/day-of-year encoded with sin/cos to capture seasonality without discontinuities.
    - **Calendar / ordinal**: year and day-of-year to allow long-term trends and coarse seasonality.
    - **Lag & rolling**: lag-1, lag-7 and 7-day rolling statistics capture persistence and short-term memory in the time series — this was the single strongest signal in EDA.
    """)
    st.caption("See the EDA page for the correlation heatmap and distribution plots that motivated each group.")

    # quick feature importance preview (top 5)
    try:
        fi_df = get_feature_importance(__import__("utils.model_utils", fromlist=["load_model_bundles"]).load_model_bundles())
        st.markdown("**Top feature importance (Random Forest)**")
        top5 = fi_df.head(5)
        for _, r in top5.iterrows():
            st.write(f"- {r['Feature']}: importance {r['Importance']:.4f}")
    except Exception:
        st.info("Feature importance not available (models not trained). Retrain models from the sidebar to see feature importances here.")

    st.divider()

    # Scope & Limitations
    st.subheader("Scope & limitations")
    st.markdown("""
    **Scope**
    - Short-range climatological forecasts of daily mean temperature for London Heathrow based on historical station records and engineered features.
    - Designed as a research prototype rather than an operational weather forecast.

    **Limitations & sources of uncertainty**
    - **Climatological inputs for forecasts**: When live atmospheric inputs are unavailable we use monthly medians — this reduces temporal fidelity.
    - **Station-only observations**: single-station biases exist and do not capture spatial fields (advection, frontal passages).
    - **Model uncertainty**: residuals and error bands (MAE, RMSE, MAPE%) reflect model limitations. Extreme events and abrupt changes will be harder to predict.
    - **Data quality**: missing values and measurement changes across decades can introduce biases; imputation reduces but does not eliminate these.
    """)
    st.caption("Always present forecast uncertainty and avoid overconfidence when publishing results.")

    st.divider()
    st.markdown("#### Quick actions")
    st.markdown("- Use the **Models** page to check baselines and time CV for model stability.\n- Use **Forecast** page to generate one-day or multi-day forecasts across models.")