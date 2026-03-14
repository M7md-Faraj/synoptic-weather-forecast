import os
import sys
import warnings
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="London Weather ML", layout="wide")

st.markdown(
    """
    <style>
    .main .block-container { padding: 2rem 2.8rem 4rem; max-width: 1100px; }
    h1 { font-size: 1.75rem !important; }
    .app-footer { margin-top: 2rem; font-size: 0.78rem; opacity: 0.7; text-align: center; }
    </style>
    """,
    unsafe_allow_html=True,
)

import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.edgecolor": "#cbd5e1",
    "text.color": "#1e293b",
    "xtick.color": "#64748b",
    "ytick.color": "#64748b",
    "grid.color": "#f1f5f9",
    "font.family": "sans-serif",
    "figure.dpi": 100,
})

from utils.data_loader import load_data
from utils.model_utils import load_model_bundles, load_results
import train_models

@st.cache_data(show_spinner="Loading dataset…")
def get_data():
    return load_data()

@st.cache_resource(show_spinner="Loading trained models…")
def get_bundles():
    try:
        return load_model_bundles()
    except FileNotFoundError:
        return None

df = get_data()
bundles = get_bundles()

with st.sidebar:
    st.markdown("### London Temperature Forecasting")
    st.caption("Synoptic ML project")
    st.divider()

    page = st.radio("Go to", options=["Home", "EDA", "Models", "Forecast"], index=0)
    st.divider()

    st.markdown("### Dataset")
    st.caption(f"Location: London Heathrow")
    st.caption(f"Records: {len(df):,}")
    st.caption("Target: mean_temp (°C)")
    st.divider()

    if bundles is None:
        st.error("Model files not found. Run training or retrain below.", icon="🚫")
    else:
        results = load_results()
        best = results.get("best_model", "Random Forest")
        best_r2 = results.get(best, {}).get("R2", 0)
        st.success(f"Models loaded — Best: {best} (R²={best_r2:.4f})", icon="✅")

    st.divider()
    st.markdown("### Model controls")
    model_choice = st.selectbox(
        "Forecast model (default)",
        options=["Linear Regression", "Decision Tree", "Random Forest"],
        index=2 if bundles is None else ["Linear Regression", "Decision Tree", "Random Forest"].index(best)
    )

    with st.expander("Hyperparameters", expanded=False):
        dt_max_depth = st.slider("DT max_depth", min_value=2, max_value=30, value=8, step=1)
        dt_min_samples_leaf = st.slider("DT min_samples_leaf", min_value=1, max_value=20, value=1, step=1)
        rf_n_estimators = st.number_input("RF n_estimators", min_value=10, max_value=1000, value=100, step=10)
        rf_max_depth = st.slider("RF max_depth", min_value=2, max_value=50, value=12, step=1)
        rf_min_samples_leaf = st.slider("RF min_samples_leaf", min_value=1, max_value=20, value=1, step=1)
        lr_fit_intercept = st.checkbox("LR fit_intercept", value=True)

    st.divider()
    # show existing training progress (if any)
    tp = st.session_state.get("training_progress")
    if tp and tp.get("percent", 100) < 100:
        st.markdown("**Training in progress**")
        st.progress(tp.get("percent", 0))
        st.caption(tp.get("message", "Training..."))

    do_retrain = st.button("Retrain models with these settings")
    if do_retrain:
        hyperparams = {
            "Decision Tree": {"max_depth": int(dt_max_depth), "min_samples_leaf": int(dt_min_samples_leaf)},
            "Random Forest": {"n_estimators": int(rf_n_estimators), "max_depth": int(rf_max_depth), "min_samples_leaf": int(rf_min_samples_leaf)},
            "Linear Regression": {"fit_intercept": bool(lr_fit_intercept)},
        }

        # create visible progress UI in the sidebar and keep session state updated
        progress_bar = st.progress(0)
        status_text = st.empty()
        st.session_state["training_progress"] = {"percent": 0, "message": "Queued"}

        def progress_cb(pct, msg):
            # update UI elements inside this rerun
            try:
                progress_bar.progress(int(pct))
                status_text.text(msg)
                st.session_state["training_progress"] = {"percent": int(pct), "message": msg}
            except Exception:
                pass

        with st.spinner("Retraining models..."):
            try:
                # invalidate cached bundles resource so models are reloaded after training
                get_bundles.clear()
                results = train_models.train_and_save(hyperparams=hyperparams, progress_callback=progress_cb)
                # reload bundles into cache
                bundles = get_bundles()
                progress_cb(100, "Training finished")
                st.success("Retraining finished and models saved.", icon="✅")
            except Exception as e:
                progress_cb(0, f"Retraining failed: {e}")
                st.exception(f"Retraining failed: {e}")

    st.divider()

from pages import page_home, page_eda, page_models, page_forecast

if page == "Home":
    page_home.render(df)

elif page == "EDA":
    page_eda.render(df)

elif page == "Models":
    if bundles is None:
        st.error("Model files not found. Retrain from the sidebar first.")
    else:
        page_models.render(df, bundles)

elif page == "Forecast":
    if bundles is None:
        st.error("Model files not found. Retrain from the sidebar first.")
    else:
        page_forecast.render(df, bundles, selected_model=model_choice)