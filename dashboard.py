import os
import sys
import warnings
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- page config ---
st.set_page_config(page_title="Synoptic Weather Forecast Dashboard", layout="wide")

# --- light styling for header & nav spacing ---
st.markdown(
    """
    <style>
    /* small visual tweaks for the app header area */
    .stApp .main .block-container { padding: 1.2rem 2rem 2.5rem; max-width: 1200px; }
    .app-brand { display:flex; align-items:center; gap:12px; margin-top:8px; margin-bottom:6px; }
    .app-logo { font-size:20px; font-weight:800; letter-spacing:0.6px; }
    .app-sub { font-size:12px; opacity:0.75; margin-top:-4px; }
    /* try to shrink top nav padding a bit (may be fragile across versions) */
    .stNavigation { padding: 6px 12px !important; margin-bottom: 6px; }
    /* make the page header stand out slightly */
    header .app-brand { margin-bottom: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.edgecolor": "#cbd5e1",
    "text.color": "#0f172a",
    "xtick.color": "#475569",
    "ytick.color": "#475569",
    "grid.color": "#f1f5f9",
    "font.family": "sans-serif",
    "figure.dpi": 100,
})

# --- project helpers (load data / models) ---
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

# import your pages modules (each should expose a render(...) function)
from pages import page_home, page_eda, page_models, page_forecast

# === Sidebar (keeps controls stateful across pages) ===
with st.sidebar:
    st.markdown("### London Temperature Forecasting")
    st.caption("Synoptic ML project")
    st.divider()

    st.markdown("### Dataset")
    st.caption(f"Location: London Heathrow")
    st.caption(f"Records: {len(df):,}")
    st.caption("Target: mean_temp (°C)")
    st.divider()

    bundles = get_bundles()
    if bundles is None:
        st.error("Model files not found. Run training or retrain below.", icon="🚫")
        results = {}
    else:
        results = load_results()
        best = results.get("best_model", "Random Forest")
        best_r2 = results.get(best, {}).get("R2", 0)
        st.success(f"Models loaded — Best: {best} (R²={best_r2:.4f})", icon="✅")

    st.divider()
    st.markdown("### Model controls")
    try:
        default_idx = ["Linear Regression", "Decision Tree", "Random Forest"].index(results.get("best_model", "Random Forest"))
    except Exception:
        default_idx = 2

    model_choice = st.selectbox(
        "Forecast model (default)",
        options=["Linear Regression", "Decision Tree", "Random Forest"],
        index=default_idx
    )

    with st.expander("Hyperparameters", expanded=False):
        dt_max_depth = st.slider("DT max_depth", min_value=2, max_value=30, value=8, step=1)
        dt_min_samples_leaf = st.slider("DT min_samples_leaf", min_value=1, max_value=20, value=1, step=1)
        rf_n_estimators = st.number_input("RF n_estimators", min_value=10, max_value=1000, value=100, step=10)
        rf_max_depth = st.slider("RF max_depth", min_value=2, max_value=50, value=12, step=1)
        rf_min_samples_leaf = st.slider("RF min_samples_leaf", min_value=1, max_value=20, value=1, step=1)
        lr_fit_intercept = st.checkbox("LR fit_intercept", value=True)

    st.divider()
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

        progress_bar = st.progress(0)
        status_text = st.empty()
        st.session_state["training_progress"] = {"percent": 0, "message": "Queued"}

        def progress_cb(pct, msg):
            try:
                progress_bar.progress(int(pct))
                status_text.text(msg)
                st.session_state["training_progress"] = {"percent": int(pct), "message": msg}
            except Exception:
                pass

        with st.spinner("Retraining models..."):
            try:
                get_bundles.clear()
                results = train_models.train_and_save(hyperparams=hyperparams, progress_callback=progress_cb)
                bundles = get_bundles()
                progress_cb(100, "Training finished")
                st.success("Retraining finished and models saved.", icon="✅")
            except Exception as e:
                progress_cb(0, f"Retraining failed: {e}")
                st.exception(f"Retraining failed: {e}")

    st.divider()
    st.markdown("<div class='app-footer'>Made for demonstration & analysis — not an operational forecast.</div>", unsafe_allow_html=True)

# === Page wrappers (zero-arg callables) ===
def _page_home():
    page_home.render(df)

def _page_eda():
    page_eda.render(df)

def _page_models():
    if bundles is None:
        st.error("Model files not found. Retrain from the sidebar first.")
    else:
        page_models.render(df, bundles)

def _page_forecast():
    if bundles is None:
        st.error("Model files not found. Retrain from the sidebar first.")
    else:
        page_forecast.render(df, bundles, selected_model=model_choice)

# --- Branded header just below the navigation (keeps the top area pretty) ---
col1, col2 = st.columns([1, 6])
with col1:
    st.markdown("<div class='app-brand'><div class='app-logo'>⛅ Synoptic</div></div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div class='app-sub'>London temperature forecasting · quick EDA · model training & forecasting</div>", unsafe_allow_html=True)

# === Build pages with nicer titles & icons using st.Page ===
# st.Page accepts a page-like object and allows supplying title & icon.
pages = [
    st.Page(_page_home, title="Home", icon="🏠"),
    st.Page(_page_eda, title="EDA", icon="📊"),
    st.Page(_page_models, title="Models", icon="🧠"),
    st.Page(_page_forecast, title="Forecast", icon="⛅"),
]

# run the navigation at the top — this prevents the default page sidebar from appearing
nav = st.navigation(pages, position="top", expanded=False)
nav.run()