import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime

# project imports
from src.data_loader import load_csv, preprocess
from src.train import train_random_forest_progress, train_sgd_progress
from src.analysis import (
    summary_stats,
    plotly_time_series,
    correlation_heatmap_plotly,
    distribution_plot,
    weather_condition_summary,
)
from src.models import (
    save_model,
    list_models,
    get_latest_by_base,
    load_model,
    extract_feature_importance,
)

# ---- CONFIG ----
st.set_page_config(page_title="Synoptic Weather Forecast", layout="wide", initial_sidebar_state="expanded")
BASE = Path.cwd()
MODELS_DIR = BASE / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ---- THEME CONTROL (sidebar will set this) ----
if "theme" not in st.session_state:
    st.session_state["theme"] = "light"  # default

def inject_theme_css(theme: str):
    """Inject CSS for dark or light theme and card styles.
    Includes broader rules for Streamlit widgets so text/buttons remain visible in dark mode.
    """
    if theme == "dark":
        page_bg = "#071128"
        sidebar_bg = "linear-gradient(180deg, #0f172a, #071128)"
        text_color = "#E6EEF8"
        card_bg = "rgba(255,255,255,0.02)"
        card_border = "rgba(255,255,255,0.04)"
        widget_bg = "rgba(255,255,255,0.03)"
        widget_border = "rgba(255,255,255,0.06)"
    else:
        page_bg = "#f7f9fc"
        sidebar_bg = "linear-gradient(180deg, #ffffff, #f4f6f9)"
        text_color = "#0b2545"
        card_bg = "rgba(0,0,0,0.03)"
        card_border = "rgba(0,0,0,0.06)"
        widget_bg = "rgba(0,0,0,0.02)"
        widget_border = "rgba(0,0,0,0.04)"

    css = f"""
    <style>
    :root{{ --page-bg: {page_bg}; --sidebar-bg: {sidebar_bg}; --text-color: {text_color}; --card-bg: {card_bg}; --card-border: {card_border}; --widget-bg: {widget_bg}; --widget-border: {widget_border}; }}

    /* page + sidebar backgrounds */
    .css-18e3th9 {{ background: var(--page-bg) !important; }} /* main page content container */
    [data-testid="stSidebar"] {{ background: {sidebar_bg}; color: {text_color}; }}

    /* Title and card styles */
    .title {{ color: var(--text-color); font-weight:700; }}
    .card {{
        background: var(--card-bg);
        padding: 12px;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        margin-bottom: 12px;
        border: 1px solid var(--card-border);
    }}
    .card h3 {{ margin: 4px 0 8px 0; color: var(--text-color); }}

    /* make core widgets readable in both themes */
    .stButton>button, button {{
        color: var(--text-color) !important;
        background: var(--widget-bg) !important;
        border: 1px solid var(--widget-border) !important;
    }}
    .stSelectbox, .stTextInput, .stNumberInput, .stMultiSelect, .stFileUploader {{
        color: var(--text-color) !important;
    }}

    /* Dataframe and charts containers */
    .stDataFrame, .stPlotlyChart {{ color: var(--text-color) !important; }}

    /* Emoji animations */
    .emoji-anim {{
        display: inline-block;
        font-size: 46px;
        transform-origin: 50% 50%;
        animation: floaty 2.5s ease-in-out infinite;
        vertical-align: middle;
    }}
    .rain-emoji {{
        position: relative;
        font-size: 46px;
        display:inline-block;
    }}
    .raindrop {{
        position: absolute;
        top: 44px;
        left: 18px;
        font-size: 16px;
        animation: drop 1s linear infinite;
        opacity: 0.9;
    }}
    @keyframes floaty {{
        0% {{ transform: translateY(0px) rotate(0deg); }}
        50% {{ transform: translateY(-6px) rotate(6deg); }}
        100% {{ transform: translateY(0px) rotate(0deg); }}
    }}
    @keyframes drop {{
        0% {{ transform: translateY(0px); opacity: 0; }}
        10% {{ opacity: 1; }}
        100% {{ transform: translateY(18px); opacity: 0; }}
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# inject initial theme CSS
inject_theme_css(st.session_state["theme"])

# ---- helper utilities ----
@st.cache_resource
def load_model_cached(path_str: str):
    return load_model(path_str)


def refresh_model_list():
    return list_models()


def safe_get(df, col, default=None):
    return df[col] if col in df.columns else default


def render_card(title: str, body_callable=None, badge: str = None):
    """Render a simple card with a title and run body_callable() to render contents."""
    header_html = f"<div class='card'><h3>{title} {badge or ''}</h3>"
    st.markdown(header_html, unsafe_allow_html=True)
    if body_callable:
        body_callable()
    st.markdown("</div>", unsafe_allow_html=True)

# ---- SIDEBAR: Controls (cleaner + moved dataset overview into an expander) ----
with st.sidebar:
    st.markdown("## Controls")

    # theme toggle
    theme_choice = st.radio("Theme", options=["dark", "light"], index=0 if st.session_state["theme"]=="dark" else 1)
    if theme_choice != st.session_state["theme"]:
        st.session_state["theme"] = theme_choice
        inject_theme_css(theme_choice)

    uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")
            df = pd.DataFrame()
    else:
        st.info("Using CSV from data/weather.csv")
        try:
            df = load_csv("data/weather.csv")
        except Exception as e:
            st.error(f"Failed to load local CSV: {e}")
            df = pd.DataFrame()

    if df.empty:
        st.stop()

    # preprocess early
    df = preprocess(df)

    # Move dataset preview + summary into sidebar expander (they're useful but not the main view)
    with st.expander("Dataset overview", expanded=True):
        st.markdown("**Head (first 10 rows)**")
        st.dataframe(df.head(10))
        st.markdown("---")
        st.markdown("**Summary statistics (numeric columns)**")
        st.write(summary_stats(df.select_dtypes(include=[np.number])))

    # target & features
    target = st.selectbox("Target variable", options=["mean_temp", "max_temp", "min_temp", "precipitation"], index=0)
    default_features = [c for c in df.columns if c not in ["date", target]]
    features = st.multiselect("Features (choose at least 3)", options=default_features, default=default_features[:6])

    # put advanced / training controls in an expander to reduce clutter
    with st.expander("Advanced / Training controls", expanded=False):
        # provide RF and SGD inputs even if not selected (prevents NameError in train_all flow)
        n_estimators = st.number_input("RF: n_estimators", min_value=10, max_value=2000, value=100, step=10)
        rf_step = st.number_input("RF: progress step (trees per update)", min_value=1, max_value=200, value=10)
        epochs = st.number_input("SGD: epochs", min_value=1, max_value=500, value=20)

        # model selection (training)
        model_choice = st.selectbox("Train model type", ["random_forest", "sgd"], index=0)

        st.markdown("---")
        st.subheader("Pretrained models")

        # session state initialization
        if "prefer_pretrained" not in st.session_state:
            st.session_state["prefer_pretrained"] = True
        if "selected_model_file" not in st.session_state:
            latest_rf = get_latest_by_base("random_forest")
            latest_sgd = get_latest_by_base("sgd")
            default = latest_rf or latest_sgd
            st.session_state["selected_model_file"] = default["filename"] if default else None

        # list available models
        model_entries = refresh_model_list()
        model_options = [m["filename"] for m in model_entries]

        if model_options:
            opts = ["(none)"] + model_options
            chosen_index = 0
            if st.session_state["selected_model_file"] in model_options:
                chosen_index = model_options.index(st.session_state["selected_model_file"]) + 1
            sel = st.selectbox("Choose pretrained model", options=opts, index=chosen_index)
            st.session_state["selected_model_file"] = None if sel == "(none)" else sel
        else:
            st.info("No pretrained models found in /models.")

        st.session_state["prefer_pretrained"] = st.checkbox(
            "Prefer pretrained model by default", value=st.session_state["prefer_pretrained"]
        )

        if st.button("Clear pretrained selection"):
            st.session_state["selected_model_file"] = None
            st.session_state["prefer_pretrained"] = False
            st.success("Cleared pretrained selection — dashboard will use newly trained models by default.")

        st.markdown("---")
        st.subheader("Training controls")
        train_rf_btn = st.button("Train RF (with progress)")
        train_sgd_btn = st.button("Train SGD (with progress)")
        train_all = st.button("Train and save both (RF + SGD)")

    st.markdown("---")
    st.subheader("Prediction")
    predict_n = st.number_input("Predict last N rows", min_value=1, max_value=365, value=10)
    predict_btn = st.button("Predict (last rows)")

# ---- MAIN LAYOUT ----
# place a nice emoji next to the title so it's immediately visible to users
st.markdown("<div class='card'><h1 class='title'><span class='emoji-anim' title='weather'>🌤️</span> Synoptic Weather Forecast App</h1></div>", unsafe_allow_html=True)

# layout: make totals + emoji the first thing, then time series below them
col1, col2 = st.columns([2, 1])

# -----------------------
# Totals row (1 x 4 metrics + emoji) - shown before time series
# -----------------------
with col1:
    def render_totals_row():
        # get totals and a small conditions figure (we'll not plot fig here; it's for the Weather overview card)
        totals, _ = weather_condition_summary(df, temp_col="mean_temp", precip_col="precipitation")

        # mapping for nicer labels
        label_map = {
            "rainy_days": "Rainy days",
            "sunny_days": "Sunny days",
            "sunshine": "Sunshine (hrs)",
            "sunshine_hours": "Sunshine (hrs)",
            "hot": "Hot days",
            "cold": "Cold days",
            "snow_days": "Snow days",
            "total_days": "Total days",
            "mixed": "Mixed days",
            "precip_days": "Precip days",
        }

        # preferred keys to try to show
        preferred = ["rainy_days", "sunny_days", "sunshine", "sunshine_hours", "hot", "cold", "snow_days", "total_days"]

        chosen = []
        for k in preferred:
            if k in totals:
                chosen.append(k)
            if len(chosen) == 4:
                break

        # fallback: pick top 4 keys by value
        if len(chosen) < 4:
            remaining_keys = [k for k in totals.keys() if k not in chosen]
            remaining_sorted = sorted(remaining_keys, key=lambda x: totals.get(x, 0), reverse=True)
            for k in remaining_sorted:
                chosen.append(k)
                if len(chosen) == 4:
                    break

        # if still fewer than 4, pad with placeholders
        while len(chosen) < 4:
            chosen.append(f"metric_{len(chosen)+1}")

        # prepare display values
        display_metrics = []
        for k in chosen:
            v = totals.get(k, 0)
            label = label_map.get(k, k.replace("_", " ").title())
            display_metrics.append((label, v))

        # layout: 4 equal columns for metrics + small column for emoji
        m1, m2, m3, m4, emo_col = st.columns([1, 1, 1, 1, 0.5])

        # display metrics using st.metric for visual clarity
        m1.metric(display_metrics[0][0], f"{display_metrics[0][1]}")
        m2.metric(display_metrics[1][0], f"{display_metrics[1][1]}")
        m3.metric(display_metrics[2][0], f"{display_metrics[2][1]}")
        m4.metric(display_metrics[3][0], f"{display_metrics[3][1]}")

        # choose emoji based on totals (same logic as overview)
        hot = totals.get("hot", 0)
        rainy = totals.get("rainy_days", 0)
        cold = totals.get("cold", 0)

        if rainy >= max(hot, cold):
            emoji_html = """
            <div style='display:flex; align-items:center; justify-content:center; gap:6px;'>
              <div class='rain-emoji'>☁️
                <div class='raindrop' style='left:6px; animation-delay:0s'>💧</div>
                <div class='raindrop' style='left:18px; animation-delay:0.25s'>💧</div>
                <div class='raindrop' style='left:30px; animation-delay:0.5s'>💧</div>
              </div>
            </div>
            """
        elif hot >= max(rainy, cold):
            emoji_html = "<div style='display:flex; align-items:center; justify-content:center;'><div class='emoji-anim'>☀️</div></div>"
        elif cold >= max(rainy, hot):
            emoji_html = "<div style='display:flex; align-items:center; justify-content:center;'><div class='emoji-anim'>❄️</div></div>"
        else:
            emoji_html = "<div style='display:flex; align-items:center; justify-content:center;'><div class='emoji-anim'>🌤️</div></div>"

        emo_col.markdown(emoji_html, unsafe_allow_html=True)

    # render the totals row card (one-row block)
    render_card("Overview — key totals", render_totals_row)

    # Time series card — next priority (below totals)
    def body_ts():
        ts_col = st.selectbox(
            "Choose column to plot",
            options=["mean_temp", "max_temp", "min_temp", "precipitation", "sunshine"],
            key="ts_col",
        )
        fig_ts = plotly_time_series(df, ts_col)
        st.plotly_chart(fig_ts, use_container_width=True)

    render_card("Time series", body_ts)

    # Correlation card
    def body_corr():
        fig_corr = correlation_heatmap_plotly(df)
        st.plotly_chart(fig_corr, use_container_width=True)
    render_card("Correlation matrix", body_corr)

    # Distribution card
    def body_dist():
        ts_col = safe_get(df, "mean_temp", df.columns[0])
        fig_dist = distribution_plot(df, ts_col)
        st.plotly_chart(fig_dist, use_container_width=True)
    render_card("Distribution", body_dist)

with col2:
    # Model & Training card
    def body_model():
        st.markdown("### Latest saved models")
        latest_rf = get_latest_by_base("random_forest")
        latest_sgd = get_latest_by_base("sgd")
        if latest_rf:
            st.write(f"- random_forest: {latest_rf['filename']} — saved {latest_rf['timestamp']} — metrics: {latest_rf.get('metrics',{})}")
        else:
            st.write("- random_forest: (none)")
        if latest_sgd:
            st.write(f"- sgd: {latest_sgd['filename']} — saved {latest_sgd['timestamp']} — metrics: {latest_sgd.get('metrics',{})}")
        else:
            st.write("- sgd: (none)")

        st.markdown("#### Live training progress")
        metrics_area = st.empty()
        progress_bar = st.empty()
        history_chart = st.empty()

        def _save_and_set_active(model_obj, base_name: str, metrics: dict):
            saved_path, meta_entry = save_model(model_obj, base_name=base_name, metrics=metrics)
            st.session_state["selected_model_file"] = meta_entry["filename"]
            st.session_state["prefer_pretrained"] = True
            return saved_path, meta_entry

        # run trainings based on button clicks
        if 'train_all' in locals() and (train_all or train_rf_btn or train_sgd_btn):
            tasks = []
            if train_all:
                tasks = ["random_forest", "sgd"]
            else:
                if train_rf_btn:
                    tasks = ["random_forest"]
                if train_sgd_btn:
                    tasks = ["sgd"]

            for task in tasks:
                st.info(f"Starting training: {task}")
                history = []
                if task == "random_forest":
                    total = int(n_estimators)
                    step = int(rf_step)
                    gen = train_random_forest_progress(df, features, target, n_estimators=total, step=step)
                    for current, model_partial, metrics, _ in gen:
                        history.append({"step": current, **metrics})
                        progress_val = min(1.0, current / total)
                        progress_bar.progress(progress_val)
                        metrics_df = pd.DataFrame(history).set_index("step")
                        metrics_area.dataframe(metrics_df)
                        history_chart.line_chart(metrics_df)
                    saved_path, meta = _save_and_set_active(model_partial, "random_forest", metrics)
                    st.success(f"RF training finished and saved to {meta['filename']}")
                    progress_bar.empty()
                elif task == "sgd":
                    total_epochs = int(epochs)
                    gen = train_sgd_progress(df, features, target, epochs=total_epochs)
                    for epoch, model_partial, metrics, _ in gen:
                        history.append({"step": epoch, **metrics})
                        progress_val = min(1.0, epoch / total_epochs)
                        progress_bar.progress(progress_val)
                        metrics_df = pd.DataFrame(history).set_index("step")
                        metrics_area.dataframe(metrics_df)
                        history_chart.line_chart(metrics_df)
                    saved_path, meta = _save_and_set_active(model_partial, "sgd", metrics)
                    st.success(f"SGD training finished and saved to {meta['filename']}")
                    progress_bar.empty()

    render_card("Model & Training", body_model)

    # Prediction card
    def body_predict():
        st.markdown("### Prediction / Forecast")
        if not features or len(features) < 1:
            st.warning("Please select at least one feature to enable prediction.")
            return

        if predict_btn:
            chosen_path = None
            if st.session_state.get("prefer_pretrained", True) and st.session_state.get("selected_model_file"):
                chosen_filename = st.session_state["selected_model_file"]
                chosen_path = MODELS_DIR / chosen_filename
            else:
                latest = get_latest_by_base(model_choice)
                chosen_path = Path(latest["path"]) if latest else None

            if not chosen_path or not chosen_path.exists():
                st.error("No model available. Please train or select a pretrained model first.")
                return
            try:
                model = load_model_cached(str(chosen_path))
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                return

            # slice X_pred and attempt to align with the model's expected features
            X_pred = df[features].tail(int(predict_n)).copy()

            # Try to infer model feature names. scikit-learn estimators often expose feature_names_in_
            model_features = getattr(model, "feature_names_in_", None)

            # Fallback: try to load nearby metadata file (common pattern in our project)
            if model_features is None:
                meta_path_json = chosen_path.with_suffix(chosen_path.suffix + ".meta.json")
                meta_path = chosen_path.with_suffix(chosen_path.suffix + ".meta")
                try:
                    if meta_path_json.exists():
                        with open(meta_path_json, "r") as fh:
                            meta = json.load(fh)
                            model_features = meta.get("features")
                    elif meta_path.exists():
                        meta = joblib.load(meta_path)
                        model_features = meta.get("features")
                except Exception:
                    model_features = None

            # If we know expected features, reindex / fill missing / drop extras so predict won't fail
            if model_features is not None:
                model_features = list(model_features)
                missing = [f for f in model_features if f not in X_pred.columns]
                extra = [f for f in X_pred.columns if f not in model_features]
                if missing:
                    st.warning(f"Model expects features {model_features}. Missing {missing}. Filling missing features with 0s.")
                    for m in missing:
                        X_pred[m] = 0
                if extra:
                    st.info(f"Dropping extra features not expected by model: {extra}")
                # ensure correct column order
                X_pred = X_pred[[c for c in model_features if c in X_pred.columns]]

            # final safety guard: try predict and catch dimensionality errors with a helpful message
            try:
                preds = model.predict(X_pred)
            except ValueError as e:
                st.error(
                    "Prediction failed due to feature mismatch.\n"
                    "Details: {}\n".format(e)
                )
                st.info("Tip: make sure the features you selected match the features used to train the model (same names & order). If you retrained, try selecting the saved model under Pretrained models.")
                return
            except Exception as e:
                st.error(f"Unexpected error during prediction: {e}")
                return

            out = X_pred.copy().reset_index(drop=True)
            out["prediction"] = preds
            out["date"] = df["date"].tail(int(predict_n)).values
            st.dataframe(out)
            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv")

            # feature importance
            try:
                fi = extract_feature_importance(model, X_pred.columns.tolist())
                if fi is not None:
                    st.subheader("Feature importance")
                    st.bar_chart(fi)
                else:
                    st.info("Feature importance not available for this model.")
            except Exception as e:
                st.info("Could not extract feature importance: " + str(e))

    render_card("Prediction", body_predict)

    # Weather overview + animated emoji card (main column still shows an emoji)
    def body_overview():
        totals, fig_conditions = weather_condition_summary(df, temp_col="mean_temp", precip_col="precipitation")

        # ensure each bar (trace) has a distinct color so different weather types are obvious
        palette = ["#FFB400", "#4DA6FF", "#A3A3A3", "#00C49A", "#C70039", "#8A2BE2"]
        try:
            for i, trace in enumerate(fig_conditions.data):
                # assign color per trace
                trace.marker.color = palette[i % len(palette)]
        except Exception:
            # if fig_conditions is not the expected object, ignore and plot normally
            pass

        st.plotly_chart(fig_conditions, use_container_width=True)
        # display totals in a friendly way here as well (small)
        st.write("Totals summary:")
        pretty = {k: v for k, v in totals.items()}
        st.write(pretty)

        # choose emoji to display based on counts — put a visible emoji block in the main view
        hot = totals.get("hot", 0)
        rainy = totals.get("rainy_days", 0)
        cold = totals.get("cold", 0)

        emoji_html = ""
        if rainy >= max(hot, cold):
            # rainy animation (cloud + raindrops)
            emoji_html = """
            <div style='display:flex; align-items:center; gap:12px;'>
              <div class='rain-emoji'>☁️
                <div class='raindrop' style='left:12px; animation-delay:0s'>💧</div>
                <div class='raindrop' style='left:24px; animation-delay:0.3s'>💧</div>
                <div class='raindrop' style='left:36px; animation-delay:0.6s'>💧</div>
              </div>
              <div style='font-size:16px; color:var(--text-color)'>Looks rainy — bring an umbrella!</div>
            </div>
            """
        elif hot >= max(rainy, cold):
            # sunny animation
            emoji_html = "<div style='display:flex; align-items:center; gap:12px;'><div class='emoji-anim'>☀️</div><div style='font-size:16px; color:var(--text-color)'>Sunny / Warm days are common.</div></div>"
        elif cold >= max(rainy, hot):
            emoji_html = "<div style='display:flex; align-items:center; gap:12px;'><div class='emoji-anim'>❄️</div><div style='font-size:16px; color:var(--text-color)'>Cold spells are frequent.</div></div>"
        else:
            emoji_html = "<div style='display:flex; align-items:center; gap:12px;'><div class='emoji-anim'>🌤️</div><div style='font-size:16px; color:var(--text-color)'>Mixed conditions observed.</div></div>"

        st.markdown(emoji_html, unsafe_allow_html=True)

    render_card("Weather overview", body_overview)

# ---- Footer quick naive forecast ----
st.markdown("<div class='card'><h3>Quick forecast (naive)</h3></div>", unsafe_allow_html=True)
left, middle, right = st.columns(3)
with left:
    days = st.number_input("Days ahead (naive average)", min_value=1, max_value=14, value=3)
with middle:
    method = st.selectbox("Method", ["last_value", "moving_average"])
with right:
    run_forecast = st.button("Run forecast")

if run_forecast:
    last_vals = df[target].tail(7)
    if method == "last_value":
        forecast = [float(last_vals.iloc[-1])] * int(days)
    else:
        forecast = [float(last_vals.mean())] * int(days)
    out = pd.DataFrame({"day_ahead": list(range(1, int(days) + 1)), "forecast": forecast})
    st.write(out)
    st.download_button("Download forecast", data=out.to_csv(index=False).encode("utf-8"), file_name="quick_forecast.csv")