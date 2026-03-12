# dashboard.py
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
import requests
from datetime import datetime, timedelta

from src.data_loader import load_csv, preprocess
from src.train import train_random_forest_progress, train_sgd_progress
from src.analysis import (
    summary_stats,
    plotly_time_series,
    correlation_heatmap_plotly,
    distribution_plot,
    weather_condition_summary,
    build_5day_forecast,
    build_hourly_preview,
    get_current_conditions,
)
from src.models import (
    save_model,
    list_models,
    get_latest_by_base,
    load_model,
    extract_feature_importance,
)

# ---- CONFIG ----
st.set_page_config(page_title="Synoptic Weather Forecast", layout="wide", initial_sidebar_state="collapsed")
BASE = Path.cwd()
MODELS_DIR = BASE / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ---- THEME / CSS (includes animation CSS) ----
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

def inject_theme_css(theme: str):
    """Inject CSS including animated weather icon helpers and heavier dark shadows."""
    if theme == "dark":
        page_bg = "#0b0d0f"
        text_color = "#eef2f6"
        card_bg = "#1f2224"   # dark gray card background
        shadow_strong = "0 30px 60px rgba(0,0,0,0.85), 0 8px 18px rgba(0,0,0,0.7)"
        accent = "#4cd137"
    else:
        page_bg = "#f3f6f9"
        text_color = "#0b2545"
        card_bg = "#ffffff"
        shadow_strong = "0 18px 30px rgba(0,0,0,0.08), 0 6px 12px rgba(0,0,0,0.04)"
        accent = "#33cc33"

    css = f"""
    <style>
    .stApp {{ background: {page_bg} !important; color: {text_color} !important; }}
    .css-18e3th9, .block-container, .stApp .main {{ background: {page_bg} !important; }}

    .big-card {{
        background: {card_bg} !important;
        color: {text_color} !important;
        border-radius: 18px !important;
        padding: 24px !important;
        box-shadow: {shadow_strong} !important;
        margin-bottom: 16px !important;
    }}
    .small-card {{
        background: {card_bg} !important;
        color: {text_color} !important;
        border-radius: 12px !important;
        padding: 12px !important;
        box-shadow: 0 12px 30px rgba(0,0,0,0.45) !important;
        margin-bottom: 12px !important;
    }}

    .big-time {{ font-size: 72px; font-weight: 800; margin:8px 0; }}
    .big-city {{ font-size: 26px; font-weight:700; }}
    .big-temp {{ font-size: 110px; font-weight:900; line-height:0.9; }}
    .deg {{ font-size: 36px; vertical-align: super; }}

    .forecast-cell {{ border-radius: 12px; padding: 12px; text-align:center; display:inline-block; min-width:140px; margin-right:12px; }}

    .stat-tile {{
        display:flex;
        flex-direction:column;
        align-items:center;
        justify-content:center;
        padding:10px;
        border-radius:12px;
        min-width:110px;
    }}

    .cards-row {{ display:flex; gap:12px; flex-wrap:wrap; }}

    /* Animated icon helpers */
    .wx-icon {{ display:inline-block; vertical-align:middle; }}
    .sun-core {{ transform-origin:50% 50%; animation: sun-rotate 12s linear infinite; }}
    @keyframes sun-rotate {{
        from {{ transform: rotate(0deg); }}
        to {{ transform: rotate(360deg); }}
    }}

    .ray {{ transform-origin: 50% 50%; animation: ray-pulse 2.8s ease-in-out infinite; }}
    @keyframes ray-pulse {{
        0% {{ opacity: 0.85; transform: scale(1); }}
        50% {{ opacity: 0.5; transform: scale(1.08); }}
        100% {{ opacity: 0.85; transform: scale(1); }}
    }}

    .cloud-move {{ animation: cloud-move 8s linear infinite; }}
    @keyframes cloud-move {{
        0% {{ transform: translateX(-5px); }}
        50% {{ transform: translateX(5px); }}
        100% {{ transform: translateX(-5px); }}
    }}

    .rain-drop {{ animation: rain-fall 1.2s linear infinite; }}
    .rain-drop:nth-child(2) {{ animation-delay: 0.18s; }}
    .rain-drop:nth-child(3) {{ animation-delay: 0.36s; }}
    @keyframes rain-fall {{
        0% {{ transform: translateY(-8px); opacity: 0; }}
        10% {{ opacity: 1; }}
        100% {{ transform: translateY(18px); opacity: 0; }}
    }}

    .snow-flake {{ animation: snow-fall 2.6s linear infinite; }}
    .snow-flake:nth-child(2) {{ animation-delay: 0.3s; }}
    .snow-flake:nth-child(3) {{ animation-delay: 0.6s; }}
    @keyframes snow-fall {{
        0% {{ transform: translateY(-6px) rotate(0deg); opacity: 0; }}
        20% {{ opacity: 1; }}
        100% {{ transform: translateY(26px) rotate(180deg); opacity: 0; }}
    }}

    .bolt {{ animation: bolt-flash 1.8s linear infinite; opacity: 0; }}
    @keyframes bolt-flash {{
        0% {{ opacity: 0; }}
        45% {{ opacity: 1; transform: translateY(0) scale(1); }}
        50% {{ opacity: 1; transform: translateY(-2px) scale(1.03); }}
        60% {{ opacity: 0; }}
        100% {{ opacity: 0; }}
    }}

    /* responsive tweaks */
    @media (max-width: 900px) {{
        .big-temp {{ font-size: 72px; }}
        .big-time {{ font-size: 48px; }}
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# inject for initial theme
inject_theme_css(st.session_state["theme"])

# ---- helpers & cached model loader ----
@st.cache_resource
def load_model_cached(path_str: str):
    return load_model(path_str)

def refresh_model_list():
    return list_models()

def safe_get(df, col, default=None):
    return df[col] if col in df.columns else default

# session variables
if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None
if "use_forecast_main" not in st.session_state:
    st.session_state["use_forecast_main"] = False

# ---- animated icon helper ----
def get_animated_icon(condition: str, size: int = 100):
    """
    Return an HTML string with an inline animated SVG for the given condition.
    condition: string (e.g. "rain", "snow", "cloud", "sunny", "partly cloudy", "thunder")
    size: pixel square size
    """
    c = (condition or "").lower()
    # thunder/storm
    if "thunder" in c or "storm" in c or "lightning" in c:
        svg = f"""
        <svg class="wx-icon" width="{size}" height="{size}" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
          <g class="cloud-move" fill="#cfd8e3">
            <ellipse cx="30" cy="26" rx="18" ry="12" />
            <ellipse cx="44" cy="28" rx="10" ry="8" />
          </g>
          <g transform="translate(22,34)" class="bolt" fill="#ffd13d">
            <path d="M6 0 L0 12 L8 12 L2 28 L18 12 L10 12 L16 0 z" fill="#ffd13d" stroke="#f2a800" stroke-width="0.5"/>
          </g>
        </svg>
        """
        return svg

    # rain/shower
    if "rain" in c or "shower" in c or "drizzle" in c:
        svg = f"""
        <svg class="wx-icon" width="{size}" height="{size}" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
          <g class="cloud-move">
            <ellipse cx="30" cy="22" rx="18" ry="12" fill="#cfd8e3"/>
            <ellipse cx="44" cy="24" rx="10" ry="8" fill="#dfe7f0"/>
          </g>
          <g transform="translate(16,34)" fill="#4DA6FF">
            <ellipse class="rain-drop" cx="6" cy="4" rx="2" ry="4" />
            <ellipse class="rain-drop" cx="16" cy="6" rx="2" ry="4" />
            <ellipse class="rain-drop" cx="26" cy="4" rx="2" ry="4" />
          </g>
        </svg>
        """
        return svg

    # snow/sleet
    if "snow" in c or "sleet" in c:
        svg = f"""
        <svg class="wx-icon" width="{size}" height="{size}" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
          <g class="cloud-move" fill="#dfe7f0">
            <ellipse cx="30" cy="22" rx="18" ry="12"/>
            <ellipse cx="44" cy="24" rx="10" ry="8"/>
          </g>
          <g transform="translate(16,36)" fill="#ffffff">
            <text class="snow-flake" x="6" y="6" font-size="10">❄</text>
            <text class="snow-flake" x="18" y="8" font-size="10">❄</text>
            <text class="snow-flake" x="30" y="6" font-size="10">❄</text>
          </g>
        </svg>
        """
        return svg

    # cloud / overcast
    if "cloud" in c or "overcast" in c:
        svg = f"""
        <svg class="wx-icon" width="{size}" height="{size}" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
          <g class="cloud-move" fill="#d0d6df">
            <ellipse cx="26" cy="26" rx="18" ry="12"/>
            <ellipse cx="42" cy="28" rx="12" ry="9"/>
          </g>
        </svg>
        """
        return svg

    # partly cloudy (sun + cloud)
    if "part" in c or ("sun" in c and "cloud" in c) or "mix" in c or "few" in c:
        svg = f"""
        <svg class="wx-icon" width="{size}" height="{size}" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
          <g transform="translate(6,6)">
            <g class="sun-core">
              <circle cx="18" cy="18" r="8" fill="#FFD33D"/>
              <g class="ray" fill="#FFD33D" opacity="0.9">
                <rect x="17.5" y="-2" width="1" height="6" />
                <rect x="34" y="17.5" width="6" height="1" />
                <rect x="17.5" y="34" width="1" height="6" />
                <rect x="-2" y="17.5" width="6" height="1" />
              </g>
            </g>
            <g transform="translate(20,10)" class="cloud-move" fill="#dfe7f0">
              <ellipse cx="18" cy="18" rx="14" ry="9"/>
              <ellipse cx="32" cy="20" rx="8" ry="6"/>
            </g>
          </g>
        </svg>
        """
        return svg

    # default: sunny
    svg = f"""
    <svg class="wx-icon" width="{size}" height="{size}" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <g transform="translate(0,0)">
        <g class="sun-core">
          <circle cx="32" cy="32" r="12" fill="#FFD33D"/>
        </g>
        <g transform="translate(32,32)" >
          <g>
            <rect class="ray" x="-1.5" y="-22" width="3" height="8" rx="1" ry="1" fill="#FFD33D" transform="rotate(0)"/>
            <rect class="ray" x="-1.5" y="-22" width="3" height="8" rx="1" ry="1" fill="#FFD33D" transform="rotate(30)"/>
            <rect class="ray" x="-1.5" y="-22" width="3" height="8" rx="1" ry="1" fill="#FFD33D" transform="rotate(60)"/>
            <rect class="ray" x="-1.5" y="-22" width="3" height="8" rx="1" ry="1" fill="#FFD33D" transform="rotate(90)"/>
            <rect class="ray" x="-1.5" y="-22" width="3" height="8" rx="1" ry="1" fill="#FFD33D" transform="rotate(120)"/>
            <rect class="ray" x="-1.5" y="-22" width="3" height="8" rx="1" ry="1" fill="#FFD33D" transform="rotate(150)"/>
            <rect class="ray" x="-1.5" y="-22" width="3" height="8" rx="1" ry="1" fill="#FFD33D" transform="rotate(180)"/>
            <rect class="ray" x="-1.5" y="-22" width="3" height="8" rx="1" ry="1" fill="#FFD33D" transform="rotate(210)"/>
            <rect class="ray" x="-1.5" y="-22" width="3" height="8" rx="1" ry="1" fill="#FFD33D" transform="rotate(240)"/>
            <rect class="ray" x="-1.5" y="-22" width="3" height="8" rx="1" ry="1" fill="#FFD33D" transform="rotate(270)"/>
            <rect class="ray" x="-1.5" y="-22" width="3" height="8" rx="1" ry="1" fill="#FFD33D" transform="rotate(300)"/>
            <rect class="ray" x="-1.5" y="-22" width="3" height="8" rx="1" ry="1" fill="#FFD33D" transform="rotate(330)"/>
          </g>
        </g>
      </g>
    </svg>
    """
    return svg

# ---- geo helpers (flag + geolocation) ----
def country_code_to_flag_emoji(code: str):
    if not code or len(code) != 2:
        return ""
    try:
        code = code.upper()
        return "".join(chr(ord(c) + 127397) for c in code)
    except Exception:
        return ""

def try_get_geolocation():
    """Best-effort geolocation using ipapi.co then ipinfo.io (non-blocking)."""
    apis = [
        ("https://ipapi.co/json/", "ipapi"),
        ("https://ipinfo.io/json", "ipinfo"),
    ]
    for url, tag in apis:
        try:
            resp = requests.get(url, timeout=3)
            if resp.status_code != 200:
                continue
            js = resp.json()
            city = js.get("city")
            region = js.get("region")
            country_name = js.get("country_name") or js.get("country")
            country_code = js.get("country") or js.get("countryCode") or js.get("country_code")
            return {"city": city, "region": region, "country": country_name, "country_code": (country_code or "").upper()}
        except Exception:
            continue
    return None

# ---- model loader / feature alignment ----
def load_model_and_features():
    chosen_path = None
    if st.session_state.get("prefer_pretrained", True) and st.session_state.get("selected_model_file"):
        chosen_filename = st.session_state["selected_model_file"]
        chosen_path = MODELS_DIR / chosen_filename
    else:
        # model_choice is set in the sidebar before predict flows are triggered
        try:
            latest = get_latest_by_base(model_choice)
            chosen_path = Path(latest["path"]) if latest else None
        except Exception:
            chosen_path = None

    if not chosen_path or not chosen_path.exists():
        return None, None
    try:
        model = load_model_cached(str(chosen_path))
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None

    model_features = getattr(model, "feature_names_in_", None)
    if model_features is None:
        meta_json = chosen_path.with_suffix(chosen_path.suffix + ".meta.json")
        meta_bin = chosen_path.with_suffix(chosen_path.suffix + ".meta")
        try:
            if meta_json.exists():
                with open(meta_json, "r") as fh:
                    meta = json.load(fh)
                    model_features = meta.get("features") or meta.get("feature_names")
            elif meta_bin.exists():
                meta = joblib.load(meta_bin)
                model_features = meta.get("features") or meta.get("feature_names")
        except Exception:
            model_features = None
    if model_features is not None:
        model_features = list(model_features)
    return model, model_features

# ---- SIDEBAR ----
with st.sidebar:
    st.markdown("# Controls")

    # theme toggle
    checked = st.checkbox("Dark", value=(st.session_state["theme"] == "dark"))
    new_theme = "dark" if checked else "light"
    if new_theme != st.session_state["theme"]:
        st.session_state["theme"] = new_theme
        inject_theme_css(new_theme)

    st.markdown("---")
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

    df = preprocess(df)

    with st.expander("Dataset overview", expanded=False):
        st.dataframe(df.head(8))
        st.write(summary_stats(df.select_dtypes(include=[np.number])))

    target = st.selectbox("Target variable", options=["mean_temp", "max_temp", "min_temp", "precipitation"], index=0)
    default_features = [c for c in df.columns if c not in ["date", target]]
    features = st.multiselect("Features (choose at least 3)", options=default_features, default=default_features[:6])

    st.session_state["use_forecast_main"] = st.checkbox("Use model forecast for main card (when available)", value=st.session_state["use_forecast_main"])

    with st.expander("Advanced / Training controls", expanded=False):
        n_estimators = st.number_input("RF: n_estimators", min_value=10, max_value=2000, value=100, step=10)
        rf_step = st.number_input("RF: progress step (trees per update)", min_value=1, max_value=200, value=10)
        epochs = st.number_input("SGD: epochs", min_value=1, max_value=500, value=20)

        model_choice = st.selectbox("Train model type", ["random_forest", "sgd"], index=0)

        if "prefer_pretrained" not in st.session_state:
            st.session_state["prefer_pretrained"] = True
        if "selected_model_file" not in st.session_state:
            latest_rf = get_latest_by_base("random_forest")
            latest_sgd = get_latest_by_base("sgd")
            default = latest_rf or latest_sgd
            st.session_state["selected_model_file"] = default["filename"] if default else None

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

        st.session_state["prefer_pretrained"] = st.checkbox("Prefer pretrained model", value=st.session_state["prefer_pretrained"])

        if st.button("Clear pretrained selection"):
            st.session_state["selected_model_file"] = None
            st.session_state["prefer_pretrained"] = False
            st.success("Cleared pretrained selection — dashboard will use newly trained models by default.")

        train_rf_btn = st.button("Train RF (with progress)")
        train_sgd_btn = st.button("Train SGD (with progress)")
        train_all = st.button("Train and save both (RF + SGD)")

    st.markdown("---")
    st.subheader("Prediction / Forecast")
    predict_n = st.number_input("Forecast next N days", min_value=1, max_value=365, value=5)
    predict_btn = st.button("Predict (forecast next N days)")

# ---- Title ----
st.markdown("<div style='display:flex; align-items:center; gap:12px;'><h1 style='margin:0; font-weight:900'>🌤️ Synoptic Weather Forecast</h1><div style='color:var(--text-color)'></div></div>", unsafe_allow_html=True)

# ---- prediction helpers & logic ----
def _create_future_dates(last_date, n):
    last_dt = pd.to_datetime(last_date, errors='coerce')
    if pd.isna(last_dt):
        last_dt = pd.Timestamp.now()
    return [(last_dt + timedelta(days=i+1)).normalize() for i in range(n)]

def _ensure_X_for_model(Xrow, model_features):
    if model_features is None:
        return Xrow
    X2 = pd.DataFrame(index=[0])
    for f in model_features:
        if f in Xrow.columns:
            X2[f] = Xrow[f].iloc[0]
        else:
            X2[f] = 0.0
    return X2

def _predict_future(df, features, n, model, model_features=None):
    start_row = df[features].tail(1).copy().fillna(0)
    future_dates = _create_future_dates(df['date'].iloc[-1], n)
    rows = []
    X_base = start_row.copy()
    for i in range(n):
        Xp = _ensure_X_for_model(X_base, model_features)
        if model_features is not None:
            try:
                Xp = Xp[model_features]
            except Exception:
                pass
        try:
            pred = float(model.predict(Xp)[0])
        except Exception as e:
            st.error(f"Prediction failed during iterative forecast: {e}")
            return None
        rec = {"date": future_dates[i], "pred_mean_temp": float(pred)}
        for col in ["precipitation", "wind", "pressure", "aqi", "humidity", "sunshine"]:
            if col in df.columns:
                rec[col] = df[col].iloc[-1]
        rows.append(rec)
        if 'mean_temp' in X_base.columns:
            X_base.at[X_base.index[0], 'mean_temp'] = pred
        else:
            for tcol in ['temp', 'max_temp', 'min_temp']:
                if tcol in X_base.columns:
                    X_base.at[X_base.index[0], tcol] = pred
                    break
    return pd.DataFrame(rows)

if predict_btn:
    if not features or len(features) < 1:
        st.error("Please select at least one feature to enable prediction.")
    else:
        model, model_feats = load_model_and_features()
        if model is None:
            st.error("No model available; check Advanced / Training controls.")
        else:
            pred_df = _predict_future(df, features, int(predict_n), model, model_features=model_feats)
            if pred_df is not None:
                st.session_state["last_prediction"] = pred_df
                st.success(f"Forecast for next {len(pred_df)} days computed and stored in session.")

# ---- MAIN VIEW ----
geo = try_get_geolocation()
current_display = get_current_conditions(df)

# city + flag
city_label = None
flag = ""
if geo and geo.get("city"):
    code = geo.get("country_code") or ""
    flag = country_code_to_flag_emoji(code)
    city_label = f"{geo.get('city')}, {geo.get('country') or geo.get('region')}"
elif current_display.get("city") and current_display.get("city") != "Unknown":
    city_label = current_display.get("city")
else:
    city_label = "Arizona, USA"
    flag = country_code_to_flag_emoji("US")

left_col, right_col = st.columns([2.2, 1.6])

with left_col:
    now = datetime.now()
    time_str = now.strftime("%I:%M %p")  # 12-hour with AM/PM
    city_html = f"<div class='big-city'>{flag} {city_label}</div>"
    time_html = f"<div class='big-time'>{time_str}</div>"
    date_html = f"<div style='font-size:16px; color:var(--text-color)'>{now.strftime('%A, %d %b %Y')}</div>"
    st.markdown(f"<div class='big-card'>{city_html}{time_html}{date_html}</div>", unsafe_allow_html=True)

# main right card (allow forecast override)
if st.session_state["use_forecast_main"] and st.session_state["last_prediction"] is not None:
    p0 = st.session_state["last_prediction"].iloc[0]
    temp_val = float(p0.get("pred_mean_temp", current_display.get("temp", 0.0)))
    feels = temp_val
    condition = current_display.get("condition", "Forecast")
    humidity = int(p0.get("humidity", current_display.get("humidity") or 0))
    wind = int(p0.get("wind", current_display.get("wind") or 0))
    pressure_val = p0.get("pressure", current_display.get("pressure"))
    precip_val = p0.get("precipitation", current_display.get("precipitation"))
    aqi_val = p0.get("aqi", current_display.get("aqi"))
else:
    temp_val = current_display.get("temp", 0.0)
    feels = current_display.get("feels_like", temp_val)
    condition = current_display.get("condition", "Unknown")
    humidity = current_display.get("humidity", 0)
    wind = current_display.get("wind", 0)
    pressure_val = current_display.get("pressure", None)
    precip_val = current_display.get("precipitation", None)
    aqi_val = current_display.get("aqi", None)

# normalize pressure -> hPa
pressure_str = "—"
if pressure_val is not None:
    try:
        ptemp = float(pressure_val)
        if ptemp > 2000:  # likely Pa
            ptemp = ptemp / 100.0
        pressure_str = f"{int(round(ptemp))} hPa"
    except Exception:
        pressure_str = str(pressure_val)

aqi_str = str(int(aqi_val)) if (aqi_val is not None and not pd.isna(aqi_val)) else "—"
precip_str = f"{precip_val} mm" if (precip_val is not None and not pd.isna(precip_val)) else "—"

# pick animated icon HTML
big_icon_html = get_animated_icon(condition, size=120)

with right_col:
    big_temp_html = (
        f"<div style='display:flex; align-items:center; justify-content:space-between'>"
        f"<div><div class='big-temp'>{int(round(temp_val))}<span class='deg'>°C</span></div>"
        f"<div style='font-size:18px; font-weight:700; margin-top:6px'>Feels like: {int(round(feels))}°C</div></div>"
        f"<div style='text-align:center; min-width:140px'>{big_icon_html}<div style='margin-top:8px; font-weight:700'>{condition}</div></div></div>"
    )

    uv_val = df['uv'].iloc[-1] if 'uv' in df.columns and not pd.isna(df['uv'].iloc[-1]) else None
    uv_str = str(int(uv_val)) if uv_val is not None else "—"

    stats_html = (
        "<div style='display:flex; gap:12px; margin-top:16px; flex-wrap:wrap'>"
        f"<div class='stat-tile' style='background:var(--card-bg)'><div style='font-size:20px'>💧</div><div style='font-weight:800; font-size:18px'>{int(humidity)}%</div><div style='font-size:12px'>Humidity</div></div>"
        f"<div class='stat-tile' style='background:var(--card-bg)'><div style='font-size:20px'>💨</div><div style='font-weight:800; font-size:18px'>{int(wind)} km/h</div><div style='font-size:12px'>Wind</div></div>"
        f"<div class='stat-tile' style='background:var(--card-bg)'><div style='font-size:20px'>⎈</div><div style='font-weight:800; font-size:18px'>{pressure_str}</div><div style='font-size:12px'>Pressure</div></div>"
        f"<div class='stat-tile' style='background:var(--card-bg)'><div style='font-size:20px'>☀️</div><div style='font-weight:800; font-size:18px'>{uv_str}</div><div style='font-size:12px'>UV</div></div>"
        "</div>"
    )

    precip_block = f"<div style='margin-top:12px; color:var(--text-color)'>🌧️ Precipitation: <strong>{precip_str}</strong> · 🩺 AQI: <strong>{aqi_str}</strong></div>"

    st.markdown(f"<div class='big-card'>{big_temp_html}{stats_html}{precip_block}</div>", unsafe_allow_html=True)

# ---- Forecast and hourly (clean layout) ----
col_a, col_b = st.columns([1, 1.6])

def _render_forecast_card(target_n=5):
    if st.session_state.get("last_prediction") is not None:
        pred_df = st.session_state["last_prediction"].copy()
        to_show = pred_df.head(int(target_n))
        cards_html = f"<div class='small-card'><h3>{target_n}-Day Forecast</h3>"
        for _, r in to_show.iterrows():
            dt_s = pd.to_datetime(r['date']).strftime('%a<br>%Y-%m-%d')
            temp = int(round(r.get('pred_mean_temp', 0)))
            icon_html = get_animated_icon('', size=24)  # use small svg based on temp/cond
            # choose cond icon by temp if cond absent
            cond = r.get('condition', '')
            if cond:
                icon_html = get_animated_icon(cond, size=28)
            else:
                # fallback by temp
                if temp >= 25:
                    icon_html = get_animated_icon('sunny', size=28)
                elif temp >= 15:
                    icon_html = get_animated_icon('partly cloudy', size=28)
                elif temp >= 5:
                    icon_html = get_animated_icon('cloud', size=28)
                else:
                    icon_html = get_animated_icon('snow', size=28)

            precip = r.get('precipitation', '—')
            wind = r.get('wind', '—')
            cards_html += (
                f"<div style='display:flex; align-items:center; justify-content:space-between; padding:8px 6px'>"
                f"<div style='display:flex; gap:12px; align-items:center'><div style='font-size:22px'>{icon_html}</div>"
                f"<div><div style='font-weight:700'>{dt_s}</div><div style='font-size:12px'>Precip: {precip} mm · Wind: {wind} km/h</div></div></div>"
                f"<div style='text-align:right'><div style='font-weight:700'>{temp}°C</div></div></div>"
            )
        cards_html += "</div>"
        st.markdown(cards_html, unsafe_allow_html=True)
    else:
        five = build_5day_forecast(df)
        cards_html = "<div class='small-card'><h3>5 Days Forecast (historical)</h3>"
        for day in five:
            icon_html = get_animated_icon(day.get('icon', ''), size=28)
            cards_html += (
                f"<div style='display:flex; align-items:center; justify-content:space-between; padding:8px 6px'>"
                f"<div style='display:flex; gap:12px; align-items:center'><div style='font-size:22px'>{icon_html}</div>"
                f"<div><div style='font-weight:700'>{day['day']}</div><div style='font-size:12px'>{day['date']}</div></div></div>"
                f"<div style='text-align:right'><div style='font-weight:700'>{day['temp']}°C</div></div></div>"
            )
        cards_html += "</div>"
        st.markdown(cards_html, unsafe_allow_html=True)

with col_a:
    _render_forecast_card(target_n=predict_n)

with col_b:
    hourly_html = f"<div class='small-card'><h3>Hourly Forecast</h3><div style='display:flex; gap:12px; overflow:auto; padding-top:8px'>"
    hourly = []
    if st.session_state.get("last_prediction") is not None:
        pred_df = st.session_state["last_prediction"].head(8)
        for _, r in pred_df.iterrows():
            t = pd.to_datetime(r['date']).strftime('%Y-%m-%d 00:00')
            hourly.append({'time': t, 'temp': int(round(r.get('pred_mean_temp', 0))), 'wind': r.get('wind', 0), 'cond': r.get('condition', ''), 'precip': r.get('precipitation', 0)})
    else:
        hist = df.tail(8)
        for _, r in hist.iterrows():
            try:
                t = pd.to_datetime(r.get('date')).strftime('%Y-%m-%d %H:%M')
            except Exception:
                t = str(r.get('date'))
            hourly.append({'time': t, 'temp': int(round(r.get('mean_temp', r.get('max_temp', r.get('min_temp', 0))))), 'wind': int(r.get('wind', 0) if 'wind' in r.index else 0), 'cond': r.get('condition',''), 'precip': r.get('precipitation', 0) if 'precipitation' in r.index else 0})
    for h in hourly:
        try:
            display_time = pd.to_datetime(h['time']).strftime('%a %H:%M')
        except Exception:
            display_time = str(h['time'])
        icon_html = get_animated_icon(h.get('cond') or ('sunny' if h['temp'] >= 15 else 'snow'), size=40)
        hourly_html += (
            f"<div style='min-width:140px; border-radius:10px; padding:10px; background:var(--card-bg); text-align:center;'>"
            f"<div style='font-weight:700'>{display_time}</div>"
            f"{icon_html}"
            f"<div style='font-weight:700'>{h['temp']}°C</div>"
            f"<div style='font-size:12px'>{int(h.get('wind',0))} km/h · {h.get('precip','—')} mm</div>"
            f"</div>"
        )
    hourly_html += "</div></div>"
    st.markdown(hourly_html, unsafe_allow_html=True)

# ---- Totals summary (only bottom) ----
st.markdown("### Totals summary")
totals, _ = weather_condition_summary(df, temp_col='mean_temp', precip_col='precipitation')
totals_df = pd.DataFrame(list(totals.items()), columns=['metric', 'value'])
totals_df['metric'] = totals_df['metric'].map({
    'hot': '🔥 Hot',
    'warm': '🌤️ Warm',
    'cool': '🌥️ Cool',
    'cold': '❄️ Cold',
    'rainy_days': '🌧️ Rainy Days'
})
st.dataframe(totals_df.set_index('metric'))
