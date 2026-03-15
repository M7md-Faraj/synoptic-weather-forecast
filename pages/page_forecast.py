import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
import streamlit.components.v1 as components

from utils.data_loader import get_monthly_medians, build_forecast_input, FEATURE_COLS as DL_FEATURES
from utils.model_utils import MODEL_NAMES, MODEL_COLORS, predict, get_best_model, load_results, load_model_bundles

# small CSS reused for emoji + table (kept from previous)
_EMOJI_CSS = """
<style>
.emoji-sunny {font-size:48px; display:inline-block; animation: sun-pulse 2.6s infinite; line-height:1;}
@keyframes sun-pulse {0%{transform:scale(1);}50%{transform:scale(1.12);}100%{transform:scale(1);} }
.emoji-rain {font-size:48px; display:inline-block; animation: rain-drop 1.2s infinite; line-height:1;}
@keyframes rain-drop {0%{transform:translateY(-6px); opacity:0.75;}50%{transform:translateY(8px); opacity:1;}100%{transform:translateY(-6px); opacity:0.75;} }
.icon-small {font-size:28px; vertical-align:middle;}
.forecast-table {border-collapse:collapse; width:100%; margin-top:8px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;}
.forecast-table th, .forecast-table td {border:1px solid #ddd; padding:8px; text-align:center;}
.forecast-table th {background:#f6f6f6; font-weight:700;}
.forecast-date {text-align:left; padding-left:12px;}
.forecast-temp {font-weight:700; text-align:right; padding-right:12px;}
</style>
"""

def _condition(temp_c: float):
    if temp_c >= 20:
        return "Warm", "☀️", "#f59e0b", "emoji-sunny"
    elif temp_c >= 14:
        return "Mild", "⛅", "#3b82f6", "emoji-sunny"
    elif temp_c >= 8:
        return "Cool", "🌥", "#64748b", "emoji-sunny"
    elif temp_c >= 2:
        return "Cold", "🌨", "#6366f1", "emoji-rain"
    else:
        return "Freezing", "❄️", "#0ea5e9", "emoji-rain"


def _ensure_2d_input(x):
    if isinstance(x, dict):
        return pd.DataFrame([x])
    if isinstance(x, pd.Series):
        return x.to_frame().T
    arr = np.asarray(x)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def _to_dataframe_with_features(x, bundles, model_name: str):
    if isinstance(x, pd.DataFrame):
        return x
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError("Model input must be 2D at this point.")
    bundle = bundles.get(model_name) if isinstance(bundles, dict) else None
    features = None
    if bundle and isinstance(bundle.get("features"), (list, tuple)):
        features = list(bundle["features"])
    if features is None and arr.shape[1] == len(DL_FEATURES):
        features = DL_FEATURES
    if features is None:
        raise ValueError("Cannot determine feature names to build a DataFrame for prediction. Ensure `build_forecast_input` returns a dict/Series or that model bundles include 'features'.")
    if arr.shape[1] != len(features):
        raise ValueError(f"Input has {arr.shape[1]} columns but model expects {len(features)} features.")
    return pd.DataFrame(arr, columns=features)


def _render_cards(day_rows: list, mae: float):
    cards_html = ""
    for row in day_rows:
        label, emoji, accent, emoji_class = _condition(row["temp"])
        emoji_html = f'<span class="{emoji_class}">{emoji}</span>'
        temp_display = row.get("temp_display", f"{row['temp']:.2f}")
        cards_html += f"""
        <div style="min-width:110px; max-width:110px; background:#fff;
                    border:1px solid #e2e8f0; border-top:3px solid {accent};
                    border-radius:10px; padding:14px 8px 12px; text-align:center;
                    flex-shrink:0; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
            <div style="font-size:0.62rem;font-weight:700;color:#64748b;text-transform:uppercase;margin-bottom:2px;">{row["day"]}</div>
            <div style="font-size:0.65rem;color:#94a3b8;margin-bottom:10px;">{row["date_short"]}</div>
            <div style="font-size:1.5rem;line-height:1;margin-bottom:5px;">{emoji_html}</div>
            <div style="font-size:0.68rem;color:{accent};font-weight:600;margin-bottom:8px;">{label}</div>
            <div style="font-size:1.65rem;font-weight:800;color:#0f172a;margin-bottom:3px;">{temp_display}&deg;</div>
            <div style="font-size:0.57rem;color:#94a3b8;border-top:1px solid #f1f5f9;padding-top:5px;margin-top:6px;">&plusmn;{mae:.2f}&deg;C</div>
        </div>"""
    full_html = f"""<!DOCTYPE html><html><body style="margin:0;padding:0;background:transparent;">
    <div style="overflow-x:auto;padding:4px 2px 10px;">
      <div style="display:flex;gap:10px;width:max-content;padding:2px 2px 4px;">
        {cards_html}
      </div>
    </div>
    <p style="font-size:0.67rem;color:#94a3b8;margin:2px 0 0;">&#8592; Scroll to browse all {len(day_rows)} days · ±{mae:.2f}°C (MAE)</p>
    </body></html>"""
    components.html(_EMOJI_CSS + full_html, height=220, scrolling=False)


def _prediction_table_html(fdf: pd.DataFrame, mae: float):
    rows = ""
    for _, r in fdf.iterrows():
        label, emoji, accent, emoji_class = _condition(r["Predicted Mean Temp (°C)"])
        emoji_html = f'<span class="{emoji_class}">{emoji}</span>'
        temp_display = r.get("Temp Display", f"{r['Predicted Mean Temp (°C)']:.2f}")
        exact2 = f"{r['Predicted Mean Temp (°C)']:.2f}"
        rows += (
            "<tr>"
            f"<td class='forecast-date'>{r['Date']}<br><small>{r['Day']}</small></td>"
            f"<td>{emoji_html}<br><small style='color:{accent};font-weight:600;'>{label}</small></td>"
            f"<td class='forecast-temp' title='{exact2}°'>{temp_display}°</td>"
            f"<td>±{mae:.2f}°C</td>"
            "</tr>"
        )
    html = f"""
    {_EMOJI_CSS}
    <div style="width:100%; max-height:420px; overflow:auto; border:1px solid #eef2f7; border-radius:8px; padding:8px; background:#ffffff;">
      <table class="forecast-table" role="table" aria-label="Full prediction table">
        <thead>
          <tr><th>Date</th><th>Condition</th><th>Predicted (°C)</th><th>Accuracy (MAE)</th></tr>
        </thead>
        <tbody>
          {rows}
        </tbody>
      </table>
    </div>
    """
    return html


def render(df, bundles, selected_model=None):
    results = load_results()
    best_name = get_best_model(results)

    st.title("Temperature Forecast")
    st.divider()

    st.subheader("Forecast configuration")
    col_model, col_horizon = st.columns(2)

    with col_model:
        default_idx = 0
        try:
            default_idx = MODEL_NAMES.index(selected_model) if selected_model in MODEL_NAMES else MODEL_NAMES.index(best_name)
        except Exception:
            default_idx = 0
        selected_model_local = st.selectbox("Select model", options=MODEL_NAMES, index=default_idx)
        m = results.get(selected_model_local, {})
        st.caption(f"{selected_model_local} — MAE={m.get('MAE', 0):.3f}°C · RMSE={m.get('RMSE', 0):.3f}°C · R²={m.get('R2', 0):.4f} · MAPE={m.get('MAPE%', 0):.3f}%")

    with col_horizon:
        horizon_options = {"Next 7 days": 7, "Next 14 days": 14, "Next 30 days": 30, "Next 90 days": 90}
        horizon_label = st.selectbox("Forecast horizon", options=list(horizon_options.keys()))
        n_days = horizon_options[horizon_label]

    # One-day combined prediction for all models
    st.markdown("### One-day prediction (all models)")
    if st.button("Predict next day (all models)"):
        today = pd.Timestamp.today().normalize()
        next_day = today + pd.Timedelta(days=1)
        monthly_meds = get_monthly_medians(df)

        # build input
        raw_input = build_forecast_input(next_day, monthly_meds)
        X_input = _ensure_2d_input(raw_input)
        if isinstance(X_input, np.ndarray):
            # convert for each model if necessary
            rows = []
            for name in MODEL_NAMES:
                try:
                    X_df = _to_dataframe_with_features(X_input, bundles, name)
                    pred_arr = predict(bundles, name, X_df)
                    pred_val = float(np.asarray(pred_arr).ravel()[0])
                except Exception as e:
                    pred_val = float("nan")
                rows.append({"Model": name, "Predicted (°C)": round(pred_val, 2)})
            st.table(pd.DataFrame(rows).set_index("Model"))
        else:
            # DataFrame: use direct prediction
            rows = []
            for name in MODEL_NAMES:
                try:
                    pred_arr = predict(bundles, name, X_input)
                    pred_val = float(np.asarray(pred_arr).ravel()[0])
                except Exception as e:
                    pred_val = float("nan")
                rows.append({"Model": name, "Predicted (°C)": round(pred_val, 2)})
            st.table(pd.DataFrame(rows).set_index("Model"))

    st.divider()

    # Multi-day forecast generation (existing flow)
    if st.button("Generate Forecast", type="primary"):
        with st.spinner("Generating predictions…"):
            today = pd.Timestamp.today().normalize()
            forecast_dates = [today + pd.Timedelta(days=i + 1) for i in range(n_days)]
            monthly_meds = get_monthly_medians(df)

            rows = []
            for d in forecast_dates:
                raw_input = build_forecast_input(d, monthly_meds)
                X_input = _ensure_2d_input(raw_input)
                if isinstance(X_input, np.ndarray):
                    try:
                        X_df = _to_dataframe_with_features(X_input, bundles, selected_model_local)
                    except Exception as e:
                        st.error(f"Prediction input error: {e}")
                        return
                else:
                    X_df = X_input

                try:
                    pred_arr = predict(bundles, selected_model_local, X_df)
                    pred_val = float(np.asarray(pred_arr).ravel()[0])
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    pred_val = float("nan")

                rows.append({
                    "Date": d.strftime("%Y-%m-%d"),
                    "Day": d.strftime("%A"),
                    "Month": d.strftime("%B"),
                    "Predicted Mean Temp (°C)": round(pred_val, 2),
                    "_day_short": f"{d.strftime('%a')}",
                    "_date_short": f"{d.day} {d.strftime('%b')}",
                })

            fdf = pd.DataFrame(rows)
            # decide display precision
            try:
                numeric_series = fdf["Predicted Mean Temp (°C)"].dropna().astype(float)
                if numeric_series.empty:
                    dp = 2
                else:
                    collapsed_unique = numeric_series.round(1).nunique()
                    dp = 2 if collapsed_unique <= max(3, len(fdf) // 4) else 1
            except Exception:
                dp = 2

            fdf["Temp Display"] = fdf["Predicted Mean Temp (°C)"].map(lambda x: f"{x:.{dp}f}")
            st.session_state["forecast_df"] = fdf
            st.session_state["forecast_model"] = selected_model_local
            st.session_state["forecast_horizon"] = n_days

    fdf = st.session_state.get("forecast_df")
    if fdf is None:
        st.info("Select model and horizon, then click Generate Forecast or use the one-day predictor above.")
        return

    forecast_model = st.session_state["forecast_model"]
    forecast_horizon = st.session_state["forecast_horizon"]
    mae = load_results().get(forecast_model, {}).get("MAE", 0)
    temps = fdf["Predicted Mean Temp (°C)"]
    colour = MODEL_COLORS.get(forecast_model, "#2563eb")

    st.divider()
    col_left, col_right = st.columns([2, 5])

    with col_left:
        first_temp = float(temps.iloc[0])
        first_temp_display = fdf["Temp Display"].iloc[0]
        first_label, first_emoji, first_accent, _ = _condition(first_temp)
        first_date = pd.to_datetime(fdf["Date"].iloc[0])
        date_str = f"{first_date.strftime('%A')}, {first_date.day} {first_date.strftime('%B %Y')}"
        st.markdown(
            f"""
            <div style="padding:20px;border-radius:10px;border:1px solid rgba(128,128,128,0.15);">
              <div style="font-size:0.72rem;opacity:0.7;margin-bottom:6px;">First predicted day</div>
              <div style="font-size:0.82rem;opacity:0.7;margin-bottom:12px;">{date_str}</div>
              <div style="font-size:4rem;font-weight:700;color:{first_accent};line-height:1;margin-bottom:6px;">{first_temp_display}°</div>
              <div style="font-size:1rem;font-weight:600;color:{first_accent};margin-bottom:6px;">{first_emoji} {first_label}</div>
              <div style="font-size:0.72rem;opacity:0.65;margin-top:12px;border-top:1px solid rgba(128,128,128,0.12);padding-top:10px;">
                Model: {forecast_model}<br>Accuracy band: ±{mae:.2f}°C<br>Horizon: {forecast_horizon} days
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_right:
        dates_plot = pd.to_datetime(fdf["Date"])
        fig, ax = plt.subplots(figsize=(9, 3.6))
        ax.plot(dates_plot, temps, marker="o", ms=3, color=colour, lw=2.0, label=forecast_model)
        ax.fill_between(dates_plot, temps - mae, temps + mae, alpha=0.13, color=colour, label=f"±MAE ({mae:.2f}°C)")
        if forecast_horizon <= 14:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        elif forecast_horizon <= 30:
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        else:
            ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        plt.xticks(rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Mean Temperature (°C)")
        ax.set_title(f"{forecast_model} — Predicted Mean Daily Temperature", fontsize=9)
        ax.legend(fontsize=8, framealpha=0)
        ax.grid(True, alpha=0.35)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("---")
    st.markdown("#### Day-by-Day Forecast")
    st.caption(f"Showing all {forecast_horizon} predicted days — scroll to browse.")
    card_data = [
        {
            "day": r["_day_short"],
            "date_short": r["_date_short"],
            "temp": r["Predicted Mean Temp (°C)"],
            "temp_display": r["Temp Display"],
        }
        for _, r in fdf.iterrows()
    ]
    _render_cards(card_data, mae)

    st.markdown("---")
    st.markdown("#### Full Prediction Table")
    html = _prediction_table_html(fdf, mae)
    row_count = len(fdf)
    height = min(100 + row_count * 44, 720)
    components.html(html, height=height, scrolling=True)

    csv_out = fdf[["Date", "Day", "Month", "Predicted Mean Temp (°C)"]].copy()
    csv_out["Predicted Mean Temp (°C)"] = csv_out["Predicted Mean Temp (°C)"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    csv_bytes = csv_out.to_csv(index=False).encode("utf-8")
    st.download_button(label="Download CSV", data=csv_bytes, file_name=f"london_forecast_{forecast_model.replace(' ','_')}_{forecast_horizon}d.csv", mime="text/csv")

    st.divider()
    with st.expander("How predictions are built"):
        st.markdown(
            """
            Inputs: monthly medians from the historical record are used as stand-ins
            for atmospheric features when live data is not provided. Lag features are approximated by month medians.
            Cyclical encodings (month/day-of-year sin/cos) are computed for each date,
            so predictions vary day-to-day even when monthly medians are used elsewhere.

            NOTE: This remains a climatological estimate when live atmospheric inputs are absent.
            """
        )