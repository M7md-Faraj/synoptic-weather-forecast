import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
import streamlit.components.v1 as components

from utils.data_loader import get_monthly_medians, build_forecast_input
from utils.model_utils import MODEL_NAMES, MODEL_COLORS, predict, get_best_model, load_results

# ----- Animated emoji CSS -----
_EMOJI_CSS = """
<style>
.emoji-sunny {font-size:48px; display:inline-block; animation: sun-pulse 2.6s infinite; line-height:1;}
@keyframes sun-pulse {0%{transform:scale(1);}50%{transform:scale(1.12);}100%{transform:scale(1);}}
.emoji-rain {font-size:48px; display:inline-block; animation: rain-drop 1.2s infinite; line-height:1;}
@keyframes rain-drop {0%{transform:translateY(-6px); opacity:0.75;}50%{transform:translateY(8px); opacity:1;}100%{transform:translateY(-6px); opacity:0.75;}}
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


def _render_cards(day_rows: list, mae: float):
    cards_html = ""
    for row in day_rows:
        label, emoji, accent, emoji_class = _condition(row["temp"])
        # wrap emoji in span with the CSS class
        emoji_html = f'<span class="{emoji_class}">{emoji}</span>'
        cards_html += f"""
        <div style="min-width:110px; max-width:110px; background:#fff;
                    border:1px solid #e2e8f0; border-top:3px solid {accent};
                    border-radius:10px; padding:14px 8px 12px; text-align:center;
                    flex-shrink:0; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
            <div style="font-size:0.62rem;font-weight:700;color:#64748b;text-transform:uppercase;margin-bottom:2px;">{row["day"]}</div>
            <div style="font-size:0.65rem;color:#94a3b8;margin-bottom:10px;">{row["date_short"]}</div>
            <div style="font-size:1.5rem;line-height:1;margin-bottom:5px;">{emoji_html}</div>
            <div style="font-size:0.68rem;color:{accent};font-weight:600;margin-bottom:8px;">{label}</div>
            <div style="font-size:1.65rem;font-weight:800;color:#0f172a;margin-bottom:3px;">{row["temp"]:.1f}&deg;</div>
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
    # simple HTML table with classes so CSS colors and spacing apply
    rows = ""
    for _, r in fdf.iterrows():
        label, emoji, accent, emoji_class = _condition(r["Predicted Mean Temp (°C)"])
        emoji_html = f'<span class="{emoji_class}">{emoji}</span>'
        rows += (
            f"<tr>"
            f"<td class='forecast-date'>{r['Date']}<br><small>{r['Day']}</small></td>"
            f"<td>{emoji_html}<br><small style='color:{accent};font-weight:600;'>{label}</small></td>"
            f"<td class='forecast-temp'>{r['Predicted Mean Temp (°C)']:.1f}°</td>"
            f"<td>±{mae:.2f}°C</td>"
            f"</tr>"
        )
    html = f"""
    <div>
      <table class="forecast-table">
        <thead>
          <tr><th>Date</th><th>Condition</th><th>Predicted (°C)</th><th>Accuracy (MAE)</th></tr>
        </thead>
        <tbody>
          {rows}
        </tbody>
      </table>
    </div>
    """
    return _EMOJI_CSS + html


def render(df, bundles, selected_model=None):
    """Render forecast UI. selected_model is optional: if provided, it is used."""
    results = load_results()
    best_name = get_best_model(results)

    st.title("Temperature Forecast")
    st.divider()

    st.subheader("Forecast configuration")
    col_model, col_horizon = st.columns(2)

    with col_model:
        default_idx = MODEL_NAMES.index(selected_model) if selected_model in MODEL_NAMES else MODEL_NAMES.index(best_name)
        selected_model_local = st.selectbox("Select model", options=MODEL_NAMES, index=default_idx)
        m = results.get(selected_model_local, {})
        st.caption(f"{selected_model_local} — MAE={m.get('MAE', 0):.3f}°C · RMSE={m.get('RMSE', 0):.3f}°C · R²={m.get('R2', 0):.4f}")

    with col_horizon:
        horizon_options = {"Next 7 days": 7, "Next 14 days": 14, "Next 30 days": 30, "Next 90 days": 90}
        horizon_label = st.selectbox("Forecast horizon", options=list(horizon_options.keys()))
        n_days = horizon_options[horizon_label]

    if st.button("Generate Forecast", type="primary"):
        with st.spinner("Generating predictions…"):
            today = pd.Timestamp.today().normalize()
            forecast_dates = [today + pd.Timedelta(days=i + 1) for i in range(n_days)]
            monthly_meds = get_monthly_medians(df)

            rows = []
            for d in forecast_dates:
                X_row = build_forecast_input(d, monthly_meds)
                pred = float(predict(bundles, selected_model_local, X_row)[0])
                rows.append({
                    "Date": d.strftime("%Y-%m-%d"),
                    "Day": d.strftime("%A"),
                    "Month": d.strftime("%B"),
                    "Predicted Mean Temp (°C)": round(pred, 2),
                    "_day_short": d.strftime("%a"),
                    "_date_short": d.strftime("%-d %b"),
                })

            fdf = pd.DataFrame(rows)
            st.session_state["forecast_df"] = fdf
            st.session_state["forecast_model"] = selected_model_local
            st.session_state["forecast_horizon"] = n_days

    fdf = st.session_state.get("forecast_df")
    if fdf is None:
        st.info("Select model and horizon, then click Generate Forecast.")
        return

    forecast_model = st.session_state["forecast_model"]
    forecast_horizon = st.session_state["forecast_horizon"]
    mae = results.get(forecast_model, {}).get("MAE", 0)
    temps = fdf["Predicted Mean Temp (°C)"]
    colour = MODEL_COLORS.get(forecast_model, "#2563eb")

    st.divider()
    col_left, col_right = st.columns([2, 5])

    with col_left:
        first_temp = float(temps.iloc[0])
        first_label, first_emoji, first_accent, _ = _condition(first_temp)
        first_date = pd.to_datetime(fdf["Date"].iloc[0])
        date_str = first_date.strftime("%A, %-d %B %Y")
        st.markdown(
            f"""
            <div style="padding:20px;border-radius:10px;border:1px solid rgba(128,128,128,0.15);">
              <div style="font-size:0.72rem;opacity:0.7;margin-bottom:6px;">First predicted day</div>
              <div style="font-size:0.82rem;opacity:0.7;margin-bottom:12px;">{date_str}</div>
              <div style="font-size:4rem;font-weight:700;color:{first_accent};line-height:1;margin-bottom:6px;">{first_temp:.1f}°</div>
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
    card_data = [{"day": r["_day_short"], "date_short": r["_date_short"], "temp": r["Predicted Mean Temp (°C)"]} for _, r in fdf.iterrows()]
    _render_cards(card_data, mae)

    st.markdown("---")
    st.markdown("#### Full Prediction Table")
    # render HTML table with emoji CSS
    html = _prediction_table_html(fdf, mae)
    st.markdown(html, unsafe_allow_html=True)

    csv_bytes = fdf[["Date", "Day", "Month", "Predicted Mean Temp (°C)"]].to_csv(index=False).encode("utf-8")
    st.download_button(label="Download CSV", data=csv_bytes, file_name=f"london_forecast_{forecast_model.replace(' ','_')}_{forecast_horizon}d.csv", mime="text/csv")

    st.divider()
    with st.expander("How predictions are built"):
        st.markdown(
            """
            Inputs: monthly medians from the historical record are used as stand-ins
            for atmospheric features. Lag features are approximated by month medians.
            This is a climatological estimate rather than a live, operational forecast.
            """
        )