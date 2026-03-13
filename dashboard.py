# dashboard.py
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import io
import calendar
from src.data_loader import load_csv, preprocess
from src.analysis import (
    summary_stats,
    missing_values_table,
    plot_time_series,
    monthly_aggregate_plot,
    correlation_heatmap_plotly,
    distribution_plot,
    predicted_vs_actual_plot
)
from src.train import train_and_evaluate
from src.models import get_model, save_model, list_models
import plotly.express as px
import json
import joblib

DATA_PATH = Path("data/weather.csv")

st.set_page_config(page_title="Synoptic Weather Forecast - ML", layout="wide")
st.title("Synoptic Weather Forecast — ML Forecasting")

# ----- Animated emoji CSS  -----
_EMOJI_CSS = """
<style>
.emoji-sunny {font-size:64px; display:inline-block; animation: sun-pulse 2.6s infinite; line-height:1;}
@keyframes sun-pulse {0%{transform:scale(1);}50%{transform:scale(1.12);}100%{transform:scale(1);}}
.emoji-rain {font-size:64px; display:inline-block; animation: rain-drop 1.2s infinite; line-height:1;}
@keyframes rain-drop {0%{transform:translateY(-6px); opacity:0.75;}50%{transform:translateY(8px); opacity:1;}100%{transform:translateY(-6px); opacity:0.75;}}
.icon-small {font-size:28px; vertical-align:middle;}
.forecast-table {border-collapse:collapse; width:100%; margin-top:8px;}
.forecast-table th, .forecast-table td {border:1px solid #ddd; padding:8px; text-align:center;}
.forecast-table th {background:#f6f6f6; font-weight:700;}
.forecast-date {text-align:left; padding-left:12px;}
.forecast-temp {font-weight:700; text-align:right; padding-right:12px;}
</style>
"""
st.markdown(_EMOJI_CSS, unsafe_allow_html=True)

# ---- Load data ----
@st.cache_data
def _load_and_prep(path=DATA_PATH):
    df = load_csv(path)
    df = preprocess(df)
    return df

if not DATA_PATH.exists():
    st.error(f"Dataset not found at {DATA_PATH}. Please place your CSV at data/weather.csv")
    st.stop()

df = _load_and_prep(DATA_PATH)

# Detect target default
detected_target = df.attrs.get("detected_target_col", None) or "mean_temp"
if detected_target not in df.columns:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    detected_target = numeric[0] if numeric else df.columns[0]

# Sidebar: settings
with st.sidebar:
    st.header("Settings")
    st.markdown("Choose project defaults and model options.")
    target = st.selectbox("Target variable", options=[c for c in df.columns if c != 'date'], index=max(0, list(df.columns).index(detected_target)) )
    test_size = st.slider("Test set fraction (chronological)", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
    rf_n = st.number_input("Random Forest trees", min_value=10, max_value=1000, value=100, step=10)
    rf_max_depth = st.number_input("RF max depth (0 = None)", min_value=0, max_value=50, value=0, step=1)
    if rf_max_depth == 0:
        rf_max_depth = None
    do_train = st.button("Train & Evaluate Models")

# Page selector
page = st.sidebar.radio("Pages", ["Home", "EDA", "Models & Evaluation", "Forecast"])

# --- HOME ---
if page == "Home":
    st.subheader("Project aim")
    st.write("A compact machine-learning forecasting pipeline for daily mean temperature. This app demonstrates preprocessing, EDA, model comparison (Linear Regression, Decision Tree, Random Forest), evaluation, and a simple forecast interface.")
    # Dataset summary
    st.markdown("### Dataset summary")
    rows = len(df)
    date_min = pd.to_datetime(df['date'], errors='coerce').min()
    date_max = pd.to_datetime(df['date'], errors='coerce').max()
    st.write(f"- Rows: **{rows}**")
    st.write(f"- Date range: **{date_min.date() if pd.notna(date_min) else 'unknown'}** → **{date_max.date() if pd.notna(date_max) else 'unknown'}**")
    st.write(f"- Target variable: **{target}**")
    st.write("- Models: **Linear Regression**, **Decision Tree**, **Random Forest**")
    st.write("- Evaluation metrics: **MAE**, **RMSE**, **MAPE**, **R²**")

    st.markdown("### Visuals")
    # Three visuals: time series, monthly mean, monthly boxplot
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_time_series(df, date_col='date', target=target), use_container_width=True)
    with col2:
        st.plotly_chart(monthly_aggregate_plot(df, date_col='date', target=target, agg='mean'), use_container_width=True)

    st.markdown("#### Monthly distribution (boxplot)")
    # monthly boxplot (added)
    df_bp = df.copy()
    try:
        df_bp['month_num'] = pd.to_datetime(df_bp['date']).dt.month
        df_bp['month_name'] = df_bp['month_num'].apply(lambda m: calendar.month_abbr[m] if m and m>0 else "")
        # keep month order
        month_order = list(calendar.month_abbr)[1:]
        fig_bp = px.box(df_bp, x='month_name', y=target, category_orders={'month_name': month_order}, title=f'Monthly distribution of {target}')
        fig_bp.update_layout(template='plotly_white', height=420)
        st.plotly_chart(fig_bp, use_container_width=True)
    except Exception:
        st.info("Monthly boxplot not available (check date parsing).")

    st.markdown("### One short limitation")
    st.info("This model uses only historical, tabular features from the dataset and a simple chronological train/test split. It does not account for exogenous weather model outputs or climate regime shifts — results should be interpreted as a demonstration, not operational forecasts.")

# --- EDA ---
elif page == "EDA":
    st.header("Exploratory Data Analysis")
    st.subheader("Summary statistics (numeric columns)")
    st.dataframe(summary_stats(df.select_dtypes(include=[np.number])))

    st.subheader("Missing values")
    mv = missing_values_table(df)
    st.dataframe(mv)

    st.subheader("Correlation heatmap")
    st.plotly_chart(correlation_heatmap_plotly(df), use_container_width=True)

    st.subheader("Distribution of target")
    st.plotly_chart(distribution_plot(df, target), use_container_width=True)

    st.subheader("Top 5 rows")
    st.dataframe(df.head(5))

    st.markdown("**Short findings (example)**")
    st.write("- Check correlations for predictors strongly associated with the target (e.g. mean_temp with max_temp/min_temp).")
    st.write("- Missing values are forward/backfilled in preprocessing; verify extreme imputation and data quality before operational use.")
    st.write("- Use monthly aggregation to inspect seasonality patterns.")

# --- Models & Evaluation ---
elif page == "Models & Evaluation":
    st.header("Models & Evaluation")
    st.markdown("Train/test strategy: chronological split (first `1-test_size` fraction used for training, last `test_size` fraction for testing). This prevents data leakage from future into past.")

    if do_train:
        with st.spinner("Training models..."):
            results = train_and_evaluate(df, features=[c for c in df.columns if c not in ['date', target]], target=target, test_size=test_size, rf_n=rf_n, rf_max_depth=rf_max_depth)
            st.session_state['train_results'] = results
            st.success("Training complete — results stored in session.")
    else:
        results = st.session_state.get('train_results', None)

    if results is None:
        st.info("Train models on the sidebar (Train & Evaluate Models).")
        st.stop()

    # Collect metrics into a table
    table = []
    for name in ['linear', 'decision_tree', 'random_forest']:
        m = results[name]['metrics']
        table.append({"model": name, "mae": round(m['mae'],3), "rmse": round(m['rmse'],3), "mape": round(m['mape'],3), "r2": round(m['r2'],3)})
    metrics_df = pd.DataFrame(table).set_index('model')
    st.subheader("Metrics (test set)")
    st.dataframe(metrics_df)

    # Bar chart for comparison
    st.plotly_chart(px.bar(metrics_df.reset_index().melt(id_vars='model', var_name='metric', value_name='value'), x='metric', y='value', color='model', barmode='group', title='Model comparison'), use_container_width=True)

    # Best model selection (by RMSE)
    best = min(table, key=lambda r: r['rmse'])
    best_name = best['model']
    st.success(f"Best model by RMSE: **{best_name}**")

    # Predicted vs Actual for best model
    y_test = results['y_test']
    y_pred = results[best_name]['pred']
    dates_test = results['test_df']['date'].tolist()
    st.subheader("Predicted vs Actual (best model)")
    st.plotly_chart(predicted_vs_actual_plot(y_test, y_pred, dates=dates_test), use_container_width=True)

    # Feature importance for Random Forest
    st.subheader("Feature importance (Random Forest)")
    rf_model = results['random_forest']['model']
    features_list = [c for c in df.columns if c not in ['date', target]]
    try:
        importances = rf_model.feature_importances_
        imp_df = pd.DataFrame({"feature": features_list, "importance": importances}).sort_values("importance", ascending=False).head(20)
        st.dataframe(imp_df.set_index('feature'))
        st.plotly_chart(px.bar(imp_df, x='feature', y='importance', title='RF feature importance'), use_container_width=True)
    except Exception as e:
        st.write("Feature importance not available:", e)

    # Option to save best model
    if st.button("Save best model"):
        model_obj = results[best_name]['model']
        path, meta = save_model(model_obj, base_name=best_name, features=features_list, metrics=results[best_name]['metrics'])
        st.success(f"Saved model to {path}")

# --- Forecast ---
elif page == "Forecast":
    st.header("Forecast")
    st.write("Use the best model saved in session (from Models page) to produce short horizon forecasts. Forecast uses last available row's features and iteratively predicts next days.")

    results = st.session_state.get('train_results', None)
    if results is None:
        st.info("Train models first on the Models & Evaluation page.")
        st.stop()

    best = min([{"model":k, **{"rmse": results[k]["metrics"]["rmse"]}} for k in ['linear','decision_tree','random_forest']], key=lambda x: x['rmse'])
    best_name = best['model']
    st.write(f"Using best model: **{best_name}**")
    model_obj = results[best_name]['model']
    features_list = [c for c in df.columns if c not in ['date', target]]

    n_days = st.number_input("Forecast horizon (days)", min_value=1, max_value=30, value=5)
    # user option: show big emoji or plain numeric table
    use_emoji = st.checkbox("Show big animated emoji (sun / rain) in forecast table", value=True)

    if st.button("Generate forecast"):
        last_row = df[features_list].tail(1).copy().fillna(0)
        out_rows = []
        last_date = pd.to_datetime(df['date'].iloc[-1], errors='coerce')
        if pd.isna(last_date):
            last_date = pd.Timestamp.now()
        for i in range(n_days):
            Xp = last_row.copy()
            try:
                Xp_num = Xp.apply(pd.to_numeric, errors='coerce').fillna(0.0).values
            except Exception:
                Xp_num = Xp.values.astype(float)
            try:
                pred = model_obj.predict(Xp_num)[0]
            except Exception:
                try:
                    pred = model_obj.predict(Xp)[0]
                except Exception:
                    pred = 0.0
            pred = float(pred)
            new_date = (last_date + pd.Timedelta(days=i+1)).date()
            # decide icon: >=15 => sunny, else rainy
            if pred >= 15:
                icon_html = "<span class='emoji-sunny' title='Sunny'>☀️</span>"
            else:
                icon_html = "<span class='emoji-rain' title='Rainy'>🌧️</span>"
            out_rows.append({"date": str(new_date), "pred_mean": round(pred, 2), "icon_html": icon_html})
            # iterative injection
            if 'mean_temp' in last_row.columns:
                last_row.at[last_row.index[0], 'mean_temp'] = pred
            elif target in last_row.columns:
                last_row.at[last_row.index[0], target] = pred

        forecast_df = pd.DataFrame(out_rows)

        # Render forecast: if emoji chosen, show HTML table with big icons; else show numeric DataFrame
        if use_emoji:
            html = "<table class='forecast-table'>"
            html += "<thead><tr><th>Date</th><th>Weather</th><th>Predicted</th></tr></thead><tbody>"
            for r in out_rows:
                html += f"<tr><td class='forecast-date'>{r['date']}</td><td>{r['icon_html']}</td><td class='forecast-temp'>{r['pred_mean']} °C</td></tr>"
            html += "</tbody></table>"
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.dataframe(forecast_df[['date','pred_mean']].rename(columns={'pred_mean':'predicted_temp'}))

        # CSV download
        csv = forecast_df[['date','pred_mean']].rename(columns={'pred_mean':'predicted_temp'}).to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv, file_name="forecast.csv", mime="text/csv")