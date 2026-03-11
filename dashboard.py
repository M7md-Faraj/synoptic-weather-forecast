"""
Advanced Streamlit dashboard for exploring the dataset, training models and making predictions.
Run:
streamlit run app/dashboard.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import io
from src.data_loader import load_csv, preprocess
from src.features import create_lag_features
from src.train import train_pipeline
from src.models import get_model, evaluate_model
from src.predict import predict_from_model
from src.analysis import summary_stats, plot_time_series, correlation_heatmap
from src.utils import to_csv_download_link

st.set_page_config(page_title='Synoptic Weather Forecast', layout='wide')

st.title('Synoptic Weather Forecast App')

# Sidebar controls
with st.sidebar:
    st.header('Controls')
    uploaded = st.file_uploader('Upload CSV (optional)', type=['csv'])
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        st.info('Using sample from data/weather.csv — replace with full dataset in `data/`')
        df = load_csv('data/weather.csv')

    df = preprocess(df)

    target = st.selectbox('Target variable', options=['mean_temp','max_temp','min_temp','precipitation'])
    default_features = [c for c in df.columns if c not in ['date', target]]
    features = st.multiselect('Features (choose at least 3)', options=default_features, default=default_features[:6])

    model_choice = st.selectbox('Model', ['random_forest', 'linear'])
    n_estimators = st.slider('RF: n_estimators', 10, 500, 100) if model_choice == 'random_forest' else None

    train_btn = st.button('Train model')
    predict_btn = st.button('Predict (last rows)')

# Main layout
col1, col2 = st.columns([2,1])

with col1:
    st.subheader('Dataset preview')
    st.dataframe(df.head(200))

    st.subheader('Summary statistics')
    st.write(summary_stats(df.select_dtypes(include=[np.number])))

    st.subheader('Time series')
    ts_col = st.selectbox('Choose column to plot', options=['mean_temp','max_temp','min_temp','precipitation','sunshine'])
    fig_ts = plot_time_series(df, ts_col)
    st.pyplot(fig_ts)

    st.subheader('Correlation matrix')
    fig_corr = correlation_heatmap(df.select_dtypes(include=[np.number]))
    st.pyplot(fig_corr)

with col2:
    st.subheader('Feature importance & models')
    st.markdown('**Model selection**')
    st.write(f'Chosen model: {model_choice}')

    if train_btn:
        st.info('Training...')
        model_kwargs = {'n_estimators': n_estimators} if model_choice == 'random_forest' else {}
        model, metrics, _ = train_pipeline(df[features + ['date', target]].dropna(), target=target, features=features, model_name=model_choice, **model_kwargs)
        st.success('Training complete')
        st.write('Metrics:')
        st.json(metrics)
        # save model file
        st.write('Model saved to disk as model_{}.joblib'.format(model_choice))

    if predict_btn:
        st.info('Generating predictions for the last 10 rows')
        # simple predict flow: use the last N rows
        X_pred = df[features].tail(10)
        model_path = f'model_{model_choice}.joblib'
        try:
            preds = predict_from_model(model_path, X_pred)
            out = X_pred.copy()
            out['prediction'] = preds
            st.dataframe(out)
            csv_bytes = to_csv_download_link(out)
            st.download_button('Download predictions CSV', data=csv_bytes, file_name='predictions.csv')
        except Exception as e:
            st.error(f'Prediction failed: {e}. You need to train a model first via the Train model button.')

# Footer: quick forecasting tool
st.markdown('---')
st.subheader('Quick forecast (naive for demonstration)')
cols = st.columns(3)
st.download_button('Download forecast', data=out.to_csv(index=False).encoderode(), file_name='forecast.csv')