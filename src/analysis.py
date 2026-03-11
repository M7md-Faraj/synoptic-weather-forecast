"""
Exploratory data analysis helpers.
Return interactive Plotly figures for use in the dashboard.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe()

def plotly_time_series(df: pd.DataFrame, col: str, title: str = None):
    """
    Interactive time series with rolling average overlay and simple animation-ready frame.
    """
    df = df.copy()
    df = df.sort_values('date')
    df['rolling_7'] = df[col].rolling(7, min_periods=1).mean()
    title = title or f"Time series of {col}"
    fig = px.line(df, x='date', y=col, title=title, labels={col: col, 'date': 'Date'})
    fig.add_traces(px.line(df, x='date', y='rolling_7').data)
    fig.update_traces(mode='lines')
    fig.update_layout(hovermode='x unified', template='plotly_white')
    return fig

def correlation_heatmap_plotly(df: pd.DataFrame):
    df_num = df.select_dtypes(include=[np.number]).copy()
    corr = df_num.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='RdBu',
        zmid=0,
        colorbar=dict(title="corr")
    ))
    fig.update_layout(title='Correlation heatmap', template='plotly_white', height=600)
    return fig

def distribution_plot(df: pd.DataFrame, col: str):
    fig = px.histogram(df, x=col, nbins=60, marginal='box', title=f'Distribution of {col}')
    fig.update_layout(template='plotly_white')
    return fig

def weather_condition_summary(df: pd.DataFrame, temp_col='mean_temp', precip_col='precipitation'):
    """
    Basic counts for hot/cold/rainy days. Thresholds can be adjusted.
    Returns dict and a small bar figure.
    """
    df = df.copy()
    hot = (df[temp_col] >= 25).sum()
    warm = ((df[temp_col] >= 15) & (df[temp_col] < 25)).sum()
    cool = ((df[temp_col] >= 5) & (df[temp_col] < 15)).sum()
    cold = (df[temp_col] < 5).sum()
    rainy = (df[precip_col] > 0).sum()
    totals = {'hot': int(hot), 'warm': int(warm), 'cool': int(cool), 'cold': int(cold), 'rainy_days': int(rainy)}
    fig = px.bar(x=list(totals.keys()), y=list(totals.values()), labels={'x': 'condition', 'y': 'count'}, title='Weather condition counts')
    fig.update_layout(template='plotly_white')
    return totals, fig