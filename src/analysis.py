import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return numeric summary statistics transposed for easier viewing."""
    return df.describe().T

def missing_values_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a table of missing counts and percentages for each column."""
    miss = df.isna().sum()
    pct = (miss / len(df) * 100).round(2)
    out = pd.DataFrame({"missing_count": miss, "missing_pct": pct})
    return out.sort_values("missing_count", ascending=False)

def plot_time_series(df: pd.DataFrame, date_col='date', target='mean_temp', title=None):
    """Line plot of a target over time using Plotly Express."""
    dfc = df.copy()
    dfc = dfc.sort_values(date_col)
    title = title or f"{target} over time"
    fig = px.line(dfc, x=date_col, y=target, title=title, labels={date_col: "Date", target: target})
    fig.update_layout(template='plotly_white', height=400)
    return fig

def monthly_aggregate_plot(df: pd.DataFrame, date_col='date', target='mean_temp', agg='mean'):
    """Monthly aggregated line plot (monthly mean/median etc.)."""
    dfc = df.copy()
    dfc['month'] = pd.to_datetime(dfc[date_col]).dt.to_period('M')
    grouped = dfc.groupby('month')[target].agg(agg).reset_index()
    grouped['month'] = grouped['month'].dt.to_timestamp()
    fig = px.line(grouped, x='month', y=target, title=f'Monthly {agg} of {target}', labels={'month': 'Month', target: target})
    fig.update_layout(template='plotly_white', height=400)
    return fig

def correlation_heatmap_plotly(df: pd.DataFrame):
    """Plot correlation heatmap for numeric columns."""
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
    fig.update_layout(title='Correlation heatmap', template='plotly_white', height=480)
    return fig

def distribution_plot(df: pd.DataFrame, col: str, nbins=50):
    """Histogram with marginal boxplot for a column."""
    fig = px.histogram(df, x=col, nbins=nbins, marginal='box', title=f'Distribution of {col}')
    fig.update_layout(template='plotly_white', height=400)
    return fig

def predicted_vs_actual_plot(y_true, y_pred, dates=None, title='Predicted vs Actual'):
    """Line plot comparing predicted vs actual values, optionally by date axis."""
    df = pd.DataFrame({"actual": y_true, "predicted": y_pred})
    if dates is not None:
        df['date'] = dates
        fig = px.line(df, x='date', y=['actual', 'predicted'], title=title)
    else:
        fig = px.line(df.reset_index(), x='index', y=['actual', 'predicted'], title=title)
    fig.update_layout(template='plotly_white', height=400)
    return fig