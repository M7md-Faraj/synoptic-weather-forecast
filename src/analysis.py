"""Exploratory data analysis helpers.
Functions return matplotlib/plotly objects that the dashboard can render.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe()


def plot_time_series(df: pd.DataFrame, col: str):
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df['date'], df[col])
    ax.set_title(f'Time series of {col}')
    ax.set_xlabel('date')
    ax.set_ylabel(col)
    fig.tight_layout()
    return fig


def correlation_heatmap(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', ax=ax)
    fig.tight_layout()
    return fig