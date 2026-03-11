"""Feature engineering helpers.
- create_lag_features(df, col, lags)
"""
import pandas as pd

def create_lag_features(df: pd.DataFrame, col: str, lags: list = [1,2,3]) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    df = df.dropna().reset_index(drop=True)
    return df