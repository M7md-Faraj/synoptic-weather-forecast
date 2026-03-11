"""
Load and preprocess the weather CSV.
Functions:
- load_csv(path)
- preprocess(df)
"""
import pandas as pd
from pathlib import Path

DATE_COL = "date"


def load_csv(path: str) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # parse date: input format appears as YYYYMMDD integer/string
    df[DATE_COL] = df[DATE_COL].astype(str)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="%Y%m%d", errors='coerce')

    # replace empty strings with NaN and convert numeric columns
    for col in df.columns:
        if col == DATE_COL:
            continue
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # simple missing value strategy: forward fill then backward fill
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    df = df.fillna(method='ffill').fillna(method='bfill')

    # basic feature: day, month, year
    df['day'] = df[DATE_COL].dt.day
    df['month'] = df[DATE_COL].dt.month
    df['year'] = df[DATE_COL].dt.year

    return df