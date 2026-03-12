# src/data_loader.py
"""
Load and preprocess the weather CSV.
Functions:
- load_csv(path)
- preprocess(df)
The preprocess function will:
- normalize column names (lowercase, strip, replace spaces -> underscore)
- auto-detect a date column (several common names + heuristic)
- auto-detect a temperature target column (common synonyms)
- parse date, coerce numeric columns, forward/backfill missing
- attach detection results in df.attrs['detected_*'] for downstream use
"""
from pathlib import Path
from typing import Optional, Tuple, List
import pandas as pd
import numpy as np

DATE_COL = "date"  # canonical name used inside the app

# candidate names for auto-detection
_POSSIBLE_DATE_COLS = [
    "date", "datetime", "timestamp", "time", "day", "obs_time", "recorded_at"
]
# candidate target temp column names (ordered preference)
_POSSIBLE_TEMP_COLS = [
    "mean_temp", "mean temperature", "avg_temp", "avg temperature",
    "temperature", "temp", "t", "air_temperature", "air temp", "max_temp", "min_temp"
]


def load_csv(path: str) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    return df


def _normalize_colname(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = s.replace(" ", "_").replace("-", "_")
    # collapse multiple underscores
    while "__" in s:
        s = s.replace("__", "_")
    return s


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {c: _normalize_colname(c) for c in df.columns}
    return df.rename(columns=mapping)


def _find_date_column(df: pd.DataFrame) -> Optional[str]:
    # try exact candidates first
    cols = list(df.columns)
    for cand in _POSSIBLE_DATE_COLS:
        if cand in cols:
            return cand
    # try to find a column that looks like a date by sampling
    for c in cols:
        # skip numeric-only columns (very unlikely to be date strings, but still test sample)
        try:
            sample = df[c].dropna().astype(str).head(10)
            if sample.empty:
                continue
            # try parsing samples
            parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
            if parsed.notna().sum() >= max(1, int(len(sample) * 0.5)):
                return c
            # try YYYYMMDD ints (common)
            parsed2 = pd.to_datetime(sample, format="%Y%m%d", errors="coerce")
            if parsed2.notna().sum() >= max(1, int(len(sample) * 0.5)):
                return c
        except Exception:
            continue
    return None


def _find_target_column(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    # exact / substring matches by preference order
    for cand in _POSSIBLE_TEMP_COLS:
        cand_norm = cand.replace(" ", "_")
        for c in cols:
            if c == cand_norm:
                return c
    # substring match (any column containing 'temp' or 'temperature')
    for c in cols:
        if "temp" in c or "temperature" in c or c in ("t",):
            return c
    # fallback: choose numeric column with 'mean' or 'avg' or the first numeric column
    for c in cols:
        if ("mean" in c or "avg" in c) and pd.api.types.is_numeric_dtype(df[c]):
            return c
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and return a DataFrame standardized for the dashboard.
    Detected columns are stored in df.attrs:
      df.attrs['detected_date_col'] -> original normalized column name chosen as date
      df.attrs['detected_target_col'] -> chosen target column (e.g. mean_temp)
      df.attrs['original_columns'] -> list(original col names normalized)
    """
    df = df.copy()

    # normalize column names so detection is easier
    df = _normalize_columns(df)
    df.attrs['original_columns'] = list(df.columns)

    # detect date column
    detected_date = _find_date_column(df)
    if detected_date is not None:
        # try parsing: consider both ISO-like and YYYYMMDD integer patterns
        # create a new canonical DATE_COL
        try:
            parsed = pd.to_datetime(df[detected_date], errors="coerce", infer_datetime_format=True)
            if parsed.isna().all():
                # try YYYYMMDD fallback
                parsed = pd.to_datetime(df[detected_date].astype(str), format="%Y%m%d", errors="coerce")
            df[DATE_COL] = parsed
        except Exception:
            df[DATE_COL] = pd.to_datetime(df[detected_date].astype(str), errors="coerce", infer_datetime_format=True)
    else:
        # no date column found: create date column from index or set to now
        try:
            # if index looks like a date index, use it
            idx = df.index
            if idx.dtype == "datetime64[ns]" or pd.to_datetime(idx, errors="coerce").notna().any():
                df[DATE_COL] = pd.to_datetime(df.index, errors="coerce")
            else:
                df[DATE_COL] = pd.Timestamp.now()
        except Exception:
            df[DATE_COL] = pd.Timestamp.now()

    df.attrs['detected_date_col'] = detected_date or DATE_COL

    # convert other columns to numeric when possible (skip the canonical DATE_COL)
    for col in df.columns:
        if col == DATE_COL:
            continue
        # keep original non-numeric columns (like 'condition') but attempt numeric coercion
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # order by date (if possible) and fill missing numerics
    try:
        df = df.sort_values(DATE_COL).reset_index(drop=True)
    except Exception:
        df = df.reset_index(drop=True)

    # simple missing value strategy: forward fill then backward fill
    df = df.fillna(method='ffill').fillna(method='bfill')

    # basic feature: day, month, year
    try:
        df['day'] = pd.to_datetime(df[DATE_COL], errors='coerce').dt.day
        df['month'] = pd.to_datetime(df[DATE_COL], errors='coerce').dt.month
        df['year'] = pd.to_datetime(df[DATE_COL], errors='coerce').dt.year
    except Exception:
        df['day'] = 0
        df['month'] = 0
        df['year'] = 0

    # detect target column (temperature) and attach to attrs
    detected_target = _find_target_column(df)
    df.attrs['detected_target_col'] = detected_target
    
    return df