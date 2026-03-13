from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np

DATE_COL = "date"

_POSSIBLE_DATE_COLS = [
    "date", "datetime", "timestamp", "time", "day", "obs_time", "recorded_at"
]

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
    while "__" in s:
        s = s.replace("__", "_")
    return s

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {c: _normalize_colname(c) for c in df.columns}
    return df.rename(columns=mapping)

def _find_date_column(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    for cand in _POSSIBLE_DATE_COLS:
        if cand in cols:
            return cand
    for c in cols:
        try:
            sample = df[c].dropna().astype(str).head(20)
            if sample.empty:
                continue
            parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
            if parsed.notna().sum() >= max(1, int(len(sample) * 0.5)):
                return c
            parsed2 = pd.to_datetime(sample, format="%Y%m%d", errors="coerce")
            if parsed2.notna().sum() >= max(1, int(len(sample) * 0.5)):
                return c
        except Exception:
            continue
    return None

def _find_target_column(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    for cand in _POSSIBLE_TEMP_COLS:
        cand_norm = cand.replace(" ", "_")
        if cand_norm in cols:
            return cand_norm
    for c in cols:
        if "temp" in c or "temperature" in c or c in ("t",):
            return c
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        # prefer mean-like
        for c in numeric_cols:
            if "mean" in c or "avg" in c:
                return c
        return numeric_cols[0]
    return None

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names, parse date, coerce numeric-ish columns only,
    fill missing values with sensible forward/backfill, add date parts,
    and store detection results in df.attrs.
    """
    df = df.copy()
    df = _normalize_columns(df)
    df.attrs['original_columns'] = list(df.columns)

    detected_date = _find_date_column(df)
    parsed = None
    if detected_date is not None:
        # try several parsing strategies
        parsed = pd.to_datetime(df[detected_date], errors="coerce", infer_datetime_format=True)
        if parsed.isna().all():
            parsed = pd.to_datetime(df[detected_date].astype(str), format="%Y%m%d", errors="coerce")
        df[DATE_COL] = parsed
    else:
        # fallback: if index is datetime-like
        try:
            idx = df.index
            if idx.dtype == "datetime64[ns]" or pd.to_datetime(idx, errors="coerce").notna().any():
                df[DATE_COL] = pd.to_datetime(idx, errors="coerce")
            else:
                df[DATE_COL] = pd.Timestamp.now()
        except Exception:
            df[DATE_COL] = pd.Timestamp.now()

    df.attrs['detected_date_col'] = detected_date or DATE_COL

    # Only coerce to numeric when a column looks numeric (heuristic)
    for col in list(df.columns):
        if col == DATE_COL:
            continue
        # Check a small sample to see if numeric
        sample = df[col].dropna().astype(str).head(20)
        numeric_likely = False
        if not sample.empty:
            parsed = pd.to_numeric(sample, errors="coerce")
            # if at least 40% of sample parse to numeric -> treat as numeric
            if parsed.notna().sum() >= max(1, int(len(parsed) * 0.4)):
                numeric_likely = True
        if numeric_likely:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            # keep as-is, but strip string whitespace
            try:
                df[col] = df[col].astype(object)
                df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
            except Exception:
                pass

    # Sort by date if possible
    try:
        df = df.sort_values(DATE_COL).reset_index(drop=True)
    except Exception:
        df = df.reset_index(drop=True)

    # Fill numeric missing values with forward then backfill
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill').fillna(0.0)
    # For object columns, forward/backfill
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        df[obj_cols] = df[obj_cols].fillna(method='ffill').fillna(method='bfill').fillna("")

    # Add date parts for feature engineering
    try:
        df['day'] = pd.to_datetime(df[DATE_COL], errors='coerce').dt.day.fillna(0).astype(int)
        df['month'] = pd.to_datetime(df[DATE_COL], errors='coerce').dt.month.fillna(0).astype(int)
        df['year'] = pd.to_datetime(df[DATE_COL], errors='coerce').dt.year.fillna(0).astype(int)
    except Exception:
        df['day'] = 0
        df['month'] = 0
        df['year'] = 0

    detected_target = _find_target_column(df)
    df.attrs['detected_target_col'] = detected_target

    return df