"""
utils/data_loader.py
====================
Centralised data loading and feature engineering for the London Weather ML project.

All preprocessing decisions are documented with their rationale so this file
can be referenced directly in a methodology chapter.

Dataset: London weather (Met Office / Kaggle), 1979–2020, daily observations.
Columns: date, cloud_cover, sunshine, global_radiation, max_temp, mean_temp,
         min_temp, precipitation, pressure, snow_depth
"""

import os
import numpy as np
import pandas as pd

# ─── File path ──────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "london_weather.csv")

# ─── Month / season label helpers ───────────────────────────────────────────
MONTH_LABELS  = ["Jan","Feb","Mar","Apr","May","Jun",
                 "Jul","Aug","Sep","Oct","Nov","Dec"]
SEASON_LABELS = ["Winter","Spring","Summer","Autumn"]

# Mapping from month number to season integer (Winter=0 … Autumn=3)
SEASON_MAP = {12:0, 1:0, 2:0,   # Winter
              3:1,  4:1, 5:1,   # Spring
              6:2,  7:2, 8:2,   # Summer
              9:3, 10:3, 11:3}  # Autumn

# ─── Engineered feature list (used by both trainer and app) ─────────────────
FEATURE_COLS = [
    # Raw atmospheric measurements
    "cloud_cover", "sunshine", "global_radiation",
    "precipitation", "pressure_hpa", "snow_depth",
    # Cyclical temporal encoding
    "month_sin", "month_cos", "doy_sin", "doy_cos",
    # Calendar / ordinal time features
    "season", "year",
    # Lag and rolling features (key autocorrelation signals)
    "mean_temp_lag1", "mean_temp_lag7", "max_temp_lag1",
    "precipitation_lag1", "temp_rolling7", "rain_rolling7",
]

TARGET_COL = "mean_temp"


def load_raw() -> pd.DataFrame:
    """
    Load the raw CSV and parse the date column.

    Returns
    -------
    pd.DataFrame with 'date' as a proper datetime object.
    """
    df = pd.read_csv(DATA_PATH)

    # date is stored as integer YYYYMMDD (e.g. 19790101)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all preprocessing and feature engineering steps.

    Steps and rationale
    -------------------
    1. Sort chronologically — MUST be done before any lag/rolling operations
       to prevent data leakage (future rows entering past calculations).

    2. Pressure Pa → hPa — standard meteorological unit; improves
       interpretability of model coefficients and feature importance.

    3. snow_depth NaN → 0 — domain-justified: a missing snow record almost
       always means there was no measurable snow, consistent with the low
       values in surrounding rows (9.4% of records affected).

    4. Calendar features — year, month, day_of_year used downstream
       for cyclical encoding and season labelling.

    5. Season ordinal — Winter=0, Spring=1, Summer=2, Autumn=3.
       Provides a coarse seasonal signal for tree-based models.

    6. Cyclical sin/cos encoding — prevents the artificial discontinuity
       that integer month/day representations create at the Dec→Jan and
       31 Dec→1 Jan boundaries. Without this, models treat Dec and Jan as
       maximally different when they are climatologically similar.

    7. Lag features — exploit strong temporal autocorrelation:
       • lag-1 temperature: r ≈ 0.97 with same-day value
       • lag-7 temperature: r ≈ 0.80
       All lags use .shift(1) so no same-day information leaks in.

    8. Rolling features — 7-day rolling mean temperature and 7-day
       cumulative rainfall capture recent weather regimes (warm spell,
       wet period) that lag-1 alone misses.

    9. Drop first ~7 rows — these have NaN lag/rolling values because
       there is no prior history available; dropping them is correct.

    Parameters
    ----------
    df : raw DataFrame from load_raw()

    Returns
    -------
    Fully engineered DataFrame ready for modelling.
    """

    # Step 1 — chronological sort (critical before any lag/rolling)
    df = df.sort_values("date").reset_index(drop=True)

    # Step 2 — pressure unit conversion
    df["pressure_hpa"] = df["pressure"] / 100

    # Step 3 — snow depth imputation
    df["snow_depth"] = df["snow_depth"].fillna(0)

    # Step 4 — calendar features
    df["year"]        = df["date"].dt.year
    df["month"]       = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear

    # Step 5 — season ordinal
    df["season"] = df["month"].map(SEASON_MAP)

    # Step 6 — cyclical encoding
    df["month_sin"] = np.sin(2 * np.pi * df["month"]        / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"]        / 12)
    df["doy_sin"]   = np.sin(2 * np.pi * df["day_of_year"]  / 365)
    df["doy_cos"]   = np.cos(2 * np.pi * df["day_of_year"]  / 365)

    # Step 7 — lag features (shift(1) = use yesterday's value only)
    df["mean_temp_lag1"]     = df["mean_temp"].shift(1)
    df["mean_temp_lag7"]     = df["mean_temp"].shift(7)
    df["max_temp_lag1"]      = df["max_temp"].shift(1)
    df["precipitation_lag1"] = df["precipitation"].shift(1)

    # Step 8 — rolling features (shift first, then roll, to prevent leakage)
    df["temp_rolling7"] = df["mean_temp"].shift(1).rolling(7).mean()
    df["rain_rolling7"] = df["precipitation"].shift(1).rolling(7).sum()

    # Step 9 — drop rows where lag/rolling features are NaN (first ~7 rows)
    df = df.dropna(subset=["mean_temp_lag1", "mean_temp_lag7",
                            "temp_rolling7",  "rain_rolling7"])

    return df.reset_index(drop=True)


def load_data() -> pd.DataFrame:
    """
    Convenience function: load raw CSV and apply all preprocessing.

    Returns
    -------
    Fully engineered DataFrame ready for EDA and modelling.
    """
    return preprocess(load_raw())


def get_monthly_medians(df: pd.DataFrame) -> dict:
    """
    Compute per-month median values for all key columns.

    Used on the Forecast page to build synthetic input rows for
    future dates (where real prior observations are unavailable).
    Clearly labelled in the UI as historical medians, not live data.

    Parameters
    ----------
    df : preprocessed DataFrame

    Returns
    -------
    dict keyed by month number (1–12), each value is a dict of medians.
    """
    medians = {}
    for m in range(1, 13):
        sub = df[df["month"] == m]
        medians[m] = {
            "mean_temp":        sub["mean_temp"].median(),
            "max_temp":         sub["max_temp"].median(),
            "min_temp":         sub["min_temp"].median(),
            "cloud_cover":      sub["cloud_cover"].median(),
            "sunshine":         sub["sunshine"].median(),
            "global_radiation": sub["global_radiation"].median(),
            "pressure_hpa":     sub["pressure_hpa"].median(),
            "precipitation":    sub["precipitation"].median(),
            "snow_depth":       sub["snow_depth"].median(),
        }
    return medians


def build_forecast_input(target_date: pd.Timestamp,
                         monthly_medians: dict) -> pd.DataFrame:
    """
    Build a single-row input DataFrame for a given future date.

    Because real lag observations are unavailable for future dates,
    we substitute historical monthly medians from the training set.
    This is explicitly a simplification that produces climatological
    estimates rather than true day-ahead forecasts.

    Parameters
    ----------
    target_date    : the date we want to predict for
    monthly_medians: output of get_monthly_medians()

    Returns
    -------
    Single-row DataFrame with all 18 feature columns.
    """
    m   = target_date.month
    doy = target_date.dayofyear
    med = monthly_medians[m]

    row = {
        # Atmospheric (from monthly medians)
        "cloud_cover":      med["cloud_cover"],
        "sunshine":         med["sunshine"],
        "global_radiation": med["global_radiation"],
        "precipitation":    med["precipitation"],
        "pressure_hpa":     med["pressure_hpa"],
        "snow_depth":       med["snow_depth"],
        # Cyclical encoding
        "month_sin":  np.sin(2 * np.pi * m   / 12),
        "month_cos":  np.cos(2 * np.pi * m   / 12),
        "doy_sin":    np.sin(2 * np.pi * doy / 365),
        "doy_cos":    np.cos(2 * np.pi * doy / 365),
        # Calendar
        "season": SEASON_MAP[m],
        "year":   target_date.year,
        # Lag features (substituted with monthly median as best proxy)
        "mean_temp_lag1":     med["mean_temp"],
        "mean_temp_lag7":     med["mean_temp"],
        "max_temp_lag1":      med["max_temp"],
        "precipitation_lag1": med["precipitation"],
        "temp_rolling7":      med["mean_temp"],
        "rain_rolling7":      med["precipitation"] * 7,
    }
    return pd.DataFrame([row])
