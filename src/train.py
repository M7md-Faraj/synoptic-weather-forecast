from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

def _eval(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else 0.0
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}

def train_linear(df, features, target, test_size=0.2, random_state=42):
    """
    Train a scaled linear regression. Returns trained pipeline, metrics, X_val,y_val,preds_val, and meta dict.
    """
    df_ = df.copy().dropna(subset=features + [target])
    X = df_[features].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
    y = df_[target].apply(pd.to_numeric, errors='coerce').fillna(0.0).values

    split_idx = int((1 - test_size) * len(df_))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    pipeline = Pipeline([("scaler", StandardScaler()), ("linear", LinearRegression())])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val)
    metrics = _eval(y_val, preds)

    # meta: store target median for fallback + any other picklables
    meta = {"target_median": float(np.median(y_train)) if len(y_train) > 0 else float(np.median(y_val) if len(y_val) > 0 else 0.0)}
    return pipeline, metrics, (X_val, y_val, preds), meta

# Slightly safer RF incremental trainer: ensure numeric conversion
def train_random_forest_progress(df, features, target,
                                 n_estimators=100, step=10, test_size=0.2, random_state=42):
    df_ = df.copy().dropna(subset=features + [target])
    X = df_[features].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
    y = df_[target].apply(pd.to_numeric, errors='coerce').fillna(0.0).values

    split_idx = int((1 - test_size) * len(df_))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    current = step
    model = RandomForestRegressor(n_estimators=step, warm_start=True, random_state=random_state, n_jobs=-1)
    while current <= n_estimators:
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        metrics = _eval(y_val, preds)
        yield current, model, metrics, (X_val, y_val, preds)
        current += step
        model.n_estimators = min(current, n_estimators)