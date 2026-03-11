# src/train.py
"""
Training utilities with progress/metrics streaming.

Functions:
- train_random_forest_progress(...)
- train_sgd_progress(...)
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def _eval(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}

def train_random_forest_progress(df, features, target,
                                 n_estimators=100, step=10, test_size=0.2, random_state=42):
    """
    Train RandomForest incrementally using warm_start and yield progress after every `step` trees.
    Yields tuples: (current_n_estimators, model, metrics, (X_val, y_val, preds_val))
    """
    df = df.copy().dropna(subset=features + [target])
    X = df[features].values
    y = df[target].values

    # Sequential split for time-series-like data: keep order, take last portion as validation
    split_idx = int((1 - test_size) * len(df))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # initial estimator
    current = step
    model = RandomForestRegressor(n_estimators=step, warm_start=True, random_state=random_state, n_jobs=-1)
    while current <= n_estimators:
        model.fit(X_train, y_train)  # warm_start will add trees after first loop
        preds = model.predict(X_val)
        metrics = _eval(y_val, preds)
        yield current, model, metrics, (X_val, y_val, preds)
        current += step
        model.n_estimators = min(current, n_estimators)

def train_sgd_progress(df, features, target, epochs=10, test_size=0.2, random_state=42):
    """
    Train an SGDRegressor using partial_fit and yield progress for each epoch.
    Yields tuples: (epoch, model, metrics, (X_val, y_val, preds_val))
    """
    df = df.copy().dropna(subset=features + [target])
    X = df[features].values
    y = df[target].values

    split_idx = int((1 - test_size) * len(df))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # initialize SGDRegressor
    model = SGDRegressor(max_iter=1, tol=None, random_state=random_state)
    # do a first call to partial_fit with the shape so it initializes correctly
    try:
        model.partial_fit(X_train, y_train)
    except Exception:
        # fallback to a simple fit if partial_fit not supported
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        metrics = _eval(y_val, preds)
        yield 1, model, metrics, (X_val, y_val, preds)
        return

    # after first partial_fit we can iterate epochs
    for epoch in range(1, epochs + 1):
        model.partial_fit(X_train, y_train)
        preds = model.predict(X_val)
        metrics = _eval(y_val, preds)
        yield epoch, model, metrics, (X_val, y_val, preds)