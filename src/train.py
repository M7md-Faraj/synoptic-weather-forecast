import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def _metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred)
    # Avoid division by zero for MAPE
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_true - y_pred) / np.where(np.abs(y_true) < 1e-8, 1e-8, np.abs(y_true)))) * 100.0
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else 0.0
    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape), "r2": float(r2)}

def time_series_split(df, features, target, test_size=0.2):
    df_ = df.copy().dropna(subset=features + [target])
    n = len(df_)
    split_idx = int((1 - test_size) * n)
    train = df_.iloc[:split_idx]
    test = df_.iloc[split_idx:]
    X_train = train[features].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
    y_train = train[target].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
    X_test = test[features].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
    y_test = test[target].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
    return X_train, X_test, y_train, y_test, train, test

def train_and_evaluate(df, features, target, test_size=0.2, random_state=42, rf_n=100, rf_max_depth=None):
    # Split using chronological split
    X_train, X_test, y_train, y_test, train_df, test_df = time_series_split(df, features, target, test_size=test_size)

    # Linear regression pipeline (scaling)
    lin = Pipeline([("scaler", StandardScaler()), ("linear", LinearRegression())])
    lin.fit(X_train, y_train)
    lin_pred = lin.predict(X_test)
    lin_metrics = _metrics(y_test, lin_pred)

    # Decision tree
    dt = DecisionTreeRegressor(random_state=random_state, max_depth=None)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    dt_metrics = _metrics(y_test, dt_pred)

    # Random forest
    rf = RandomForestRegressor(n_estimators=rf_n, random_state=random_state, n_jobs=-1, max_depth=rf_max_depth)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_metrics = _metrics(y_test, rf_pred)

    # package results
    results = {
        "linear": {"model": lin, "pred": lin_pred, "metrics": lin_metrics},
        "decision_tree": {"model": dt, "pred": dt_pred, "metrics": dt_metrics},
        "random_forest": {"model": rf, "pred": rf_pred, "metrics": rf_metrics},
        "X_test": X_test,
        "y_test": y_test,
        "train_df": train_df,
        "test_df": test_df
    }
    return results