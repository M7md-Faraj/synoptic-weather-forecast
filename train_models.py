import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import sys
import time

warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data_loader import load_data, FEATURE_COLS, TARGET_COL

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def compute_metrics(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}


def compute_baselines(df, split_idx):
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]
    y_test = df_test[TARGET_COL].values

    # Persistence baseline: yesterday's value
    y_persist = df.iloc[split_idx - 1 : split_idx + len(df_test) - 1][TARGET_COL].values
    n = min(len(y_test), len(y_persist))
    mask = ~(np.isnan(y_test[:n]) | np.isnan(y_persist[:n]))
    persist_m = compute_metrics(y_test[:n][mask], y_persist[:n][mask])

    # Climatology baseline: monthly mean from training set
    monthly_mean = df_train.groupby("month")[TARGET_COL].mean()
    y_clim = df_test["month"].map(monthly_mean).values
    clim_mask = ~(np.isnan(y_test) | np.isnan(y_clim))
    clim_m = compute_metrics(y_test[clim_mask], y_clim[clim_mask])

    return {
        "Persistence (lag-1)": persist_m,
        "Climatology (monthly mean)": clim_m,
    }


def build_model_defs(hyperparams=None):
    """Return a dict of model name -> instantiated model.
    hyperparams should be a dict keyed by model name with param dicts.
    """
    hp = hyperparams or {}
    lr_hp = hp.get("Linear Regression", {})
    dt_hp = hp.get("Decision Tree", {})
    rf_hp = hp.get("Random Forest", {})

    lr = LinearRegression(**lr_hp)

    dt = DecisionTreeRegressor(
        max_depth=dt_hp.get("max_depth", 8),
        min_samples_leaf=dt_hp.get("min_samples_leaf", 1),
        random_state=42,
    )

    rf = RandomForestRegressor(
        n_estimators=rf_hp.get("n_estimators", 100),
        max_depth=rf_hp.get("max_depth", 12),
        min_samples_leaf=rf_hp.get("min_samples_leaf", 1),
        random_state=42,
        n_jobs=-1,
    )

    return {
        "Linear Regression": lr,
        "Decision Tree": dt,
        "Random Forest": rf,
    }


def train_and_save(hyperparams=None, progress_callback=None):
    """
    Run the training pipeline. If hyperparams provided, use them.
    progress_callback: function(percent:int, message:str) - optional.
    Returns results dict on success.
    """
    def _report(pct, msg):
        try:
            if progress_callback:
                progress_callback(int(pct), str(msg))
        except Exception:
            # silently ignore callback errors so training still runs
            pass

    _report(0, "Starting training pipeline...")
    df = load_data()
    _report(5, f"Loaded data ({len(df):,} rows)")

    # drop rows where target is missing
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    n = len(df)
    split = int(n * 0.8)

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    _report(10, "Imputing missing values (median)...")
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    X_train, X_test = X_imp[:split], X_imp[split:]
    y_train, y_test = y[:split], y[split:]

    _report(15, "Scaling features for linear model...")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    _report(20, "Computing baseline metrics...")
    baselines = compute_baselines(df, split)
    with open(os.path.join(MODELS_DIR, "baselines.json"), "w") as f:
        json.dump(baselines, f, indent=2)
    _report(30, "Baselines saved")

    model_defs = build_model_defs(hyperparams)
    results = {}
    bundles = {}
    best_r2 = -999
    best_name = None

    model_names = list(model_defs.keys())
    n_models = len(model_names)
    # allocate progress window 35..85 for model training/evaluation
    start_pct = 35
    end_pct = 85
    for i, (name, model) in enumerate(model_defs.items(), start=1):
        pct_start = start_pct + (i - 1) * (end_pct - start_pct) / n_models
        pct_end = start_pct + i * (end_pct - start_pct) / n_models
        _report(pct_start, f"Training {name}...")
        use_scaler = (name == "Linear Regression")
        Xtr = X_train_sc if use_scaler else X_train
        Xte = X_test_sc if use_scaler else X_test

        model.fit(Xtr, y_train)
        # small sleep to make progress visible when called from UI
        time.sleep(0.2)

        y_pred = model.predict(Xte)
        m = compute_metrics(y_test, y_pred)
        results[name] = m

        if m["R2"] > best_r2:
            best_r2 = m["R2"]
            best_name = name

        bundles[name] = {
            "model": model,
            "scaler": scaler if use_scaler else None,
            "imputer": imputer,
            "features": FEATURE_COLS,
            "use_scaler": use_scaler,
            "target": TARGET_COL,
        }

        _report(pct_end - 1, f"Evaluated {name}: MAE={m['MAE']:.3f}, RMSE={m['RMSE']:.3f}, R²={m['R2']:.4f}")

    results["best_model"] = best_name
    _report(90, f"Best model: {best_name} (R²={best_r2:.4f})")

    _report(92, "Saving model bundles and results...")
    joblib.dump(bundles, os.path.join(MODELS_DIR, "academic_models.pkl"))
    with open(os.path.join(MODELS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    _report(98, "Finalising and cleaning up...")
    time.sleep(0.2)
    _report(100, "Training complete")
    return results


if __name__ == "__main__":
    train_and_save()