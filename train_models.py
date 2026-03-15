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


def compute_mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > 1e-8)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def compute_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return {"MAE": None, "RMSE": None, "R2": None, "MAPE%": None}
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = float("nan")
    mape = compute_mape(y_true, y_pred)
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4) if np.isfinite(r2) else None, "MAPE%": round(mape, 3)}


def compute_baselines(df, split_idx):
    df = df.reset_index(drop=True)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]
    y_test = df_test[TARGET_COL].values if TARGET_COL in df_test.columns else np.array([])

    # Persistence baseline (lag-1) aligned to test rows using shift
    df['persist'] = df[TARGET_COL].shift(1)
    y_persist = df['persist'].iloc[split_idx:].values
    # align and mask NaNs
    mask = np.isfinite(y_test) & np.isfinite(y_persist)
    if mask.any():
        persist_m = compute_metrics(y_test[mask], y_persist[mask])
    else:
        persist_m = {"MAE": None, "RMSE": None, "R2": None, "MAPE%": None}

    # Climatology: monthly mean from training set (if month exists)
    clim_m = {"MAE": None, "RMSE": None, "R2": None, "MAPE%": None}
    if "month" in df.columns and len(df_train) > 0:
        monthly_mean = df_train.groupby("month")[TARGET_COL].mean()
        y_clim = df_test["month"].map(monthly_mean).values
        mask2 = np.isfinite(y_test) & np.isfinite(y_clim)
        if mask2.any():
            clim_m = compute_metrics(y_test[mask2], y_clim[mask2])

    return {
        "Persistence (lag-1)": persist_m,
        "Climatology (monthly mean)": clim_m,
    }


def build_model_defs(hyperparams=None):
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
    def _report(pct, msg):
        try:
            if progress_callback:
                progress_callback(int(pct), str(msg))
        except Exception:
            pass

    _report(0, "Starting training pipeline...")
    df = load_data()
    if df is None or df.empty:
        raise RuntimeError("No data loaded for training.")

    # Ensure target exists and drop rows where target is missing
    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found in dataset.")
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    n = len(df)
    split = max(1, int(n * 0.8))

    # Build feature matrix using only features that exist in df
    features_present = [c for c in FEATURE_COLS if c in df.columns]
    if len(features_present) == 0:
        # fall back: use all numeric columns except target and date
        features_present = [c for c in df.select_dtypes(include=[np.number]).columns.tolist() if c not in [TARGET_COL]]

    X_raw = df[features_present].values
    y = df[TARGET_COL].values

    _report(10, "Imputing missing values (median)...")
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_raw)

    # split after imputation so the imputer is fit on full data (consistent with CV)
    X_train, X_test = X_imp[:split], X_imp[split:]
    y_train, y_test = y[:split], y[split:]

    _report(20, "Scaling features for linear model...")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    _report(25, "Computing baseline metrics...")
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
    start_pct = 35
    end_pct = 85

    for i, (name, model) in enumerate(model_defs.items(), start=1):
        pct_start = start_pct + (i - 1) * (end_pct - start_pct) / n_models
        pct_end = start_pct + i * (end_pct - start_pct) / n_models
        _report(pct_start, f"Training {name}...")
        use_scaler = (name == "Linear Regression")
        Xtr = X_train_sc if use_scaler else X_train
        Xte = X_test_sc if use_scaler else X_test

        # fit model (Xtr, y_train should have no NaNs)
        model.fit(Xtr, y_train)
        time.sleep(0.15)

        y_pred = model.predict(Xte)
        m = compute_metrics(y_test, y_pred)
        results[name] = m

        if (m.get("R2") is not None) and (m["R2"] > best_r2):
            best_r2 = m["R2"]
            best_name = name

        # Save bundle: include imputer & scaler for consistent preprocessing at inference time
        bundles[name] = {
            "model": model,
            "scaler": scaler if use_scaler else None,
            "imputer": imputer,
            "features": features_present,
            "use_scaler": use_scaler,
            "target": TARGET_COL,
        }

        _report(pct_end - 1, f"Evaluated {name}: MAE={m.get('MAE')}, RMSE={m.get('RMSE')}, R²={m.get('R2')}, MAPE%={m.get('MAPE%')}")

    results["best_model"] = best_name
    results["timestamp"] = pd.Timestamp.now().isoformat()
    _report(90, f"Best model: {best_name} (R²={best_r2:.4f})" if best_name else "No best model")

    _report(92, "Saving model bundles and results...")
    joblib.dump(bundles, os.path.join(MODELS_DIR, "trained_models.pkl"))
    with open(os.path.join(MODELS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    _report(98, "Finalising and cleaning up...")
    time.sleep(0.1)
    _report(100, "Training complete")
    return results


if __name__ == "__main__":
    train_and_save()