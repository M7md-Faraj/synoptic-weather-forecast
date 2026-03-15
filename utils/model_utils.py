"""
Model loading, prediction, evaluation utilities.

Trained model bundles live in models/trained_models.pkl.
Each bundle is expected to be a dict with keys:
  - model: fitted sklearn-like estimator with .predict()
  - imputer: fitted SimpleImputer (or similar) used at training (optional)
  - scaler: fitted scaler (or None) used for linear model (optional)
  - features: list of feature column names used at training (in order)
  - use_scaler: bool
  - target: (optional) name of the target column used at training
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Tuple, List, Any, Optional

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
MODELS_PATH = os.path.join(MODELS_DIR, "trained_models.pkl")
RESULTS_PATH   = os.path.join(MODELS_DIR, "results.json")
BASELINES_PATH = os.path.join(MODELS_DIR, "baselines.json")

MODEL_NAMES = ["Linear Regression", "Decision Tree", "Random Forest"]

MODEL_COLORS = {
    "Linear Regression": "#3b82f6",
    "Decision Tree":     "#f59e0b",
    "Random Forest":     "#16a34a",
}


def load_model_bundles() -> dict:
    """Load model bundles saved by the training script."""
    if not os.path.exists(MODELS_PATH):
        raise FileNotFoundError(f"Model file not found at {MODELS_PATH}. Please run train_models.py first.")
    bundles = joblib.load(MODELS_PATH)
    if not isinstance(bundles, dict):
        raise ValueError("Expected model bundles to be a dict keyed by model name.")
    return bundles


def load_results() -> dict:
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            return json.load(f)
    return {}


def load_baselines() -> dict:
    if os.path.exists(BASELINES_PATH):
        with open(BASELINES_PATH, "r") as f:
            return json.load(f)
    return {}


def _align_input_df_to_features(input_df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Return DataFrame with columns in 'features' order.
    If a feature is missing in input_df, create column filled with NaN.
    """
    input_df = input_df.copy()
    out = pd.DataFrame(index=input_df.index)
    for f in features:
        if f in input_df.columns:
            out[f] = input_df[f]
        else:
            # missing feature: fill with NaN so imputer can handle it
            out[f] = np.nan
            print(f"[model_utils] Warning: feature '{f}' not found in input; filling with NaN before imputation.")
    return out


def _safe_median_impute(X: np.ndarray) -> np.ndarray:
    """
    Fill NaNs in X (2D) by column medians. If a column is all-NaN, fill with 0.0.
    """
    X = X.astype(float)
    # compute column medians ignoring nan
    col_meds = np.nanmedian(X, axis=0)
    # replace NaNs in medians with 0.0
    col_meds = np.where(np.isfinite(col_meds), col_meds, 0.0)
    inds = np.where(~np.isfinite(X))
    if inds[0].size:
        X[inds] = np.take(col_meds, inds[1])
    return X


def _apply_bundle_predict(bundle: dict, X_raw: np.ndarray) -> np.ndarray:
    """
    Given bundle and raw X array (n_samples x n_features as in bundle['features']),
    impute/scale if needed and return predictions.
    Robustly handles the case where bundle['imputer'] is None or transform fails by
    falling back to simple column-median imputation.
    """
    X = np.asarray(X_raw, dtype=float).copy()  # ensure numpy array of floats
    imputer = bundle.get("imputer", None)

    # If an imputer exists, try to use it. If it fails, fallback to median impute.
    if imputer is not None:
        try:
            X = imputer.transform(X)
        except Exception as e:
            print(f"[model_utils] Imputer.transform failed: {e} — falling back to column median imputation.")
            X = _safe_median_impute(X)
    else:
        # No imputer: if there are NaNs, do median imputation on the input chunk
        if not np.isfinite(X).all():
            X = _safe_median_impute(X)

    # scaling if required
    if bundle.get("use_scaler") and bundle.get("scaler") is not None:
        scaler = bundle.get("scaler")
        try:
            X = scaler.transform(X)
        except Exception as e:
            # scaler might fail if shape mismatch or NaNs remain. attempt to clean and retry once.
            print(f"[model_utils] Scaler.transform failed: {e} — attempting to median-impute then retry.")
            X = _safe_median_impute(X)
            try:
                X = scaler.transform(X)
            except Exception as e2:
                print(f"[model_utils] Scaler retry failed: {e2} — skipping scaling.")
                # proceed without scaling

    # final predict
    try:
        preds = bundle["model"].predict(X)
    except Exception as e:
        # If model.predict fails (e.g., still NaNs), try a safer fallback of predicting NaN array
        print(f"[model_utils] Model.predict failed: {e} — returning NaN predictions for this call.")
        preds = np.array([np.nan] * X.shape[0])
    return np.asarray(preds)


def predict(bundles: dict, model_name: str, input_df: pd.DataFrame) -> np.ndarray:
    """
    Predict using a named bundle on a pandas DataFrame.
    Aligns columns to bundle['features'], imputes and scales as needed.
    """
    if model_name not in bundles:
        raise KeyError(f"Model '{model_name}' not found in bundles.")
    b = bundles[model_name]
    features = b.get("features", [])
    if not features:
        raise ValueError(f"Bundle for {model_name} has no 'features' recorded.")
    X_df = _align_input_df_to_features(input_df, features)
    return _apply_bundle_predict(b, X_df.values)


def get_best_model(results: dict) -> str:
    if not results:
        return "Random Forest"
    best_name = max(MODEL_NAMES, key=lambda m: (results.get(m, {}) or {}).get("R2", -999))
    return best_name


def build_results_table(results: dict) -> pd.DataFrame:
    """
    Build a DataFrame summarising model metrics including MAPE%.
    Handles missing / None metrics gracefully.
    """
    rows = []
    best = get_best_model(results)
    for name in MODEL_NAMES:
        d = results.get(name) or {}
        def _safe(key):
            v = d.get(key, None)
            return round(v, 4) if isinstance(v, (int, float)) and not pd.isna(v) else None
        rows.append({
            "Model":     name,
            "MAE (°C)":  _safe("MAE"),
            "RMSE (°C)": _safe("RMSE"),
            "MAPE (%)":  (round(d.get("MAPE%", None), 3) if isinstance(d.get("MAPE%", None), (int, float)) and not pd.isna(d.get("MAPE%", None)) else None),
            "R²":        _safe("R2"),
            "Best ✓":    "★ Best" if name == best else "",
        })
    return pd.DataFrame(rows)


def _choose_target_column_for_df_and_bundle(df: pd.DataFrame, bundle: Optional[dict]) -> Optional[str]:
    """
    Prefer bundle['target'] if present, else df.attrs detected target, else common names including 'mean_temp', 'temp'.
    """
    if bundle is not None:
        t = bundle.get("target")
        if t and t in df.columns:
            return t
        if t:
            # try normalized match
            t_norm = t.strip().lower().replace(" ", "_")
            for c in df.columns:
                if c.strip().lower().replace(" ", "_") == t_norm:
                    return c
            print(f"[model_utils] Warning: bundle target '{t}' not found in df columns.")
    t2 = df.attrs.get("detected_target_col") if hasattr(df, "attrs") else None
    if t2 and t2 in df.columns:
        return t2
    for cand in ["mean_temp", "temp", "temperature", "max_temp", "min_temp"]:
        if cand in df.columns:
            return cand
    return None


def get_test_predictions(bundles: dict, df: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Run all models on the held-out test set (last 20% chronologically).
    Returns dict: model_name -> (y_actual, y_predicted, dates)
    """
    n = len(df)
    split = int(n * 0.8)
    df_test = df.iloc[split:].copy().reset_index(drop=True)
    out = {}
    for name in MODEL_NAMES:
        if name not in bundles:
            out[name] = (np.array([]), np.array([]), np.array([]))
            continue
        b = bundles[name]
        features = b.get("features", [])
        if not features:
            print(f"[model_utils] Warning: bundle {name} has empty 'features'. Skipping.")
            out[name] = (np.array([]), np.array([]), np.array([]))
            continue
        X_df = _align_input_df_to_features(df_test, features)
        y_pred = _apply_bundle_predict(b, X_df.values)
        target_col = _choose_target_column_for_df_and_bundle(df_test, b)
        if target_col is None:
            print(f"[model_utils] Warning: no target column found for predictions; returning NaNs for actuals.")
            y_actual = np.array([np.nan] * len(df_test))
        else:
            y_actual = df_test[target_col].values
        dates = df_test.get("date", pd.Series(pd.NaT, index=df_test.index)).values
        out[name] = (np.asarray(y_actual), np.asarray(y_pred), np.asarray(dates))
    return out


def get_feature_importance(bundles: dict) -> pd.DataFrame:
    """
    Return feature importance DataFrame for Random Forest if available.
    Fallback: for non-tree models return coefficients if present (LinearRegression.coef_).
    """
    if "Random Forest" in bundles:
        b = bundles["Random Forest"]
        model = b.get("model")
        features = b.get("features", [])
        fi = getattr(model, "feature_importances_", None)
        if fi is not None and len(fi) == len(features):
            df = pd.DataFrame({"Feature": features, "Importance": fi}).sort_values("Importance", ascending=False).reset_index(drop=True)
            return df
    # try linear coef fallback
    for name in ["Linear Regression", "Decision Tree"]:
        if name in bundles:
            b = bundles[name]
            model = b.get("model")
            coef = getattr(model, "coef_", None)
            features = b.get("features", [])
            if coef is not None and len(coef) == len(features):
                df = pd.DataFrame({"Feature": features, "Importance": np.abs(coef)}).sort_values("Importance", ascending=False).reset_index(drop=True)
                return df
    return pd.DataFrame({"Feature": [], "Importance": []})


def time_series_cv(df: pd.DataFrame, bundles: dict, n_splits: int = 5) -> Dict[str, pd.DataFrame]:
    """
    Perform expanding-window time CV across the training portion (first 80%).
    For each fold we evaluate saved models (do not re-fit).
    Returns dict model_name -> DataFrame(rows=folds) with fold metrics.
    """
    if df is None or df.shape[0] == 0:
        raise ValueError("Empty dataframe passed to time_series_cv")

    n = len(df)
    split = int(n * 0.8)
    df_train = df.iloc[:split].reset_index(drop=True)
    N = len(df_train)
    if N < n_splits + 1:
        raise ValueError("Not enough training samples for the requested number of splits")

    # compute fold boundaries (consecutive chunks)
    fold_sizes = [N // n_splits] * n_splits
    for i in range(N % n_splits):
        fold_sizes[i] += 1
    boundaries = []
    start = 0
    for fs in fold_sizes:
        end = start + fs
        boundaries.append((start, end))
        start = end

    # metric helpers
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    def safe_mape(y_t, y_p):
        y_t = np.asarray(y_t, dtype=float)
        y_p = np.asarray(y_p, dtype=float)
        mask = np.isfinite(y_t) & np.isfinite(y_p) & (np.abs(y_t) > 1e-8)
        if not mask.any():
            return float("nan")
        return float(np.mean(np.abs((y_t[mask] - y_p[mask]) / y_t[mask])) * 100.0)

    results = {m: [] for m in MODEL_NAMES}

    # for each fold: evaluate on that chunk using saved bundles (imputer+scaler)
    for i, (s, e) in enumerate(boundaries):
        df_fold = df_train.iloc[s:e].copy().reset_index(drop=True)
        if df_fold.empty:
            continue
        for name in MODEL_NAMES:
            if name not in bundles:
                continue
            b = bundles[name]
            features = b.get("features", [])
            if not features:
                print(f"[model_utils] Warning: no features in bundle {name}; skipping fold {i+1}")
                continue
            # align features
            X_df = _align_input_df_to_features(df_fold, features)
            # use central predict path which handles imputation/fallbacks
            try:
                y_pred = _apply_bundle_predict(b, X_df.values)
            except Exception as e:
                print(f"[model_utils] Prediction failed on fold {i+1} for {name}: {e}")
                y_pred = np.array([np.nan] * len(df_fold))
            # choose true target (bundle target preferred)
            target_col = _choose_target_column_for_df_and_bundle(df_fold, b)
            if target_col is None:
                print(f"[model_utils] Warning: no target column found for fold {i+1}; metrics will be NaN.")
                y_true = np.array([np.nan] * len(df_fold))
            else:
                y_true = df_fold[target_col].values
            # mask invalids
            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            if not mask.any():
                mae = float("nan"); rmse = float("nan"); r2 = float("nan"); mape_v = float("nan")
            else:
                try:
                    mae = float(mean_absolute_error(y_true[mask], y_pred[mask]))
                    rmse = float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))
                    try:
                        r2 = float(r2_score(y_true[mask], y_pred[mask]))
                    except Exception:
                        r2 = float("nan")
                    mape_v = safe_mape(y_true[mask], y_pred[mask])
                except Exception as e:
                    print(f"[model_utils] Metric computation failed on fold {i+1} for {name}: {e}")
                    mae = float("nan"); rmse = float("nan"); r2 = float("nan"); mape_v = float("nan")
            results[name].append({
                "fold": i + 1,
                "start_idx": int(s),
                "end_idx": int(e),
                "MAE": mae, "RMSE": rmse, "MAPE%": mape_v, "R2": r2
            })

    # convert lists to DataFrames
    out = {}
    for name, rows in results.items():
        out[name] = pd.DataFrame(rows)
    return out