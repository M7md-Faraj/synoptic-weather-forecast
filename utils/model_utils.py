"""
utils/model_utils.py
====================
Model loading, prediction, and evaluation utilities.

All trained model bundles live in models/trained_models.pkl.
Each bundle is a dict with: model, scaler, imputer, features, use_scaler.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
MODELS_PATH = os.path.join(MODELS_DIR, "trained_models.pkl")
RESULTS_PATH   = os.path.join(MODELS_DIR, "results.json")
BASELINES_PATH = os.path.join(MODELS_DIR, "baselines.json")

# The three models used in this project (display order)
MODEL_NAMES = ["Linear Regression", "Decision Tree", "Random Forest"]

# Friendly colour palette for charts (one per model)
MODEL_COLORS = {
    "Linear Regression": "#3b82f6",   # blue
    "Decision Tree":     "#f59e0b",   # amber
    "Random Forest":     "#16a34a",   # green
}


def load_model_bundles() -> dict:
    """
    Load all three trained model bundles from disk.

    Returns
    -------
    dict: { model_name -> bundle_dict }
    Each bundle contains: model, scaler, imputer, features, use_scaler.
    """
    if not os.path.exists(MODELS_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODELS_PATH}. "
            "Please run train_models.py first."
        )
    return joblib.load(MODELS_PATH)


def load_results() -> dict:
    """Load evaluation metrics from results.json."""
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            return json.load(f)
    return {}


def load_baselines() -> dict:
    """Load naive baseline metrics from baselines.json."""
    if os.path.exists(BASELINES_PATH):
        with open(BASELINES_PATH, "r") as f:
            return json.load(f)
    return {}


def predict(bundles: dict, model_name: str,
            input_df: pd.DataFrame) -> np.ndarray:
    """
    Generate predictions for one model given an input DataFrame.

    Parameters
    ----------
    bundles    : output of load_model_bundles()
    model_name : one of MODEL_NAMES
    input_df   : DataFrame with the correct feature columns

    Returns
    -------
    np.ndarray of predictions (length = len(input_df))
    """
    b = bundles[model_name]
    X = input_df[b["features"]].values

    # Impute any remaining NaN values (uses the training-set imputer)
    X = b["imputer"].transform(X)

    # Scale only for Linear Regression (tree models do not need scaling)
    if b["use_scaler"] and b["scaler"] is not None:
        X = b["scaler"].transform(X)

    return b["model"].predict(X)


def get_best_model(results: dict) -> str:
    """
    Return the name of the best model (highest R²) from results dict.
    Falls back to 'Random Forest' if results are unavailable.
    """
    if not results:
        return "Random Forest"

    best_name = max(
        MODEL_NAMES,
        key=lambda m: results.get(m, {}).get("R2", -999)
    )
    return best_name


def build_results_table(results: dict) -> pd.DataFrame:
    """
    Build a clean DataFrame summarising the three model metrics.

    Parameters
    ----------
    results : output of load_results()

    Returns
    -------
    DataFrame with columns: Model, MAE (°C), RMSE (°C), R²
    """
    rows = []
    best = get_best_model(results)
    for name in MODEL_NAMES:
        d = results.get(name, {})
        rows.append({
            "Model":     name,
            "MAE (°C)":  round(d.get("MAE",  0), 4),
            "RMSE (°C)": round(d.get("RMSE", 0), 4),
            "R²":        round(d.get("R2",   0), 4),
            "Best ✓":    "★ Best" if name == best else "",
        })
    return pd.DataFrame(rows)


def get_test_predictions(bundles: dict, df: pd.DataFrame) -> dict:
    """
    Run all three models on the held-out test set (last 20% chronologically).

    Parameters
    ----------
    bundles : loaded model bundles
    df      : fully preprocessed DataFrame (from data_loader.load_data)

    Returns
    -------
    dict: { model_name -> (y_actual, y_predicted, dates) }
    """
    # Replicate the same split used during training
    n     = len(df)
    split = int(n * 0.8)
    df_test = df.iloc[split:].copy()

    out = {}
    for name in MODEL_NAMES:
        b  = bundles[name]
        X  = df_test[b["features"]].values
        X  = b["imputer"].transform(X)
        if b["use_scaler"] and b["scaler"] is not None:
            X = b["scaler"].transform(X)
        y_pred  = b["model"].predict(X)
        y_actual = df_test["mean_temp"].values
        dates    = df_test["date"].values
        out[name] = (y_actual, y_pred, dates)

    return out


def get_feature_importance(bundles: dict) -> pd.DataFrame:
    """
    Extract Random Forest feature importances.

    Returns
    -------
    DataFrame with columns: Feature, Importance
    Sorted descending by Importance.
    """
    b  = bundles["Random Forest"]
    fi = b["model"].feature_importances_
    df = pd.DataFrame({
        "Feature":    b["features"],
        "Importance": fi,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    return df
