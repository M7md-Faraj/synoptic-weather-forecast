from pathlib import Path
import joblib
import json
from datetime import datetime
import pandas as pd
import numpy as np

BASE = Path.cwd()
MODELS_DIR = BASE / "models"
META_FILE = MODELS_DIR / "models_meta.json"
MODELS_DIR.mkdir(exist_ok=True, parents=True)

# --- model factory (keep if you want) ---
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor

def get_model(name: str, **kwargs):
    name = name.lower()
    if name in ('linear', 'linear_regression'):
        return LinearRegression()
    if name in ('random_forest', 'rf'):
        return RandomForestRegressor(n_estimators=kwargs.get('n_estimators', 100),
                                     random_state=kwargs.get('random_state', 42),
                                     n_jobs=kwargs.get('n_jobs', -1),
                                     warm_start=kwargs.get('warm_start', False))
    if name in ('sgd', 'sgd_regressor'):
        return SGDRegressor(max_iter=kwargs.get('max_iter', 1000), tol=kwargs.get('tol', 1e-3),
                            random_state=kwargs.get('random_state', 42))
    raise ValueError(f'Unknown model: {name}')

# --- utilities ---
def _read_meta():
    if not META_FILE.exists():
        return {}
    with open(META_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_meta(meta: dict):
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def save_model(model, base_name: str, metrics: dict = None):
    """
    Save model to models/ with timestamp version.
    base_name: short name e.g. 'random_forest' or 'sgd'
    metrics: optional dict of evaluation metrics (mae, rmse, r2)
    Returns: saved_path (str) and metadata entry (dict)
    """
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = f"model_{base_name}_{timestamp}.joblib"
    path = MODELS_DIR / filename
    joblib.dump(model, path)

    entry = {
        "filename": filename,
        "base_name": base_name,
        "path": str(path),
        "timestamp": timestamp,
        "size_kb": round(path.stat().st_size / 1024, 2),
        "metrics": metrics or {}
    }

    meta = _read_meta()
    meta[filename] = entry
    _write_meta(meta)
    return str(path), entry

def load_model(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(p)

def list_models():
    """
    Return a list of metadata entries (most recent first).
    """
    meta = _read_meta()
    entries = list(meta.values())
    entries_sorted = sorted(entries, key=lambda e: e.get("timestamp", ""), reverse=True)
    return entries_sorted

def get_latest_by_base(base_name: str):
    """
    Return the most recent metadata entry for base_name (or None).
    """
    entries = list_models()
    for e in entries:
        if e.get("base_name") == base_name:
            return e
    return None

def model_info(path: str):
    p = Path(path)
    if not p.exists():
        return None
    return {
        "path": str(p),
        "size_kb": round(p.stat().st_size / 1024, 2),
        "modified": p.stat().st_mtime
    }

def extract_feature_importance(model, feature_names):
    """
    Return pandas Series indexed by feature_names for feature importance or coefficients.
    """
    try:
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            return pd.Series(imp, index=feature_names).sort_values(ascending=False)
        if hasattr(model, "coef_"):
            coef = np.ravel(model.coef_)
            return pd.Series(coef, index=feature_names).sort_values(key=abs, ascending=False)
    except Exception:
        return None
    return None