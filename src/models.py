from pathlib import Path
import joblib
import json
from datetime import datetime
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

BASE = Path.cwd()
MODELS_DIR = BASE / "models"
META_FILE = MODELS_DIR / "models_meta.json"
MODELS_DIR.mkdir(exist_ok=True, parents=True)

def get_model(name: str, **kwargs):
    """Return ready-to-train model. 'linear' returns a pipeline with scaling."""
    name = name.lower()
    if name in ('linear', 'linear_regression'):
        # pipeline ensures prediction inputs are scaled the same as training
        return Pipeline([
            ("scaler", StandardScaler()),
            ("linear", LinearRegression())
        ])
    if name in ('random_forest', 'rf'):
        return RandomForestRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            random_state=kwargs.get('random_state', 42),
            n_jobs=kwargs.get('n_jobs', -1),
            warm_start=kwargs.get('warm_start', False)
        )
    raise ValueError(f'Unknown model: {name}')

def _read_meta():
    if not META_FILE.exists():
        return {}
    with open(META_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_meta(meta: dict):
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def save_model(model, base_name: str, features: list, metrics: dict = None, meta_obj: dict = None):
    """
    Save model and a joblib meta container.
    - model: trained estimator (can be pipeline)
    - base_name: short name e.g. 'random_forest' or 'linear'
    - features: list of feature names used for training
    - metrics: optional metrics dict
    - meta_obj: optional dict (picklable) to save alongside model (e.g., target_median)
    Returns: (path_str, metadata_entry)
    """
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = f"model_{base_name}_{timestamp}.joblib"
    path = MODELS_DIR / filename
    joblib.dump(model, path)

    # Save binary meta (picklable) next to model so we can store scalers, arrays, etc.
    meta_bin_path = path.with_suffix(path.suffix + ".meta.joblib")
    joblib.dump(meta_obj or {}, meta_bin_path)

    entry = {
        "filename": filename,
        "base_name": base_name,
        "path": str(path),
        "timestamp": timestamp,
        "size_kb": round(path.stat().st_size / 1024, 2),
        "metrics": metrics or {},
        "features": list(features or []),
        "meta_bin": str(meta_bin_path.name)
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

def load_model_meta_bin(model_path: str):
    p = Path(model_path)
    meta_bin = p.with_suffix(p.suffix + ".meta.joblib")
    if meta_bin.exists():
        return joblib.load(meta_bin)
    return {}

def list_models():
    meta = _read_meta()
    entries = list(meta.values())
    entries_sorted = sorted(entries, key=lambda e: e.get("timestamp", ""), reverse=True)
    return entries_sorted

def get_latest_by_base(base_name: str):
    entries = list_models()
    for e in entries:
        if e.get("base_name") == base_name:
            return e
    return None

def extract_feature_importance(model, feature_names):
    try:
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            return pd.Series(imp, index=feature_names).sort_values(ascending=False)
        if hasattr(model, "named_steps") and "linear" in model.named_steps:
            coef = np.ravel(model.named_steps["linear"].coef_)
            return pd.Series(coef, index=feature_names).sort_values(key=abs, ascending=False)
        if hasattr(model, "coef_"):
            coef = np.ravel(model.coef_)
            return pd.Series(coef, index=feature_names).sort_values(key=abs, ascending=False)
    except Exception:
        return None
    return None