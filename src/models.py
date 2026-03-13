from pathlib import Path
import json
import joblib
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

BASE = Path.cwd()
MODELS_DIR = BASE / "models"
MODELS_DIR.mkdir(exist_ok=True, parents=True)
META_FILE = MODELS_DIR / "models_meta.json"

def get_model(name: str, **kwargs):
    name = name.lower()
    if name in ("linear", "linear_regression"):
        return LinearRegression()
    if name in ("decision_tree", "decisiontree", "dt"):
        return DecisionTreeRegressor(random_state=kwargs.get("random_state", 42), max_depth=kwargs.get("max_depth", None))
    if name in ("random_forest", "rf"):
        return RandomForestRegressor(n_estimators=kwargs.get("n_estimators", 100), random_state=kwargs.get("random_state", 42), n_jobs=kwargs.get("n_jobs", -1))
    raise ValueError(f"Unknown model: {name}")

def _read_meta():
    if not META_FILE.exists():
        return {}
    try:
        with open(META_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}

def _write_meta(meta: dict):
    with open(META_FILE, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

def save_model(model, base_name: str, features: list, metrics: dict=None, meta_obj: dict=None):
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = f"model_{base_name}_{timestamp}.joblib"
    path = MODELS_DIR / filename
    joblib.dump(model, path)
    meta_entry = {
        "filename": filename,
        "path": str(path),
        "base_name": base_name,
        "timestamp": timestamp,
        "features": list(features or []),
        "metrics": metrics or {},
    }
    all_meta = _read_meta()
    all_meta[filename] = meta_entry
    _write_meta(all_meta)
    # store separate meta object if provided
    if meta_obj is not None:
        meta_bin_path = path.with_suffix(path.suffix + ".meta.json")
        with open(meta_bin_path, "w", encoding="utf-8") as fh:
            json.dump(meta_obj, fh, default=str)
    return str(path), meta_entry

def load_model(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    return joblib.load(p)

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