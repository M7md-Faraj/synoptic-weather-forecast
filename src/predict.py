"""Prediction utilities."""
import pandas as pd
from src.models import load_model


def predict_from_model(model_path: str, X: pd.DataFrame):
    model = load_model(model_path)
    preds = model.predict(X)
    return preds