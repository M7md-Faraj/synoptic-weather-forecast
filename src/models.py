"""Model training and helper functions.
Supports: Linear Regression, Random Forest
"""
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


def get_model(name: str, **kwargs):
    if name == 'linear':
        return LinearRegression()
    if name == 'random_forest':
        return RandomForestRegressor(n_estimators=kwargs.get('n_estimators', 100), random_state=42)
    raise ValueError('Unknown model')


def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


def save_model(model, path: str):
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)