"""Train pipeline script. Import and run from CLI or from dashboard."""
import pandas as pd
from sklearn.model_selection import train_test_split
from src.models import get_model, evaluate_model, save_model


def train_pipeline(df, target='mean_temp', features=None, model_name='random_forest', **model_kwargs):
    df = df.copy()
    if features is None:
        features = [c for c in df.columns if c not in ['date','mean_temp']]

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = get_model(model_name, **model_kwargs)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = evaluate_model(y_test, preds)

    # save model
    save_model(model, f'model_{model_name}.joblib')

    return model, metrics, (X_test, y_test, preds)