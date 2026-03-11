import pandas as pd

def to_csv_download_link(df: pd.DataFrame, filename: str = 'predictions.csv') -> bytes:
    return df.to_csv(index=False).encode('utf-8')