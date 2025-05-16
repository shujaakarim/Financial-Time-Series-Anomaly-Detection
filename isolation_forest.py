import pandas as pd
from sklearn.ensemble import IsolationForest
import pandas as pd


def detect_anomalies(df, stock_symbol):
    df = df.copy()

    features = ['Close', 'High', 'Low', 'Open', 'Volume',
                'sma_20', 'ema_20', 'rsi_14', 'bollinger_upper', 'bollinger_lower']

    # Fill NA values forward to avoid errors
    df[features] = df[features].fillna(method='ffill').fillna(method='bfill')

    iso_forest = IsolationForest(contamination=0.02, random_state=42)
    df['anomaly'] = iso_forest.fit_predict(df[features])

    # Anomalies have value -1
    anomalies = df[df['anomaly'] == -1]

    print(f"Anomalies detected for {stock_symbol}:")
    print(anomalies[['Date', 'Close', 'anomaly']])

    return anomalies
