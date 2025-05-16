from sklearn.ensemble import IsolationForest
import pandas as pd

def detect_anomalies(df):
    features = ['Close', 'SMA_20', 'EMA_20', 'Bollinger_Upper', 'Bollinger_Lower', 'RSI_14']

    # Drop rows where features are NaN (due to rolling windows)
    df_clean = df.dropna(subset=features).copy()

    # Isolation Forest model
    iso = IsolationForest(contamination=0.05, random_state=42)
    df_clean['anomaly'] = iso.fit_predict(df_clean[features])

    # anomaly = -1 means anomaly, 1 means normal
    anomalies = df_clean[df_clean['anomaly'] == -1][['Date', 'Close', 'anomaly']]

    return anomalies
