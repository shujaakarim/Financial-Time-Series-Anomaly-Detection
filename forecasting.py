from prophet import Prophet
import pandas as pd

def run_forecast(df, periods=30):
    # Prepare dataframe for Prophet
    df_prophet = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'}).dropna()

    model = Prophet()
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    return forecast
