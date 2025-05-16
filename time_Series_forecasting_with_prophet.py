import pandas as pd
from prophet import Prophet
import pandas as pd


def run_prophet_forecast(df, stock_symbol=None, periods=30):
    try:
        df = df.copy()
        df.columns = df.columns.str.strip().str.lower()

        df_prophet = df[['date', 'close']].rename(columns={'date': 'ds', 'close': 'y'})
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

        model = Prophet()
        model.fit(df_prophet)

        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        forecast['stock'] = stock_symbol

        print(f"Forecast completed for {stock_symbol}")

        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'stock']]

    except Exception as e:
        print(f"❌ Failed to forecast {stock_symbol}: {e}")
        raise
