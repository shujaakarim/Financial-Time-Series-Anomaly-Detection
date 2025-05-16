import os
import matplotlib.pyplot as plt
from dataset import load_stock_data
from financial_indicators import add_indicators
from processed_stock import save_processed_data
from isolation_forest import detect_anomalies
from time_Series_forecasting_with_prophet import run_prophet_forecast
import pandas as pd


def process_and_detect(stock_symbol, input_path):
    output_path = f"processed_stock_data/{stock_symbol}_with_indicators.csv"

    df = load_stock_data(input_path)
    df = add_indicators(df)
    save_processed_data(df, output_path)

    anomalies = detect_anomalies(df, stock_symbol)

    forecast = run_prophet_forecast(df, stock_symbol, periods=30)
    forecast['stock'] = stock_symbol

    print(f"✅ Completed processing for {stock_symbol}\n")
    return forecast


if __name__ == "__main__":
    os.makedirs("processed_stock_data", exist_ok=True)
    os.makedirs("forecast_plots", exist_ok=True)

    raw_folder = "raw_stock"
    all_forecasts = []

    # Find all CSV files in raw_stock folder
    csv_files = [f for f in os.listdir(raw_folder) if f.lower().endswith('.csv')]

    if not csv_files:
        print(f"No CSV files found in '{raw_folder}' folder.")
    else:
        for file in csv_files:
            # Extract stock symbol from filename (assumes format SYMBOL_stock_data.csv)
            stock_symbol = file.split('_')[0]
            input_path = os.path.join(raw_folder, file)

            try:
                forecast_df = process_and_detect(stock_symbol, input_path)
                all_forecasts.append(forecast_df)
            except Exception as e:
                print(f"❌ Failed to process {stock_symbol}: {e}")

    # Plot combined forecasts if any
    if all_forecasts:
        plt.figure(figsize=(14, 8))
        for fcast in all_forecasts:
            plt.plot(fcast['ds'], fcast['yhat'], label=fcast['stock'])

        plt.title("Stock Price Forecast (Combined)")
        plt.xlabel("Date")
        plt.ylabel("Predicted Close Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("forecast_plots/combined_forecast.png")
        plt.show()
