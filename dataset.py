import os
print(f"dataset.py loaded from: {os.path.abspath(__file__)}")
import pandas as pd
print("pandas imported successfully")


def load_stock_data(filepath):
    """
    Load stock data from a CSV file, ensure 'Date' is a datetime column and data is sorted.
    """
    df = pd.read_csv(filepath)

    # Rename date column if needed
    if 'Date' not in df.columns:
        for col in df.columns:
            if 'date' in col.lower():
                df.rename(columns={col: 'Date'}, inplace=True)
                break
        else:
            raise ValueError("No date column found in dataset.")

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

# ✅ Only runs when you execute this file directly (for testing)
if __name__ == "__main__":
    # 👇 Make sure this path is correct based on your project structure
    raw_folder = "raw_stock"
    filenames = ["AAPL_stock_data.csv", "GOOG_stock_data.csv", "MSFT_stock_data.csv"]

    for fname in filenames:
        path = os.path.join(raw_folder, fname)
        print(f"\n--- Loading: {fname} ---")
        try:
            df = load_stock_data(path)
            print(df.head())  # Print first few rows
        except Exception as e:
            print(f"❌ Failed to load {fname}: {e}")
