def save_processed_data(df, filepath):
    df.to_csv(filepath, index=False)
