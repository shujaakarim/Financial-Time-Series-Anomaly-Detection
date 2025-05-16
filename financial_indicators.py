def add_indicators(df):
    # Simple Moving Average (SMA) 20 days
    df['SMA_20'] = df['Close'].rolling(window=20).mean()

    # Exponential Moving Average (EMA) 20 days
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # Bollinger Bands
    rolling_std = df['Close'].rolling(window=20).std()
    df['Bollinger_Upper'] = df['SMA_20'] + (rolling_std * 2)
    df['Bollinger_Lower'] = df['SMA_20'] - (rolling_std * 2)

    # Relative Strength Index (RSI) 14 days
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    return df
