import pandas as pd
import numpy as np

def compute_indicators(df):
    df = df.copy()

    # Simple Moving Averages (SMA)
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()

    # Relative Strength Index (RSI)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Moving Average Convergence Divergence (MACD)
    short_ema = df["Close"].ewm(span=12, adjust=False).mean()
    long_ema = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = short_ema - long_ema
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df["BB_Middle"] = df["Close"].rolling(window=20).mean()
    df["BB_Upper"] = df["BB_Middle"] + (df["Close"].rolling(window=20).std() * 2)
    df["BB_Lower"] = df["BB_Middle"] - (df["Close"].rolling(window=20).std() * 2)

    # Average Directional Index (ADX)
    high_diff = df["High"].diff()
    low_diff = df["Low"].diff()

    df["+DM"] = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    df["-DM"] = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

    tr1 = df["High"] - df["Low"]
    tr2 = abs(df["High"] - df["Close"].shift(1))
    tr3 = abs(df["Low"] - df["Close"].shift(1))
    df["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    df["+DI"] = 100 * (df["+DM"].rolling(window=14).mean() / df["TR"].rolling(window=14).mean())
    df["-DI"] = 100 * (df["-DM"].rolling(window=14).mean() / df["TR"].rolling(window=14).mean())
    df["DX"] = (abs(df["+DI"] - df["-DI"]) / (df["+DI"] + df["-DI"])) * 100
    df["ADX"] = df["DX"].rolling(window=14).mean()

    # Volume Weighted Average Price (VWAP)
    df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()

    # Stochastic Oscillator
    df["14-high"] = df["High"].rolling(14).max()
    df["14-low"] = df["Low"].rolling(14).min()
    df["%K"] = 100 * ((df["Close"] - df["14-low"]) / (df["14-high"] - df["14-low"]))
    df["%D"] = df["%K"].rolling(3).mean()

    # Overall Sentiment
    df["Sentiment"] = df.apply(determine_sentiment, axis=1)

    return df

def determine_sentiment(row):
    """Determines the overall sentiment based on multiple indicators."""
    buy_signals = 0
    sell_signals = 0

    # SMA Strategy
    if row["SMA_50"] > row["SMA_200"]:
        buy_signals += 1
    else:
        sell_signals += 1

    # RSI Strategy
    if row["RSI"] < 30:
        buy_signals += 1
    elif row["RSI"] > 70:
        sell_signals += 1

    # MACD Strategy
    if row["MACD"] > row["Signal_Line"]:
        buy_signals += 1
    else:
        sell_signals += 1

    # Bollinger Bands Strategy
    if row["Close"] < row["BB_Lower"]:
        buy_signals += 1
    elif row["Close"] > row["BB_Upper"]:
        sell_signals += 1

    # ADX Strategy
    if row["ADX"] > 25 and row["+DI"] > row["-DI"]:
        buy_signals += 1
    elif row["ADX"] > 25 and row["-DI"] > row["+DI"]:
        sell_signals += 1

    # Stochastic Oscillator Strategy
    if row["%K"] > row["%D"]:
        buy_signals += 1
    else:
        sell_signals += 1

    # Final Decision
    if buy_signals > sell_signals:
        return "Buy"
    elif sell_signals > buy_signals:
        return "Sell"
    else:
        return "Neutral"
