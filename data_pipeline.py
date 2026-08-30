"""
Shared data preparation for the Gold Price forecasting project.

This reproduces the (identical) preprocessing block that appears at the top
of GBR.ipynb, RFR.ipynb and SVR.ipynb, so that all three tree/kernel models
train on exactly the same engineered features -- one source of truth instead
of three copy-pasted versions.

MLP.ipynb uses a different, simpler feature set / detrending approach and is
left as its own self-contained pipeline in train_models.py.
"""
import pandas as pd
import numpy as np

RAW_CSV = "Gold Price.csv"
CLEANED_CSV = "Gold_Price_Cleaned.csv"

FEATURE_COLS = [
    'DayOfWeek',
    'Return_Today',
    'Return_Lag_1', 'Return_Lag_2', 'Return_Lag_3', 'Return_Lag_5', 'Return_Lag_10',
    'MA_Ratio_3', 'MA_Ratio_5', 'MA_Ratio_10', 'MA_Ratio_20',
    'Vol_3', 'Vol_5', 'Vol_10', 'Vol_20',
    'Daily_Range_Pct', 'Price_Open_Ratio',
    'Volume_Ratio', 'Volume_MA_5', 'Volume_MA_10',
    'Volume_Lag_1', 'Volume_Lag_3', 'Volume_Lag_5',
]


def build_features(raw_csv_path: str = RAW_CSV) -> pd.DataFrame:
    """Recreate the cleaned + feature-engineered dataframe used by GBR/RFR/SVR."""
    df = pd.read_csv(raw_csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['Quarter'] = df['Date'].dt.quarter
    df['Return'] = df['Price'].pct_change()
    df['Return_Today'] = df['Return']

    for lag in [1, 2, 3, 5, 10]:
        df[f'Return_Lag_{lag}'] = df['Return'].shift(lag)

    for window in [3, 5, 10, 20]:
        df[f'MA_{window}'] = df['Price'].rolling(window).mean()
        df[f'MA_Ratio_{window}'] = df['Price'] / df[f'MA_{window}'] - 1
        df[f'Vol_{window}'] = df['Return'].rolling(window).std()

    df['Daily_Range_Pct'] = (df['High'] - df['Low']) / df['Low'] * 100
    df['Price_Open_Ratio'] = df['Price'] / df['Open'] - 1

    for lag in [1, 3, 5]:
        df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
    df['Volume_MA_5'] = df['Volume'].rolling(5).mean()
    df['Volume_MA_10'] = df['Volume'].rolling(10).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_10']

    df['Target'] = df['Price'].shift(-1)          # next-day price
    df['Target_Return'] = df['Return'].shift(-1)  # next-day return

    lo = df["Daily_Range_Pct"].quantile(0.25)
    hi = df["Daily_Range_Pct"].quantile(0.75)
    inter = hi - lo
    lbound = lo - (inter * 1.5)
    hbound = hi + (inter * 1.5)
    df['Is_Extreme_Day'] = ((df['Daily_Range_Pct'] > hbound) |
                             (df['Daily_Range_Pct'] < lbound)).astype(int)

    df_clean = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    return df_clean


if __name__ == "__main__":
    cleaned = build_features()
    cleaned.to_csv(CLEANED_CSV, index=False)
    print(f"Saved {CLEANED_CSV} with {len(cleaned)} rows.")
