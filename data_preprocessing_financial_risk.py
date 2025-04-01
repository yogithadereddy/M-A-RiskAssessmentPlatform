import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from scipy.stats.mstats import winsorize

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_FOLDER, exist_ok=True)

# Mapping of company names to stock tickers
COMPANY_TICKERS = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Google": "GOOGL",
    "Amazon": "AMZN",
    "Tesla": "TSLA"
}

def fetch_stock_data(company_name):
    """Fetch historical stock data and save as CSV."""
    if company_name not in COMPANY_TICKERS:
        print("❌ Company not found!")
        return None
    
    ticker = COMPANY_TICKERS[company_name]
    print(f"📊 Fetching stock data for {company_name} ({ticker})...")
    stock = yf.Ticker(ticker)
    stock_data = stock.history(period="10y")[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    if stock_data.empty:
        print("⚠️ No stock data found!")
        return None
    
    stock_data.reset_index(inplace=True)
    stock_data["Date"] = pd.to_datetime(stock_data["Date"]).dt.tz_localize(None)
    
    file_path = os.path.join(DATA_FOLDER, f"{company_name}_stock_data.csv")
    stock_data.to_csv(file_path, index=False)
    
    print(f"✅ Stock data saved: {file_path}")
    return file_path

def load_data(file_path):
    """Load stock data and preprocess missing values."""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    df.interpolate(method='linear', inplace=True)

    return df

def remove_outliers(df):
    """Winsorize data instead of dropping outliers."""
    for col in df.columns:
        df[col] = winsorize(df[col], limits=[0.05, 0.05])  # Caps extreme values at 5% tail
    return df


def feature_engineering(df):
    """Add moving averages and volatility features."""
    df['5_day_MA'] = df['Close'].rolling(window=5).mean()
    df['10_day_MA'] = df['Close'].rolling(window=10).mean()
    df['30_day_MA'] = df['Close'].rolling(window=30).mean()
    df['Volatility'] = df['Close'].pct_change().rolling(window=10).std()
    
    df.interpolate(method='linear', inplace=True)
    return df

def prepare_lstm_data(data, time_steps=30):
    
    X, y = [], []
    
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps, :])  # Select past `time_steps` data points
        y.append(data[i + time_steps, -1])  # Predict the last column (e.g., closing price)

    return np.array(X), np.array(y)


def process_data(company_name, time_steps=30):
    """Fetch, clean, preprocess, and prepare LSTM data for training."""
    file_path = fetch_stock_data(company_name)
    if not file_path:
        return None, None, None, None
    
    df = load_data(file_path)
    df = remove_outliers(df)
    df = feature_engineering(df)

    # Drop NaN rows after feature engineering
    df.dropna(inplace=True)

    # Normalize using only training data
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)

    # Prepare data for LSTM
    X_train, y_train = prepare_lstm_data(train_scaled, time_steps)
    X_test, y_test = prepare_lstm_data(test_scaled, time_steps)

    # Ensure y_train is binary (0 or 1)
    y_train = (y_train > np.median(y_train)).astype(int)
    y_test = (y_test > np.median(y_test)).astype(int)

    print(f"✅ Data processed for {company_name} - X_train: {X_train.shape}, y_train: {y_train.shape}")
    return X_train, y_train, X_test, y_test



if __name__ == "__main__":
    company = input("Enter company name (Apple, Microsoft, Google, Amazon, Tesla): ").strip()
    X_train, y_train = process_data(company)