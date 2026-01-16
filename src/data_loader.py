"""
Data Loader Module for Stock Price Prediction

This module handles data downloading, feature engineering, and preprocessing.

WHY PREDICT RETURNS INSTEAD OF PRICES?
- Raw stock prices are non-stationary (they trend up/down over time)
- Returns (percent changes) are closer to stationary, making patterns easier to learn
- Technical indicators normalize price information into comparable signals

WHY ADD TECHNICAL INDICATORS?
- Moving averages capture trend direction and momentum
- Volatility measures risk and regime changes
- Volume indicators show market participation strength
- RSI-like features detect overbought/oversold conditions
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
import requests
import io
from datetime import datetime

def _download_manual_fallback(ticker, start_date, end_date):
    """
    Fallback method: Download directly from Yahoo Query API using requests.
    This bypasses yfinance library internals which might be blocked or erroring.
    """
    try:
        # Convert dates to unix timestamps
        start_ts = int(pd.Timestamp(start_date).timestamp())
        end_ts = int(pd.Timestamp(end_date).timestamp())
        
        url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
        params = {
            'period1': start_ts,
            'period2': end_ts,
            'interval': '1d',
            'events': 'history',
            'includeAdjustedClose': 'true'
        }
        
        # Rotated user agents to look like real browsers
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0'
        ]
        
        headers = {
            'User-Agent': user_agents[0],
            'Accept': '*/*'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text))
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            return df
            
        return pd.DataFrame()
        
    except Exception as e:
        print(f"Manual fallback failed: {e}")
        return pd.DataFrame()


def _download_with_retry(ticker, start=None, end=None, period=None, max_retries=3):
    """
    Download data with retry logic and fallbacks for Streamlit Cloud compatibility.
    """
    # Attempt 1-3: Standard yfinance with backoff
    for attempt in range(max_retries):
        try:
            if period:
                df = yf.download(ticker, period=period, progress=False)
            else:
                df = yf.download(ticker, start=start, end=end, progress=False)
            
            if not df.empty:
                return df
            
            time.sleep(1 + attempt)
                
        except Exception as e:
            time.sleep(1 + attempt)
    
    # Attempt 4: Manual Fallback (The Nuclear Option)
    if start and end and not period:
        # If period is requested, we can't easily map to start/end for manual, but usually start/end is used by app
        print(f"yfinance failed, attempting manual fallback for {ticker}...")
        return _download_manual_fallback(ticker, start, end)
            
    return pd.DataFrame()


def download_data(ticker, start_date, end_date):
    """
    Downloads historical stock data from Yahoo Finance.
    
    Automatically adjusts start_date if it's before the stock's first trading day.
    Uses retry logic to handle Streamlit Cloud rate limiting.
    
    Returns:
        tuple: (DataFrame, adjusted_start_date or None, message or None)
               - DataFrame: The stock data, or None if download failed
               - adjusted_start_date: The actual start date used if different from requested
               - message: Info message about date adjustment, or error message
    """
    try:
        # Try downloading with the requested date range (with retry)
        df = _download_with_retry(ticker, start=start_date, end=end_date)
        
        if df.empty:
            # If empty, try to find the stock's first available date
            # Download max history to find the earliest date (with retry)
            df_max = _download_with_retry(ticker, period="max")
            
            if df_max.empty:
                return None, None, f"No data available for ticker '{ticker}'. Please verify the symbol or try again."
            
            # Get the first available date
            first_available = df_max.index[0]
            
            # Check if start_date is before the first available date
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            if start_dt < first_available:
                # Adjust start date to first available
                adjusted_start = first_available.strftime('%Y-%m-%d')
                
                if first_available > end_dt:
                    return None, None, f"'{ticker}' started trading on {adjusted_start}, which is after your end date."
                
                # Re-download with adjusted dates (with retry)
                df = _download_with_retry(ticker, start=adjusted_start, end=end_date)
                
                if df.empty:
                    return None, None, f"Could not download data for '{ticker}'. Yahoo Finance may be rate-limiting. Try again in a moment."
                
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                df.reset_index(inplace=True)
                
                return df, adjusted_start, f"📅 '{ticker}' started trading on {adjusted_start}. Date range adjusted automatically."
            else:
                return None, None, f"No data available for '{ticker}' in the selected date range."
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df.reset_index(inplace=True)
        return df, None, None
        
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        return None, None, f"Error downloading data: {str(e)}. Yahoo Finance may be temporarily unavailable."


def augment_data(df):
    """
    Adds technical indicators to the DataFrame.
    
    Features added:
    - Log_Return: Daily log return (target variable)
    - MA_10, MA_30: Price moving averages (trend)
    - Volatility: 10-day rolling std of returns (risk)
    - Volume_Log: Log-transformed volume
    - Volume_MA_10, Volume_MA_30: Volume moving averages
    - ROC_10: 10-day rate of change (momentum)
    - RSI_14: Approximate Relative Strength Index
    - Price_to_MA: Price relative to 30-day MA (mean reversion signal)
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)
    
    # === RETURNS (Target) ===
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # === PRICE-BASED FEATURES ===
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_30'] = df['Close'].rolling(window=30).mean()
    
    # Price relative to moving average (mean reversion signal)
    df['Price_to_MA'] = df['Close'] / df['MA_30']
    
    # Rate of Change (momentum)
    df['ROC_10'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)
    
    # === VOLATILITY ===
    df['Volatility'] = df['Log_Return'].rolling(window=10).std()
    
    # === VOLUME FEATURES ===
    df['Volume_Log'] = np.log(df['Volume'].replace(0, 1))
    df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()
    df['Volume_MA_30'] = df['Volume'].rolling(window=30).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_30']
    
    # === RSI-LIKE INDICATOR (Approximate) ===
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)  # Avoid division by zero
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Drop NaNs created by rolling windows
    df.dropna(inplace=True)
    
    return df


# Define the feature columns to use (order matters for indexing)
FEATURE_COLUMNS = [
    'Log_Return',      # Index 0 - This is also our target
    'MA_10', 
    'MA_30',
    'Volatility',
    'Volume_Log',
    'Volume_Ratio',
    'ROC_10',
    'RSI_14',
    'Price_to_MA'
]


def preprocess_pipeline(df, train_ratio=0.7, val_ratio=0.15):
    """
    Splits data and fits scalers ONLY on training data to prevent leakage.
    
    WHY FIT SCALER ON TRAIN ONLY?
    - If we fit on all data, we leak future information into training
    - This causes overly optimistic metrics that don't reflect real performance
    
    Returns:
        scaled_train, scaled_val, scaled_test: Scaled feature arrays
        scaler_features: Fitted scaler for features
        scaler_target: Fitted scaler for target (Log Return)
        train_data, val_data, test_data: Original DataFrames
    """
    target = 'Log_Return'
    
    n = len(df)
    train_split = int(n * train_ratio)
    val_split = int(n * (train_ratio + val_ratio))
    
    train_data = df.iloc[:train_split]
    val_data = df.iloc[train_split:val_split]
    test_data = df.iloc[val_split:]
    
    # Fit scalers ONLY on training data
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    
    scaler_features.fit(train_data[FEATURE_COLUMNS])
    scaler_target.fit(train_data[[target]])
    
    # Transform all sets using training-fitted scalers
    train_scaled = scaler_features.transform(train_data[FEATURE_COLUMNS])
    val_scaled = scaler_features.transform(val_data[FEATURE_COLUMNS])
    test_scaled = scaler_features.transform(test_data[FEATURE_COLUMNS])
    
    return (train_scaled, val_scaled, test_scaled, 
            scaler_features, scaler_target, 
            train_data, val_data, test_data)


def create_sequences(data, target_col_idx, window_size):
    """
    Creates sequences for LSTM training.
    
    Args:
        data: Scaled feature array (samples, features)
        target_col_idx: Index of target column (0 for Log_Return)
        window_size: Number of past days to use as input
        
    Returns:
        X: (samples, window_size, features)
        y: (samples,) - Next step's target value
    """
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i, target_col_idx])
    return np.array(X), np.array(y)
