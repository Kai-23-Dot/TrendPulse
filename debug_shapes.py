import sys
import os
import numpy as np
import pandas as pd
from datetime import date, timedelta
from src.data_loader import download_data, augment_data, preprocess_pipeline, create_sequences
from src.model import build_lstm_model

# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

def debug_pipeline():
    print("--- Starting Debug Pipeline ---")
    ticker = "SPY"
    start_date = (date.today() - timedelta(days=365*2)).strftime('%Y-%m-%d')
    end_date = date.today().strftime('%Y-%m-%d')
    
    print(f"Downloading data for {ticker}...")
    raw_df = download_data(ticker, start_date, end_date)
    if raw_df is None:
        print("Error: DataFrame is None")
        return

    print("Augmenting data...")
    df_augmented = augment_data(raw_df)
    print(f"Augmented shape: {df_augmented.shape}")
    print(f"Columns: {df_augmented.columns}")

    print("Preprocessing...")
    (train_scaled, val_scaled, test_scaled, 
     scaler_features, scaler_target, 
     train_data, val_data, test_data) = preprocess_pipeline(df_augmented)
    
    print(f"Train Scaled Shape: {train_scaled.shape}")
    
    target_col_idx = 0
    window_size = 60
    
    print("Creating Sequences...")
    X_train, y_train = create_sequences(train_scaled, target_col_idx, window_size)
    
    print(f"X_train type: {type(X_train)}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_train dtype: {X_train.dtype}")
    
    if len(X_train.shape) != 3:
        print("CRITICAL: X_train is not 3D!")
        return

    # Try building model
    print("Building Model...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"Input Shape passed to model: {input_shape}")
    
    model = build_lstm_model(input_shape=input_shape, units=50, dropout=0.2)
    model.summary()
    
    # Try a dummy fit with one batch
    print("Attempting dummy fit...")
    try:
        X_dummy = X_train[:32].astype(np.float32)
        y_dummy = y_train[:32].astype(np.float32)
        model.fit(X_dummy, y_dummy, epochs=1, batch_size=32, verbose=1)
        print("Dummy fit SUCCESS")
    except Exception as e:
        print(f"Dummy fit FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_pipeline()
