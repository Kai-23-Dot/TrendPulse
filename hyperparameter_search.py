"""
Hyperparameter Search Script for Stock Predictor

This script performs a systematic grid search over hyperparameters
and reports the best configuration based on test set performance.

Run with: python3 hyperparameter_search.py

Output: CSV file with results and printed summary of best configs.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU

import numpy as np
import pandas as pd
from datetime import date, timedelta
from itertools import product
import warnings
warnings.filterwarnings('ignore')

from src.data_loader import (
    download_data, augment_data, preprocess_pipeline, 
    create_sequences, FEATURE_COLUMNS
)
from src.model import build_lstm_model, train_model
from src.utils import calculate_metrics, calculate_naive_metrics


# ================================
# HYPERPARAMETER GRID
# ================================
PARAM_GRID = {
    'window_size': [30, 60],
    'units': [32, 64],
    'dropout': [0.2, 0.3],
    'batch_size': [32, 64]
}

# Fixed parameters
TICKER = "SPY"
YEARS_OF_DATA = 5
MAX_EPOCHS = 50


def run_single_config(X_train, y_train, X_val, y_val, X_test, y_test,
                      test_data, scaler_target, window_size,
                      units, dropout, batch_size):
    """Train and evaluate a single hyperparameter configuration."""
    
    n_features = X_train.shape[2]
    
    model = build_lstm_model(
        input_shape=(window_size, n_features),
        units=units,
        dropout=dropout
    )
    
    # Train
    history = train_model(
        model, X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=batch_size
    )
    
    # Get best validation loss
    best_val_loss = min(history.history['val_loss'])
    epochs_trained = len(history.history['loss'])
    
    # Evaluate on test set
    pred_returns_scaled = model.predict(X_test, verbose=0)
    pred_returns = scaler_target.inverse_transform(pred_returns_scaled).flatten()
    
    # Reconstruct prices
    base_prices = test_data['Close'].iloc[window_size-1:-1].values
    pred_prices = base_prices * np.exp(pred_returns)
    actual_prices = test_data['Close'].iloc[window_size:].values
    
    # Calculate metrics
    lstm_metrics = calculate_metrics(actual_prices, pred_prices)
    naive_metrics = calculate_naive_metrics(actual_prices, base_prices)
    
    return {
        'window_size': window_size,
        'units': units,
        'dropout': dropout,
        'batch_size': batch_size,
        'epochs_trained': epochs_trained,
        'best_val_loss': best_val_loss,
        'test_rmse': lstm_metrics['RMSE'],
        'test_mae': lstm_metrics['MAE'],
        'test_r2': lstm_metrics['R2'],
        'naive_rmse': naive_metrics['RMSE'],
        'improvement_pct': (naive_metrics['RMSE'] - lstm_metrics['RMSE']) / naive_metrics['RMSE'] * 100
    }


def main():
    print("=" * 60)
    print("HYPERPARAMETER SEARCH FOR STOCK PREDICTOR")
    print("=" * 60)
    
    # Download and prepare data once
    end_date = date.today()
    start_date = end_date - timedelta(days=365 * YEARS_OF_DATA)
    
    print(f"\nDownloading {TICKER} data from {start_date} to {end_date}...")
    raw_df = download_data(TICKER, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
    if raw_df is None:
        print("Error: Could not download data")
        return
    
    print("Augmenting data with features...")
    df_augmented = augment_data(raw_df)
    print(f"Dataset size: {len(df_augmented)} samples")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    
    # Generate all combinations
    param_names = list(PARAM_GRID.keys())
    param_values = list(PARAM_GRID.values())
    all_combinations = list(product(*param_values))
    
    print(f"\nTotal configurations to test: {len(all_combinations)}")
    print("-" * 60)
    
    results = []
    
    for i, combo in enumerate(all_combinations):
        params = dict(zip(param_names, combo))
        window_size = params['window_size']
        
        print(f"\n[{i+1}/{len(all_combinations)}] Testing: {params}")
        
        # Preprocess with current window size
        (train_scaled, val_scaled, test_scaled, 
         scaler_features, scaler_target, 
         train_data, val_data, test_data) = preprocess_pipeline(df_augmented)
        
        # Create sequences
        X_train, y_train = create_sequences(train_scaled, 0, window_size)
        X_val, y_val = create_sequences(val_scaled, 0, window_size)
        X_test, y_test = create_sequences(test_scaled, 0, window_size)
        
        # Cast to float32
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        y_val = y_val.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_test = y_test.astype(np.float32)
        
        # Run training and evaluation
        result = run_single_config(
            X_train, y_train, X_val, y_val, X_test, y_test,
            test_data, scaler_target, **params
        )
        results.append(result)
        
        print(f"  -> Test RMSE: ${result['test_rmse']:.2f}, R²: {result['test_r2']:.4f}, "
              f"Improvement vs Naive: {result['improvement_pct']:.1f}%")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('test_rmse', ascending=True)
    
    # Save to CSV
    csv_path = "hyperparameter_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n\nResults saved to: {csv_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("TOP 5 CONFIGURATIONS (sorted by Test RMSE)")
    print("=" * 60)
    
    top5 = results_df.head(5)
    for idx, row in top5.iterrows():
        print(f"\nWindow: {int(row['window_size'])}, Units: {int(row['units'])}, "
              f"Dropout: {row['dropout']}, Batch: {int(row['batch_size'])}")
        print(f"  Test RMSE: ${row['test_rmse']:.2f}")
        print(f"  Test MAE:  ${row['test_mae']:.2f}")
        print(f"  Test R²:   {row['test_r2']:.4f}")
        print(f"  Improvement vs Naive: {row['improvement_pct']:.1f}%")
    
    # Best config
    best = results_df.iloc[0]
    print("\n" + "=" * 60)
    print("RECOMMENDED DEFAULTS FOR app.py:")
    print("=" * 60)
    print(f"window_size = {int(best['window_size'])}")
    print(f"units = {int(best['units'])}")
    print(f"dropout = {best['dropout']}")
    print(f"batch_size = {int(best['batch_size'])}")


if __name__ == "__main__":
    main()
