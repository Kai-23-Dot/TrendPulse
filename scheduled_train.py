#!/usr/bin/env python3
"""
Scheduled Training Script for TrendPulse

This script is designed to be run via cron job after market close (e.g., 5 PM EST).
It trains models for specified tickers and saves them for the web app to load.

Usage:
    python3 scheduled_train.py                    # Train all default tickers
    python3 scheduled_train.py AAPL TSLA NVDA    # Train specific tickers

Cron Example (run at 5:30 PM EST every weekday):
    30 17 * * 1-5 cd /path/to/TrendPulse && python3 scheduled_train.py >> logs/training.log 2>&1
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
import json
import logging
from datetime import date, timedelta, datetime
from pathlib import Path
import numpy as np

# Ensure directories exist BEFORE logging setup
from pathlib import Path
Path('logs').mkdir(exist_ok=True)
Path('models').mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/scheduled_train.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

from src.data_loader import download_data, augment_data, preprocess_pipeline, create_sequences, FEATURE_COLUMNS
from src.model import build_lstm_model, train_model
from src.utils import calculate_metrics, calculate_naive_metrics

# Default configuration
DEFAULT_TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "JPM"]
YEARS_OF_DATA = 5
WINDOW_SIZE = 60
UNITS = 64
DROPOUT = 0.2
EPOCHS = 50
BATCH_SIZE = 32


def train_and_save_model(ticker: str) -> dict:
    """
    Train a model for the given ticker and save it.
    
    Returns:
        dict with training results and metrics
    """
    logger.info(f"Starting training for {ticker}")
    
    # Download data
    end_date = date.today()
    start_date = end_date - timedelta(days=365 * YEARS_OF_DATA)
    
    raw_df = download_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
    if raw_df is None or len(raw_df) < 100:
        logger.error(f"Insufficient data for {ticker}")
        return {"ticker": ticker, "status": "failed", "error": "insufficient_data"}
    
    # Feature engineering
    df_augmented = augment_data(raw_df)
    
    # Preprocessing
    (train_scaled, val_scaled, test_scaled, 
     scaler_features, scaler_target, 
     train_data, val_data, test_data) = preprocess_pipeline(df_augmented)
    
    # Create sequences
    X_train, y_train = create_sequences(train_scaled, 0, WINDOW_SIZE)
    X_val, y_val = create_sequences(val_scaled, 0, WINDOW_SIZE)
    X_test, y_test = create_sequences(test_scaled, 0, WINDOW_SIZE)
    
    if len(X_train) < 50:
        logger.error(f"Not enough training samples for {ticker}")
        return {"ticker": ticker, "status": "failed", "error": "insufficient_samples"}
    
    # Cast to float32
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    y_val = y_val.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)
    
    # Build and train model
    model = build_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        units=UNITS,
        dropout=DROPOUT
    )
    
    history = train_model(
        model, X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    # Evaluate on test set
    pred_scaled = model.predict(X_test, verbose=0)
    pred_returns = scaler_target.inverse_transform(pred_scaled).flatten()
    
    base_prices = test_data['Close'].iloc[WINDOW_SIZE-1:-1].values
    pred_prices = base_prices * np.exp(pred_returns)
    actual_prices = test_data['Close'].iloc[WINDOW_SIZE:].values
    
    lstm_metrics = calculate_metrics(actual_prices, pred_prices)
    naive_metrics = calculate_naive_metrics(actual_prices, base_prices)
    
    # Save model
    model_path = f"models/{ticker}_model.keras"
    model.save(model_path)
    logger.info(f"Saved model to {model_path}")
    
    # Save metadata
    metadata = {
        "ticker": ticker,
        "trained_at": datetime.now().isoformat(),
        "data_range": f"{start_date} to {end_date}",
        "samples": len(X_train),
        "epochs_run": len(history.history['loss']),
        "final_val_loss": float(history.history['val_loss'][-1]),
        "test_rmse": float(lstm_metrics['RMSE']),
        "test_mae": float(lstm_metrics['MAE']),
        "test_r2": float(lstm_metrics['R2']),
        "naive_rmse": float(naive_metrics['RMSE']),
        "beats_naive": lstm_metrics['RMSE'] < naive_metrics['RMSE']
    }
    
    metadata_path = f"models/{ticker}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Completed {ticker}: RMSE=${lstm_metrics['RMSE']:.2f}, Beats Naive: {metadata['beats_naive']}")
    
    return {"ticker": ticker, "status": "success", **metadata}


def main():
    """Main entry point for scheduled training."""
    logger.info("=" * 60)
    logger.info("SCHEDULED TRAINING STARTED")
    logger.info(f"Time: {datetime.now()}")
    logger.info("=" * 60)
    
    # Get tickers from command line or use defaults
    if len(sys.argv) > 1:
        tickers = [t.upper() for t in sys.argv[1:]]
    else:
        tickers = DEFAULT_TICKERS
    
    logger.info(f"Training {len(tickers)} tickers: {tickers}")
    
    results = []
    for ticker in tickers:
        try:
            result = train_and_save_model(ticker)
            results.append(result)
        except Exception as e:
            logger.error(f"Error training {ticker}: {e}")
            results.append({"ticker": ticker, "status": "error", "error": str(e)})
    
    # Summary
    successful = sum(1 for r in results if r.get('status') == 'success')
    failed = len(results) - successful
    
    logger.info("=" * 60)
    logger.info(f"TRAINING COMPLETE: {successful} successful, {failed} failed")
    logger.info("=" * 60)
    
    # Save summary
    summary_path = "models/last_training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "completed_at": datetime.now().isoformat(),
            "tickers_trained": successful,
            "tickers_failed": failed,
            "results": results
        }, f, indent=2)
    
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
