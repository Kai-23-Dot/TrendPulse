"""
Model Module for Stock Price Prediction

This module defines the LSTM architecture and training logic.

WHY EARLY STOPPING?
- Prevents overfitting by stopping when validation loss stops improving
- Saves training time by not running unnecessary epochs
- restore_best_weights ensures we keep the best model, not the last

WHY LEARNING RATE SCHEDULING?
- ReduceLROnPlateau automatically lowers learning rate when stuck
- Helps fine-tune convergence in later epochs
- Prevents overshooting optimal weights
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def build_lstm_model(input_shape, units=64, dropout=0.2, learning_rate=0.001):
    """
    Builds a compiled LSTM model.
    
    Architecture:
    - LSTM layer 1 with return_sequences=True (for stacking)
    - Dropout for regularization
    - LSTM layer 2 
    - Dropout
    - Dense output layer (1 unit for regression)
    
    Args:
        input_shape: (time_steps, features)
        units: Number of LSTM units per layer
        dropout: Dropout rate (0.2-0.3 typical for stocks)
        learning_rate: Adam optimizer learning rate
    """
    model = Sequential([
        LSTM(units=units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        LSTM(units=units, return_sequences=False),
        Dropout(dropout),
        Dense(units=1)
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    return model


def get_training_callbacks(patience_early=7, patience_lr=3):
    """
    Returns standard callbacks for training.
    
    Args:
        patience_early: Epochs to wait before early stopping
        patience_lr: Epochs to wait before reducing learning rate
    """
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=patience_early,
        restore_best_weights=True,
        verbose=0
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=patience_lr,
        min_lr=1e-6,
        verbose=0
    )
    
    return [early_stop, reduce_lr]


def train_model(model, X_train, y_train, validation_data=None, 
                epochs=50, batch_size=32, callbacks=None):
    """
    Trains the LSTM model.
    
    Args:
        model: Compiled Keras model
        X_train, y_train: Training data
        validation_data: Tuple of (X_val, y_val)
        epochs: Maximum epochs (early stopping may stop earlier)
        batch_size: Batch size for training
        callbacks: Additional callbacks (e.g., Streamlit progress)
    
    Returns:
        history: Training history object
    """
    if callbacks is None:
        callbacks = []
    
    # Add standard callbacks
    callbacks.extend(get_training_callbacks())
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        verbose=0,
        callbacks=callbacks
    )
    return history
