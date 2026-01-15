from src.data_loader import download_data, preprocess_data, create_sequences, train_test_split_time
from src.model import build_lstm_model, train_model
from src.utils import calculate_metrics

def test_pipeline():
    print("Testing pipeline on AAPL...")
    
    # 1. Data Loading
    df = download_data('AAPL', '2020-01-01', '2021-01-01')
    if df is None or df.empty:
        print("FAIL: Data download failed.")
        return
    print(f"Data downloaded: {len(df)} rows.")
    
    # 2. Preprocessing
    scaled_data, scaler, processed_df = preprocess_data(df)
    if scaled_data.shape[0] != len(df):
        print("FAIL: Preprocessing size mismatch.")
        return
    
    # 3. Sequences
    window_size = 10
    X, y = create_sequences(scaled_data, window_size)
    if len(X) != len(df) - window_size:
        print(f"FAIL: Sequence length mismatch. Expected {len(df) - window_size}, got {len(X)}")
        return
        
    X_train, X_test, y_train, y_test = train_test_split_time(X, y)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # 4. Model Training
    print("Building and training model (fast run)...")
    model = build_lstm_model(input_shape=(X_train.shape[1], 1), units=10) # Small units for speed
    train_model(model, X_train, y_train, epochs=1, batch_size=32)
    
    # 5. Prediction & Metrics
    preds = model.predict(X_test)
    metrics = calculate_metrics(y_test, preds)
    print("Metrics:", metrics)
    
    print("SUCCESS: Pipeline verification passed.")

if __name__ == "__main__":
    test_pipeline()
