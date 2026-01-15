from src.data_loader import download_data, preprocess_data, create_sequences, train_val_test_split
import numpy as np

def test_split_logic():
    print("Testing 3-way split logic...")
    
    # Create dummy data
    data_len = 1000
    X = np.random.rand(data_len, 1)
    y = np.random.rand(data_len)
    
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, train_ratio=0.7, val_ratio=0.15)
    
    print(f"Total: {data_len}")
    print(f"Train: {len(X_train)} ({len(X_train)/data_len:.2f})")
    print(f"Val: {len(X_val)} ({len(X_val)/data_len:.2f})")
    print(f"Test: {len(X_test)} ({len(X_test)/data_len:.2f})")
    
    expected_train = 700
    expected_val = 150
    expected_test = 150
    
    if len(X_train) == expected_train and len(X_val) == expected_val and len(X_test) == expected_test:
        print("SUCCESS: Split proportions are correct.")
    else:
        print("FAIL: Split proportions are incorrect.")

if __name__ == "__main__":
    test_split_logic()
