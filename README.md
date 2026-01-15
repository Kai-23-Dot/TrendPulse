# Stock Price Predictor

A production-quality web application that predicts future stock prices using Deep Learning (LSTM) and visualizes the results.

## Overview
This project demonstrates the application of Long Short-Term Memory (LSTM) recurrent neural networks for time-series forecasting. It allows users to select any US stock ticker, fetch historical data, train a custom model on-the-fly, and visualize the predicted vs. actual prices.

**Key Features:**
- **Dynamic Data Loading**: Fetches real-time OHLCV data from Yahoo Finance.
- **Deep Learning Model**: Uses a multi-layer LSTM architecture built with TensorFlow/Keras.
- **Interactive Visualization**: Interactive charts using Plotly.
- **Metric Evaluation**: Computes RMSE, MAE, and R² score to evaluate model performance without bias.

## Tech Stack
- **Language**: Python 3.10+
- **Frontend**: Streamlit
- **ML/DL**: TensorFlow, Keras, Scikit-Learn
- **Data**: yfinance, Pandas, NumPy
- **Visualization**: Plotly

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/stock-predictor.git
   cd stock-predictor
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Interact with the App:**
   - Enter a stock ticker (e.g., `AAPL`, `TSLA`, `SPY`).
   - Select a date range for training.
   - Adjust hyperparameters (Window Size, Epochs) in the sidebar.
   - Click **"Train & Predict"**.

## Model Architecture
The core model is an LSTM network designed to capture temporal dependencies in stock price movements:
- **Input Layer**: Sequences of sliding windows (e.g., past 60 days).
- **LSTM Layers**: Two stacked LSTM layers with Dropout to prevent overfitting.
- **Dense Layer**: Single output neuron predicting the next day's closing price.
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)

## Disclaimer
This project is for **educational purposes only**. Stock market prediction is inherently uncertain, and this model should not be used for financial trading or investment decisions.
