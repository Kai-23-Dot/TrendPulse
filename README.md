# 📊 TrendPulse

> AI-Powered Stock Price Prediction using LSTM Neural Networks

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

## Overview

TrendPulse is a production-quality web application that predicts future stock prices using Deep Learning. It allows users to select any US stock ticker, train custom LSTM models on-the-fly, and visualize predictions with professional metrics.

**Key Features:**
- 📈 **Dynamic Data Loading** - Real-time OHLCV data from Yahoo Finance
- 🧠 **Deep Learning** - Multi-layer LSTM with dropout regularization
- 📊 **9 Technical Indicators** - MA, RSI, volatility, volume patterns
- 🎯 **Rigorous Evaluation** - Naive baseline comparison, train/val/test splits
- 🔮 **Next-Day Forecasts** - With confidence intervals

## Tech Stack

| Category | Technologies |
|----------|-------------|
| Frontend | Streamlit, Plotly |
| ML/DL | TensorFlow, Keras, Scikit-Learn |
| Data | yfinance, Pandas, NumPy |

## Quick Start

```bash
# Clone
git clone https://github.com/Kai-23-Dot/TrendPulse.git
cd TrendPulse

# Install
pip install -r requirements.txt

# Run
streamlit run app.py
```

---

## Why LSTM? (Model Selection & Trade-offs)

**Why LSTM over simpler models (Random Forest, XGBoost)?**  
LSTMs capture long-term temporal dependencies in sequential data. Unlike tree-based models that treat features independently, LSTMs understand that yesterday's price influences today's, and patterns can repeat over weeks or months.

**Why LSTM over traditional time-series (ARIMA)?**  
ARIMA assumes linear relationships and stationarity. LSTMs learn non-linear patterns directly from data without manual differencing or stationarity transformations.

**Trade-off:**  
LSTMs require more data and compute but offer flexibility. Stock data has only ~250 trading days/year, so we augment with returns + 9 technical indicators to help the model generalize.

---

## Limitations & Lessons Learned

> *Honest limitations demonstrate critical thinking—essential for finance portfolios.*

### 1. External Shocks Are Invisible
Market crashes, geopolitical events, earnings surprises—these are not in historical price data. The model learns patterns from calm periods but struggles during crises. (See: [Black Swan Theory](https://en.wikipedia.org/wiki/Black_swan_theory))

### 2. Regime Shifts Break Patterns
Even with returns, market regimes change (bull ≠ bear markets). A model trained on 2021–2023 may not generalize to 2024–2025 if conditions shift. (See: [Efficient Market Hypothesis](https://en.wikipedia.org/wiki/Efficient-market_hypothesis))

### 3. Data Leakage Risks
We carefully fit scalers only on training data and use time-ordered splits. Real trading would require daily retraining on new data to avoid look-ahead bias.

### 4. Lagging Indicators
Moving averages and RSI are *lagging* indicators—they describe what happened, not what will happen. Heavy reliance can lead to overfitting past patterns.

### 5. Limited Prediction Horizon
Next-day prediction is easier than weeks ahead. Beyond 5–10 days, uncertainty grows exponentially. This model is optimized for 1-day forecasts.

### 6. Not a Trading System
This is a demonstration of LSTM time-series forecasting, not a deployed strategy. Real trading requires risk management, position sizing, and handling slippage/commissions.

---

## Future Work

- 📰 **Sentiment Analysis** - Integrate news/social media sentiment
- 🤖 **Ensemble Methods** - Combine LSTM with XGBoost for robustness
- 💰 **Portfolio Optimization** - Multi-asset allocation with RL
- ☁️ **Cloud Deployment** - Scheduled retraining on AWS/GCP

---

## Model Architecture

```
Input (60 days × 9 features)
    ↓
LSTM (64 units, return_sequences=True)
    ↓
Dropout (0.2)
    ↓
LSTM (64 units)
    ↓
Dropout (0.2)
    ↓
Dense (1) → Predicted Log Return
    ↓
Reconstruct Price: Price_t = Price_{t-1} × exp(Return)
```

---

## Disclaimer

⚠️ **This project is for educational purposes only.** Stock market prediction is inherently uncertain. Past performance does not guarantee future results. This model should not be used for financial trading or investment decisions.

---

## License

MIT License - See [LICENSE](LICENSE) for details.
