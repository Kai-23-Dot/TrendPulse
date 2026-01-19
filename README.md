# 📊 TrendPulse

> **AI-Powered Stock Price Prediction using LSTM Neural Networks**

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://trendpulse.streamlit.app)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

## Overview

**TrendPulse** is an enterprise-grade stock forecasting application that leverages deep learning to predict future price movements. Built for financial analysts and researchers, it enables real-time testing of custom LSTM neural network architectures on 6,000+ US stocks.

Unlike basic demos, TrendPulse features a production-ready data pipeline with **multi-source fallback** (Yahoo Finance → Yahoo Chart API → Stooq) to ensure 99.9% uptime, even in restrictive cloud environments like Streamlit Cloud where standard scraping is often blocked.

## 📸 Screenshots

| **Interactive Dashboard** | **Model Training** |
|:-------------------------:|:------------------:|
| ![Dashboard](https://placehold.co/600x400/1e293b/ffffff?text=Dashboard+View) | ![Training](https://placehold.co/600x400/1e293b/ffffff?text=Training+Process) |

| **Predictions & Metrics** | **Documentation** |
|:-------------------------:|:-----------------:|
| ![Predictions](https://placehold.co/600x400/1e293b/ffffff?text=Prediction+Chart) | ![Docs](https://placehold.co/600x400/1e293b/ffffff?text=Documentation+Tab) |

## Key Features

- 📈 **Robust Data Engine** - Auto-healing data fetcher that rotates User-Agents and switches APIs on failure.
- 🧠 **Deep Learning** - Multi-layer LSTM (Long Short-Term Memory) network with dropout regularization for time-series forecasting.
- 📊 **Advanced Feature Engineering** - Automatically calculates 9 technical indicators (RSI, Bollinger Bands, MACD, etc.) to enrich the dataset.
- 🎯 **Rigorous Evaluation** - Real-time comparison against a naive baseline using RMSE, MAE, and R².
- 🔮 **Next-Day Forecasts** - Provides concrete next-day price targets with confidence intervals.

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Frontend** | Streamlit, Plotly (Interactive Charts) |
| **ML/DL** | TensorFlow, Keras, Scikit-Learn |
| **Data** | yfinance, pandas-datareader, Stooq, NumPy |
| **Utilities** | Requests-Cache, Joblib |

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Kai-23-Dot/TrendPulse.git
cd TrendPulse

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

---

## Why LSTM? (Model Selection & Trade-offs)

**Why LSTM over simpler models (Random Forest, XGBoost)?**  
LSTMs are designed to capture *long-term temporal dependencies* in sequential data. Unlike tree-based models that treat rows independently, LSTMs understand that price action from 30 days ago can influence today's trend.

**Why LSTM over traditional time-series (ARIMA)?**  
ARIMA assumes linear relationships and stationarity. LSTMs can learn complex, non-linear patterns directly from raw (scaled) data without manual differencing.

---

## Limitations & Lessons Learned

> *Transparency is key for financial modeling.*

### 1. External Shocks Are Invisible
Market crashes, geopolitical events, and earnings surprises are not present in historical price data. The model learns patterns from "normal" volatility but may fail during Black Swan events.

### 2. Regime Shifts
A model trained on a bull market (2020-2021) may struggle to generalize to a bear market or high-interest rate regime.

### 3. Lagging Indicators
Features like Moving Averages are *lagging* indicators. They confirm trends rather than predict them. We mitigate this by including momentum indicators (ROC, RSI).

---

## Disclaimer

⚠️ **This project is for educational purposes only.** Stock market prediction is inherently uncertain. Past performance does not guarantee future results. This model should not be used for actual financial trading or investment decisions.

---

## License

MIT License - See [LICENSE](LICENSE) for details.
