import streamlit as st
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
try:
    tf.config.set_visible_devices([], 'GPU')
except:
    pass

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta
from pathlib import Path
from src.data_loader import download_data, augment_data, preprocess_pipeline, create_sequences, FEATURE_COLUMNS
from src.model import build_lstm_model, train_model
from src.utils import calculate_metrics, calculate_naive_metrics

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="TrendPulse | AI Stock Predictor", 
    layout="wide", 
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# ================================
# CUSTOM CSS
# ================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary: #6366f1;
        --accent: #10b981;
        --bg-card: #1e293b;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1e2e 50%, #0f172a 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        padding: 0.5rem 0;
    }
    
    .sub-header {
        color: #94a3b8;
        text-align: center;
        font-size: 1rem;
        margin-bottom: 1rem;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #1e293b, #2d3748);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.5);
    }
    
    .stat-card {
        background: #1e293b;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #334155;
    }
    
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f8fafc;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: #94a3b8;
    }
    
    .winner-badge {
        background: linear-gradient(90deg, #10b981, #059669);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .disclaimer-box {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid #f59e0b;
        border-radius: 10px;
        padding: 1rem;
        color: #fbbf24;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ================================
# HEADER
# ================================
st.markdown('<h1 class="main-header">📊 TrendPulse</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Stock Price Prediction • LSTM Neural Networks</p>', unsafe_allow_html=True)

# ================================
# SIDEBAR
# ================================
with st.sidebar:
    st.markdown("## 🎯 Configuration")
    st.markdown("---")
    
    # Load full US stock ticker database
    @st.cache_data
    def load_ticker_database():
        """Load all US stock tickers from database file."""
        ticker_file = Path("data/us_tickers.txt")
        if ticker_file.exists():
            with open(ticker_file, 'r') as f:
                tickers = [line.strip() for line in f if line.strip()]
            return sorted(tickers)
        else:
            # Fallback to popular stocks if database not available
            return sorted(["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", 
                          "JPM", "V", "MA", "UNH", "JNJ", "WMT", "HD", "DIS", "NFLX", "BA", "CAT"])
    
    ALL_TICKERS = load_ticker_database()
    
    # Stock Selection
    st.markdown("### 📈 Stock Selection")
    st.caption(f"📊 {len(ALL_TICKERS):,} stocks available")
    
    use_custom = st.checkbox("Enter custom ticker", value=False)
    
    if use_custom:
        ticker = st.text_input("Custom Ticker", "SPY", help="Enter any valid stock symbol").upper()
    else:
        ticker = st.selectbox(
            "Select Stock",
            options=ALL_TICKERS,
            index=ALL_TICKERS.index("SPY") if "SPY" in ALL_TICKERS else 0,
            help="Type to search from 6,600+ US stocks"
        )
    
    st.markdown("### 📅 Date Range")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", date.today() - timedelta(days=365*5))
    with col2:
        end_date = st.date_input("End", date.today())
    
    st.markdown("---")
    st.markdown("### ⚙️ Hyperparameters")
    
    window_size = st.slider("🔢 Window Size", 10, 90, 60)
    units = st.slider("🧠 LSTM Units", 16, 128, 64)
    dropout = st.slider("💧 Dropout", 0.0, 0.5, 0.2)
    epochs = st.slider("🔄 Max Epochs", 10, 100, 50)
    batch_size = st.select_slider("📦 Batch Size", options=[16, 32, 64], value=32)
    
    st.markdown("---")
    run_analysis = st.button("🚀 Run Analysis", use_container_width=True)
    
    st.markdown("---")
    st.caption("⚠️ For educational purposes only. Not financial advice.")

# ================================
# MODEL LOADING HELPERS
# ================================
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model
from pathlib import Path
import json

def check_pretrained_model(ticker: str):
    """Check if a pre-trained model exists for this ticker."""
    model_path = Path(f"models/{ticker}_model.keras")
    metadata_path = Path(f"models/{ticker}_metadata.json")
    
    if model_path.exists() and metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return True, model_path, metadata
    return False, None, None

def load_pretrained_model(model_path):
    """Load a pre-trained model from disk."""
    return load_model(model_path)

class StreamlitCallback(Callback):
    def __init__(self, progress_bar, status_text, total_epochs):
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.total_epochs = total_epochs

    def on_train_begin(self, logs=None):
        self.status_text.markdown("🔄 **Training started...**")

    def on_epoch_end(self, epoch, logs=None):
        progress = (epoch + 1) / self.total_epochs
        self.progress_bar.progress(progress)
        self.status_text.markdown(f"**Epoch {epoch + 1}/{self.total_epochs}** | Loss: `{logs['loss']:.5f}` | Val: `{logs['val_loss']:.5f}`")

# ================================
# MAIN TABS
# ================================
tab1, tab2, tab3 = st.tabs(["📊 Data Exploration", "🚀 Model Training & Evaluation", "📈 Predictions & Results"])

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
    st.session_state.model = None
    st.session_state.history = None
    st.session_state.metrics = None
    st.session_state.predictions = None

# ================================
# DATA LOADING (runs on button click)
# ================================

# Check for pre-trained model
has_pretrained, pretrained_path, pretrained_meta = check_pretrained_model(ticker)

if has_pretrained:
    with st.sidebar:
        st.markdown("---")
        st.success(f"✅ Pre-trained model available!")
        st.caption(f"Trained: {pretrained_meta.get('trained_at', 'Unknown')[:10]}")
        st.caption(f"RMSE: ${pretrained_meta.get('test_rmse', 0):.2f}")
        use_pretrained = st.checkbox("Use pre-trained model (faster)", value=True)
else:
    use_pretrained = False

if run_analysis:
    with st.spinner(f"📥 Downloading {ticker} data..."):
        raw_df, adjusted_start, message = download_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
    if raw_df is None:
        st.error(f"❌ {message if message else f'Could not download data for {ticker}'}")
    else:
        # Show info message if date was adjusted
        if message:
            st.info(message)
        
        with st.spinner("🔧 Engineering features..."):
            df_augmented = augment_data(raw_df)
        
        # Store in session state
        st.session_state.raw_df = raw_df
        st.session_state.df_augmented = df_augmented
        st.session_state.ticker = ticker
        st.session_state.adjusted_start = adjusted_start
        
        # Preprocessing
        with st.spinner("⚡ Preprocessing..."):
            (train_scaled, val_scaled, test_scaled, 
             scaler_features, scaler_target, 
             train_data, val_data, test_data) = preprocess_pipeline(df_augmented)
            
            X_train, y_train = create_sequences(train_scaled, 0, window_size)
            X_val, y_val = create_sequences(val_scaled, 0, window_size)
            X_test, y_test = create_sequences(test_scaled, 0, window_size)
        
        st.session_state.preprocessing = {
            'X_train': X_train.astype(np.float32),
            'y_train': y_train.astype(np.float32),
            'X_val': X_val.astype(np.float32),
            'y_val': y_val.astype(np.float32),
            'X_test': X_test.astype(np.float32),
            'y_test': y_test.astype(np.float32),
            'scaler_features': scaler_features,
            'scaler_target': scaler_target,
            'test_data': test_data,
            'window_size': window_size
        }
        
        st.session_state.data_loaded = True
        st.rerun()

# ================================
# TAB 1: DATA EXPLORATION
# ================================
with tab1:
    if 'df_augmented' in st.session_state:
        df = st.session_state.df_augmented
        ticker_name = st.session_state.ticker
        
        st.markdown(f"### 📊 Historical Data: {ticker_name}")
        
        # Price Chart
        price_fig = go.Figure()
        price_fig.add_trace(go.Scatter(
            x=df.index, y=df['Close'], 
            mode='lines', name='Close Price',
            line=dict(color='#6366f1', width=2)
        ))
        price_fig.add_trace(go.Scatter(
            x=df.index, y=df['MA_30'], 
            mode='lines', name='30-Day MA',
            line=dict(color='#10b981', width=1, dash='dash')
        ))
        price_fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Date", yaxis_title="Price ($)",
            hoverlabel=dict(bgcolor="#1e293b", font_color="#f8fafc"),
            legend=dict(x=0.02, y=0.98)
        )
        st.plotly_chart(price_fig, use_container_width=True)
        
        # Basic Stats
        st.markdown("### 📈 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Min Price", f"${df['Close'].min():.2f}")
        col2.metric("Max Price", f"${df['Close'].max():.2f}")
        col3.metric("Mean Price", f"${df['Close'].mean():.2f}")
        col4.metric("Volatility (Ann.)", f"{df['Volatility'].mean() * np.sqrt(252) * 100:.1f}%")
        
        # Feature Display
        st.markdown("### 🔧 Engineered Features")
        st.info("💡 These features help the LSTM understand **trend** (moving averages), **momentum** (ROC, RSI), and **market conditions** (volatility, volume).")
        
        display_cols = ['Close', 'Log_Return', 'MA_10', 'MA_30', 'RSI_14', 'Volatility', 'Volume_Ratio']
        st.dataframe(df[display_cols].tail(10).style.format("{:.4f}"), use_container_width=True)
        
    else:
        st.info("👈 Configure parameters and click **Run Analysis** to load data.")

# ================================
# TAB 2: MODEL TRAINING & EVALUATION
# ================================
with tab2:
    if 'preprocessing' in st.session_state:
        prep = st.session_state.preprocessing
        
        # Hyperparameter Summary
        st.markdown("### ⚙️ Current Configuration")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Window", f"{prep['window_size']} days")
        col2.metric("LSTM Units", units)
        col3.metric("Dropout", f"{dropout:.0%}")
        col4.metric("Max Epochs", epochs)
        col5.metric("Batch Size", batch_size)
        
        st.markdown("---")
        
        # Training Section
        if not st.session_state.model_trained:
            st.markdown("### 🏋️ Model Training")
            
            if len(prep['X_train']) < 50:
                st.error("❌ Not enough training data. Increase date range.")
            else:
                train_progress = st.progress(0)
                train_status = st.empty()
                
                train_status.markdown("🔄 Building model...")
                model = build_lstm_model(
                    input_shape=(prep['X_train'].shape[1], prep['X_train'].shape[2]), 
                    units=units, dropout=dropout
                )
                
                cb = StreamlitCallback(train_progress, train_status, epochs)
                
                history = train_model(
                    model, prep['X_train'], prep['y_train'],
                    validation_data=(prep['X_val'], prep['y_val']),
                    epochs=epochs, batch_size=batch_size,
                    callbacks=[cb]
                )
                
                train_status.success("✅ Training Complete!")
                train_progress.empty()
                
                # Store results
                st.session_state.model = model
                st.session_state.history = history
                st.session_state.model_trained = True
                
                # Calculate metrics
                pred_scaled = model.predict(prep['X_test'], verbose=0)
                pred_returns = prep['scaler_target'].inverse_transform(pred_scaled).flatten()
                
                test_data = prep['test_data']
                ws = prep['window_size']
                base_prices = test_data['Close'].iloc[ws-1:-1].values
                pred_prices = base_prices * np.exp(pred_returns)
                actual_prices = test_data['Close'].iloc[ws:].values
                test_dates = test_data.index[ws:]
                
                lstm_metrics = calculate_metrics(actual_prices, pred_prices)
                naive_metrics = calculate_naive_metrics(actual_prices, base_prices)
                
                st.session_state.metrics = {
                    'lstm': lstm_metrics,
                    'naive': naive_metrics
                }
                st.session_state.predictions = {
                    'dates': test_dates,
                    'actual': actual_prices,
                    'predicted': pred_prices,
                    'baseline': base_prices
                }
                
                st.rerun()
        
        # Show Training Results
        if st.session_state.model_trained and st.session_state.history:
            st.markdown("### 📉 Learning Curve")
            h = st.session_state.history.history
            loss_fig = go.Figure()
            loss_fig.add_trace(go.Scatter(y=h['loss'], name='Train Loss', line=dict(color='#6366f1', width=2)))
            loss_fig.add_trace(go.Scatter(y=h['val_loss'], name='Val Loss', line=dict(color='#10b981', width=2)))
            loss_fig.update_layout(
                template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Epoch", yaxis_title="Loss",
                hoverlabel=dict(bgcolor="#1e293b", font_color="#f8fafc")
            )
            st.plotly_chart(loss_fig, use_container_width=True)
            
            # Metrics Comparison
            st.markdown("### 📊 Metrics Comparison: LSTM vs Naive Baseline")
            st.caption("*Naive Baseline: Predict tomorrow's price = today's price*")
            
            metrics = st.session_state.metrics
            lstm = metrics['lstm']
            naive = metrics['naive']
            
            col1, col2, col3 = st.columns(3)
            
            # RMSE
            with col1:
                st.markdown("#### RMSE (Lower is Better)")
                rmse_winner = "LSTM" if lstm['RMSE'] < naive['RMSE'] else "Baseline"
                st.metric("Naive Baseline", f"${naive['RMSE']:.2f}")
                st.metric("LSTM Model", f"${lstm['RMSE']:.2f}", 
                         delta=f"{lstm['RMSE'] - naive['RMSE']:.2f}", delta_color="inverse")
                if rmse_winner == "LSTM":
                    st.markdown('<span class="winner-badge">✓ LSTM Wins</span>', unsafe_allow_html=True)
            
            # MAE
            with col2:
                st.markdown("#### MAE (Lower is Better)")
                st.metric("Naive Baseline", f"${naive['MAE']:.2f}")
                st.metric("LSTM Model", f"${lstm['MAE']:.2f}", 
                         delta=f"{lstm['MAE'] - naive['MAE']:.2f}", delta_color="inverse")
            
            # R²
            with col3:
                st.markdown("#### R² (Higher is Better)")
                st.metric("Naive Baseline", f"{naive['R2']:.4f}")
                st.metric("LSTM Model", f"{lstm['R2']:.4f}", 
                         delta=f"{lstm['R2'] - naive['R2']:.4f}")
            
            st.info("💡 **Interpretation:** Model is better than baseline if RMSE < Baseline RMSE. Green deltas indicate improvement.")
            
    else:
        st.info("👈 Click **Run Analysis** to train the model.")

# ================================
# TAB 3: PREDICTIONS & RESULTS
# ================================
with tab3:
    if st.session_state.model_trained and st.session_state.predictions:
        preds = st.session_state.predictions
        
        st.markdown("### 🔮 Test Set Predictions")
        
        # Actual vs Predicted Chart
        pred_fig = go.Figure()
        pred_fig.add_trace(go.Scatter(
            x=preds['dates'], y=preds['actual'],
            mode='lines', name='Actual Price',
            line=dict(color='#10b981', width=2)
        ))
        pred_fig.add_trace(go.Scatter(
            x=preds['dates'], y=preds['predicted'],
            mode='lines', name='Predicted Price',
            line=dict(color='#6366f1', width=2)
        ))
        # Error band (±1 RMSE)
        rmse = st.session_state.metrics['lstm']['RMSE']
        pred_fig.add_trace(go.Scatter(
            x=list(preds['dates']) + list(preds['dates'])[::-1],
            y=list(preds['predicted'] + rmse) + list(preds['predicted'] - rmse)[::-1],
            fill='toself', fillcolor='rgba(99, 102, 241, 0.1)',
            line=dict(color='rgba(0,0,0,0)'),
            name='±1 RMSE Band'
        ))
        pred_fig.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Date", yaxis_title="Price ($)",
            hoverlabel=dict(bgcolor="#1e293b", font_color="#f8fafc"),
            hovermode="x unified",
            legend=dict(x=0.02, y=0.98)
        )
        st.plotly_chart(pred_fig, use_container_width=True)
        
        # Next Day Forecast
        st.markdown("---")
        st.markdown("### 🎯 Next Day Forecast")
        
        prep = st.session_state.preprocessing
        df = st.session_state.df_augmented
        model = st.session_state.model
        
        full_scaled = prep['scaler_features'].transform(df[FEATURE_COLUMNS])
        n_features = len(FEATURE_COLUMNS)
        last_seq = full_scaled[-prep['window_size']:].reshape(1, prep['window_size'], n_features).astype(np.float32)
        
        next_return_scaled = model.predict(last_seq, verbose=0)
        next_return = prep['scaler_target'].inverse_transform(next_return_scaled)[0][0]
        
        last_close = df['Close'].iloc[-1]
        last_date = df.index[-1]
        next_price = last_close * np.exp(next_return)
        pct_change = ((next_price / last_close) - 1) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("📅 Last Close", f"${last_close:.2f}", help=f"As of {last_date.date()}")
        col2.metric("🎯 Predicted Next Close", f"${next_price:.2f}", 
                   delta=f"{pct_change:+.2f}%")
        col3.metric("📊 Model Confidence", f"±${rmse:.2f}", help="Based on test RMSE")
        
        # Detailed Predictions Table
        st.markdown("---")
        st.markdown("### 📋 Recent Predictions Detail")
        
        detail_df = pd.DataFrame({
            'Date': preds['dates'][-10:],
            'Actual': preds['actual'][-10:],
            'Predicted': preds['predicted'][-10:],
            'Error': preds['actual'][-10:] - preds['predicted'][-10:]
        }).set_index('Date')
        st.dataframe(detail_df.style.format({'Actual': '${:.2f}', 'Predicted': '${:.2f}', 'Error': '${:.2f}'}), 
                    use_container_width=True)
        
        # Disclaimer
        st.markdown("---")
        st.markdown("""
        <div class="disclaimer-box">
            ⚠️ <strong>Important Disclaimer</strong><br><br>
            These predictions are from a deep learning model trained on historical patterns. 
            Stock prices are influenced by unforeseen events, news, and market sentiment that 
            the model cannot capture. Past performance does not guarantee future results. 
            <strong>This is not financial advice.</strong>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.info("👈 Complete the training in the **Model Training** tab to see predictions.")
