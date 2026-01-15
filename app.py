import streamlit as st
import os
# Force CPU usage to avoid MacOS Metal/GPU hangs
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
from src.data_loader import download_data, augment_data, preprocess_pipeline, create_sequences, FEATURE_COLUMNS
from src.model import build_lstm_model, train_model
from src.utils import calculate_metrics, calculate_naive_metrics, plot_test_results

# Page Config
st.set_page_config(
    page_title="TrendPulse | AI Stock Predictor", 
    layout="wide", 
    page_icon="�",
    initial_sidebar_state="expanded"
)

# ================================
# CUSTOM CSS STYLING
# ================================
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root Variables */
    :root {
        --primary: #6366f1;
        --primary-dark: #4f46e5;
        --accent: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --bg-dark: #0f172a;
        --bg-card: #1e293b;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --border: #334155;
    }
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1e2e 50%, #0f172a 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Smooth transitions for all elements */
    * {
        transition: all 0.3s ease;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 0.5rem;
        animation: fadeInDown 0.8s ease-out;
    }
    
    .sub-header {
        color: #94a3b8;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        animation: fadeInUp 0.8s ease-out;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid #334155;
    }
    
    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #f8fafc !important;
        font-weight: 600;
    }
    
    /* Card-like containers */
    .stDataFrame, .stPlotlyChart {
        background: #1e293b;
        border-radius: 12px;
        border: 1px solid #334155;
        padding: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Metric Cards */
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #1e293b 0%, #2d3748 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.3);
        border-color: #6366f1;
    }
    
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6);
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #6366f1 0%, #10b981 100%);
        border-radius: 10px;
    }
    
    /* Sliders */
    .stSlider > div > div > div {
        background: #6366f1;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        border-radius: 10px;
        animation: fadeIn 0.5s ease-out;
    }
    
    .stError {
        background: linear-gradient(90deg, #ef4444 0%, #dc2626 100%);
        border-radius: 10px;
    }
    
    /* Section Headers */
    .section-header {
        color: #f8fafc;
        font-size: 1.5rem;
        font-weight: 600;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #6366f1;
        margin-bottom: 1.5rem;
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    /* Loading Animation */
    .loading-pulse {
        animation: pulse 1.5s infinite;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #334155, transparent);
        margin: 2rem 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e293b;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #6366f1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #4f46e5;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# HEADER
# ================================
st.markdown('<h1 class="main-header">📊 TrendPulse</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Stock Price Prediction • Powered by LSTM Neural Networks</p>', unsafe_allow_html=True)

# ================================
# SIDEBAR
# ================================
with st.sidebar:
    st.markdown("## 🎯 Configuration")
    st.markdown("---")
    
    ticker = st.text_input("📈 Stock Ticker", "SPY", help="Enter any valid stock symbol").upper()
    
    st.markdown("### 📅 Date Range")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", date.today() - timedelta(days=365*5))
    with col2:
        end_date = st.date_input("End", date.today())
    
    st.markdown("---")
    st.markdown("### ⚙️ Hyperparameters")
    
    window_size = st.slider("🔢 Window Size", 10, 90, 60, help="Days of history to consider")
    units = st.slider("🧠 LSTM Units", 16, 128, 64, help="Neurons per layer")
    dropout = st.slider("💧 Dropout", 0.0, 0.5, 0.2, help="Regularization strength")
    epochs = st.slider("🔄 Max Epochs", 10, 100, 50, help="Training iterations")
    batch_size = st.select_slider("📦 Batch Size", options=[16, 32, 64], value=32)
    
    st.markdown("---")
    predict_btn = st.button("🚀 Train & Predict", use_container_width=True)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #64748b; font-size: 0.8rem;'>
        <p>⚠️ For educational purposes only</p>
        <p>Not financial advice</p>
    </div>
    """, unsafe_allow_html=True)

# ================================
# CALLBACK CLASS
# ================================
from tensorflow.keras.callbacks import Callback

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
        self.status_text.markdown(f"📈 **Epoch {epoch + 1}/{self.total_epochs}** | Loss: `{logs['loss']:.5f}` | Val Loss: `{logs['val_loss']:.5f}`")

# ================================
# MAIN EXECUTION
# ================================
if predict_btn:
    with st.spinner(f"📥 Downloading data for **{ticker}**..."):
        raw_df = download_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
    if raw_df is not None:
        # Data Preview
        st.markdown("### 📊 Historical Data")
        st.dataframe(raw_df.tail(), use_container_width=True)
        
        # Feature Engineering
        with st.spinner("🔧 Engineering features..."):
            df_augmented = augment_data(raw_df)
        
        # Preprocessing
        with st.spinner("⚡ Preprocessing..."):
            (train_scaled, val_scaled, test_scaled, 
             scaler_features, scaler_target, 
             train_data, val_data, test_data) = preprocess_pipeline(df_augmented)
            
            target_col_idx = 0 
            X_train, y_train = create_sequences(train_scaled, target_col_idx, window_size)
            X_val, y_val = create_sequences(val_scaled, target_col_idx, window_size)
            X_test, y_test = create_sequences(test_scaled, target_col_idx, window_size)

        if len(X_train) < 50:
            st.error("❌ Not enough training data. Please increase the date range.")
        else:
            # Model Training
            st.markdown("---")
            st.markdown("### ⚙️ Model Training")
            
            X_train = X_train.astype(np.float32)
            y_train = y_train.astype(np.float32)
            X_val = X_val.astype(np.float32)
            y_val = y_val.astype(np.float32)
            
            train_progress = st.progress(0)
            train_status = st.empty()
            
            model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]), units=units, dropout=dropout)
            cb_streamlit = StreamlitCallback(train_progress, train_status, epochs)
            
            history = train_model(
                model, X_train, y_train, 
                validation_data=(X_val, y_val),
                epochs=epochs, 
                batch_size=batch_size,
                callbacks=[cb_streamlit]
            )
            
            train_status.success("✅ Training Complete!")
            train_progress.empty()
            
            # Learning Curve
            st.markdown("---")
            st.markdown("### 📉 Learning Curve")
            loss_fig = go.Figure()
            loss_fig.add_trace(go.Scatter(y=history.history['loss'], mode='lines', name='Train Loss', 
                                          line=dict(color='#6366f1', width=2)))
            loss_fig.add_trace(go.Scatter(y=history.history['val_loss'], mode='lines', name='Val Loss',
                                          line=dict(color='#10b981', width=2)))
            loss_fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Epoch", 
                yaxis_title="Loss",
                font=dict(family="Inter", color="#f8fafc"),
                legend=dict(x=0.8, y=0.95),
                hoverlabel=dict(
                    bgcolor="#1e293b",
                    font_size=14,
                    font_family="Inter",
                    font_color="#f8fafc",
                    bordercolor="#6366f1"
                )
            )
            st.plotly_chart(loss_fig, use_container_width=True)
            
            # Evaluation
            st.markdown("---")
            st.markdown("### � Test Set Performance")
            
            X_test = X_test.astype(np.float32)
            y_test = y_test.astype(np.float32)
            
            pred_returns_scaled = model.predict(X_test)
            pred_returns = scaler_target.inverse_transform(pred_returns_scaled).flatten()
            
            test_dates = test_data.index[window_size:]
            base_prices = test_data['Close'].iloc[window_size-1:-1].values
            pred_prices = base_prices * np.exp(pred_returns)
            actual_prices = test_data['Close'].iloc[window_size:].values
            
            lstm_metrics = calculate_metrics(actual_prices, pred_prices)
            naive_metrics = calculate_naive_metrics(actual_prices, base_prices)
            
            # Metrics Display
            col1, col2, col3 = st.columns(3)
            col1.metric("🎯 Test RMSE", f"${lstm_metrics['RMSE']:.2f}", 
                       delta=f"{lstm_metrics['RMSE'] - naive_metrics['RMSE']:.2f} vs baseline", 
                       delta_color="inverse")
            col2.metric("📊 Test MAE", f"${lstm_metrics['MAE']:.2f}", 
                       delta=f"{lstm_metrics['MAE'] - naive_metrics['MAE']:.2f} vs baseline", 
                       delta_color="inverse")
            col3.metric("📈 Test R²", f"{lstm_metrics['R2']:.4f}", 
                       delta=f"{lstm_metrics['R2'] - naive_metrics['R2']:.4f} vs baseline")
            
            st.caption("🟢 Green delta = better than naive baseline (tomorrow = today)")
            
            # Prediction Chart
            st.markdown("---")
            st.markdown("### 🔮 Predictions vs Actual")
            pred_fig = go.Figure()
            pred_fig.add_trace(go.Scatter(x=test_dates, y=actual_prices, mode='lines', name='Actual',
                                          line=dict(color='#10b981', width=2)))
            pred_fig.add_trace(go.Scatter(x=test_dates, y=pred_prices, mode='lines', name='Predicted',
                                          line=dict(color='#6366f1', width=2)))
            pred_fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Date",
                yaxis_title="Price ($)",
                font=dict(family="Inter", color="#f8fafc"),
                legend=dict(x=0.02, y=0.98),
                hoverlabel=dict(
                    bgcolor="#1e293b",
                    font_size=14,
                    font_family="Inter",
                    font_color="#f8fafc",
                    bordercolor="#6366f1"
                ),
                hovermode="x unified"
            )
            st.plotly_chart(pred_fig, use_container_width=True)
            
            # Next Day Forecast
            st.markdown("---")
            st.markdown("### 🎯 Next Day Forecast")
            
            full_scaled = scaler_features.transform(df_augmented[FEATURE_COLUMNS])
            n_features = len(FEATURE_COLUMNS)
            last_sequence = full_scaled[-window_size:].reshape(1, window_size, n_features).astype(np.float32)
            
            next_return_scaled = model.predict(last_sequence, verbose=0)
            next_return = scaler_target.inverse_transform(next_return_scaled)[0][0]
            
            last_close = df_augmented['Close'].iloc[-1]
            last_date = df_augmented.index[-1]
            next_day_price = last_close * np.exp(next_return)
            pct_change = ((next_day_price/last_close)-1)*100
            
            col1, col2 = st.columns(2)
            col1.metric("📅 Last Close", f"${last_close:.2f}", help=f"As of {last_date.date()}")
            col2.metric("🎯 Tomorrow's Prediction", f"${next_day_price:.2f}", 
                       delta=f"{pct_change:+.2f}%",
                       delta_color="normal" if pct_change >= 0 else "inverse")
            
            st.info(f"⚠️ Prediction based on data through **{last_date.date()}**. This is not financial advice.")
            
    else:
        st.error(f"❌ Could not download data for **{ticker}**. Please check the symbol.")
