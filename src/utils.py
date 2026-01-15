import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go

def calculate_metrics(y_true, y_pred):
    """
    Calculates evaluation metrics.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

def calculate_naive_metrics(y_true, y_prev_day):
    """
    Calculates metrics for a naive baseline (prediction = previous day's close).
    """
    return calculate_metrics(y_true, y_prev_day)

def plot_test_results(dates, y_true, y_pred, y_naive=None, title="Stock Price: Actual vs Predicted (Test Set)"):
    """
    Creates a Plotly figure for actual vs predicted prices.
    Includes explicit Baseline comparison if provided.
    """
    fig = go.Figure()
    
    # Actual Price
    fig.add_trace(go.Scatter(
        x=dates, y=y_true.flatten(),
        mode='lines', name='Actual Price',
        line=dict(color='#00CC96', width=2) # Green
    ))
    
    # Predicted Price
    fig.add_trace(go.Scatter(
        x=dates, y=y_pred.flatten(),
        mode='lines', name='Predicted Price (LSTM)',
        line=dict(color='#EF553B', width=2) # Red
    ))
    
    # Naive Baseline (optional visual)
    # Usually naive is just shifting actual by 1 day, visually cluttering.
    # We might just rely on the metrics table for naive comparison.
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        legend=dict(x=0, y=1)
    )
    
    return fig
