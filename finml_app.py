# finml_app.py

# 1. REQUIRED: Streamlit import FIRST
import streamlit as st

# 2. REQUIRED: Page config SECOND
st.set_page_config(
    page_title="FinML Pro - Financial Machine Learning",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/hamnaiimran/financial-ml-app',
        'Report a bug': "https://github.com/hamnaiimran/financial-ml-app/issues",
        'About': "# Financial ML App v2.0"
    }
)

# 3. Other imports (AFTER page config)
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 4. Custom functions (NO Streamlit commands here)
def load_stock_data(ticker, start_date, end_date):
    """Fetch stock data from Yahoo Finance with technical indicators"""
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        return None
    
    # Add technical indicators
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = compute_rsi(data['Close'])
    data['MACD'], data['Signal_Line'] = compute_macd(data['Close'])
    return data.dropna()

def compute_rsi(series, window=14):
    """Calculate Relative Strength Index"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, slow=26, fast=12, signal=9):
    """Calculate MACD indicator"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def train_ml_model(X, y, model_type='linear'):
    """Train and evaluate ML model"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    if model_type == 'linear':
        model = make_pipeline(
            StandardScaler(),
            LinearRegression()
        )
    elif model_type == 'random_forest':
        model = make_pipeline(
            StandardScaler(),
            RandomForestRegressor(n_estimators=100, random_state=42)
        )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return {
        'model': model,
        'mse': mean_squared_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'actual': y_test,
        'predicted': y_pred
    }

# 5. Main application function
def main():
    """Core application logic"""
    
    # Sidebar controls
    with st.sidebar:
        st.title("Configuration")
        ticker = st.text_input("Stock Ticker", "AAPL")
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
        end_date = st.date_input("End Date", datetime.now())
        model_type = st.radio("ML Model", ["Linear Regression", "Random Forest"])
        st.markdown("---")
        st.caption("AF3005 - Programming for Finance")
        st.caption("Dr. Usama Arshad - FAST-NUCES")
    
    # Main dashboard
    st.title(f"📊 {ticker} Financial Analysis")
    
    # Data loading section
    if st.sidebar.button("Load & Analyze"):
        with st.spinner("Processing data..."):
            try:
                data = load_stock_data(ticker, start_date, end_date)
                
                if data is None:
                    st.error("No data found for this ticker/date range")
                    return
                
                # Show raw data
                with st.expander("View Raw Data"):
                    st.dataframe(data.tail(10))
                
                # Price chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data.index, y=data['Close'],
                    name='Close Price', line=dict(color='#1f77b4'))
                fig.add_trace(go.Scatter(
                    x=data.index, y=data['SMA_20'],
                    name='20-day SMA', line=dict(color='#ff7f0e', dash='dot')))
                fig.add_trace(go.Scatter(
                    x=data.index, y=data['SMA_50'],
                    name='50-day SMA', line=dict(color='#2ca02c', dash='dot')))
                fig.update_layout(
                    title=f"{ticker} Price and Moving Averages",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Technical indicators
                st.subheader("Technical Indicators")
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_rsi = px.line(data, y='RSI', title="RSI (14-day)")
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                    st.plotly_chart(fig_rsi, use_container_width=True)
                
                with col2:
                    fig_macd = go.Figure()
                    fig_macd.add_trace(go.Scatter(
                        x=data.index, y=data['MACD'],
                        name='MACD', line=dict(color='#17becf')))
                    fig_macd.add_trace(go.Scatter(
                        x=data.index, y=data['Signal_Line'],
                        name='Signal Line', line=dict(color='#ff7f0e')))
                    fig_macd.update_layout(title="MACD Indicator")
                    st.plotly_chart(fig_macd, use_container_width=True)
                
                # Machine Learning Section
                st.subheader("Machine Learning Prediction")
                X = data[['SMA_20', 'SMA_50', 'RSI', 'MACD']]
                y = data['Close']
                
                model_results = train_ml_model(
                    X, y, 
                    model_type='linear' if model_type == "Linear Regression" else 'random_forest'
                )
                
                # Results display
                col_metrics1, col_metrics2 = st.columns(2)
                with col_metrics1:
                    st.metric("Mean Squared Error", f"{model_results['mse']:.2f}")
                with col_metrics2:
                    st.metric("R² Score", f"{model_results['r2']:.2f}")
                
                # Actual vs Predicted plot
                fig_results = px.scatter(
                    x=model_results['actual'],
                    y=model_results['predicted'],
                    labels={'x': 'Actual Price', 'y': 'Predicted Price'},
                    title="Actual vs Predicted Prices"
                )
                fig_results.add_shape(
                    type="line",
                    x0=min(model_results['actual']),
                    y0=min(model_results['actual']),
                    x1=max(model_results['actual']),
                    y1=max(model_results['actual']),
                    line=dict(color="Red", dash="dash")
                )
                st.plotly_chart(fig_results, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

# 6. REQUIRED: Execution guard at the end
if __name__ == "__main__":
    main()
