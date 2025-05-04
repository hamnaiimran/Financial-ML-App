st.set_page_config(
    page_title="Financial ML App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)
@st.cache_data  # Must come AFTER set_page_config
def load_data():
    return pd.read_csv(...)
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (mean_squared_error, r2_score, 
                           accuracy_score, confusion_matrix, 
                           silhouette_score)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from datetime import datetime, timedelta
import time
import requests
from io import StringIO
import base64
import pytz
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# Custom Theme and Professional Styling
# ------------------------------

# Custom CSS with professional dark theme
def set_custom_theme():
    st.markdown(f"""
    <style>
    :root {{
        --primary: #1f77b4;
        --secondary: #ff7f0e;
        --accent: #2ca02c;
        --background: #0e1117;
        --card: #1a1d24;
        --text: #f0f2f6;
        --border: #2a2e36;
    }}
    
    .main {{
        background-color: var(--background);
        color: var(--text);
    }}
    
    .sidebar .sidebar-content {{
        background-color: var(--card) !important;
        border-right: 1px solid var(--border);
    }}
    
    .stButton>button {{
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
        border-radius: 8px;
        border: none;
        padding: 12px 28px;
        font-weight: 600;
        font-size: 16px;
        margin: 8px 0;
        transition: all 0.3s;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        background: linear-gradient(135deg, var(--secondary), var(--primary));
    }}
    
    .stSelectbox, .stTextInput, .stDateInput, .stNumberInput {{
        background-color: var(--card);
        border: 1px solid var(--border);
        border-radius: 6px;
        color: var(--text);
    }}
    
    .stDataFrame {{
        background-color: var(--card) !important;
        border: 1px solid var(--border) !important;
    }}
    
    .stProgress > div > div > div {{
        background: linear-gradient(90deg, var(--primary), var(--accent));
    }}
    
    .stAlert {{
        border-radius: 8px;
    }}
    
    .stSuccess {{
        background-color: rgba(46, 125, 50, 0.2) !important;
        border-left: 4px solid #2e7d32;
    }}
    
    .stInfo {{
        background-color: rgba(2, 136, 209, 0.2) !important;
        border-left: 4px solid #0288d1;
    }}
    
    .stWarning {{
        background-color: rgba(245, 124, 0, 0.2) !important;
        border-left: 4px solid #f57c00;
    }}
    
    .stError {{
        background-color: rgba(211, 47, 47, 0.2) !important;
        border-left: 4px solid #d32f2f;
    }}
    
    .css-1aumxhk {{
        background-color: var(--card);
    }}
    
    .ticker-tape {{
        display: flex;
        align-items: center;
        background-color: var(--card);
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 16px;
        overflow: hidden;
        border: 1px solid var(--border);
    }}
    
    .ticker-item {{
        display: inline-block;
        margin-right: 30px;
        font-family: 'Courier New', monospace;
        font-weight: bold;
        white-space: nowrap;
    }}
    
    .model-card {{
        background-color: var(--card);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid var(--border);
        transition: transform 0.3s, box-shadow 0.3s;
    }}
    
    .model-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
    }}
    
    .metric-card {{
        background-color: var(--card);
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid var(--primary);
    }}
    
    .footer {{
        text-align: center;
        padding: 20px;
        margin-top: 30px;
        color: var(--text);
        font-size: 14px;
        border-top: 1px solid var(--border);
    }}
    
    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to {{ opacity: 1; }}
    }}
    
    .fade-in {{
        animation: fadeIn 0.5s ease-in;
    }}
    </style>
    """, unsafe_allow_html=True)

# ------------------------------
# Utility Functions
# ------------------------------

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_bg(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-attachment: fixed;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# ------------------------------
# Data Loading Functions
# ------------------------------

def load_kragle_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
        
        # Add data quality check
        if df.empty:
            st.error("Uploaded file is empty.")
            return None
            
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def fetch_live_market_data(tickers):
    try:
        data = yf.download(tickers, period="1d", interval="1m")
        if data.empty:
            st.error("No live data available for the selected tickers.")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching live market data: {str(e)}")
        return None

def fetch_historical_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.error("No data found for the given ticker and date range.")
            return None
        
        # Add technical indicators
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        data['RSI'] = compute_rsi(data['Close'])
        data['MACD'], data['Signal_Line'] = compute_macd(data['Close'])
        
        return data
    except Exception as e:
        st.error(f"Error fetching historical data: {str(e)}")
        return None

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, slow=26, fast=12, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# ------------------------------
# Data Preprocessing
# ------------------------------

def preprocess_data(df):
    try:
        # Create a copy to avoid SettingWithCopyWarning
        processed_df = df.copy()
        
        # Missing value handling
        missing_values = processed_df.isnull().sum()
        if missing_values.sum() > 0:
            with st.expander("Missing Value Handling"):
                st.write("Missing values before handling:")
                st.dataframe(missing_values[missing_values > 0])
                
                method = st.radio("Select imputation method:", 
                                ["Forward Fill", "Backward Fill", "Mean", "Median", "Drop"])
                
                if method == "Forward Fill":
                    processed_df = processed_df.fillna(method='ffill')
                elif method == "Backward Fill":
                    processed_df = processed_df.fillna(method='bfill')
                elif method == "Mean":
                    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
                    processed_df[numeric_cols] = processed_df[numeric_cols].fillna(processed_df[numeric_cols].mean())
                elif method == "Median":
                    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
                    processed_df[numeric_cols] = processed_df[numeric_cols].fillna(processed_df[numeric_cols].median())
                elif method == "Drop":
                    processed_df = processed_df.dropna()
                
                st.success(f"Applied {method} imputation method.")
                st.write("Missing values after handling:")
                st.dataframe(processed_df.isnull().sum()[processed_df.isnull().sum() > 0])
        
        # Outlier detection and handling
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            with st.expander("Outlier Detection & Handling"):
                st.write("Outlier detection using IQR method:")
                
                outlier_col = st.selectbox("Select column for outlier analysis:", numeric_cols)
                
                Q1 = processed_df[outlier_col].quantile(0.25)
                Q3 = processed_df[outlier_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = processed_df[(processed_df[outlier_col] < lower_bound) | 
                                       (processed_df[outlier_col] > upper_bound)]
                
                st.write(f"Found {len(outliers)} outliers in {outlier_col} (out of {len(processed_df)} records)")
                
                if len(outliers) > 0:
                    action = st.radio("Select outlier handling method:", 
                                    ["Cap Values", "Remove Outliers", "Keep Outliers"])
                    
                    if action == "Cap Values":
                        processed_df[outlier_col] = processed_df[outter_col].clip(lower_bound, upper_bound)
                        st.success(f"Capped outliers in {outlier_col} to IQR bounds.")
                    elif action == "Remove Outliers":
                        processed_df = processed_df[(processed_df[outlier_col] >= lower_bound) & 
                                                  (processed_df[outlier_col] <= upper_bound)]
                        st.success(f"Removed {len(outliers)} outliers from {outlier_col}.")
        
        # Feature scaling
        with st.expander("Feature Scaling"):
            scale_method = st.radio("Select scaling method:", 
                                  ["None", "Standardization", "Normalization"])
            
            if scale_method != "None":
                numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    if scale_method == "Standardization":
                        scaler = StandardScaler()
                        processed_df[numeric_cols] = scaler.fit_transform(processed_df[numeric_cols])
                        st.success("Applied StandardScaler to numeric features.")
                    elif scale_method == "Normalization":
                        scaler = MinMaxScaler()
                        processed_df[numeric_cols] = scaler.fit_transform(processed_df[numeric_cols])
                        st.success("Applied MinMaxScaler to numeric features.")
        
        # Date conversion
        date_cols = [col for col in processed_df.columns if 'date' in col.lower()]
        for col in date_cols:
            processed_df[col] = pd.to_datetime(processed_df[col])
            processed_df[f'{col}_year'] = processed_df[col].dt.year
            processed_df[f'{col}_month'] = processed_df[col].dt.month
            processed_df[f'{col}_day'] = processed_df[col].dt.day
            processed_df[f'{col}_dayofweek'] = processed_df[col].dt.dayofweek
        
        return processed_df
    except Exception as e:
        st.error(f"Error during preprocessing: {str(e)}")
        return None

# ------------------------------
# Feature Engineering
# ------------------------------

def feature_engineering(df, target_col=None):
    try:
        st.subheader("Feature Analysis & Selection")
        
        # Correlation analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            with st.expander("Correlation Analysis"):
                corr_method = st.selectbox("Select correlation method:", 
                                         ["Pearson", "Spearman", "Kendall"])
                
                if corr_method == "Pearson":
                    corr_matrix = df[numeric_cols].corr(method='pearson')
                elif corr_method == "Spearman":
                    corr_matrix = df[numeric_cols].corr(method='spearman')
                elif corr_method == "Kendall":
                    corr_matrix = df[numeric_cols].corr(method='kendall')
                
                fig = px.imshow(corr_matrix, 
                               text_auto=True, 
                               aspect="auto",
                               color_continuous_scale='RdBu_r',
                               title=f"{corr_method} Correlation Matrix")
                st.plotly_chart(fig, use_container_width=True)
                
                if target_col and target_col in numeric_cols:
                    target_corr = corr_matrix[target_col].sort_values(key=abs, ascending=False)
                    st.write("Correlation with target variable:")
                    st.dataframe(target_corr)
        
        # PCA for dimensionality reduction
        with st.expander("Dimensionality Reduction (PCA)"):
            if len(numeric_cols) > 2:
                n_components = st.slider("Number of PCA components", 2, min(10, len(numeric_cols)), 3)
                
                if st.button("Run PCA"):
                    pca = PCA(n_components=n_components)
                    pca_result = pca.fit_transform(df[numeric_cols].fillna(0))
                    
                    # Create a DataFrame with the PCA results
                    pca_df = pd.DataFrame(pca_result, columns=[f"PC{i+1}" for i in range(n_components)])
                    
                    # Visualize the explained variance
                    fig = px.bar(x=[f"PC{i+1}" for i in range(n_components)],
                                y=pca.explained_variance_ratio_,
                                labels={'x': 'Principal Component', 'y': 'Explained Variance'},
                                title="PCA Explained Variance Ratio")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Visualize first two components
                    if n_components >= 2:
                        fig = px.scatter(pca_df, x='PC1', y='PC2',
                                        title="PCA Components Visualization (PC1 vs PC2)")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    return pca_df.columns.tolist()
        
        # Feature selection
        with st.expander("Feature Selection"):
            if target_col:
                features = [col for col in numeric_cols if col != target_col]
            else:
                features = list(numeric_cols)
            
            # Automatic feature selection
            if target_col and len(features) > 1:
                k = st.slider("Select top k features", 1, len(features), min(5, len(features)))
                
                selector = SelectKBest(score_func=f_regression, k=k)
                X_new = selector.fit_transform(df[features], df[target_col])
                
                selected_features = [features[i] for i in selector.get_support(indices=True)]
                scores = selector.scores_[selector.get_support()]
                
                st.write("Top selected features based on statistical tests:")
                feature_scores = pd.DataFrame({'Feature': selected_features, 'Score': scores})
                feature_scores = feature_scores.sort_values('Score', ascending=False)
                st.dataframe(feature_scores)
                
                fig = px.bar(feature_scores, x='Score', y='Feature', orientation='h',
                            title="Feature Importance Scores")
                st.plotly_chart(fig, use_container_width=True)
                
                return selected_features
            else:
                st.warning("Target variable not selected or insufficient features for automatic selection.")
                return st.multiselect("Select features manually:", features, default=features)
    
    except Exception as e:
        st.error(f"Error during feature engineering: {str(e)}")
        return None

# ------------------------------
# Model Training & Evaluation
# ------------------------------

def train_model(model_type, X_train, X_test, y_train, y_test):
    try:
        with st.spinner(f"Training {model_type} model..."):
            start_time = time.time()
            
            if model_type == "Linear Regression":
                model = LinearRegression()
                param_grid = {
                    'fit_intercept': [True, False],
                    'normalize': [True, False]
                }
            elif model_type == "Random Forest":
                model = RandomForestRegressor(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 5, 10],
                    'min_samples_split': [2, 5]
                }
            elif model_type == "Gradient Boosting":
                model = GradientBoostingRegressor(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                }
            elif model_type == "K-Means Clustering":
                model = KMeans(random_state=42)
                param_grid = {
                    'n_clusters': range(2, 6)
                }
            
            # Hyperparameter tuning with GridSearchCV
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            # Training time
            training_time = time.time() - start_time
            
            # Make predictions
            y_pred = best_model.predict(X_test)
            
            # Model evaluation
            if model_type != "K-Means Clustering":
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Feature importance
                if hasattr(best_model, 'feature_importances_'):
                    feature_importance = pd.DataFrame({
                        'Feature': X_train.columns,
                        'Importance': best_model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                elif hasattr(best_model, 'coef_'):
                    feature_importance = pd.DataFrame({
                        'Feature': X_train.columns,
                        'Coefficient': best_model.coef_
                    }).sort_values('Coefficient', ascending=False)
                else:
                    feature_importance = None
                
                return {
                    'model': best_model,
                    'mse': mse,
                    'r2': r2,
                    'training_time': training_time,
                    'feature_importance': feature_importance,
                    'best_params': grid_search.best_params_
                }
            else:
                silhouette = silhouette_score(X_test, best_model.predict(X_test))
                return {
                    'model': best_model,
                    'silhouette_score': silhouette,
                    'training_time': training_time,
                    'best_params': grid_search.best_params_
                }
    
    except Exception as e:
        st.error(f"Error during model training: {str(e)}")
        return None

# ------------------------------
# Visualization Functions
# ------------------------------

def plot_actual_vs_predicted(y_test, y_pred, model_name):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_test,
        y=y_pred,
        mode='markers',
        name='Predictions',
        marker=dict(color='#1f77b4', size=8, opacity=0.7)
    ))
    
    fig.add_trace(go.Scatter(
        x=[y_test.min(), y_test.max()],
        y=[y_test.min(), y_test.max()],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='#ff7f0e', dash='dash')
    ))
    
    fig.update_layout(
        title=f"{model_name} - Actual vs Predicted",
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values",
        hovermode='closest',
        template='plotly_dark',
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_feature_importance(feature_importance, model_name):
    fig = px.bar(feature_importance, 
                 x='Importance' if 'Importance' in feature_importance.columns else 'Coefficient', 
                 y='Feature',
                 orientation='h',
                 title=f"{model_name} - Feature Importance",
                 color='Importance' if 'Importance' in feature_importance.columns else 'Coefficient',
                 color_continuous_scale='Viridis')
    
    fig.update_layout(
        template='plotly_dark',
        height=600,
        yaxis={'categoryorder':'total ascending'}
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_residuals(y_test, y_pred, model_name):
    residuals = y_test - y_pred
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode='markers',
        name='Residuals',
        marker=dict(color='#2ca02c', size=8, opacity=0.7)
    ))
    
    fig.add_trace(go.Scatter(
        x=[y_pred.min(), y_pred.max()],
        y=[0, 0],
        mode='lines',
        name='Zero Residual',
        line=dict(color='#d62728', dash='dash')
    ))
    
    fig.update_layout(
        title=f"{model_name} - Residual Analysis",
        xaxis_title="Predicted Values",
        yaxis_title="Residuals",
        hovermode='closest',
        template='plotly_dark',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_clusters(X, labels, model_name):
    if X.shape[1] >= 2:
        # If more than 2 features, use PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        fig = px.scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            color=labels,
            title=f"{model_name} - Cluster Visualization (PCA Reduced)",
            labels={'x': 'Principal Component 1', 'y': 'Principal Component 2'},
            color_continuous_scale='Viridis'
        )
    else:
        # If only 1 feature, plot histogram
        fig = px.histogram(
            x=X.iloc[:, 0],
            color=labels,
            title=f"{model_name} - Cluster Visualization",
            nbins=50,
            color_continuous_scale='Viridis'
        )
    
    fig.update_layout(
        template='plotly_dark',
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# Main Application
# ------------------------------

def main():
    # Set custom theme
    set_custom_theme()
    
    # Set page config with professional title and icon
    st.set_page_config(
        page_title="FinML Pro - Financial Machine Learning Platform",
        page_icon="💹",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'data_preprocessed' not in st.session_state:
        st.session_state.data_preprocessed = False
    if 'features_selected' not in st.session_state:
        st.session_state.features_selected = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1
    
    # ------------------------------
    # Sidebar Navigation
    # ------------------------------
    
    st.sidebar.title("FinML Pro")
    st.sidebar.markdown("---")
    
    # Progress tracker
    st.sidebar.subheader("Workflow Progress")
    steps = ["Data Loading", "Data Preprocessing", "Feature Engineering", "Model Training", "Results"]
    for i, step in enumerate(steps):
        if i+1 < st.session_state.current_step:
            st.sidebar.success(f"✓ {step}")
        elif i+1 == st.session_state.current_step:
            st.sidebar.info(f"➤ {step}")
        else:
            st.sidebar.write(f"○ {step}")
    
    st.sidebar.markdown("---")
    
    # Quick stats
    if st.session_state.data_loaded and st.session_state.df is not None:
        st.sidebar.subheader("Dataset Stats")
        st.sidebar.write(f"📊 Rows: {st.session_state.df.shape[0]}")
        st.sidebar.write(f"📈 Columns: {st.session_state.df.shape[1]}")
        st.sidebar.write(f"🔢 Numeric: {len(st.session_state.df.select_dtypes(include=[np.number]).columns)}")
        st.sidebar.write(f"📅 Date: {len([col for col in st.session_state.df.columns if 'date' in col.lower()])}")
    
    st.sidebar.markdown("---")
    
    # Market data ticker
    st.sidebar.subheader("Live Market Data")
    live_tickers = st.sidebar.text_input("Watchlist (comma separated)", "AAPL,MSFT,GOOG,AMZN")
    
    if st.sidebar.button("Refresh Market Data"):
        with st.spinner("Fetching live market data..."):
            live_data = fetch_live_market_data(live_tickers.split(','))
            if live_data is not None:
                st.session_state.live_data = live_data
    
    if 'live_data' in st.session_state and st.session_state.live_data is not None:
        last_prices = st.session_state.live_data.groupby(level=0).last()['Close']
        for ticker, price in last_prices.items():
            st.sidebar.metric(ticker, f"${price:.2f}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div class="footer">
        <p>FinML Pro v1.0</p>
        <p>© 2025 FAST-NUCES Financial Technology</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ------------------------------
    # Main Content Area
    # ------------------------------
    
    # Header with animated ticker tape
    st.markdown("""
    <div class="ticker-tape">
        <div class="ticker-item">📈 S&P 500: 4,567.89 (+1.2%)</div>
        <div class="ticker-item">💵 USD/GBP: 0.82 (-0.3%)</div>
        <div class="ticker-item">🛢️ Crude Oil: $78.45 (+2.1%)</div>
        <div class="ticker-item">🏦 10-Yr Yield: 3.45% (+0.05)</div>
        <div class="ticker-item">📱 NASDAQ: 14,256.34 (+0.8%)</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Current step display
    st.title(f"{steps[st.session_state.current_step-1]}")
    st.markdown("---")
    
    # ------------------------------
    # Data Loading Section
    # ------------------------------
    
    if st.session_state.current_step == 1:
        st.subheader("Load Financial Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="model-card">
                <h3>📁 Upload Dataset</h3>
                <p>Upload your financial dataset in CSV or Excel format from Kragle or other sources.</p>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"], key="file_uploader")
            
            if uploaded_file is not None:
                with st.spinner("Loading dataset..."):
                    df = load_kragle_data(uploaded_file)
                    if df is not None:
                        st.session_state.df = df
                        st.session_state.data_loaded = True
                        st.session_state.current_step = 2
                        st.success("Dataset loaded successfully!")
                        st.experimental_rerun()
        
        with col2:
            st.markdown("""
            <div class="model-card">
                <h3>🌐 Fetch Market Data</h3>
                <p>Download historical market data directly from Yahoo Finance API.</p>
            </div>
            """, unsafe_allow_html=True)
            
            ticker = st.text_input("Stock Ticker (e.g., AAPL)", "AAPL")
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
            end_date = st.date_input("End Date", datetime.now())
            
            if st.button("Fetch Data"):
                with st.spinner("Downloading market data..."):
                    df = fetch_historical_data(ticker, start_date, end_date)
                    if df is not None:
                        st.session_state.df = df
                        st.session_state.data_loaded = True
                        st.session_state.current_step = 2
                        st.success("Market data fetched successfully!")
                        st.experimental_rerun()
        
        # Sample data option
        st.markdown("---")
        st.subheader("Or try with sample data")
        
        sample_options = {
            "S&P 500 Stocks": "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv",
            "Cryptocurrency Prices": "https://raw.githubusercontent.com/plotly/datasets/master/crypto.csv",
            "Stock Market Data": "https://raw.githubusercontent.com/plotly/datasets/master/stockdata.csv"
        }
        
        sample_choice = st.selectbox("Select sample dataset", list(sample_options.keys()))
        
        if st.button("Load Sample Data"):
            with st.spinner(f"Loading {sample_choice}..."):
                try:
                    df = pd.read_csv(sample_options[sample_choice])
                    st.session_state.df = df
                    st.session_state.data_loaded = True
                    st.session_state.current_step = 2
                    st.success("Sample data loaded successfully!")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Failed to load sample data: {str(e)}")
    
    # ------------------------------
    # Data Preprocessing Section
    # ------------------------------
    
    elif st.session_state.current_step == 2:
        if not st.session_state.data_loaded:
            st.warning("Please load data first.")
            st.session_state.current_step = 1
            st.experimental_rerun()
        
        st.subheader("Data Cleaning & Preparation")
        
        # Show data summary
        st.markdown("""
        <div class="metric-card">
            <h4>Dataset Summary</h4>
            <p>Rows: {:,} | Columns: {:,} | Numeric: {:,} | Date: {:,}</p>
        </div>
        """.format(
            st.session_state.df.shape[0],
            st.session_state.df.shape[1],
            len(st.session_state.df.select_dtypes(include=[np.number]).columns),
            len([col for col in st.session_state.df.columns if 'date' in col.lower()])
        ), unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Preview:**")
            st.dataframe(st.session_state.df.head())
        
        with col2:
            st.write("**Data Types:**")
            dtype_df = pd.DataFrame(st.session_state.df.dtypes, columns=['Data Type'])
            st.dataframe(dtype_df)
        
        st.markdown("---")
        
        # Data preprocessing
        st.subheader("Data Preprocessing")
        
        if st.button("Run Data Preprocessing"):
            with st.spinner("Processing data..."):
                processed_df = preprocess_data(st.session_state.df)
                if processed_df is not None:
                    st.session_state.df = processed_df
                    st.session_state.data_preprocessed = True
                    st.session_state.current_step = 3
                    st.success("Data preprocessing completed!")
                    st.experimental_rerun()
        
        # Skip preprocessing option
        if st.button("Skip Preprocessing"):
            st.session_state.data_preprocessed = True
            st.session_state.current_step = 3
            st.warning("Skipped data preprocessing. Ensure your data is clean.")
            st.experimental_rerun()
    
    # ------------------------------
    # Feature Engineering Section
    # ------------------------------
    
    elif st.session_state.current_step == 3:
        if not st.session_state.data_preprocessed:
            st.warning("Please preprocess data first.")
            st.session_state.current_step = 2
            st.experimental_rerun()
        
        st.subheader("Feature Analysis & Selection")
        
        # Select target variable
        numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            target_col = st.selectbox("Select target variable (for supervised learning)", 
                                    [None] + list(numeric_cols))
            st.session_state.target_col = target_col
        else:
            st.warning("No numeric columns found for modeling.")
            target_col = None
        
        st.markdown("---")
        
        # Run feature engineering
        if st.button("Analyze Features"):
            with st.spinner("Analyzing features..."):
                selected_features = feature_engineering(st.session_state.df, target_col)
                if selected_features:
                    st.session_state.selected_features = selected_features
                    st.session_state.features_selected = True
                    st.session_state.current_step = 4
                    st.success("Feature engineering completed!")
                    st.experimental_rerun()
    
    # ------------------------------
    # Model Training Section
    # ------------------------------
    
    elif st.session_state.current_step == 4:
        if not st.session_state.features_selected:
            st.warning("Please complete feature engineering first.")
            st.session_state.current_step = 3
            st.experimental_rerun()
        
        st.subheader("Model Training")
        
        # Model selection
        model_options = {
            "Linear Regression": "For predicting continuous values",
            "Random Forest": "Powerful ensemble method for regression",
            "Gradient Boosting": "State-of-the-art for many tasks",
            "K-Means Clustering": "For unsupervised clustering"
        }
        
        st.write("**Select Model Type:**")
        model_type = st.radio("", list(model_options.keys()), 
                             format_func=lambda x: f"{x} - {model_options[x]}")
        
        st.markdown("---")
        
        # Train/test split
        st.subheader("Train/Test Split")
        
        col1, col2 = st.columns(2)
        
        with col1:
            test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
        
        with col2:
            random_state = st.number_input("Random State", value=42)
        
        st.markdown("---")
        
        # Model training
        if st.button("Train Model"):
            with st.spinner("Training in progress..."):
                # Prepare data
                X = st.session_state.df[st.session_state.selected_features]
                
                if st.session_state.target_col:
                    y = st.session_state.df[st.session_state.target_col]
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state
                    )
                else:
                    # For unsupervised learning
                    X_train, X_test = train_test_split(
                        X, test_size=test_size, random_state=random_state
                    )
                    y_train, y_test = None, None
                
                # Train model
                if st.session_state.target_col:
                    results = train_model(model_type, X_train, X_test, y_train, y_test)
                else:
                    results = train_model(model_type, X_train, X_test, X_train, X_test)
                
                if results:
                    st.session_state.model_results = results
                    st.session_state.model_trained = True
                    st.session_state.current_step = 5
                    st.success("Model training completed!")
                    st.experimental_rerun()
    
    # ------------------------------
    # Results Section
    # ------------------------------
    
    elif st.session_state.current_step == 5:
        if not st.session_state.model_trained:
            st.warning("Please train a model first.")
            st.session_state.current_step = 4
            st.experimental_rerun()
        
        st.subheader("Model Results")
        
        # Display model performance
        st.markdown("""
        <div class="metric-card">
            <h4>Model Performance</h4>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'mse' in st.session_state.model_results:
                st.metric("Mean Squared Error", f"{st.session_state.model_results['mse']:.4f}")
            
            if 'r2' in st.session_state.model_results:
                st.metric("R-squared Score", f"{st.session_state.model_results['r2']:.4f}")
            
            if 'silhouette_score' in st.session_state.model_results:
                st.metric("Silhouette Score", f"{st.session_state.model_results['silhouette_score']:.4f}")
        
        with col2:
            st.metric("Training Time", f"{st.session_state.model_results['training_time']:.2f} seconds")
            st.write("**Best Parameters:**")
            st.write(st.session_state.model_results['best_params'])
        
        st.markdown("---")
        
        # Visualizations
        if 'mse' in st.session_state.model_results:  # Supervised learning
            # Prepare test data predictions
            model = st.session_state.model_results['model']
            X_test = st.session_state.df[st.session_state.selected_features].iloc[
                -int(len(st.session_state.df) * 0.2):]  # Last 20% as test set
            y_test = st.session_state.df[st.session_state.target_col].iloc[
                -int(len(st.session_state.df) * 0.2):]
            y_pred = model.predict(X_test)
            
            # Actual vs Predicted
            plot_actual_vs_predicted(y_test, y_pred, model_type)
            
            # Residuals plot
            plot_residuals(y_test, y_pred, model_type)
            
            # Feature importance
            if st.session_state.model_results['feature_importance'] is not None:
                plot_feature_importance(
                    st.session_state.model_results['feature_importance'], 
                    model_type
                )
        
        else:  # Unsupervised learning
            # Cluster visualization
            X = st.session_state.df[st.session_state.selected_features]
            labels = st.session_state.model_results['model'].predict(X)
            plot_clusters(X, labels, model_type)
        
        st.markdown("---")
        
        # Download results
        st.subheader("Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export predictions
            if 'mse' in st.session_state.model_results:
                results_df = pd.DataFrame({
                    'Actual': y_test,
                    'Predicted': y_pred,
                    'Residual': y_test - y_pred
                })
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions (CSV)",
                    data=csv,
                    file_name="model_predictions.csv",
                    mime="text/csv"
                )
        
        with col2:
            # Export model summary
            report = f"""
            FinML Pro - Model Report
            =======================
            
            Model Type: {model_type}
            Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            Parameters:
            {st.session_state.model_results['best_params']}
            
            Performance Metrics:
            """
            
            if 'mse' in st.session_state.model_results:
                report += f"""
                - Mean Squared Error: {st.session_state.model_results['mse']:.4f}
                - R-squared Score: {st.session_state.model_results['r2']:.4f}
                """
            else:
                report += f"""
                - Silhouette Score: {st.session_state.model_results['silhouette_score']:.4f}
                """
            
            report += f"""
            Training Time: {st.session_state.model_results['training_time']:.2f} seconds
            
            Features Used:
            {', '.join(st.session_state.selected_features)}
            """
            
            if st.session_state.target_col:
                report += f"""
                Target Variable: {st.session_state.target_col}
                """
            
            st.download_button(
                label="Download Report (TXT)",
                data=report,
                file_name="model_report.txt",
                mime="text/plain"
            )
        
        # Restart workflow
        st.markdown("---")
        if st.button("Start New Analysis"):
            st.session_state.clear()
            st.session_state.current_step = 1
            st.experimental_rerun()

# ------------------------------
# Run the Application
# ------------------------------

if __name__ == "__main__":
    main()
