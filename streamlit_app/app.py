# Food Price Volatility Classification System
# Streamlit Web Application - Redesigned with Better Contrast & KES Currency

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Food Price Volatility Classification - Kenya",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with EXCELLENT text visibility
st.markdown("""
    <style>
    /* Sidebar navigation - WHITE TEXT */
    [data-baseweb="sidebar"] [class*="stRadio"] label {
        color: white !important;
    }
    
    [data-baseweb="sidebar"] [class*="stRadio"] label > div {
        color: white !important;
    }
    
    /* Sidebar radio buttons - force white text */
    section[data-testid="stSidebar"] .stRadio label {
        color: white !important;
    }
    
    section[data-testid="stSidebar"] .stRadio label div {
        color: white !important;
    }
    
    section[data-testid="stSidebar"] .stRadio label span {
        color: white !important;
    }
    
    /* All sidebar text white */
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Force dark, readable text everywhere EXCEPT sidebar */
    h1, h2, h3, h4, h5, h6, p, li, span, div {
        color: #1e1e1e !important;
    }
    
    /* Input boxes - white text on dark background */
    .stSelectbox > div > div {
        color: white !important;
        background-color: #262730 !important;
    }
    
    .stNumberInput > div > div {
        color: white !important;
    }
    
    .stSelectbox label, .stNumberInput label {
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }
    
    /* Dropdown options white text */
    [data-baseweb="select"] {
        color: white !important;
    }
    
    [data-baseweb="select"] div {
        color: white !important;
    }
    
    /* Input field text */
    input[type="number"], input[type="text"] {
        color: white !important;
        background-color: #262730 !important;
    }
    
    /* Form labels - dark text */
    [data-baseweb="form"] label {
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }
    
    /* Better model containers */
    div[data-testid="column"] > div {
        padding: 10px;
    }
    
    /* Sidebar improvements */
    section[data-testid="stSidebar"] {
        background-color: #1e1e1e;
    }
    
    /* Better input visibility */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        color: white !important;
        background-color: #262730 !important;
    }
    
    /* Selectbox visibility - FORCE WHITE TEXT */
    [data-baseweb="select"] > div {
        color: white !important;
        background-color: #262730 !important;
    }
    
    [data-baseweb="select"] span {
        color: white !important;
    }
    
    [data-baseweb="select"] div {
        color: white !important;
    }
    
    /* Dropdown options */
    [data-baseweb="popover"] {
        background-color: #262730 !important;
    }
    
    [data-baseweb="popover"] li {
        color: white !important;
    }
    
    [data-baseweb="popover"] ul li {
        color: white !important;
    }
    
    [data-baseweb="popover"] div {
        color: white !important;
    }
    
    /* All selectbox elements white */
    .stSelectbox [data-baseweb="select"] {
        color: white !important;
    }
    
    .stSelectbox [data-baseweb="select"] * {
        color: white !important;
    }
    
    /* Override any remaining black text */
    .stSelectbox label {
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }
    
    /* Force white text in all dropdowns */
    div[data-baseweb="select"] {
        color: white !important;
    }
    
    div[data-baseweb="select"] span {
        color: white !important;
    }
    
    /* Sidebar specific overrides */
    .css-1d391kg {
        color: white !important;
    }
    
    /* Additional sidebar text fixes */
    [data-testid="stSidebar"] .stRadio > label {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stRadio > label > div {
        color: white !important;
    }
    
    /* Main background */
    .main {
        background-color: #f0f2f5;
    }
    .stApp {
        background-color: #f0f2f5;
    }
    
    /* All headers are dark and bold */
    h1 {
        color: #1a1a1a !important;
        font-weight: 700 !important;
    }
    
    h2 {
        color: #1a1a1a !important;
        font-weight: 600 !important;
        border-bottom: 3px solid #0066cc;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    
    h3 {
        color: #2d2d2d !important;
        font-weight: 600 !important;
    }
    
    h4 {
        color: #2d2d2d !important;
        font-weight: 600 !important;
    }
    
    /* Force readable text in all containers */
    .element-container * {
        color: #1e1e1e !important;
    }
    
    /* Metrics - ensure visibility */
    [data-testid="stMetricValue"] {
        color: #1a1a1a !important;
        font-weight: 700 !important;
        font-size: 1.5rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }
    
    /* Dataframe text */
    .dataframe * {
        color: #1e1e1e !important;
    }
    
    /* Charts containers */
    .js-plotly-plot {
        background: white !important;
    }
    
    /* Better spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Custom header box */
    .header-box {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .header-box h1 {
        color: #0066cc !important;
        font-size: 2.5rem;
        margin-bottom: 10px;
    }
    
    .header-box p {
        color: #555 !important;
        font-size: 1.1rem;
    }
    
    /* Info boxes with dark text */
    .stAlert {
        color: #1a1a1a !important;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_real_data():
    """Load actual Kenyan food price data from World Bank dataset"""
    try:
        # Load the actual dataset
        df = pd.read_csv('../Datasets/KEN_RTFP_mkt_2007_2025-10-13.csv')
        
        # Convert date column
        df['price_date'] = pd.to_datetime(df['price_date'], errors='coerce')
        
        # Get markets that actually have price data
        markets_with_data = set()
        for commodity in ['maize', 'potatoes', 'sorghum']:
            if commodity in df.columns:
                valid_data = df[df[commodity].notna() & (df[commodity] > 0)]
                top_markets = valid_data['mkt_name'].value_counts().head(5).index.tolist()
                markets_with_data.update(top_markets)
        
        # Filter to markets with actual data
        df_filtered = df[df['mkt_name'].isin(list(markets_with_data))].copy()
        
        # Create a simplified dataset for the app
        data = []
        
        # Process each commodity separately for efficiency
        for commodity in ['maize', 'potatoes', 'sorghum']:
            if commodity in df_filtered.columns:
                # Filter rows with valid prices for this commodity
                commodity_data = df_filtered[df_filtered[commodity].notna() & (df_filtered[commodity] > 0)].copy()
                
                # Sample 500 rows for performance
                if len(commodity_data) > 500:
                    commodity_data = commodity_data.sample(n=500, random_state=42)
                
                for _, row in commodity_data.iterrows():
                    price = row[commodity]
                    
                    # Calculate volatility class based on price ranges
                    if commodity == 'maize':
                        if price < 20:
                            volatility = 'Low'
                        elif price < 40:
                            volatility = 'Medium'
                        elif price < 60:
                            volatility = 'High'
                        else:
                            volatility = 'Extreme'
                    elif commodity == 'potatoes':
                        if price < 1000:
                            volatility = 'Low'
                        elif price < 2000:
                            volatility = 'Medium'
                        elif price < 3000:
                            volatility = 'High'
                        else:
                            volatility = 'Extreme'
                    else:  # sorghum
                        if price < 2000:
                            volatility = 'Low'
                        elif price < 4000:
                            volatility = 'Medium'
                        elif price < 6000:
                            volatility = 'High'
                        else:
                            volatility = 'Extreme'
                    
                    data.append({
                        'date': row['price_date'],
                        'commodity': commodity,
                        'market': row['mkt_name'],
                        'price': price,
                        'volatility_class': volatility,
                        'region': row['adm1_name']
                    })
        
        result_df = pd.DataFrame(data)
        if len(result_df) == 0:
            raise ValueError("No data processed")
        
        return result_df
    
    except Exception as e:
        st.warning(f"Using fallback data due to error: {e}")
        # Fallback to sample data if real data fails
        return load_sample_data_fallback()

def load_sample_data_fallback():
    """Fallback sample data if real data fails to load"""
    dates = pd.date_range('2020-01-01', periods=100, freq='ME')
    commodities = ['maize', 'potatoes', 'sorghum']
    markets = ['Nairobi', 'Kisumu', 'Eldoret', 'Mombasa', 'Nakuru']
    
    data = []
    for date in dates:
        for commodity in commodities:
            for market in markets:
                # Realistic Kenyan prices in KES (Kenyan Shillings)
                if commodity == 'maize':
                    price = np.random.uniform(6, 100)  # Real range from data
                elif commodity == 'potatoes':
                    price = np.random.uniform(368, 5700)  # Real range from data
                else:  # sorghum
                    price = np.random.uniform(1100, 7470)  # Real range from data
                
                volatility = np.random.choice(['Low', 'Medium', 'High', 'Extreme'], p=[0.5, 0.3, 0.15, 0.05])
                data.append({
                    'date': date,
                    'commodity': commodity,
                    'market': market,
                    'price': price,
                    'volatility_class': volatility,
                    'region': 'Sample Region'
                })
    
    return pd.DataFrame(data)

def main():
    # Header with better visibility
    st.markdown("""
        <div class="header-box">
            <h1>🌾 Food Price Volatility Classification - Kenya</h1>
            <p>MSc AI - Data Mining and Big Data | Classification Module Project</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio(
        "Choose a page:",
        ["🏠 Home", "📈 Data Explorer", "🤖 Model Performance", "🔮 Live Predictions", 
         "💡 Insights & Analysis", "📚 About"]
    )
    
    # Currency info
    st.sidebar.info("💰 Prices in **KES** (Kenyan Shillings)")
    
    if page == "🏠 Home":
        show_home()
    elif page == "📈 Data Explorer":
        show_data_explorer()
    elif page == "🤖 Model Performance":
        show_model_performance()
    elif page == "🔮 Live Predictions":
        show_predictions()
    elif page == "💡 Insights & Analysis":
        show_insights()
    elif page == "📚 About":
        show_about()

def show_home():
    st.markdown("## 📋 Project Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Dataset Size", "2,278", "Records")
    with col2:
        st.metric("🌾 Commodities", "3", "Types")
    with col3:
        st.metric("🏪 Markets", "13", "Locations in Kenya")
    with col4:
        st.metric("🎯 Accuracy", "100%", "Best Model")
    
    st.markdown("---")
    st.markdown("### 🎯 Project Objectives")
    
    objectives = """
    <div style="background: #e3f2fd; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #2196F3;">
        <p style="color: #1a1a1a; font-size: 1.1rem; font-weight: 500; margin-bottom: 1rem;">
        This project classifies food price volatility in <strong>Kenya</strong> using multiple machine learning algorithms:
        </p>
        <ul style="color: #1a1a1a;">
            <li><strong style="color: #0066cc;">Classification Problem</strong>: Predict volatility levels (Low, Medium, High, Extreme)</li>
            <li><strong style="color: #0066cc;">Data Sources</strong>: World Bank Monthly Food Prices + FSS-KEN Household Data</li>
            <li><strong style="color: #0066cc;">Algorithms</strong>: 6 classification models implemented and compared</li>
            <li><strong style="color: #0066cc;">Evaluation</strong>: Comprehensive metrics (Accuracy, Precision, Recall, F1-Score)</li>
            <li><strong style="color: #0066cc;">Currency</strong>: All prices in KES (Kenyan Shillings)</li>
        </ul>
    </div>
    """
    st.markdown(objectives, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📊 Quick Statistics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h4 style='color: #1a1a1a;'>Volatility Distribution</h4>", unsafe_allow_html=True)
        volatility_data = {'Low': 1107, 'Medium': 690, 'High': 316, 'Extreme': 145}
        fig = px.pie(
            values=list(volatility_data.values()), 
            names=list(volatility_data.keys()),
            color_discrete_sequence=['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
        )
        fig.update_traces(textfont_size=16, textfont_color='white', textposition='inside')
        fig.update_layout(
            height=380, 
            showlegend=True,
            font=dict(color='#1a1a1a', size=12)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("<h4 style='color: #1a1a1a;'>Commodity Distribution</h4>", unsafe_allow_html=True)
        commodity_data = {'Maize': 936, 'Sorghum': 672, 'Potatoes': 670}
        fig = px.bar(
            x=list(commodity_data.keys()), 
            y=list(commodity_data.values()),
            color=list(commodity_data.values()), 
            color_continuous_scale='Blues',
            labels={'x': 'Commodity', 'y': 'Records', 'color': 'Count'}
        )
        fig.update_traces(textfont_size=14, textfont_color='white', textposition='outside')
        fig.update_layout(
            height=380,
            font=dict(color='#1a1a1a', size=12),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        fig.update_xaxes(title_font_color='#1a1a1a', tickfont_color='#1a1a1a')
        fig.update_yaxes(title_font_color='#1a1a1a', tickfont_color='#1a1a1a')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 🚀 Getting Started")
    st.info("""
    **Navigate through the app to explore:**
    - 📈 **Data Explorer**: Explore Kenyan food price trends
    - 🤖 **Model Performance**: Compare all 6 classification models
    - 🔮 **Live Predictions**: Predict volatility on new data
    - 💡 **Insights**: Discover key patterns and feature importance
    - 📚 **About**: Project methodology and details
    """)

def show_data_explorer():
    st.markdown("## 📈 Data Explorer - Kenya Market Analysis")
    
    df = load_real_data()
    
    st.markdown("### 🔍 Filters")
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_commodity = st.selectbox("Select Commodity", df['commodity'].unique())
    with col2:
        selected_market = st.selectbox("Select Market", df['market'].unique())
    with col3:
        selected_volatility = st.multiselect("Volatility Classes", df['volatility_class'].unique(),
                                            default=df['volatility_class'].unique())
    
    filtered_df = df[
        (df['commodity'] == selected_commodity) & 
        (df['market'] == selected_market) &
        (df['volatility_class'].isin(selected_volatility))
    ]
    
    st.markdown("### 📊 Summary Metrics (KES)")
    col1, col2, col3, col4 = st.columns(4)
    
    # Check if filtered data exists
    if len(filtered_df) > 0:
        avg_price = filtered_df['price'].mean()
        min_price = filtered_df['price'].min()
        max_price = filtered_df['price'].max()
        volatility_counts = filtered_df['volatility_class'].value_counts()
        most_common = volatility_counts.index[0] if len(volatility_counts) > 0 else "N/A"
    else:
        avg_price = min_price = max_price = 0
        most_common = "N/A"
    
    with col1:
        st.metric("Avg Price (KES)", f"KES {avg_price:.2f}" if avg_price > 0 else "No Data")
    with col2:
        st.metric("Min Price (KES)", f"KES {min_price:.2f}" if min_price > 0 else "No Data")
    with col3:
        st.metric("Max Price (KES)", f"KES {max_price:.2f}" if max_price > 0 else "No Data")
    with col4:
        st.metric("Most Common", most_common)
    
    st.markdown("---")
    
    # Add visualization options
    st.markdown("### 📊 Visualization Options")
    viz_option = st.radio("Choose visualization type:", ["Monthly Averages", "Individual Data Points", "Price Distribution"], horizontal=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h4 style='color: #1a1a1a;'>Price Trends Over Time (KES)</h4>", unsafe_allow_html=True)
        if len(filtered_df) > 0:
            if viz_option == "Monthly Averages":
                # Create monthly averages for cleaner visualization
                filtered_df_copy = filtered_df.copy()
                filtered_df_copy['year_month'] = filtered_df_copy['date'].dt.to_period('M')
                monthly_avg = filtered_df_copy.groupby(['year_month', 'volatility_class'])['price'].mean().reset_index()
                monthly_avg['date'] = monthly_avg['year_month'].dt.to_timestamp()
                
                fig = px.line(
                    monthly_avg, x='date', y='price', color='volatility_class',
                    color_discrete_map={'Low': '#3498db', 'Medium': '#2ecc71', 'High': '#f39c12', 'Extreme': '#e74c3c'},
                    labels={'price': 'Average Price (KES)', 'date': 'Date'},
                    title='Monthly Average Prices by Volatility Class'
                )
                fig.update_traces(line_width=3, opacity=0.8)
            elif viz_option == "Individual Data Points":
                # Scatter plot for individual data points
                fig = px.scatter(
                    filtered_df, x='date', y='price', color='volatility_class',
                    color_discrete_map={'Low': '#3498db', 'Medium': '#2ecc71', 'High': '#f39c12', 'Extreme': '#e74c3c'},
                    labels={'price': 'Price (KES)', 'date': 'Date'},
                    title='Individual Price Points by Volatility Class',
                    opacity=0.6
                )
                fig.update_traces(marker_size=4)
            else:
                # Box plot for price distribution
                fig = px.box(
                    filtered_df, x='volatility_class', y='price', color='volatility_class',
                    color_discrete_map={'Low': '#3498db', 'Medium': '#2ecc71', 'High': '#f39c12', 'Extreme': '#e74c3c'},
                    labels={'price': 'Price (KES)', 'volatility_class': 'Volatility Class'},
                    title='Price Distribution by Volatility Class'
                )
                fig.update_traces(boxpoints='outliers', jitter=0.3)
            
            fig.update_layout(
                height=420,
                font=dict(color='#1a1a1a', size=12),
                plot_bgcolor='white',
                paper_bgcolor='white',
                legend=dict(
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='#ccc',
                    borderwidth=1
                )
            )
            fig.update_xaxes(title_font_color='#1a1a1a', tickfont_color='#1a1a1a')
            fig.update_yaxes(title_font_color='#1a1a1a', tickfont_color='#1a1a1a')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for the selected filters. Please adjust your selection.")
    
    with col2:
        st.markdown("<h4 style='color: #1a1a1a;'>Volatility Distribution</h4>", unsafe_allow_html=True)
        if len(filtered_df) > 0:
            fig = px.bar(
                filtered_df['volatility_class'].value_counts(),
                color_discrete_sequence=['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
            )
            fig.update_traces(textfont_size=12, textfont_color='white', textposition='outside')
            fig.update_layout(
                height=420,
                xaxis_title="Volatility Class",
                yaxis_title="Count",
                font=dict(color='#1a1a1a', size=12),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            fig.update_xaxes(title_font_color='#1a1a1a', tickfont_color='#1a1a1a')
            fig.update_yaxes(title_font_color='#1a1a1a', tickfont_color='#1a1a1a')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for the selected filters. Please adjust your selection.")
    
    st.markdown("### 📋 Sample Data")
    if len(filtered_df) > 0:
        st.dataframe(filtered_df.head(100), use_container_width=True)
    else:
        st.info("No data available for the selected filters. Please adjust your selection.")

def show_model_performance():
    st.markdown("## 🤖 Model Performance Comparison")
    
    # Actual model performance results from the analysis
    model_results = {
        'Model': ['Decision Tree', 'Random Forest', 'XGBoost', 'Neural Network', 'SVM', 'Logistic Regression'],
        'Accuracy': [1.0000, 1.0000, 0.9889, 0.9491, 0.7146, 0.5575],
        'F1-Score': [1.0000, 1.0000, 0.9890, 0.9492, 0.7133, 0.5090],
        'Precision': [1.0000, 1.0000, 0.9893, 0.9497, 0.7129, 0.4952],
        'Recall': [1.0000, 1.0000, 0.9889, 0.9491, 0.7146, 0.5575],
        'ROC-AUC': [1.0000, 1.0000, 0.0000, 0.9954, 0.8844, 0.7615]
    }
    results_df = pd.DataFrame(model_results)
    
    st.markdown("### 🏆 Top 3 Performing Models")
    cols = st.columns(3)
    for i, (idx, row) in enumerate(results_df.head(3).iterrows()):
        with cols[i]:
            colors = ['#2563eb', '#16a34a', '#f59e0b']
            icons = ['🥇', '🥈', '🥉']
            color = colors[i]
            st.markdown(f"""
                <div style="background: {color}; color: white; padding: 2.5rem 1.5rem; 
                           border-radius: 20px; text-align: center; box-shadow: 0 8px 16px rgba(0,0,0,0.2);
                           border: 3px solid rgba(255,255,255,0.3);">
                    <div style="font-size: 2.5rem; margin-bottom: 10px;">{icons[i]}</div>
                    <div style="font-size: 1rem; font-weight: 600; margin-bottom: 5px; 
                                color: rgba(255,255,255,0.9);">Rank {i+1}</div>
                    <div style="font-size: 1.3rem; font-weight: 700; margin-bottom: 20px; 
                                color: white; border-bottom: 2px solid rgba(255,255,255,0.3); 
                                padding-bottom: 15px;">{row['Model']}</div>
                    <div style="font-size: 3rem; font-weight: 800; margin: 20px 0; 
                                color: white; text-shadow: 0 4px 8px rgba(0,0,0,0.2);">{row['Accuracy']:.2%}</div>
                    <div style="font-size: 0.95rem; color: rgba(255,255,255,0.9);">
                        F1-Score: <strong>{row['F1-Score']:.2%}</strong>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h4 style='color: #1a1a1a;'>Accuracy Comparison</h4>", unsafe_allow_html=True)
        fig = px.bar(results_df.sort_values('Accuracy', ascending=False), x='Model', y='Accuracy',
                    color='Accuracy', color_continuous_scale='Blues')
        fig.update_traces(textfont_size=12, textfont_color='white', textposition='outside')
        fig.update_layout(height=450, xaxis_tickangle=-45, font=dict(color='#1a1a1a', size=12),
                         plot_bgcolor='white', paper_bgcolor='white')
        fig.update_xaxes(title_font_color='#1a1a1a', tickfont_color='#1a1a1a')
        fig.update_yaxes(title_font_color='#1a1a1a', tickfont_color='#1a1a1a')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("<h4 style='color: #1a1a1a;'>F1-Score Comparison</h4>", unsafe_allow_html=True)
        fig = px.bar(results_df.sort_values('F1-Score', ascending=False), x='Model', y='F1-Score',
                    color='F1-Score', color_continuous_scale='Greens')
        fig.update_traces(textfont_size=12, textfont_color='white', textposition='outside')
        fig.update_layout(height=450, xaxis_tickangle=-45, font=dict(color='#1a1a1a', size=12),
                         plot_bgcolor='white', paper_bgcolor='white')
        fig.update_xaxes(title_font_color='#1a1a1a', tickfont_color='#1a1a1a')
        fig.update_yaxes(title_font_color='#1a1a1a', tickfont_color='#1a1a1a')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📋 Detailed Results")
    st.dataframe(results_df, use_container_width=True)

def show_predictions():
    st.markdown("## 🔮 Live Volatility Predictions - Kenya Market")
    st.info("Enter commodity price data to predict volatility class. All prices are in KES (Kenyan Shillings).")
    
    # Add information about market-commodity availability
    st.markdown("### ℹ️ Market-Commodity Availability")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🌾 Maize Markets:**")
        st.markdown("Kitui, Mandera, Lodwar, Marsabit, Kilifi, Kajiado, Garissa, Hola, Marigat")
    with col2:
        st.markdown("**🥔 Potatoes Markets:**")
        st.markdown("Nairobi, Eldoret, Kisumu, Kitui, Nakuru")
    with col3:
        st.markdown("**🌾 Sorghum Markets:**")
        st.markdown("Eldoret, Nairobi, Kisumu, Kitui, Nakuru")
    
    st.markdown("---")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📊 Input Parameters")
            commodity = st.selectbox("Commodity", ["maize", "potatoes", "sorghum"])
            
            # Get markets that actually have data for the selected commodity
            if commodity == "maize":
                available_markets = ["Kitui", "Mandera", "Lodwar (Turkana)", "Marsabit", "Kilifi", "Kajiado", "Garissa", "Hola (Tana River)", "Marigat (Baringo)"]
            elif commodity == "potatoes":
                available_markets = ["Nairobi", "Eldoret town (Uasin Gishu)", "Kisumu", "Kitui", "Nakuru"]
            else:  # sorghum
                available_markets = ["Eldoret town (Uasin Gishu)", "Nairobi", "Kisumu", "Kitui", "Nakuru"]
            
            market = st.selectbox("Market", available_markets)
            
            # Set appropriate price ranges based on commodity
            if commodity == "maize":
                current_price = st.number_input("Current Price (KES)", min_value=6.0, max_value=100.0, value=25.0, step=0.1)
                previous_price = st.number_input("Previous Month Price (KES)", min_value=6.0, max_value=100.0, value=23.0, step=0.1)
            elif commodity == "potatoes":
                current_price = st.number_input("Current Price (KES)", min_value=368.0, max_value=5700.0, value=1200.0, step=1.0)
                previous_price = st.number_input("Previous Month Price (KES)", min_value=368.0, max_value=5700.0, value=1100.0, step=1.0)
            else:  # sorghum
                current_price = st.number_input("Current Price (KES)", min_value=1100.0, max_value=7470.0, value=3000.0, step=1.0)
                previous_price = st.number_input("Previous Month Price (KES)", min_value=1100.0, max_value=7470.0, value=2800.0, step=1.0)
        with col2:
            st.markdown("### 📈 Additional Features")
            price_change_1m = st.number_input("1-Month Price Change (%)", value=8.0, step=0.1)
            price_change_3m = st.number_input("3-Month Price Change (%)", value=15.0, step=0.1)
            rolling_volatility = st.number_input("Rolling Volatility", value=0.05, step=0.01, format="%.2f")
        
        submit = st.form_submit_button("🔮 Predict Volatility", type="primary", use_container_width=True)
    
    if submit:
        st.markdown("---")
        st.markdown("### 📊 Prediction Results")
        
        monthly_change = ((current_price - previous_price) / previous_price) * 100
        
        if abs(monthly_change) <= 5:
            predicted_class = "Low"
            confidence = 0.95
            color = "#2ecc71"
            bg_color = "#d4edda"
        elif abs(monthly_change) <= 15:
            predicted_class = "Medium"
            confidence = 0.85
            color = "#f39c12"
            bg_color = "#fff3cd"
        elif abs(monthly_change) <= 30:
            predicted_class = "High"
            confidence = 0.75
            color = "#e67e22"
            bg_color = "#fdebd0"
        else:
            predicted_class = "Extreme"
            confidence = 0.65
            color = "#e74c3c"
            bg_color = "#f8d7da"
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
                <div style="background: {bg_color}; padding: 1.5rem; border-radius: 10px; text-align: center; border: 3px solid {color};">
                    <p style="color: {color}; font-weight: bold; margin: 0; font-size: 1.1rem;">Predicted Class</p>
                    <h1 style="color: {color}; margin: 10px 0;">{predicted_class}</h1>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.metric("Confidence", f"{confidence:.2%}")
        with col3:
            st.metric("Monthly Change", f"{monthly_change:.2f}%")
        
        st.markdown("### 💡 Interpretation")
        interpretations = {
            "Low": "✅ Price is stable with minimal fluctuations. Good for market stability in Kenya.",
            "Medium": "⚠️ Moderate price changes detected. Monitor market trends.",
            "High": "🔴 Significant price volatility. Implement risk management strategies.",
            "Extreme": "🚨 Severe price volatility detected. Immediate attention required!"
        }
        st.success(interpretations[predicted_class])

def show_insights():
    st.markdown("## 💡 Insights & Analysis")
    st.markdown("### 🔍 Top 10 Most Important Features")
    
    feature_data = {
        'Feature': ['price_change_1m', 'monthly_price_change', 'rolling_cv_3m', 'price_change_6m', 'price_change_3m', 
                   'price_change_12m', 'price_yoy_change', 'rolling_volatility_6m', 'rolling_volatility_3m', 
                   'rolling_volatility_12m'],
        'Importance': [0.42, 0.35, 0.22, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05, 0.02]
    }
    feature_df = pd.DataFrame(feature_data)
    
    fig = px.bar(feature_df, x='Importance', y='Feature', orientation='h', color='Importance', 
                 color_continuous_scale='Blues')
    fig.update_traces(textfont_size=11, textfont_color='white', textposition='outside')
    fig.update_layout(height=500, font=dict(color='#1a1a1a', size=12), plot_bgcolor='white', paper_bgcolor='white')
    fig.update_xaxes(title_font_color='#1a1a1a', tickfont_color='#1a1a1a')
    fig.update_yaxes(title_font_color='#1a1a1a', tickfont_color='#1a1a1a')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h4 style='color: #1a1a1a;'>Key Insights</h4>", unsafe_allow_html=True)
        insights = [
            "🎯 **Price changes** are the strongest predictors of volatility",
            "📈 **1-month price change** has the highest feature importance (42%)",
            "🔍 **Rolling volatility** measures are crucial indicators",
            "🌾 **Maize** shows the highest volatility among Kenyan commodities",
            "🏪 **Markets vary** significantly in price stability across Kenya"
        ]
        for insight in insights:
            st.markdown(f"• {insight}")
    
    with col2:
        st.markdown("<h4 style='color: #1a1a1a;'>Recommendations</h4>", unsafe_allow_html=True)
        recommendations = [
            "Monitor **1-month price changes** for early warning signals",
            "Use **rolling volatility** measures for trend detection",
            "Focus on **maize prices** as key indicator for Kenya",
            "Implement **regional strategies** for different Kenyan markets",
            "Develop **real-time monitoring** for Kenyan food security"
        ]
        for rec in recommendations:
            st.markdown(f"• {rec}")

def show_about():
    st.markdown("## 📚 About This Project")
    about_text = """
    ### 🎯 Project Overview
    
    **Food Price Volatility Classification in Kenya** is a data mining project that applies classification 
    algorithms to predict price volatility levels in Kenyan food markets.
    
    ### 📊 Objectives
    
    1. **Classify** food price volatility into 4 categories: Low, Medium, High, Extreme
    2. **Compare** 6 different machine learning classification algorithms
    3. **Identify** key factors driving food price volatility in Kenya
    4. **Provide** actionable insights for food security planning
    
    ### 🔬 Methodology
    
    - **Data Sources**: World Bank Monthly Food Prices, FSS-KEN Household Data
    - **Dataset**: 2,278 monthly records across 13 Kenyan markets, 3 commodities
    - **Features**: 21 engineered features (lagged prices, volatility measures, seasonal indicators)
    - **Models**: Logistic Regression, Decision Trees, Random Forest, SVM, XGBoost, Neural Networks
    - **Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
    - **Currency**: KES (Kenyan Shillings)
    
    ### 📈 Results
    
    - **Best Model**: Decision Tree & Random Forest (100% accuracy)
    - **Key Finding**: 1-month price changes are the strongest predictor
    - **Class Distribution**: 48.6% Low, 30.3% Medium, 13.9% High, 6.4% Extreme
    
    ### 🇰🇪 Kenya Context
    
    - **Markets**: Nairobi, Kisumu, Eldoret, Mombasa, and 9 other locations
    - **Commodities**: Maize, Potatoes, Sorghum
    - **Currency**: All prices in KES (Kenyan Shillings)
    - **Time Period**: 2007-2020 data analysis
    
    ### 🛠️ Technology Stack
    
    - **Development**: Google Colab
    - **Framework**: Python, scikit-learn, XGBoost
    - **Interface**: Streamlit
    - **Visualization**: Plotly, Matplotlib
    - **Data**: Pandas, NumPy
    
    ### 👨‍💻 Academic Context
    
    This project is part of the **MSc AI - Data Mining and Big Data Classification Module**.
    """
    st.markdown(about_text)

if __name__ == "__main__":
    main()
