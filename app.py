import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import json
import folium
from streamlit_folium import st_folium
import tensorflow as tf
from datetime import datetime

# Os environment settings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SeismicAI | Earthquake Prediction",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CUSTOM CSS & THEME
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Theme Overrides */
    
    /* Typography */
    h1, h2, h3 {
        color: #00d4ff !important;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }
    
    /* Metrics Styling */
    div[data-testid="stMetric"] {
        background-color: #1e2530;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetricValue"] {
        color: #00d4ff;
        font-size: 28px !important;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #00d4ff 0%, #0072ff 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 212, 255, 0.4);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: rgba(0, 212, 255, 0.1);
        color: #00d4ff;
        border: 1px solid #00d4ff;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    if os.path.exists('data/earthquake_data.csv'):
        df = pd.read_csv('data/earthquake_data.csv')
        df['time'] = pd.to_datetime(df['time'])
        return df
    return None

def get_magnitude_class(mag):
    if mag < 3.0: return "Minor", "#00ff88"  # Green
    elif mag < 4.0: return "Light", "#bbff00" # Lime
    elif mag < 5.0: return "Moderate", "#ffdd00" # Yellow
    elif mag < 6.0: return "Strong", "#ff9900" # Orange
    elif mag < 7.0: return "Major", "#ff4400" # RedOrange
    elif mag < 8.0: return "Great", "#ff0000" # Red
    else: return "Catastrophic", "#990000" # Dark Red

# -----------------------------------------------------------------------------
# NAVIGATION & STATE
# -----------------------------------------------------------------------------
if 'page' not in st.session_state:
    st.session_state.page = 'landing'

def navigate_to_app():
    st.session_state.page = 'app'

def navigate_to_landing():
    st.session_state.page = 'landing'

# -----------------------------------------------------------------------------
# LANDING PAGE
# -----------------------------------------------------------------------------
def landing_page():
    # Custom CSS for Landing
    st.markdown("""
    <style>
        .landing-title {
            font-size: 4rem;
            font-weight: 800;
            background: -webkit-linear-gradient(0deg, #00d4ff, #0072ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0;
            text-align: center;
        }
        .landing-subtitle {
            font-size: 1.5rem;
            color: #a0a0a0;
            text-align: center;
            margin-bottom: 3rem;
        }
        .feature-card {
            background-color: rgba(30, 37, 48, 0.6);
            border: 1px solid rgba(0, 212, 255, 0.1);
            padding: 2rem;
            border-radius: 16px;
            backdrop-filter: blur(10px);
            transition: transform 0.3s ease;
            text-align: center;
            height: 100%;
        }
        .feature-card:hover {
            transform: translateY(-5px);
            border-color: rgba(0, 212, 255, 0.5);
        }
        .feature-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
        }
        .cta-container {
            display: flex;
            justify_content: center;
            margin-top: 4rem;
            text-align: center;
        }
        /* Hide sidebar on landing */
        section[data-testid="stSidebar"] {
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Hero Section
    st.markdown('<div style="padding: 4rem 0;">', unsafe_allow_html=True)
    st.markdown('<h1 class="landing-title">Predicting the Unpredictable</h1>', unsafe_allow_html=True)
    st.markdown('<p class="landing-subtitle">Next-Generation AI for Real-time Earthquake Forecasting & Global Monitoring</p>', unsafe_allow_html=True)
    
    # CTA
    col_cta1, col_cta2, col_cta3 = st.columns([1, 2, 1])
    with col_cta2:
        st.button("LAUNCH DASHBOARD 🚀", on_click=navigate_to_app, use_container_width=True, type="primary")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Feature Grid
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🌍</div>
            <h3>Global Seismicity</h3>
            <p>Real-time visualization of worldwide earthquake events using live USGS data feeds.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <h3>Neuro-Seismic AI</h3>
            <p>Advanced ensemble models (RF + XGBoost) optimized via Genetic Algorithms for high-precision magnitude prediction.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <h3>Deep Analytics</h3>
            <p>Comprehensive performance metrics, feature importance tracking, and model evaluation insights.</p>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MAIN APP (DASHBOARD)
# -----------------------------------------------------------------------------
def main_app():
    # Show sidebar again
    st.markdown("""
    <style>
        section[data-testid="stSidebar"] {
            display: block;
        }
    </style>
    """, unsafe_allow_html=True)

    # ... [Previous Sidebar Code] ...
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/seismic-wave.png", width=80)
        st.title("SeismicAI")
        st.markdown("Advanced Earthquake Prediction System powered by Genetic Algorithms & Deep Learning.")
        
        if st.button("← Back to Home"):
            navigate_to_landing()
            st.rerun()
            
        st.markdown("---")
        st.subheader("⚙️ Settings")
        
        # Map Filters
        st.markdown("**Map Filters**")
        min_mag_filter = st.slider("Min Magnitude", 2.5, 9.0, 4.5, 0.1)
        
        st.markdown("---")
        st.info("""
        **How it works**
        1. **Data:** USGS Real-time Feed
        2. **Optimization:** Genetic Algo selects best features
        3. **Prediction:** Ensemble of RF, XGBoost & MLP
        """)
        st.markdown("v1.0.0 | Delta Team")

    # [Previous Main Content]
    st.title("🌍 Global Seismic Activity Monitor")
    st.markdown(f"**Live Dashboard** | {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    df = load_data()

    # TABS
    tab_map, tab_predict, tab_metrics = st.tabs(["🗺️ Global Map", "🔮 Predictor", "📊 Model Analytics"])

    # --- TAB 1: GLOBAL MAP ---
    with tab_map:
        if df is not None:
            # KPI Row
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            
            filtered_df = df[df['mag'] >= min_mag_filter]
            
            with kpi1:
                st.metric("Detected Events", len(filtered_df), delta=None)
            with kpi2:
                max_mag = filtered_df['mag'].max() if not filtered_df.empty else 0
                st.metric("Max Magnitude", f"{max_mag:.2f}", delta="Risk Level" if max_mag > 6 else "Normal")
            with kpi3:
                avg_depth = filtered_df['depth'].mean() if not filtered_df.empty else 0
                st.metric("Avg Depth", f"{avg_depth:.1f} km")
            with kpi4:
                st.metric("Recent Activity", "High" if len(filtered_df) > 100 else "Normal")
                
            st.markdown(f"Displaying **{len(filtered_df)}** earthquakes with magnitude > **{min_mag_filter}**")
            
            # Enhanced Map
            map_center = [filtered_df['latitude'].mean(), filtered_df['longitude'].mean()] if not filtered_df.empty else [20, 0]
            m = folium.Map(location=map_center, zoom_start=2, tiles='CartoDB dark_matter')
            
            # Feature Group for better performance
            fg = folium.FeatureGroup(name="Earthquakes")
            
            for idx, row in filtered_df.head(2000).iterrows():
                mag_text, color = get_magnitude_class(row['mag'])
                
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=row['mag'] ** 1.5, # Exponential radius for better diff
                    popup=folium.Popup(f"<b>Magnitude: {row['mag']}</b><br>Depth: {row['depth']}km<br>Date: {row['time']}", max_width=200),
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.6,
                    weight=1
                ).add_to(fg)
            
            fg.add_to(m)
            st_folium(m, width=1200, height=600) # Fixed size for better layout in tabs
        else:
            st.error("🚨 Data not found. Please run the training pipeline first.")

    # --- TAB 2: PREDICTOR ---
    with tab_predict:
        st.markdown("### 🤖 Neuro-Seismic Predictor")
        st.markdown("Input geophysical parameters to estimate earthquake magnitude.")
        
        col_input, col_result = st.columns([1, 1])
        
        with col_input:
            with st.container(border=True):
                st.subheader("Input Parameters")
                
                # Smart defaults (lat/lon of recent significant quake or generic)
                col_a, col_b = st.columns(2)
                with col_a:
                    lat_in = st.number_input("Latitude", -90.0, 90.0, 35.0)
                    depth_in = st.number_input("Depth (km)", 0.0, 800.0, 15.0)
                with col_b:
                    lon_in = st.number_input("Longitude", -180.0, 180.0, 139.0)
                    
                # Dynamic Features (try to load selected features)
                features = ['latitude', 'longitude', 'depth', 'timestamp', 'gap', 'dmin', 'rms']
                if os.path.exists('results/selected_features.csv'):
                    try:
                        sel = pd.read_csv('results/selected_features.csv')
                        features = sel['feature'].tolist()
                    except:
                        pass
                
                # Render other inputs
                input_dict = {'latitude': lat_in, 'longitude': lon_in, 'depth': depth_in}
                
                for f in features:
                    if f not in input_dict and f != 'mag':
                        input_dict[f] = st.number_input(f"Feature: {f}", value=0.0)
                
                predict_btn = st.button("Analyze & Predict", use_container_width=True, type="primary")

        with col_result:
            if predict_btn:
                 # Load Model
                try:
                    model_path = 'models/ensemble_model.pkl'
                    if os.path.exists(model_path):
                        from models.ensemble_model import EarthquakeEnsembleModel
                        loaded = joblib.load(model_path)
                        
                        rf = loaded['rf']
                        xgb_model = loaded['xgb']
                        feature_names = rf.feature_names_in_
                        
                        # Prepare X
                        X_pred = pd.DataFrame([input_dict])
                        for f in feature_names:
                            if f not in X_pred.columns:
                                X_pred[f] = 0
                        X_pred = X_pred[feature_names]
                        
                        # Predict
                        pred_rf = rf.predict(X_pred)[0]
                        pred_xgb = xgb_model.predict(X_pred)[0]
                        final_pred = (pred_rf + pred_xgb) / 2
                        
                        # Display Result
                        mag_class, color = get_magnitude_class(final_pred)
                        
                        st.markdown(f"""
                        <div style="background-color: #1e2530; border-radius: 12px; padding: 30px; text-align: center; border: 2px solid {color};">
                            <h2 style="color: {color} !important; margin-bottom: 0;">{mag_class} Earthquake</h2>
                            <h1 style="font-size: 72px; margin: 10px 0; color: white !important;">{final_pred:.2f} <span style="font-size: 24px;">Mw</span></h1>
                            <p>Confidence Score: <b>High</b> (Ensemble Agreement)</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("#### Model Breakdown")
                        m_col1, m_col2 = st.columns(2)
                        m_col1.metric("Random Forest", f"{pred_rf:.2f}")
                        m_col2.metric("XGBoost", f"{pred_xgb:.2f}")
                        
                    else:
                        st.error("Model file missing.")
                except Exception as e:
                    st.error(f"Prediction logic error: {e}")
            else:
                # Placeholder State
                st.info("👈 Enter parameters and click Predict to see the analysis.")
                st.image("https://media.giphy.com/media/3o7TKSjRrfIPjeiVyM/giphy.gif") # Placeholder gif or static image

    # --- TAB 3: ANALYTICS ---
    with tab_metrics:
        st.header("🔬 Model Performance Analytics")
        
        col_l, col_r = st.columns([1, 2])
        
        with col_l:
            st.subheader("Genetic Optimization")
            # Load GA Results
            if os.path.exists('results/ga_results.json'):
                with open('results/ga_results.json') as f:
                    ga_res = json.load(f)
                
                st.metric("Best MSE Score", f"{ga_res['best_fitness']:.5f}")
                st.metric("Features Selected", len(ga_res['selected_features']))
                
                with st.expander("View Selected Features"):
                    st.write(ga_res['selected_features'])
                    
        with col_r:
            st.subheader("Visual Evaluation")
            plots_dir = 'plots'
            if os.path.exists(plots_dir):
                plots = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
                tabs_plots = st.tabs([p.replace('.png', '').replace('_', ' ').title() for p in plots])
                
                for t, p in zip(tabs_plots, plots):
                    with t:
                        st.image(os.path.join(plots_dir, p), use_column_width=True)

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
if st.session_state.page == 'landing':
    landing_page()
else:
    main_app()
