import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go

st.set_page_config(page_title="Exoplanet Detection", page_icon="🪐", layout="wide")

@st.cache_resource
def load_model():
    with open('lgbm_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv('tess_predict.csv')
    return df

# Header
st.title("🪐 Exoplanet Detection System")
st.markdown("### Machine Learning-based classification of TESS exoplanet candidates")

# Quick stats
col1, col2, col3, col4 = st.columns(4)

try:
    model_data = load_model()
    model = model_data['model']
    params = model.get_params()
    
    df = load_data()
    
    with col1:
        st.metric("Model Type", "LightGBM")
    with col2:
        st.metric("Features", len(model_data['feature_names']))
    with col3:
        st.metric("Training Samples", len(df))
    with col4:
        class_balance = (df['tfopwg_disp'].sum() / len(df)) * 100
        st.metric("Planet Ratio", f"{class_balance:.1f}%")

except:
    st.warning("Model or data not loaded. Please ensure files are in place.")

st.divider()

# Overview sections
col1, col2 = st.columns(2)

with col1:
    st.header("🎯 Project Overview")
    st.markdown("""
    This application uses a LightGBM classifier to identify potential exoplanets from TESS 
    (Transiting Exoplanet Survey Satellite) observations.
    
    **Key Features:**
    - Advanced gradient boosting classification
    - Class-based median imputation for missing values
    - Real-time probability visualization
    - Custom model training and comparison
    
    **Dataset Features:**
    - Stellar parameters (temperature, magnitude, radius)
    - Orbital characteristics (period, transit duration)
    - Planetary properties (radius, insolation, equilibrium temperature)
    """)
    
    st.header("📊 Model Performance")
    try:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Estimators", params['n_estimators'])
        col_b.metric("Learning Rate", f"{params['learning_rate']:.4f}")
        col_c.metric("Num Leaves", params['num_leaves'])
    except:
        pass

with col2:
    st.header("🧭 Navigation")
    
    st.markdown("""
    #### 🔭 Visualization
    - Explore probability decision boundaries
    - Classify candidate planets
    - Interactive feature analysis
    
    #### 📈 Performance
    - Detailed model metrics
    - ROC and Precision-Recall curves
    - Feature importance analysis
    - Confusion matrix and classification report
    
    #### 🔧 Custom Training
    - Train your own LightGBM model
    - Adjust hyperparameters
    - Compare with baseline model
    - Save custom models
    """)
    
    st.info("👈 Use the sidebar to navigate between pages")

st.divider()

# Feature importance preview
st.header("🔍 Top Features")

try:
    importance = model.feature_importances_
    feature_names = model_data['feature_names']
    
    df_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(8)
    
    fig = go.Figure(go.Bar(
        y=df_imp['feature'][::-1],
        x=df_imp['importance'][::-1],
        orientation='h',
        marker=dict(
            color=df_imp['importance'][::-1],
            colorscale='Viridis',
            showscale=False
        )
    ))
    
    fig.update_layout(
        title="Most Important Features for Classification",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
except Exception as e:
    st.error(f"Could not display feature importance: {str(e)}")

# Quick start guide
st.header("🚀 Quick Start")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **1. Explore**
    
    Visit the Visualization page to see how the model classifies exoplanets 
    based on different feature combinations.
    """)

with col2:
    st.markdown("""
    **2. Analyze**
    
    Check the Performance page to understand model metrics and see detailed 
    evaluation charts.
    """)

with col3:
    st.markdown("""
    **3. Experiment**
    
    Train your own model with custom parameters and compare results with 
    the baseline model.
    """)

st.divider()

# Data source
st.header("📁 Data Source")
st.markdown("""
This file was produced by the NASA Exoplanet Archive  
**http://exoplanetarchive.ipac.caltech.edu**  
Retrieved: Saturday, October 4, 2025 13:11:29
""")

# Footer
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Built with Streamlit • Powered by LightGBM • Data from NASA Exoplanet Archive</p>
</div>

""", unsafe_allow_html=True)
