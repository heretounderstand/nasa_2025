import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

st.set_page_config(page_title="Exoplanet Visualization", layout="wide")

# Load model
@st.cache_resource
def load_model():
    with open('lgbm_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_candidates():
    return pd.read_csv('candidates.csv')

model_data = load_model()
model = model_data['model']
feature_names = model_data['feature_names']
medians_0 = model_data['medians_class_0']
medians_1 = model_data['medians_class_1']

st.title("🪐 Exoplanet Detection Visualization")

# Two columns layout
col1, col2 = st.columns(2)

with col1:
    st.header("Probability Decision Boundary")
    
    # Select 2 features for axes
    x_feat = st.selectbox("X-axis feature", feature_names, index=0)
    y_feat = st.selectbox("Y-axis feature", feature_names, index=1)
    
    # Get ranges for selected features
    x_range = st.slider(f"{x_feat} range", 
                        float(medians_0[x_feat] * 0.1), 
                        float(medians_0[x_feat] * 10),
                        (float(medians_0[x_feat] * 0.5), float(medians_0[x_feat] * 2)))
    
    y_range = st.slider(f"{y_feat} range",
                        float(medians_0[y_feat] * 0.1),
                        float(medians_0[y_feat] * 10),
                        (float(medians_0[y_feat] * 0.5), float(medians_0[y_feat] * 2)))
    
    # Sliders for other features
    st.subheader("Other Features")
    feature_values = {}
    for feat in feature_names:
        if feat not in [x_feat, y_feat]:
            median_val = (medians_0[feat] + medians_1[feat]) / 2
            min_val = median_val * 0.1 if median_val > 0 else median_val - 10
            max_val = median_val * 10 if median_val > 0 else median_val + 10
            feature_values[feat] = st.slider(feat, 
                                            float(min_val),
                                            float(max_val),
                                            float(median_val),
                                            key=f"slider_{feat}")
    
    # Generate prediction grid
    if st.button("Generate Probability Map"):
        x_grid = np.linspace(x_range[0], x_range[1], 50)
        y_grid = np.linspace(y_range[0], y_range[1], 50)
        xx, yy = np.meshgrid(x_grid, y_grid)
        
        # Create feature matrix
        grid_points = []
        for i in range(len(x_grid)):
            for j in range(len(y_grid)):
                point = {}
                for feat in feature_names:
                    if feat == x_feat:
                        point[feat] = xx[j, i]
                    elif feat == y_feat:
                        point[feat] = yy[j, i]
                    else:
                        point[feat] = feature_values[feat]
                grid_points.append(point)
        
        X_grid = pd.DataFrame(grid_points)
        proba = model.predict_proba(X_grid)[:, 1].reshape(xx.shape)
        
        # Plot
        fig = go.Figure(data=go.Contour(
            x=x_grid,
            y=y_grid,
            z=proba,
            colorscale='RdYlGn',
            contours=dict(
                start=0,
                end=1,
                size=0.1,
            ),
            colorbar=dict(title="P(Planet)")
        ))
        
        fig.update_layout(
            title="Probability of being a Planet",
            xaxis_title=x_feat,
            yaxis_title=y_feat,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.header("Candidate Planets Classification")
    
    try:
        candidates = load_candidates()
        
        if 'tid' in candidates.columns:
            candidate_features = candidates.drop('tid', axis=1)
            
            # Handle missing values using class 1 medians (optimistic)
            for col in candidate_features.columns:
                if candidate_features[col].isnull().any():
                    candidate_features[col].fillna(medians_1.get(col, medians_0.get(col, 0)), inplace=True)
            
            # Predict
            predictions = model.predict(candidate_features)
            probas = model.predict_proba(candidate_features)[:, 1]
            
            candidates['prediction'] = predictions
            candidates['probability'] = probas
            
            # Show stats
            st.metric("Total Candidates", len(candidates))
            st.metric("Predicted Planets", int(predictions.sum()))
            st.metric("Predicted False Positives", int((1-predictions).sum()))
            
            # Select features for scatter plot
            x_scatter = st.selectbox("X-axis", feature_names, index=0, key="scatter_x")
            y_scatter = st.selectbox("Y-axis", feature_names, index=1, key="scatter_y")
            
            # Scatter plot
            fig_scatter = go.Figure()
            
            planets = candidates[candidates['prediction'] == 1]
            false_pos = candidates[candidates['prediction'] == 0]
            
            fig_scatter.add_trace(go.Scatter(
                x=false_pos[x_scatter],
                y=false_pos[y_scatter],
                mode='markers',
                name='False Positive',
                marker=dict(color='red', size=8, opacity=0.6),
                text=false_pos['tid'],
                hovertemplate='<b>%{text}</b><br>Probability: ' + 
                              false_pos['probability'].round(3).astype(str)
            ))
            
            fig_scatter.add_trace(go.Scatter(
                x=planets[x_scatter],
                y=planets[y_scatter],
                mode='markers',
                name='Planet',
                marker=dict(color='green', size=8, opacity=0.6),
                text=planets['tid'],
                hovertemplate='<b>%{text}</b><br>Probability: ' + 
                              planets['probability'].round(3).astype(str)
            ))
            
            fig_scatter.update_layout(
                title="Candidate Classification",
                xaxis_title=x_scatter,
                yaxis_title=y_scatter,
                height=600
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Show candidates table
            st.subheader("Candidates Details")
            st.dataframe(candidates[['tid', 'prediction', 'probability']].sort_values('probability', ascending=False))
            
    except FileNotFoundError:
        st.warning("candidates.csv file not found. Please upload your candidates dataset.")