import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve, 
                             auc, precision_recall_curve, f1_score)

st.set_page_config(page_title="Model Performance", layout="wide")

@st.cache_resource
def load_model():
    with open('lgbm_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    # Load your training/test data
    df = pd.read_csv('tess_predict.csv')  # Replace with actual file
    X = df.drop('tfopwg_disp', axis=1)
    y = df['tfopwg_disp']
    return X, y

model_data = load_model()
model = model_data['model']

st.title("📊 Model Performance Analysis")

# Model Parameters Section
st.header("Model Parameters")
params = model.get_params()
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Learning Rate", params['learning_rate'])
    st.metric("N Estimators", params['n_estimators'])
    st.metric("Num Leaves", params['num_leaves'])
    
with col2:
    st.metric("Min Child Samples", params['min_child_samples'])
    st.metric("Min Split Gain", params['min_split_gain'])
    st.metric("Subsample", params['subsample'])
    
with col3:
    st.metric("Colsample Bytree", params['colsample_bytree'])
    st.metric("Reg Alpha", params['reg_alpha'])
    st.metric("Reg Lambda", params['reg_lambda'])

st.divider()

# Load data and make predictions
try:
    X, y = load_data()
    
    # Apply imputation
    X_imputed = X.copy()
    medians_0 = model_data['medians_class_0']
    medians_1 = model_data['medians_class_1']
    
    for col in X.columns:
        if X[col].isnull().any():
            mask_0 = (y == 0) & (X[col].isnull())
            mask_1 = (y == 1) & (X[col].isnull())
            X_imputed.loc[mask_0, col] = medians_0[col]
            X_imputed.loc[mask_1, col] = medians_1[col]
    
    y_pred = model.predict(X_imputed)
    y_proba = model.predict_proba(X_imputed)[:, 1]
    
    # Performance Metrics
    st.header("Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = f1_score(y, y_pred)
    
    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("Precision", f"{precision:.4f}")
    col3.metric("Recall", f"{recall:.4f}")
    col4.metric("F1 Score", f"{f1:.4f}")
    
    # Visualizations
    st.header("Performance Visualizations")
    
    # Row 1: Confusion Matrix & ROC Curve
    col1, col2 = st.columns(2)
    
    with col1:
        # Confusion Matrix
        fig_cm = go.Figure(data=go.Heatmap(
            z=[[tn, fp], [fn, tp]],
            x=['Predicted 0', 'Predicted 1'],
            y=['Actual 0', 'Actual 1'],
            text=[[tn, fp], [fn, tp]],
            texttemplate='%{text}',
            colorscale='Blues',
            showscale=False
        ))
        fig_cm.update_layout(title="Confusion Matrix", height=400)
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col2:
        # ROC Curve
        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_auc = auc(fpr, tpr)
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC curve (AUC = {roc_auc:.4f})',
            line=dict(color='blue', width=2)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(color='red', dash='dash')
        ))
        fig_roc.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=400
        )
        st.plotly_chart(fig_roc, use_container_width=True)
    
    # Row 2: Precision-Recall Curve & Feature Importance
    col1, col2 = st.columns(2)
    
    with col1:
        # Precision-Recall Curve
        precision_curve, recall_curve, _ = precision_recall_curve(y, y_proba)
        
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(
            x=recall_curve, y=precision_curve,
            mode='lines',
            fill='tozeroy',
            name='Precision-Recall curve',
            line=dict(color='green', width=2)
        ))
        fig_pr.update_layout(
            title="Precision-Recall Curve",
            xaxis_title="Recall",
            yaxis_title="Precision",
            height=400
        )
        st.plotly_chart(fig_pr, use_container_width=True)
    
    with col2:
        # Feature Importance
        importance = model.feature_importances_
        feature_names = model_data['feature_names']
        
        df_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True).tail(10)
        
        fig_imp = go.Figure(go.Bar(
            x=df_importance['importance'],
            y=df_importance['feature'],
            orientation='h',
            marker=dict(color='purple')
        ))
        fig_imp.update_layout(
            title="Top 10 Feature Importances",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=400
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    
    # Classification Report
    st.header("Classification Report")
    report = classification_report(y, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.dataframe(df_report.style.format("{:.4f}"))
    
    # Probability Distribution
    st.header("Prediction Probability Distribution")
    
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=y_proba[y == 0],
        name='Class 0 (False)',
        opacity=0.7,
        marker=dict(color='red'),
        nbinsx=50
    ))
    fig_dist.add_trace(go.Histogram(
        x=y_proba[y == 1],
        name='Class 1 (Planet)',
        opacity=0.7,
        marker=dict(color='green'),
        nbinsx=50
    ))
    fig_dist.update_layout(
        title="Distribution of Predicted Probabilities by True Class",
        xaxis_title="Predicted Probability",
        yaxis_title="Count",
        barmode='overlay',
        height=400
    )
    st.plotly_chart(fig_dist, use_container_width=True)

except FileNotFoundError:
    st.error("Data file not found. Please ensure 'your_data.csv' exists in the project directory.")
except Exception as e:
    st.error(f"Error loading data or computing metrics: {str(e)}")