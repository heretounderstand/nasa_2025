import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from lightgbm import LGBMClassifier
from sklearn.metrics import (confusion_matrix, roc_curve, auc, f1_score, 
                             precision_recall_curve, accuracy_score)
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Custom Training", layout="wide")

@st.cache_resource
def load_baseline_model():
    with open('lgbm_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv('tess_predict.csv')  # Replace with actual file
    X = df.drop('tfopwg_disp', axis=1)
    y = df['tfopwg_disp']
    return X, y

def impute_data(X, y, medians_0=None, medians_1=None):
    X_imputed = X.copy()
    
    if medians_0 is None:
        medians_0 = {col: X.loc[y == 0, col].median() for col in X.columns}
        medians_1 = {col: X.loc[y == 1, col].median() for col in X.columns}
    
    for col in X.columns:
        if X[col].isnull().any():
            mask_0 = (y == 0) & (X[col].isnull())
            mask_1 = (y == 1) & (X[col].isnull())
            X_imputed.loc[mask_0, col] = medians_0[col]
            X_imputed.loc[mask_1, col] = medians_1[col]
    
    return X_imputed, medians_0, medians_1

st.title("🧪 Custom Model Training")

baseline_data = load_baseline_model()
baseline_model = baseline_data['model']

# Sidebar for parameters
st.sidebar.header("Model Parameters")

col1, col2 = st.sidebar.columns(2)

with col1:
    n_estimators = st.number_input("N Estimators", 100, 2000, 900, 50)
    learning_rate = st.number_input("Learning Rate", 0.001, 0.3, 0.0375, 0.005)
    num_leaves = st.number_input("Num Leaves", 20, 150, 60, 10)
    min_child_samples = st.number_input("Min Child Samples", 5, 100, 10, 5)

with col2:
    subsample = st.slider("Subsample", 0.1, 1.0, 0.7, 0.1)
    colsample_bytree = st.slider("Colsample Bytree", 0.1, 1.0, 0.7, 0.1)
    reg_alpha = st.number_input("Reg Alpha", 0.0, 10.0, 0.1, 0.1)
    reg_lambda = st.number_input("Reg Lambda", 0.0, 10.0, 0.005, 0.005)

min_split_gain = st.sidebar.number_input("Min Split Gain", 0.0, 5.0, 1.0, 0.1)
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, 0.05)

train_button = st.sidebar.button("🚀 Train Model", type="primary")

# Load data
try:
    X, y = load_data()
    
    if train_button:
        with st.spinner("Training model..."):
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Impute
            X_train_imp, med_0, med_1 = impute_data(X_train, y_train)
            X_test_imp, _, _ = impute_data(X_test, y_test, med_0, med_1)
            
            # Train custom model
            custom_model = LGBMClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                min_child_samples=min_child_samples,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                min_split_gain=min_split_gain,
                random_state=42,
                force_col_wise=True,
                verbose=-1
            )
            
            custom_model.fit(X_train_imp, y_train)
            
            # Store in session state
            st.session_state.custom_model = custom_model
            st.session_state.X_test = X_test_imp
            st.session_state.y_test = y_test
            st.session_state.X_train = X_train_imp
            st.session_state.y_train = y_train
            
        st.success("Model trained successfully!")
    
    # Display results if model exists
    if 'custom_model' in st.session_state:
        custom_model = st.session_state.custom_model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        
        # Predictions
        y_pred_custom = custom_model.predict(X_test)
        y_proba_custom = custom_model.predict_proba(X_test)[:, 1]
        
        y_pred_baseline = baseline_model.predict(X_test)
        y_proba_baseline = baseline_model.predict_proba(X_test)[:, 1]
        
        # Metrics comparison
        st.header("Model Comparison")
        
        col1, col2, col3 = st.columns(3)
        
        # Custom metrics
        acc_custom = accuracy_score(y_test, y_pred_custom)
        f1_custom = f1_score(y_test, y_pred_custom)
        fpr_c, tpr_c, _ = roc_curve(y_test, y_proba_custom)
        auc_custom = auc(fpr_c, tpr_c)
        
        # Baseline metrics
        acc_baseline = accuracy_score(y_test, y_pred_baseline)
        f1_baseline = f1_score(y_test, y_pred_baseline)
        fpr_b, tpr_b, _ = roc_curve(y_test, y_proba_baseline)
        auc_baseline = auc(fpr_b, tpr_b)
        
        with col1:
            st.metric("Custom Accuracy", f"{acc_custom:.4f}", 
                     delta=f"{acc_custom - acc_baseline:.4f}")
        with col2:
            st.metric("Custom F1 Score", f"{f1_custom:.4f}",
                     delta=f"{f1_custom - f1_baseline:.4f}")
        with col3:
            st.metric("Custom AUC", f"{auc_custom:.4f}",
                     delta=f"{auc_custom - auc_baseline:.4f}")
        
        # Visualizations
        st.header("Performance Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # ROC Curves
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr_c, y=tpr_c,
                mode='lines',
                name=f'Custom (AUC={auc_custom:.4f})',
                line=dict(color='blue', width=2)
            ))
            fig_roc.add_trace(go.Scatter(
                x=fpr_b, y=tpr_b,
                mode='lines',
                name=f'Baseline (AUC={auc_baseline:.4f})',
                line=dict(color='orange', width=2)
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random',
                line=dict(color='red', dash='dash')
            ))
            fig_roc.update_layout(
                title="ROC Curve Comparison",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=400
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        
        with col2:
            # Confusion Matrices
            cm_custom = confusion_matrix(y_test, y_pred_custom)
            cm_baseline = confusion_matrix(y_test, y_pred_baseline)
            
            fig_cm = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Custom Model", "Baseline Model")
            )
            
            fig_cm.add_trace(go.Heatmap(
                z=cm_custom,
                text=cm_custom,
                texttemplate='%{text}',
                colorscale='Blues',
                showscale=False
            ), row=1, col=1)
            
            fig_cm.add_trace(go.Heatmap(
                z=cm_baseline,
                text=cm_baseline,
                texttemplate='%{text}',
                colorscale='Oranges',
                showscale=False
            ), row=1, col=2)
            
            fig_cm.update_layout(title="Confusion Matrix Comparison", height=400)
            st.plotly_chart(fig_cm, use_container_width=True)
        
        # Feature Importance Comparison
        st.header("Feature Importance Comparison")
        
        importance_custom = custom_model.feature_importances_
        importance_baseline = baseline_model.feature_importances_
        feature_names = baseline_data['feature_names']
        
        df_imp = pd.DataFrame({
            'feature': feature_names,
            'custom': importance_custom,
            'baseline': importance_baseline
        }).sort_values('custom', ascending=False).head(10)
        
        fig_imp = go.Figure()
        fig_imp.add_trace(go.Bar(
            x=df_imp['feature'],
            y=df_imp['custom'],
            name='Custom',
            marker=dict(color='blue')
        ))
        fig_imp.add_trace(go.Bar(
            x=df_imp['feature'],
            y=df_imp['baseline'],
            name='Baseline',
            marker=dict(color='orange')
        ))
        fig_imp.update_layout(
            title="Top 10 Feature Importances",
            xaxis_title="Feature",
            yaxis_title="Importance",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        
        # Precision-Recall Curves
        st.header("Precision-Recall Comparison")
        
        prec_c, rec_c, _ = precision_recall_curve(y_test, y_proba_custom)
        prec_b, rec_b, _ = precision_recall_curve(y_test, y_proba_baseline)
        
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(
            x=rec_c, y=prec_c,
            mode='lines',
            name='Custom',
            line=dict(color='blue', width=2)
        ))
        fig_pr.add_trace(go.Scatter(
            x=rec_b, y=prec_b,
            mode='lines',
            name='Baseline',
            line=dict(color='orange', width=2)
        ))
        fig_pr.update_layout(
            title="Precision-Recall Curve Comparison",
            xaxis_title="Recall",
            yaxis_title="Precision",
            height=400
        )
        st.plotly_chart(fig_pr, use_container_width=True)
        
        # Save custom model
        st.header("Save Custom Model")
        model_name = st.text_input("Model filename", "custom_lgbm_model.pkl")
        
        if st.button("💾 Save Model"):
            custom_model_data = {
                'model': custom_model,
                'medians_class_0': st.session_state.get('med_0', {}),
                'medians_class_1': st.session_state.get('med_1', {}),
                'feature_names': feature_names
            }
            with open(model_name, 'wb') as f:
                pickle.dump(custom_model_data, f)
            st.success(f"Model saved as {model_name}")

except FileNotFoundError:
    st.error("Data file not found. Please ensure 'your_data.csv' exists.")
except Exception as e:
    st.error(f"Error: {str(e)}")