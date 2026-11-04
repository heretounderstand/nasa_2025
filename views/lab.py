import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve

from exoplanet.features import RAW_FEATURES, build_features
from exoplanet.model import DEFAULT_PARAMS, cross_validate, metrics_at
from views.common import LEGACY, PLANET, get_oof, get_training_data, style

st.title("🔬 Training lab")
st.caption(
    "Retrain the model with your own settings. Training uses the same 5-fold cross-validation as the "
    "production model, so the comparison below is apples-to-apples."
)

raw, y = get_training_data()
oof = get_oof()

with st.form("params"):
    c1, c2, c3 = st.columns(3)
    with c1:
        n_estimators = st.slider("Trees", 100, 2000, DEFAULT_PARAMS["n_estimators"], 50)
        learning_rate = st.select_slider("Learning rate", [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2],
                                         DEFAULT_PARAMS["learning_rate"])
        num_leaves = st.slider("Leaves per tree", 4, 128, DEFAULT_PARAMS["num_leaves"])
    with c2:
        min_child = st.slider("Min samples per leaf", 5, 100, DEFAULT_PARAMS["min_child_samples"])
        subsample = st.slider("Row subsample", 0.3, 1.0, DEFAULT_PARAMS["subsample"], 0.05)
        colsample = st.slider("Feature subsample", 0.2, 1.0, DEFAULT_PARAMS["colsample_bytree"], 0.05)
    with c3:
        reg_lambda = st.select_slider("L2 regularisation", [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                                      DEFAULT_PARAMS["reg_lambda"])
        feature_set = st.radio("Inputs", ["Physics-informed features", "Raw catalogue columns only"])
        balanced = st.toggle("Balance classes", False)
    go_train = st.form_submit_button("Train with 5-fold CV", type="primary", use_container_width=True)

if go_train:
    X = build_features(raw) if feature_set.startswith("Physics") else raw[list(RAW_FEATURES)]
    params = dict(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
                  min_child_samples=min_child, subsample=subsample, colsample_bytree=colsample,
                  reg_lambda=reg_lambda, class_weight="balanced" if balanced else None)
    bar = st.progress(0.0, "Training folds…")
    res = cross_validate(X, y, params=params, progress=lambda f: bar.progress(f, "Training folds…"))
    bar.empty()
    st.session_state.lab = {"oof": res.oof, "params": params, "features": feature_set}

if "lab" not in st.session_state:
    st.info("Pick some settings and press **Train**. Tip: try *Raw catalogue columns only* to see what the "
            "physics features bring.", icon="🧭")
    st.stop()

lab = st.session_state.lab
yv = y.values
mine, ref = metrics_at(yv, lab["oof"]), metrics_at(yv, oof.oof.values)

st.subheader("Your model vs. production model")
cols = st.columns(5)
for col, (k, n) in zip(cols, [("roc_auc", "ROC-AUC"), ("accuracy", "Accuracy"), ("f1", "F1"),
                               ("log_loss", "Log-loss"), ("brier", "Brier")]):
    lower_better = k in ("log_loss", "brier")
    d = mine[k] - ref[k]
    col.metric(n, f"{mine[k]:.4f}", delta=f"{d:+.4f}" if abs(d) >= 5e-5 else None,
               delta_color="inverse" if lower_better else "normal")

fig = go.Figure()
for prob, name, colr in [(oof.oof.values, "Production", PLANET), (lab["oof"], "Yours", LEGACY)]:
    fpr, tpr, _ = roc_curve(yv, prob)
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC {metrics_at(yv, prob)['roc_auc']:.4f})",
                             line=dict(color=colr, width=2.5)))
fig.update_xaxes(title="False positive rate")
fig.update_yaxes(title="True positive rate")
st.plotly_chart(style(fig, title="ROC curve (out-of-fold)", height=420), use_container_width=True)

diff = np.abs(lab["oof"] - oof.oof.values)
st.caption(f"Predictions differ by more than 10 points on {(diff > 0.1).mean():.1%} of TOIs.")
