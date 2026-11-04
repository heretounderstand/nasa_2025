import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from exoplanet.model import metrics_at
from views.common import FALSE_POS, LEGACY, NEUTRAL, PLANET, get_model, get_oof, get_training_data, label, style

model = get_model()
meta = model.meta
oof = get_oof()
y, p, p_legacy = oof.y.values, oof.oof.values, oof.legacy_oof.values

st.title("📈 Model performance")
st.markdown(
    "All numbers below are **out-of-fold**: each TOI is scored by the fold model that did *not* train on it, "
    "so they estimate performance on new candidates. The *legacy* curve is the original hackathon setup "
    "(raw columns, original hyper-parameters) re-evaluated on the same folds."
)

thr = st.slider("Decision threshold", 0.05, 0.95, float(round(model.threshold, 2)), 0.01,
                help=f"Default {model.threshold:.2f} maximises F1. Raise it to trade recall for purity.")
m = metrics_at(y, p, thr)
ml = metrics_at(y, p_legacy, 0.5)

cols = st.columns(6)
for col, (k, name) in zip(cols, [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("accuracy", "Accuracy"),
                                  ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]):
    col.metric(name, f"{m[k]:.3f}", delta=f"{m[k] - ml[k]:+.3f} vs legacy")

a, b = st.columns(2)
with a:
    tn, fp, fn, tp = confusion_matrix(y, p >= thr).ravel()
    z = np.array([[tn, fp], [fn, tp]])
    fig = go.Figure(go.Heatmap(
        z=z, x=["Predicted FP", "Predicted planet"], y=["True FP", "True planet"],
        text=[[f"{v}<br>{v / z.sum():.0%}" for v in r] for r in z], texttemplate="%{text}",
        colorscale=[[0, "#151C33"], [1, PLANET]], showscale=False, textfont=dict(size=18),
    ))
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(style(fig, title="Confusion matrix"), use_container_width=True)
with b:
    fig = go.Figure()
    for prob, name, colr in [(p_legacy, "Legacy", LEGACY), (p, "New", PLANET)]:
        fpr, tpr, _ = roc_curve(y, prob)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC {metrics_at(y, prob)['roc_auc']:.3f})",
                                 line=dict(color=colr, width=2.5)))
    fpr_t = fp / (fp + tn)
    tpr_t = tp / (tp + fn)
    fig.add_trace(go.Scatter(x=[fpr_t], y=[tpr_t], mode="markers", name="Threshold",
                             marker=dict(size=12, color="#FFF", line=dict(color=PLANET, width=2))))
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color=NEUTRAL, dash="dot"))
    fig.update_xaxes(title="False positive rate")
    fig.update_yaxes(title="True positive rate")
    st.plotly_chart(style(fig, title="ROC curve"), use_container_width=True)

a, b = st.columns(2)
with a:
    fig = go.Figure()
    for prob, name, colr in [(p_legacy, "Legacy", LEGACY), (p, "New", PLANET)]:
        pr, rc, _ = precision_recall_curve(y, prob)
        fig.add_trace(go.Scatter(x=rc, y=pr, name=name, line=dict(color=colr, width=2.5)))
    fig.add_trace(go.Scatter(x=[m["recall"]], y=[m["precision"]], mode="markers", name="Threshold",
                             marker=dict(size=12, color="#FFF", line=dict(color=PLANET, width=2))))
    fig.update_xaxes(title="Recall (planets recovered)")
    fig.update_yaxes(title="Precision (purity)", range=[0.4, 1.02])
    st.plotly_chart(style(fig, title="Precision–recall"), use_container_width=True)
with b:
    frac, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig = go.Figure()
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color=NEUTRAL, dash="dot"))
    fig.add_trace(go.Scatter(x=mean_pred, y=frac, mode="lines+markers", name="New",
                             line=dict(color=PLANET, width=2.5)))
    fig.update_xaxes(title="Predicted probability", tickformat=".0%")
    fig.update_yaxes(title="Observed planet fraction", tickformat=".0%")
    st.plotly_chart(style(fig, title=f"Calibration (Brier {m['brier']:.3f})"), use_container_width=True)
    st.caption("Points on the diagonal mean a '70 %' really is a planet ~70 % of the time.")

a, b = st.columns(2)
with a:
    fig = go.Figure()
    for cls, name, colr in [(0, "True false positives", FALSE_POS), (1, "True planets", PLANET)]:
        fig.add_trace(go.Histogram(x=p[y == cls], name=name, marker_color=colr, opacity=0.7,
                                   xbins=dict(start=0, end=1, size=0.025)))
    fig.add_vline(x=thr, line=dict(color="#FFF", dash="dash"))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Predicted probability")
    st.plotly_chart(style(fig, title="Score distribution"), use_container_width=True)
with b:
    @st.cache_data(show_spinner="Computing SHAP importance…")
    def global_importance() -> pd.Series:
        raw, _ = get_training_data()
        contrib, _ = model.explain(raw)
        return contrib.abs().mean().sort_values()

    imp = global_importance().tail(12)
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=PLANET,
                           hovertext=[label(f) for f in imp.index]))
    fig.update_xaxes(title="Mean |SHAP| (log-odds)")
    st.plotly_chart(style(fig, title="What drives the decisions", height=380), use_container_width=True)

with st.expander("Per-fold stability and hyper-parameters"):
    folds = pd.DataFrame(meta["fold_metrics"]).rename_axis("fold")
    st.dataframe(folds.style.format("{:.4f}"), use_container_width=True)
    st.json(meta["params"], expanded=False)

with st.expander("What changed compared with the hackathon version?"):
    st.markdown(
        """
* **Evaluation leak fixed.** The old performance page scored the model on the data it was trained on
  (≈98 % accuracy displayed). The honest, cross-validated figure for that model is ≈84 %.
* **Label leak in imputation removed.** Missing values were filled with the median of the *true class*,
  i.e. the answer was used to build the input. Unlabelled candidates were filled with the *planet* median,
  biasing them towards "planet". LightGBM now handles missing values natively.
* **Physics-informed features** (depth/radius consistency, Kepler-based expected duration, stellar density,
  a/R★…) carry most of the gain.
* **Sky position (RA/Dec) dropped** – it is not a property of the signal and only lets the model memorise
  survey footprints.
* **Smaller, more regularised trees** (15 leaves instead of 60) → better calibrated probabilities.
* **5-model ensemble** with uncertainty, and **per-prediction explanations**.
        """
    )
