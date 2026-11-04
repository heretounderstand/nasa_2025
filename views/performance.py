import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from exoplanet.model import metrics_at
from views.common import (
    FALSE_POS,
    GRID,
    LEGACY,
    PLANET,
    SEQ_SCALE,
    TEXT,
    TEXT_2,
    get_model,
    get_oof,
    get_training_data,
    label,
    page_header,
    short,
    style,
)

model = get_model()
meta = model.meta
oof = get_oof()
y, p, p_legacy = oof.y.values, oof.oof.values, oof.legacy_oof.values

page_header(
    "Model",
    "Performance",
    "Every number here is <b>out-of-fold</b>: each labelled TOI is scored by a model that never trained on it, "
    "which estimates performance on new candidates. The grey <i>legacy</i> curves show the original "
    "hackathon model evaluated on the same folds; the green deltas compare against it.",
)

with st.container(border=True):
    thr = st.slider("Decision threshold: a signal is called a planet when P(planet) ≥ this value",
                    0.05, 0.95, float(round(model.threshold, 2)), 0.01,
                    help=f"The default {model.threshold:.2f} maximises F1. Raise it for a purer, smaller list.")
    m = metrics_at(y, p, thr)
    ml = metrics_at(y, p_legacy, 0.5)
    cols = st.columns(6)
    for col, (k, name) in zip(cols, [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("accuracy", "Accuracy"),
                                      ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]):
        col.metric(name, f"{m[k]:.3f}", delta=f"{m[k] - ml[k]:+.3f}", help="Change vs. the legacy model at 0.5")

tn, fp, fn, tp = confusion_matrix(y, p >= thr).ravel()
marker = dict(size=12, color=TEXT, line=dict(color="#0B1020", width=2))


def card(fig, caption=None, height=360):
    with st.container(border=True):
        st.plotly_chart(style(fig, height=height), use_container_width=True, config={"displayModeBar": False})
        if caption:
            st.caption(caption)


a, b = st.columns(2)
with a:
    z = np.array([[tn, fp], [fn, tp]])
    txt = [[f"<b>{v:,}</b><br>{v / z.sum():.0%}" for v in r] for r in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=["Called false positive", "Called planet"], y=["Truly false positive", "Truly planet"],
        text=txt, texttemplate="%{text}", colorscale=SEQ_SCALE, showscale=False, xgap=3, ygap=3,
        textfont=dict(size=17, color=TEXT), hovertemplate="%{y} · %{x}<br><b>%{z:,}</b> TOIs<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="top")
    fig.update_layout(title="Confusion matrix")
    card(fig, f"At threshold {thr:.2f}: {fp} false alarms and {fn} missed planets out of {len(y):,} TOIs.")
with b:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GRID, width=1), hoverinfo="skip",
                             showlegend=False))
    for prob, name, colr, w in [(p_legacy, "Legacy", LEGACY, 1.5), (p, "New", PLANET, 2.5)]:
        fpr, tpr, _ = roc_curve(y, prob)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} · AUC {metrics_at(y, prob)['roc_auc']:.3f}",
                                 line=dict(color=colr, width=w),
                                 hovertemplate="FPR %{x:.2f} · TPR %{y:.2f}<extra>" + name + "</extra>"))
    fig.add_trace(go.Scatter(x=[fp / (fp + tn)], y=[tp / (tp + fn)], mode="markers", name="Current threshold",
                             marker=marker))
    fig.update_xaxes(title="False positive rate", range=[-0.01, 1])
    fig.update_yaxes(title="True positive rate (recall)", range=[0, 1.02])
    fig.update_layout(title="ROC curve")
    card(fig, "Closer to the top-left corner is better.")

a, b = st.columns(2)
with a:
    fig = go.Figure()
    for prob, name, colr, w in [(p_legacy, "Legacy", LEGACY, 1.5), (p, "New", PLANET, 2.5)]:
        pr, rc, _ = precision_recall_curve(y, prob)
        fig.add_trace(go.Scatter(x=rc, y=pr, name=name, line=dict(color=colr, width=w),
                                 hovertemplate="recall %{x:.2f} · precision %{y:.2f}<extra>" + name + "</extra>"))
    fig.add_trace(go.Scatter(x=[m["recall"]], y=[m["precision"]], mode="markers", name="Current threshold",
                             marker=marker))
    fig.update_xaxes(title="Recall: share of real planets recovered")
    fig.update_yaxes(title="Precision: purity of the list", range=[0.45, 1.02])
    fig.update_layout(title="Precision vs. recall")
    card(fig, "Moving the threshold slides the white dot along this curve.")
with b:
    frac, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                             line=dict(color=GRID, width=1), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=mean_pred, y=frac, mode="lines+markers", name="New model",
                             line=dict(color=PLANET, width=2), marker=dict(size=9, line=dict(color="#0B1020", width=2)),
                             hovertemplate="predicted %{x:.0%} → observed %{y:.0%}<extra></extra>"))
    fig.update_xaxes(title="Predicted probability", tickformat=".0%")
    fig.update_yaxes(title="Observed share of planets", tickformat=".0%")
    fig.update_layout(title=f"Calibration · Brier score {m['brier']:.3f}")
    card(fig, "On the diagonal, a 70 % score means a planet about 70 % of the time.")

a, b = st.columns(2)
with a:
    bins = np.linspace(0, 1, 41)
    centers = (bins[:-1] + bins[1:]) / 2
    fig = go.Figure()
    for cls, name, colr in [(0, "True false positives", FALSE_POS), (1, "True planets", PLANET)]:
        counts, _ = np.histogram(p[y == cls], bins)
        fig.add_trace(go.Bar(x=centers, y=counts, name=name, marker_color=colr, width=0.025 * 0.8,
                             hovertemplate="%{x:.0%}: <b>%{y}</b><extra>" + name + "</extra>"))
    fig.add_vline(x=thr, line=dict(color=TEXT_2, width=1))
    fig.update_layout(barmode="stack", title="Scores by true class")
    fig.update_xaxes(title="Predicted probability", tickformat=".0%")
    fig.update_yaxes(title="TOIs")
    card(fig, "The vertical line marks the threshold. Orange to its right are false alarms; blue to its left "
              "are missed planets.")
with b:
    @st.cache_data(show_spinner="Computing SHAP importance…")
    def global_importance() -> pd.Series:
        raw, _ = get_training_data()
        contrib, _ = model.explain(raw)
        return contrib.abs().mean().sort_values()

    imp = global_importance().tail(12)
    fig = go.Figure(go.Bar(
        x=imp.values, y=[short(f) for f in imp.index], orientation="h", marker_color=PLANET,
        customdata=[label(f) for f in imp.index],
        hovertemplate="%{customdata}<br>mean |SHAP| <b>%{x:.3f}</b><extra></extra>"))
    fig.update_xaxes(title="Average impact on the score (mean |SHAP|, log-odds)")
    fig.update_yaxes(showgrid=False)
    fig.update_layout(title="What drives the decisions", bargap=0.35)
    card(fig, "The two physics consistency checks and the planet's temperature and size dominate.")

with st.expander("Per-fold stability and hyper-parameters", icon=":material/tune:"):
    folds = pd.DataFrame(meta["fold_metrics"]).rename_axis("fold")
    st.dataframe(folds.style.format("{:.4f}"), use_container_width=True)
    st.json(meta["params"], expanded=False)

with st.expander("What changed compared with the hackathon version?", icon=":material/history:"):
    st.markdown(
        """
* **Evaluation leak fixed.** The old performance page scored the model on its own training data
  (about 98 % accuracy shown). Measured honestly with cross-validation, that model reaches about 84 %.
* **Label leak in imputation removed.** Missing values were filled with the median of the *true class*,
  so the inputs contained the answer. Unlabelled candidates were filled with the *planet* median, which
  biased them towards "planet". LightGBM now handles missing values natively.
* **Physics-informed features** (depth/radius consistency, Kepler-based expected duration, stellar density,
  a/R★, …) provide most of the gain.
* **Sky position (RA/Dec) dropped.** It is not a property of the signal; it only lets the model memorise
  survey footprints.
* **Smaller, more regularised trees** (15 leaves instead of 60) give better-calibrated probabilities.
* **Five-model ensemble** with an uncertainty estimate, plus **per-prediction explanations**.
        """
    )
