import numpy as np
import plotly.graph_objects as go
import streamlit as st

from views.common import FALSE_POS, PLANET, get_model, get_scored_candidates, label, style

model = get_model()
meta = model.meta
cv = meta["cv_metrics"]
cands = get_scored_candidates()
thr = model.threshold

st.title("🪐 Exoplanet Hunter")
st.markdown(
    "#### Triage of TESS Objects of Interest: which transit signals are real planets?\n"
    "TESS has flagged thousands of periodic dips in starlight. Only a fraction are planets – the rest are "
    "eclipsing binaries, blended stars or instrumental artefacts, and confirming each one needs expensive "
    "ground-based follow-up. This tool ranks every open candidate by how *planet-like* its signal is, and "
    "explains why."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Candidates scored", f"{len(cands):,}")
c2.metric("Predicted planets", f"{(cands.probability >= thr).sum():,}", help=f"Probability ≥ {thr:.2f}")
c3.metric("Cross-validated ROC-AUC", f"{cv['roc_auc']:.3f}", help="Out-of-fold, never seen by the model")
c4.metric("Cross-validated accuracy", f"{cv['accuracy']:.1%}")

st.divider()
left, right = st.columns([1.1, 1])

with left:
    st.subheader("Where the candidates fall")
    df = cands.dropna(subset=["pl_orbper", "pl_rade"])
    fig = go.Figure(
        go.Scattergl(
            x=df.pl_orbper,
            y=df.pl_rade,
            mode="markers",
            showlegend=False,
            marker=dict(
                size=5,
                color=df.probability,
                colorscale=[[0, FALSE_POS], [0.5, "#EDEDED"], [1, PLANET]],
                cmin=0,
                cmax=1,
                opacity=0.75,
                colorbar=dict(title="P(planet)", tickformat=".0%"),
            ),
            customdata=np.stack([df.tid, df.probability], axis=1),
            hovertemplate="TIC %{customdata[0]}<br>P = %{x:.2f} d, R = %{y:.1f} R⊕"
            "<br>P(planet) = %{customdata[1]:.0%}<extra></extra>",
        )
    )
    fig.update_xaxes(type="log", dtick=1, title=label("pl_orbper"))
    fig.update_yaxes(type="log", dtick=1, title=label("pl_rade"))
    xr = [df.pl_orbper.min(), df.pl_orbper.max()]
    for r, name in [(1, "Earth"), (3.9, "Neptune"), (11.2, "Jupiter")]:
        fig.add_trace(go.Scatter(x=xr, y=[r, r], mode="lines+text", text=["", name], textposition="top left",
                                 line=dict(color="rgba(255,255,255,0.3)", dash="dot"), hoverinfo="skip",
                                 showlegend=False, textfont=dict(color="rgba(255,255,255,0.6)")))
    st.plotly_chart(style(fig, height=460), use_container_width=True)
    st.caption("Very large 'planets' (> ~2 R♃) are almost always eclipsing binaries – the model learns this.")

with right:
    st.subheader("How it works")
    st.markdown(
        """
1. **Physics-informed features.** On top of the catalogue values the model derives what a human vetter
   would check: does the **transit depth match** the planet/star radii? Is the **transit duration
   consistent** with Kepler's third law for this star? What is the star's density?
2. **Gradient-boosted trees (LightGBM)** trained on 2,562 TOIs already dispositioned by the TESS
   Follow-up Observing Program (confirmed/known planets vs. false positives/alarms).
   Missing values are handled natively – no guess-work imputation.
3. **Honest evaluation.** 5-fold stratified cross-validation; every score shown in the app is computed
   on objects the model never trained on.
4. **Ensemble + uncertainty.** The 5 fold models vote; their disagreement is reported as uncertainty.
5. **Explanations.** Each prediction is broken down into per-feature contributions (TreeSHAP).
        """
    )
    st.info("Start with the **Candidate explorer** to browse the most promising targets.", icon="🔭")

st.divider()
st.subheader("Most promising candidates")
top = cands.head(10)[["rank", "tid", "probability", "uncertainty", "pl_orbper", "pl_rade", "pl_eqt", "st_tmag"]]
st.dataframe(
    top,
    hide_index=True,
    use_container_width=True,
    column_config={
        "rank": "Rank",
        "tid": st.column_config.NumberColumn("TIC ID", format="%d"),
        "probability": st.column_config.ProgressColumn("P(planet)", min_value=0, max_value=1, format="percent"),
        "uncertainty": st.column_config.NumberColumn("± (fold spread)", format="%.3f"),
        "pl_orbper": st.column_config.NumberColumn("Period [d]", format="%.3f"),
        "pl_rade": st.column_config.NumberColumn("Radius [R⊕]", format="%.2f"),
        "pl_eqt": st.column_config.NumberColumn("Teq [K]", format="%.0f"),
        "st_tmag": st.column_config.NumberColumn("Tmag", format="%.2f"),
    },
)
