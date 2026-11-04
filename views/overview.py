import numpy as np
import plotly.graph_objects as go
import streamlit as st

from views.common import (
    FALSE_POS,
    GRID,
    MUTED,
    PLANET,
    PROB_SCALE,
    TEXT_2,
    get_model,
    get_scored_candidates,
    sky_figure,
    step_card,
    style,
)

model = get_model()
cv = model.meta["cv_metrics"]
cands = get_scored_candidates()
thr = model.threshold
n_strong = int((cands.probability >= 0.9).sum())

st.markdown(
    f"""
<div class="exo-hero">
  <div class="exo-planet"></div>
  <div class="exo-eyebrow">NASA Space Apps 2025 · TESS</div>
  <h1 class="exo-title">Which of these signals<br>are real planets?</h1>
  <div class="exo-sub">TESS has flagged thousands of periodic dips in starlight. Many are eclipsing binaries,
  blended stars or artefacts, and confirming each one needs telescope time. Exoplanet Hunter ranks
  <b>{len(cands):,} open candidates</b> by how planet-like their signal is, and explains every score.</div>
</div>
""",
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Open candidates scored", f"{len(cands):,}")
c2.metric("High-confidence planets", f"{n_strong:,}", help="Probability ≥ 90 %")
c3.metric("ROC-AUC (cross-validated)", f"{cv['roc_auc']:.3f}", help="Measured on TOIs the model never saw")
c4.metric("Accuracy (cross-validated)", f"{cv['accuracy']:.1%}", help="At a 0.5 threshold")

st.write("")
with st.container(border=True):
    st.markdown("#### The candidate population")
    tab_pr, tab_sky, tab_hist = st.tabs(["Period vs. radius", "Sky map", "Score distribution"])

    with tab_pr:
        df = cands.dropna(subset=["pl_orbper", "pl_rade"])
        fig = go.Figure()
        for r, name in [(1, "Earth"), (3.9, "Neptune"), (11.2, "Jupiter")]:
            fig.add_trace(go.Scatter(
                x=[0.15, 2500], y=[r, r], mode="lines+text", text=["", name], textposition="top left",
                line=dict(color=GRID, width=1), hoverinfo="skip", showlegend=False,
                textfont=dict(color=MUTED, size=12)))
        fig.add_trace(go.Scattergl(
            x=df.pl_orbper, y=df.pl_rade, mode="markers", showlegend=False,
            marker=dict(size=6, color=df.probability, colorscale=PROB_SCALE, cmin=0, cmax=1, opacity=0.85,
                        line=dict(width=0),
                        colorbar=dict(title=dict(text="P(planet)", side="right"), tickformat=".0%", thickness=10,
                                      outlinewidth=0, len=0.9)),
            customdata=np.stack([df.tid, df.probability], axis=1),
            hovertemplate="<b>%{customdata[1]:.0%}</b> planet<br>TIC %{customdata[0]}"
                          "<br>Period %{x:.2f} d · Radius %{y:.1f} × Earth<extra></extra>",
        ))
        fig.update_xaxes(type="log", dtick=1, title="Orbital period [days]", range=[np.log10(0.15), np.log10(2500)])
        fig.update_yaxes(type="log", dtick=1, title="Planet radius [Earth radii]")
        st.plotly_chart(style(fig, height=470, margin=dict(l=8, r=8, t=16, b=8)), use_container_width=True)
        st.caption("Blue = planet-like, orange = false-positive-like. Objects much larger than Jupiter "
                   "(top band) are mostly eclipsing binaries, and the model learned that.")

    with tab_sky:
        fig = sky_figure(
            cands.ra, cands.dec, cands.probability,
            np.stack([cands.tid, cands.probability, cands.ra, cands.dec], axis=1),
            "<b>%{customdata[1]:.0%}</b> planet<br>TIC %{customdata[0]}"
            "<br>RA %{customdata[2]:.1f}° · Dec %{customdata[3]:.1f}°<extra></extra>",
        )
        st.plotly_chart(style(fig, height=470, margin=dict(l=0, r=0, t=10, b=0)), use_container_width=True)
        st.caption("Equatorial coordinates (Mollweide projection). The dense patches near the ecliptic poles are "
                   "TESS's continuous viewing zones. Sky position is deliberately *not* a model input.")

    with tab_hist:
        bins = np.linspace(0, 1, 41)
        counts, _ = np.histogram(cands.probability, bins)
        centers = (bins[:-1] + bins[1:]) / 2
        fig = go.Figure(go.Bar(
            x=centers, y=counts, width=0.025 * 0.85,
            marker_color=[PLANET if c >= thr else FALSE_POS for c in centers],
            hovertemplate="P(planet) %{x:.0%}<br><b>%{y}</b> candidates<extra></extra>"))
        fig.add_vline(x=thr, line=dict(color=TEXT_2, width=1),
                      annotation_text=f"decision threshold {thr:.2f}", annotation_font_color=TEXT_2)
        fig.update_xaxes(title="Planet probability", tickformat=".0%")
        fig.update_yaxes(title="Candidates")
        st.plotly_chart(style(fig, height=420, showlegend=False), use_container_width=True)
        st.caption("Most candidates are scored confidently at either end; the middle is where follow-up "
                   "observations matter most.")

st.write("")
st.markdown("#### How it works")
steps = [
    ("Physics checks", "Derive what a human vetter checks: does the <b>transit depth</b> match the radii? Is the "
                       "<b>duration</b> consistent with Kepler's law for this star?"),
    ("Learn from labels", "LightGBM trained on <b>2,562 TOIs</b> already confirmed or rejected by the TESS "
                          "follow-up programme. Missing values handled natively."),
    ("Honest scores", "Every metric comes from <b>5-fold cross-validation</b>, measured on objects the model "
                      "never trained on."),
    ("Explain & rank", "Five fold-models vote, their spread is the <b>uncertainty</b>, and SHAP shows "
                       "<b>why</b> each score came out as it did."),
]
for col, (i, (t, d)) in zip(st.columns(4), enumerate(steps, 1)):
    col.markdown(step_card(i, t, d), unsafe_allow_html=True)

st.write("")
with st.container(border=True):
    h1, h2 = st.columns([3, 1], vertical_alignment="center")
    h1.markdown("#### Top 10 candidates")
    h2.page_link("views/candidates.py", label="Open the explorer", icon=":material/arrow_forward:")
    top = cands.head(10)[["rank", "tid", "probability", "uncertainty", "pl_orbper", "pl_rade", "pl_eqt", "st_tmag"]]
    st.dataframe(
        top,
        hide_index=True,
        use_container_width=True,
        column_config={
            "rank": st.column_config.NumberColumn("#", width="small"),
            "tid": st.column_config.NumberColumn("TIC ID", format="%d"),
            "probability": st.column_config.ProgressColumn("P(planet)", min_value=0, max_value=1, format="percent"),
            "uncertainty": st.column_config.NumberColumn("± spread", format="%.3f"),
            "pl_orbper": st.column_config.NumberColumn("Period [d]", format="%.2f"),
            "pl_rade": st.column_config.NumberColumn("Radius (× Earth)", format="%.2f"),
            "pl_eqt": st.column_config.NumberColumn("Teq [K]", format="%.0f"),
            "st_tmag": st.column_config.NumberColumn("TESS mag", format="%.1f"),
        },
    )
