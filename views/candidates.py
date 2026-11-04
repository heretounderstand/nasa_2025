import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from exoplanet.features import RAW_FEATURES, build_features
from views.common import (
    FALSE_POS,
    PLANET,
    gauge,
    get_model,
    get_scored_candidates,
    get_training_data,
    label,
    style,
    verdict,
    waterfall,
)

model = get_model()
thr = model.threshold
cands = get_scored_candidates()

st.title("🔭 Candidate explorer")
st.caption(
    f"{len(cands):,} TESS Objects of Interest still awaiting a disposition, ranked by the model. "
    "Select a row to see why the model scored it that way."
)

with st.container(border=True):
    f1, f2, f3, f4 = st.columns([1.2, 1, 1, 1])
    p_range = f1.slider("P(planet)", 0.0, 1.0, (0.0, 1.0), 0.05, format="%.2f")
    rmax = float(np.nanpercentile(cands.pl_rade, 99.5))
    r_range = f2.slider("Radius [R⊕] (max = no limit)", 0.0, round(rmax, 0), (0.0, round(rmax, 0)))
    max_unc = f3.slider("Max uncertainty", 0.0, 0.3, 0.3, 0.01, help="Spread of the 5 fold models")
    tic = f4.text_input("Search TIC ID", placeholder="e.g. 88863718")

mask = cands.probability.between(*p_range) & (cands.uncertainty <= max_unc)
r_hi = np.inf if r_range[1] >= round(rmax, 0) else r_range[1]  # top of slider = no upper limit
mask &= cands.pl_rade.between(r_range[0], r_hi) | cands.pl_rade.isna()
if tic.strip():
    mask &= cands.tid.astype(str).str.contains(tic.strip())
view = cands[mask]

m1, m2, m3 = st.columns(3)
m1.metric("Shown", f"{len(view):,}")
m2.metric("Above decision threshold", f"{(view.probability >= thr).sum():,}", help=f"Threshold = {thr:.2f} (max F1)")
m3.metric("Median P(planet)", f"{view.probability.median():.0%}" if len(view) else "–")

cols = ["rank", "tid", "probability", "uncertainty", "pl_orbper", "pl_trandurh", "pl_trandep",
        "pl_rade", "pl_eqt", "st_teff", "st_tmag"]
event = st.dataframe(
    view[cols],
    hide_index=True,
    use_container_width=True,
    height=360,
    on_select="rerun",
    selection_mode="single-row",
    column_config={
        "rank": "Rank",
        "tid": st.column_config.NumberColumn("TIC ID", format="%d"),
        "probability": st.column_config.ProgressColumn("P(planet)", min_value=0, max_value=1, format="percent"),
        "uncertainty": st.column_config.NumberColumn("±", format="%.3f"),
        "pl_orbper": st.column_config.NumberColumn("Period [d]", format="%.3f"),
        "pl_trandurh": st.column_config.NumberColumn("Duration [h]", format="%.2f"),
        "pl_trandep": st.column_config.NumberColumn("Depth [ppm]", format="%.0f"),
        "pl_rade": st.column_config.NumberColumn("Radius [R⊕]", format="%.2f"),
        "pl_eqt": st.column_config.NumberColumn("Teq [K]", format="%.0f"),
        "st_teff": st.column_config.NumberColumn("Teff [K]", format="%.0f"),
        "st_tmag": st.column_config.NumberColumn("Tmag", format="%.2f"),
    },
)
st.download_button(
    "Download filtered list (CSV)",
    view.to_csv(index=False).encode(),
    "tess_candidates_scored.csv",
    "text/csv",
    icon=":material/download:",
)

rows = event.selection.rows
if not rows:
    st.info("Select a candidate in the table to open its vetting report.", icon="👆")
    st.stop()

row = view.iloc[[rows[0]]]
obj = row.iloc[0]
st.divider()
st.subheader(f"Vetting report · TIC {int(obj.tid)}")

contrib, base = model.explain(row)
feats = build_features(row).iloc[0]
title, color = verdict(obj.probability, thr)

a, b = st.columns([1, 1.6])
with a:
    st.plotly_chart(gauge(obj.probability, thr), use_container_width=True)
    st.markdown(f"<h4 style='text-align:center;color:{color};margin-top:-10px'>{title}</h4>", unsafe_allow_html=True)
    st.caption(f"Rank {int(obj['rank'])} of {len(cands):,} · fold-model spread ±{obj.uncertainty:.3f}")

    st.markdown("**Physical sanity checks**")

    def check(name, value, ok, fmt, hint):
        if pd.isna(value):
            st.markdown(f"⚪ {name}: *not available*")
        else:
            st.markdown(f"{'🟢' if ok else '🟠'} {name}: **{fmt.format(value)}** — {hint}")

    dc = feats.depth_consistency
    check("Depth vs. radii", dc, pd.notna(dc) and 0.5 <= dc <= 2, "{:.2f}×", "≈1 if the depth matches Rp/R★")
    dr = feats.dur_ratio
    check("Duration vs. Kepler", dr, pd.notna(dr) and 0.3 <= dr <= 1.3, "{:.2f}×",
          "≤1 for a planet orbiting this star")
    rp = obj.pl_rade
    check("Companion size", rp, pd.notna(rp) and rp < 22, "{:.1f} R⊕", "> ~2 R♃ suggests a stellar companion")

with b:
    st.markdown("**Why this score?** Each bar is how much a feature pushed the probability up (blue) or down (orange).")
    st.plotly_chart(waterfall(contrib.iloc[0], base[0], feats), use_container_width=True)

with st.expander("All parameters"):
    raw = row[list(RAW_FEATURES)].T.rename(columns={row.index[0]: "value"})
    raw["description"] = [label(c) for c in raw.index]
    eng = feats.drop(raw.index, errors="ignore").to_frame("value")
    eng["description"] = [label(c) for c in eng.index]
    st.dataframe(pd.concat([raw, eng]), use_container_width=True)

# Context: where does this object sit among labelled TOIs?
st.markdown("**In context** – this candidate (★) against TOIs with a known outcome")
train_raw, y = get_training_data()
tf = build_features(train_raw)
c1, c2 = st.columns(2)
for col, (fx, fy, lx, ly) in zip(
    (c1, c2),
    [("pl_orbper", "pl_rade", True, True), ("depth_consistency", "dur_ratio", True, True)],
):
    fig = go.Figure()
    for cls, name, colr in [(0, "False positive", FALSE_POS), (1, "Planet", PLANET)]:
        sub = tf[y == cls]
        fig.add_trace(go.Scattergl(x=sub[fx], y=sub[fy], mode="markers", name=name,
                                   marker=dict(color=colr, size=4, opacity=0.45)))
    fig.add_trace(go.Scatter(x=[feats[fx]], y=[feats[fy]], mode="markers", name=f"TIC {int(obj.tid)}",
                             marker=dict(symbol="star", size=20, color="#FFFFFF", line=dict(color="#000", width=1))))
    fig.update_xaxes(type="log" if lx else "linear", dtick=1 if lx else None, title=label(fx))
    fig.update_yaxes(type="log" if ly else "linear", dtick=1 if ly else None, title=label(fy))
    col.plotly_chart(style(fig, height=380), use_container_width=True)
