import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from exoplanet.features import RAW_FEATURES, build_features
from views.common import (
    FALSE_POS,
    PLANET,
    TEXT,
    badge,
    check_row,
    fmt,
    gauge,
    get_model,
    get_scored_candidates,
    get_training_data,
    label,
    page_header,
    short,
    style,
    transit_schematic,
    verdict,
    waterfall,
)

model = get_model()
thr = model.threshold
cands = get_scored_candidates()

page_header(
    "Explore",
    "Candidate explorer",
    f"{len(cands):,} TESS Objects of Interest still awaiting a verdict, ranked by planet probability. "
    "Pick a shortlist, then click a row to open its vetting report.",
)

# ---------------------------------------------------------------- filters
PRESETS = {
    "All": lambda d: d.probability >= 0,
    "Top picks": lambda d: (d.probability >= 0.9) & (d.uncertainty <= 0.03),
    "Small planets": lambda d: d.pl_rade < 2,
    "Temperate": lambda d: d.pl_eqt.between(180, 350),
    "Bright hosts": lambda d: d.st_tmag < 9,
    "Uncertain": lambda d: d.probability.between(0.35, 0.65),
}
HELP = {
    "Top picks": "P ≥ 90 % and the five fold-models agree",
    "Small planets": "Radius < 2 Earth radii",
    "Temperate": "Equilibrium temperature 180–350 K",
    "Bright hosts": "TESS magnitude < 9: easiest follow-up",
    "Uncertain": "35–65 %: where new observations help most",
}

with st.container(border=True):
    preset = st.pills("Shortlist", list(PRESETS), default="All", help=" · ".join(f"{k}: {v}" for k, v in HELP.items()))
    f1, f2, f3 = st.columns([1.3, 1.3, 1])
    p_range = f1.slider("Planet probability", 0.0, 1.0, (0.0, 1.0), 0.05, format="%.2f")
    per = f2.select_slider("Orbital period [days]", options=[0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 365, 1000, 3000],
                           value=(0.1, 3000))
    tic = f3.text_input("Search TIC ID", placeholder="e.g. 112115898")

mask = PRESETS[preset or "All"](cands) & cands.probability.between(*p_range)
mask &= cands.pl_orbper.between(per[0], per[1] if per[1] < 3000 else np.inf) | cands.pl_orbper.isna()
if tic.strip():
    mask &= cands.tid.astype(str).str.contains(tic.strip(), regex=False)
view = cands[mask]

m1, m2, m3, m4 = st.columns(4)
m1.metric("In shortlist", f"{len(view):,}")
m2.metric("Predicted planets", f"{(view.probability >= thr).sum():,}", help=f"P ≥ {thr:.2f} (threshold maximising F1)")
m3.metric("Median probability", f"{view.probability.median():.0%}" if len(view) else "–")
m4.metric("Median radius", f"{view.pl_rade.median():.1f} Earth radii" if view.pl_rade.notna().any() else "–")

cols = ["rank", "tid", "probability", "verdict", "uncertainty", "pl_orbper", "pl_rade", "pl_trandep", "pl_eqt",
        "st_teff", "st_tmag"]
event = st.dataframe(
    view[cols],
    hide_index=True,
    use_container_width=True,
    height=380,
    on_select="rerun",
    selection_mode="single-row",
    key="cand_table",
    column_config={
        "rank": st.column_config.NumberColumn("#", width="small"),
        "tid": st.column_config.NumberColumn("TIC ID", format="%d"),
        "probability": st.column_config.ProgressColumn("P(planet)", min_value=0, max_value=1, format="percent"),
        "verdict": st.column_config.TextColumn("Verdict"),
        "uncertainty": st.column_config.NumberColumn("± spread", format="%.3f"),
        "pl_orbper": st.column_config.NumberColumn("Period [d]", format="%.2f"),
        "pl_rade": st.column_config.NumberColumn("Radius (× Earth)", format="%.2f"),
        "pl_trandep": st.column_config.NumberColumn("Depth [ppm]", format="%d"),
        "pl_eqt": st.column_config.NumberColumn("Teq [K]", format="%d"),
        "st_teff": st.column_config.NumberColumn("Star Teff [K]", format="%d"),
        "st_tmag": st.column_config.NumberColumn("TESS mag", format="%.1f"),
    },
)
st.download_button("Download shortlist (CSV)", view.to_csv(index=False).encode(), "tess_shortlist.csv", "text/csv",
                   icon=":material/download:")

rows = event.selection.rows
if not rows:
    st.info("Click a row in the table to open its vetting report.", icon=":material/touch_app:")
    st.stop()

# ---------------------------------------------------------------- vetting report
row = view.iloc[[rows[0]]]
obj = row.iloc[0]
feats = build_features(row).iloc[0]
contrib, base = model.explain(row)
title, color = verdict(obj.probability, thr)

st.write("")
with st.container(border=True):
    st.markdown(
        f"<div class='exo-eyebrow'>Vetting report</div>"
        f"<div style='display:flex;align-items:center;gap:1rem;flex-wrap:wrap'>"
        f"<h2 style='margin:0;color:{TEXT}'>TIC {int(obj.tid)}</h2>{badge(title, color)}"
        f"<span class='exo-kv'>Rank <b>{int(obj['rank'])}</b> of {len(cands):,} · model spread "
        f"<b>±{obj.uncertainty:.3f}</b></span></div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 2.1], gap="large")
    with left:
        st.plotly_chart(gauge(obj.probability, thr), use_container_width=True, config={"displayModeBar": False})
        st.caption(f"White tick = decision threshold ({thr:.0%}).")
        kv = [("Period", fmt(obj.pl_orbper, "pl_orbper")), ("Radius", fmt(obj.pl_rade, "pl_rade")),
              ("Depth", fmt(obj.pl_trandep, "pl_trandep")), ("Duration", fmt(obj.pl_trandurh, "pl_trandurh")),
              ("Teq", fmt(obj.pl_eqt, "pl_eqt")), ("Host Teff", fmt(obj.st_teff, "st_teff")),
              ("Host radius", fmt(obj.st_rad, "st_rad")), ("Distance", fmt(obj.st_dist, "st_dist"))]
        st.markdown(
            "<div style='display:grid;grid-template-columns:1fr 1fr;gap:.5rem 1rem;margin-top:.6rem'>"
            + "".join(f"<div class='exo-kv'>{k}<br><b>{v}</b></div>" for k, v in kv)
            + "</div>",
            unsafe_allow_html=True,
        )

    with right:
        t_why, t_lc, t_ctx, t_all = st.tabs(["Why this score", "Transit check", "Compared with known TOIs",
                                             "All parameters"])
        with t_why:
            st.caption("Starting from the average TOI, each feature pushes the probability towards "
                       "planet (blue) or false positive (orange).")
            st.plotly_chart(waterfall(contrib.iloc[0], base[0], feats), use_container_width=True,
                            config={"displayModeBar": False})

        with t_lc:
            fig = transit_schematic(feats, obj)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                st.caption("Schematic, built from catalogue values (not the raw light curve). A real planet's "
                           "measured dip should resemble the dip expected from its radius and its star.")
            dc, dr, rp = feats.depth_consistency, feats.dur_ratio, obj.pl_rade
            st.markdown(
                check_row("Depth matches the radii", fmt(dc, "depth_consistency"),
                          None if pd.isna(dc) else bool(0.5 <= dc <= 2),
                          "Measured depth ÷ (Rp/R★)². About 1× for a planet; far off can mean a blend or a "
                          "wrong stellar radius.")
                + check_row("Duration fits Kepler's law", fmt(dr, "dur_ratio"),
                            None if pd.isna(dr) else bool(0.3 <= dr <= 1.3),
                            "Measured ÷ maximum duration for this star and period. A ratio well above 1 suggests "
                            "the eclipsed object is larger or the host is not the star assumed.")
                + check_row("Size is planetary", fmt(rp, "pl_rade"), None if pd.isna(rp) else bool(rp < 22),
                            "Above about 2 Jupiter radii (22 Earth radii), the companion is almost certainly a star."),
                unsafe_allow_html=True,
            )

        with t_ctx:
            train_raw, y = get_training_data()
            tf = build_features(train_raw)
            fx, fy = st.columns(2)
            x_feat = fx.selectbox("X axis", tf.columns, index=list(tf.columns).index("pl_orbper"), format_func=short)
            y_feat = fy.selectbox("Y axis", tf.columns, index=list(tf.columns).index("pl_rade"), format_func=short)
            fig = go.Figure()
            for cls, name, colr in [(0, "Known false positives", FALSE_POS), (1, "Confirmed / known planets", PLANET)]:
                sub = tf[y == cls]
                fig.add_trace(go.Scattergl(x=sub[x_feat], y=sub[y_feat], mode="markers", name=name,
                                           marker=dict(color=colr, size=5, opacity=0.55),
                                           hovertemplate=f"{name}<br>%{{x:.3g}}, %{{y:.3g}}<extra></extra>"))
            fig.add_trace(go.Scatter(x=[feats.get(x_feat, np.nan)], y=[feats.get(y_feat, np.nan)], mode="markers",
                                     name=f"TIC {int(obj.tid)}",
                                     marker=dict(symbol="star", size=22, color="#FFFFFF",
                                                 line=dict(color="#0B1020", width=2))))

            def axis_type(f):
                v = tf[f].dropna()
                return "log" if (v > 0).all() and v.max() / max(v.min(), 1e-12) > 100 else "linear"

            tx, ty = axis_type(x_feat), axis_type(y_feat)
            fig.update_xaxes(type=tx, dtick=1 if tx == "log" else None, title=label(x_feat))
            fig.update_yaxes(type=ty, dtick=1 if ty == "log" else None, title=label(y_feat))
            st.plotly_chart(style(fig, height=420), use_container_width=True)

        with t_all:
            raw = row[list(RAW_FEATURES)].T.rename(columns={row.index[0]: "value"})
            eng = feats.drop(raw.index, errors="ignore").to_frame("value")
            table = pd.concat([raw.assign(kind="catalogue"), eng.assign(kind="derived")])
            table.insert(0, "parameter", [label(c) for c in table.index])
            st.dataframe(table, hide_index=True, use_container_width=True,
                         column_config={"value": st.column_config.NumberColumn(format="%.4g")})

