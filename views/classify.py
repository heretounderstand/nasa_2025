import pandas as pd
import streamlit as st

from exoplanet.features import EXCLUDED_RAW, RAW_FEATURES, build_features
from views.common import (
    TEXT,
    badge,
    gauge,
    get_model,
    get_training_data,
    label,
    page_header,
    transit_schematic,
    verdict,
    waterfall,
)

model = get_model()
thr = model.threshold
train_raw, _ = get_training_data()

page_header("Explore", "Vet a signal",
            "Score a transit signal of your own: type its parameters, or upload a CSV to score many at once.")

tab_one, tab_many = st.tabs([":material/edit: Single signal", ":material/upload_file: Batch (CSV)"])

PRESETS = {
    "Hot Jupiter": dict(pl_orbper=3.5, pl_trandurh=2.8, pl_trandep=10000, pl_rade=11.5, pl_insol=900,
                        pl_eqt=1400, st_tmag=10.5, st_dist=300, st_teff=5800, st_logg=4.4, st_rad=1.1,
                        st_pmra=-5, st_pmdec=-5),
    "Eclipsing binary": dict(pl_orbper=1.2, pl_trandurh=5.5, pl_trandep=60000, pl_rade=35, pl_insol=6000,
                             pl_eqt=2400, st_tmag=11, st_dist=900, st_teff=6500, st_logg=4.0, st_rad=1.8,
                             st_pmra=1, st_pmdec=-2),
    "Warm sub-Neptune": dict(pl_orbper=12.0, pl_trandurh=2.6, pl_trandep=900, pl_rade=2.6, pl_insol=40,
                             pl_eqt=650, st_tmag=9.0, st_dist=60, st_teff=4500, st_logg=4.65, st_rad=0.7,
                             st_pmra=60, st_pmdec=-80),
}
GROUPS = {
    ":material/orbit: Transit & planet": ["pl_orbper", "pl_trandurh", "pl_trandep", "pl_rade", "pl_insol", "pl_eqt"],
    ":material/star: Host star": ["st_teff", "st_logg", "st_rad", "st_tmag", "st_dist", "st_pmra", "st_pmdec"],
}

with tab_one:
    form_col, out_col = st.columns([1, 1.25], gap="large")
    with form_col:
        preset = st.segmented_control("Example", list(PRESETS), default=list(PRESETS)[0])
        values = PRESETS[preset or list(PRESETS)[0]]
        inputs = {}
        with st.form("signal", border=True):
            for g, feats in GROUPS.items():
                st.markdown(f"**{g}**")
                cols = st.columns(2)
                for i, f in enumerate(feats):
                    inputs[f] = cols[i % 2].number_input(label(f), value=float(values[f]), format="%.4g",
                                                         key=f"{preset}_{f}")
            submitted = st.form_submit_button("Score this signal", type="primary", use_container_width=True,
                                              icon=":material/rocket_launch:")
    if submitted:
        st.session_state.last_signal = inputs

    with out_col:
        if "last_signal" not in st.session_state:
            st.info("Pick an example or type your values, then press **Score this signal**.",
                    icon=":material/arrow_back:")
        else:
            sig = dict(st.session_state.last_signal)
            sig.update({c: 0.0 for c in EXCLUDED_RAW})
            row = pd.DataFrame([sig])
            pred = model.predict(row).iloc[0]
            contrib, base = model.explain(row)
            feats = build_features(row).iloc[0]
            title, color = verdict(pred.probability, thr)
            with st.container(border=True):
                st.markdown(
                    f"<div class='exo-eyebrow'>Result</div><div style='display:flex;gap:1rem;align-items:center;"
                    f"flex-wrap:wrap'><h3 style='margin:0;color:{TEXT}'>{pred.probability:.1%} planet</h3>"
                    f"{badge(title, color)}<span class='exo-kv'>model spread <b>±{pred.uncertainty:.3f}</b>"
                    "</span></div>",
                    unsafe_allow_html=True,
                )
                _, g, _ = st.columns([1, 4, 1])
                g.plotly_chart(gauge(pred.probability, thr), use_container_width=True,
                               config={"displayModeBar": False})
                t1, t2 = st.tabs(["Why this score", "Transit check"])
                with t1:
                    st.plotly_chart(waterfall(contrib.iloc[0], base[0], feats), use_container_width=True,
                                    config={"displayModeBar": False})
                with t2:
                    fig = transit_schematic(feats, pd.Series(sig))
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with tab_many:
    needed = [c for c in RAW_FEATURES if c not in EXCLUDED_RAW]
    with st.container(border=True):
        st.markdown("**1. Get the template.** Use the same units as the NASA Exoplanet Archive TOI table; "
                    "blank cells are fine.")
        st.download_button("Download CSV template", train_raw[needed].head(5).to_csv(index=False).encode(),
                           "template.csv", "text/csv", icon=":material/download:")
        st.markdown("**2. Upload your file.**")
        up = st.file_uploader("CSV file", type="csv", label_visibility="collapsed")
    if up is not None:
        df = pd.read_csv(up)
        missing = [c for c in needed if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            for c in EXCLUDED_RAW:
                if c not in df.columns:
                    df[c] = 0.0
            out = pd.concat([df, model.predict(df)], axis=1)
            out["verdict"] = [verdict(p, thr)[0] for p in out.probability]
            a, b, c = st.columns(3)
            a.metric("Signals scored", f"{len(out):,}")
            b.metric("Predicted planets", f"{(out.probability >= thr).sum():,}")
            c.metric("Median probability", f"{out.probability.median():.0%}")
            st.dataframe(
                out.sort_values("probability", ascending=False),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "probability": st.column_config.ProgressColumn(
                        "P(planet)", min_value=0, max_value=1, format="percent"
                    )
                },
            )
            st.download_button("Download results", out.to_csv(index=False).encode(), "scored.csv", "text/csv",
                               type="primary", icon=":material/download:")
