import pandas as pd
import streamlit as st

from exoplanet.features import EXCLUDED_RAW, RAW_FEATURES, build_features
from views.common import gauge, get_model, get_training_data, label, verdict, waterfall

model = get_model()
thr = model.threshold
train_raw, _ = get_training_data()

st.title("🧪 Vet a signal")
st.caption("Score your own transit signal – type its parameters, or upload a CSV with many of them.")

tab_one, tab_many = st.tabs(["Single signal", "Batch (CSV)"])

PRESETS = {
    "Typical hot Jupiter": dict(pl_orbper=3.5, pl_trandurh=2.8, pl_trandep=10000, pl_rade=11.5, pl_insol=900,
                                 pl_eqt=1400, st_tmag=10.5, st_dist=300, st_teff=5800, st_logg=4.4, st_rad=1.1,
                                 st_pmra=-5, st_pmdec=-5),
    "Eclipsing-binary-like": dict(pl_orbper=1.2, pl_trandurh=5.5, pl_trandep=60000, pl_rade=35, pl_insol=6000,
                                   pl_eqt=2400, st_tmag=11, st_dist=900, st_teff=6500, st_logg=4.0, st_rad=1.8,
                                   st_pmra=1, st_pmdec=-2),
    "Small warm planet": dict(pl_orbper=12.0, pl_trandurh=2.6, pl_trandep=900, pl_rade=2.6, pl_insol=40,
                               pl_eqt=650, st_tmag=9.0, st_dist=60, st_teff=4500, st_logg=4.65, st_rad=0.7,
                               st_pmra=60, st_pmdec=-80),
}

with tab_one:
    preset = st.segmented_control("Start from", list(PRESETS), default=list(PRESETS)[0])
    values = PRESETS[preset or list(PRESETS)[0]]
    inputs = {}
    with st.form("signal"):
        groups = {
            "Transit": ["pl_orbper", "pl_trandurh", "pl_trandep", "pl_rade", "pl_insol", "pl_eqt"],
            "Host star": ["st_teff", "st_logg", "st_rad", "st_tmag", "st_dist", "st_pmra", "st_pmdec"],
        }
        for (g, feats), col in zip(groups.items(), st.columns(2)):
            with col:
                st.markdown(f"**{g}**")
                for f in feats:
                    inputs[f] = st.number_input(label(f), value=float(values[f]), format="%.4g", key=f"{preset}_{f}")
        submitted = st.form_submit_button("Score signal", type="primary", use_container_width=True)

    if submitted or "last_signal" in st.session_state:
        if submitted:
            st.session_state.last_signal = inputs
        sig = dict(st.session_state.last_signal)
        sig.update({c: 0.0 for c in EXCLUDED_RAW})
        row = pd.DataFrame([sig])
        pred = model.predict(row).iloc[0]
        contrib, base = model.explain(row)
        title, color = verdict(pred.probability, thr)
        a, b = st.columns([1, 1.6])
        with a:
            st.plotly_chart(gauge(pred.probability, thr), use_container_width=True)
            st.markdown(f"<h4 style='text-align:center;color:{color}'>{title}</h4>", unsafe_allow_html=True)
            st.caption(f"Fold-model spread ±{pred.uncertainty:.3f} · decision threshold {thr:.2f}")
        with b:
            st.plotly_chart(waterfall(contrib.iloc[0], base[0], build_features(row).iloc[0]), use_container_width=True)

with tab_many:
    needed = [c for c in RAW_FEATURES if c not in EXCLUDED_RAW]
    st.markdown(
        "Upload a CSV with these columns (same units as the NASA Exoplanet Archive TOI table, blanks allowed): "
        + ", ".join(f"`{c}`" for c in needed)
    )
    template = train_raw[needed].head(5)
    st.download_button("Download a template", template.to_csv(index=False).encode(), "template.csv", "text/csv",
                       icon=":material/download:")
    up = st.file_uploader("CSV file", type="csv")
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
            out["predicted_planet"] = out.probability >= thr
            st.success(f"Scored {len(out):,} signals – {out.predicted_planet.sum():,} above threshold.")
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
