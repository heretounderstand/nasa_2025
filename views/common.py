"""Shared loaders, design tokens, page chrome and chart helpers."""
from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from exoplanet.features import FEATURE_DESCRIPTIONS
from exoplanet.model import ARTIFACT_DIR, CANDIDATES_CSV, VettingModel, load_training_data

# ---------------------------------------------------------------------------
# Design tokens
# Planet / false-positive hues are a validated colour-blind-safe pair
# (CVD ΔE 26.8, contrast ≥ 3:1 on the app surface). Anything that is only a
# reference (legacy model, diagonals) is neutral grey so the eye goes to the data.
# ---------------------------------------------------------------------------
SURFACE = "#0B1020"
CARD = "#121933"
BORDER = "rgba(255,255,255,0.08)"
TEXT = "#E8EAF2"
TEXT_2 = "#A9B0C3"
MUTED = "#6E7690"
GRID = "rgba(255,255,255,0.07)"

PLANET = "#3987E5"
FALSE_POS = "#D95926"
NEUTRAL = "#8B93A7"
LEGACY = NEUTRAL
MIDPOINT = "#5B6072"

# FP ← grey → planet: probability is a polarity, so a diverging scale.
PROB_SCALE = [[0.0, FALSE_POS], [0.5, MIDPOINT], [1.0, PLANET]]
# One-hue sequential ramp (counts)
SEQ_SCALE = [[0.0, "#16203d"], [0.5, "#256abf"], [1.0, "#86b6ef"]]

SHORT_LABELS = {
    "st_pmra": "Proper motion RA",
    "st_pmdec": "Proper motion Dec",
    "pl_orbper": "Orbital period",
    "pl_trandurh": "Transit duration",
    "pl_trandep": "Transit depth",
    "pl_rade": "Planet radius",
    "pl_insol": "Insolation",
    "pl_eqt": "Equilibrium temp.",
    "st_tmag": "TESS magnitude",
    "st_dist": "Distance",
    "st_teff": "Star temperature",
    "st_logg": "Star surface gravity",
    "st_rad": "Star radius",
    "st_mass": "Star mass",
    "st_dens": "Star density",
    "a_over_rs": "Orbit size a/R★",
    "rp_rs": "Radius ratio Rp/R★",
    "rp_rs_from_depth": "√depth",
    "depth_consistency": "Depth consistency",
    "dur_ratio": "Duration consistency",
    "dur_over_period": "Duty cycle",
    "pm_total": "Total proper motion",
    "depth_snr_proxy": "Signal strength",
}

UNITS = {
    "pl_orbper": "d", "pl_trandurh": "h", "pl_trandep": "ppm", "pl_rade": "× Earth", "pl_insol": "× Earth",
    "pl_eqt": "K", "st_dist": "pc", "st_teff": "K", "st_rad": "R☉", "st_mass": "M☉", "st_dens": "ρ☉",
    "st_pmra": "mas/yr", "st_pmdec": "mas/yr", "pm_total": "mas/yr", "dur_ratio": "×",
    "depth_consistency": "×",
}

pio.templates["exo"] = go.layout.Template(
    layout=dict(
        font=dict(family="Inter, system-ui, sans-serif", size=13, color=TEXT_2),
        title=dict(font=dict(size=15, color=TEXT), x=0, xanchor="left", pad=dict(l=4)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=[PLANET, FALSE_POS, NEUTRAL],
        xaxis=dict(gridcolor=GRID, linecolor=GRID, zeroline=False, ticks="", title_font=dict(color=TEXT_2)),
        yaxis=dict(gridcolor=GRID, linecolor=GRID, zeroline=False, ticks="", title_font=dict(color=TEXT_2)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)",
                    font=dict(color=TEXT_2)),
        hoverlabel=dict(bgcolor=CARD, bordercolor=BORDER, font=dict(color=TEXT, family="Inter, sans-serif")),
        margin=dict(l=8, r=8, t=48, b=8),
        barcornerradius=4,
        bargap=0.25,
    )
)
pio.templates.default = "plotly_dark+exo"


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def get_model() -> VettingModel:
    return VettingModel.load()


@st.cache_data(show_spinner=False)
def get_training_data() -> tuple[pd.DataFrame, pd.Series]:
    return load_training_data()


@st.cache_data(show_spinner=False)
def get_oof() -> pd.DataFrame:
    return pd.read_csv(ARTIFACT_DIR / "oof.csv")


@st.cache_data(show_spinner="Scoring TESS candidates…")
def get_scored_candidates() -> pd.DataFrame:
    model = get_model()
    cands = pd.read_csv(CANDIDATES_CSV)
    scored = pd.concat([cands, model.predict(cands)], axis=1)
    scored["rank"] = scored["probability"].rank(ascending=False, method="first").astype(int)
    scored["verdict"] = [verdict(p, model.threshold)[0] for p in scored.probability]
    return scored.sort_values("rank").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def label(feature: str) -> str:
    return FEATURE_DESCRIPTIONS.get(feature, feature)


def short(feature: str) -> str:
    return SHORT_LABELS.get(feature, feature)


def fmt(v, feature: str | None = None) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    v = float(v)
    if v == 0:
        s = "0"
    elif abs(v) >= 1e5 or abs(v) < 1e-2:
        s = f"{v:.1e}"
    elif abs(v) >= 100:
        s = f"{v:,.0f}"
    else:
        s = f"{v:.3g}"
    unit = UNITS.get(feature or "", "")
    return f"{s}×" if unit == "×" else f"{s} {unit}".strip()


def verdict(p: float, threshold: float) -> tuple[str, str]:
    if p >= max(threshold, 0.8):
        return "Strong candidate", PLANET
    if p >= threshold:
        return "Likely planet", PLANET
    if p >= min(threshold, 0.2):
        return "Likely false positive", FALSE_POS
    return "False-positive signature", FALSE_POS


# ---------------------------------------------------------------------------
# Page chrome
# ---------------------------------------------------------------------------
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea {{
  font-family: 'Inter', system-ui, sans-serif !important;
}}
.block-container {{ padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1280px; }}
h1, h2, h3, h4 {{ letter-spacing: -0.01em; }}
[data-testid="stSidebar"] {{ border-right: 1px solid {BORDER}; }}

/* page header */
.exo-header {{ margin: 0 0 1.4rem 0; }}
.exo-eyebrow {{ color: {PLANET}; font-size: .78rem; font-weight: 600; letter-spacing: .12em;
  text-transform: uppercase; margin-bottom: .35rem; }}
.exo-title {{ color: {TEXT}; font-size: 2.1rem; font-weight: 700; line-height: 1.15; margin: 0; }}
.exo-sub {{ color: {TEXT_2}; font-size: 1.02rem; margin-top: .5rem; max-width: 52rem; line-height: 1.55; }}

/* hero */
.exo-hero {{ position: relative; overflow: hidden; border-radius: 18px; padding: 2.4rem 2.2rem 2.1rem;
  border: 1px solid {BORDER}; margin-bottom: 1.4rem;
  background:
    radial-gradient(1px 1px at 12% 22%, #ffffffaa 50%, transparent 51%),
    radial-gradient(1px 1px at 32% 68%, #ffffff77 50%, transparent 51%),
    radial-gradient(1.5px 1.5px at 58% 18%, #ffffffaa 50%, transparent 51%),
    radial-gradient(1px 1px at 74% 54%, #ffffff66 50%, transparent 51%),
    radial-gradient(1px 1px at 88% 28%, #ffffff99 50%, transparent 51%),
    radial-gradient(1px 1px at 46% 84%, #ffffff55 50%, transparent 51%),
    radial-gradient(1.5px 1.5px at 92% 80%, #ffffff88 50%, transparent 51%),
    radial-gradient(circle at 105% 120%, #3987e555 0%, transparent 45%),
    radial-gradient(circle at -10% -40%, #6250d633 0%, transparent 50%),
    {CARD}; }}
.exo-hero .exo-title {{ font-size: 2.6rem; }}
.exo-hero .exo-sub {{ font-size: 1.08rem; max-width: calc(100% - 230px); }}
@media (max-width: 900px) {{ .exo-hero .exo-sub {{ max-width: none; }} }}
.exo-planet {{ position: absolute; right: 2.4rem; top: 50%; transform: translateY(-50%);
  width: 150px; height: 150px; border-radius: 50%;
  background: radial-gradient(circle at 35% 30%, #86b6ef 0%, #2a78d6 35%, #0d366b 75%, #050a1a 100%);
  box-shadow: 0 0 60px #3987e544; }}
.exo-planet::after {{ content: ""; position: absolute; left: -38px; right: -38px; top: 64px; height: 22px;
  border-radius: 50%; border: 2px solid #c3d8f499; transform: rotate(-14deg); }}
@media (max-width: 900px) {{ .exo-planet {{ display: none; }} }}

/* metric tiles */
[data-testid="stMetric"] {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 14px;
  padding: 1rem 1.1rem .9rem; }}
[data-testid="stMetricLabel"] p {{ color: {TEXT_2}; font-size: .85rem; font-weight: 500; }}
[data-testid="stMetricValue"] {{ font-weight: 650; font-size: 1.9rem; color: {TEXT}; }}

/* bordered containers as cards */
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 14px; }}

/* badges & sanity checks */
.exo-badge {{ display: inline-flex; align-items: center; gap: .45rem; padding: .3rem .75rem; border-radius: 999px;
  font-weight: 600; font-size: .9rem; border: 1px solid; }}
.exo-badge .dot {{ width: .55rem; height: .55rem; border-radius: 50%; }}
.exo-check {{ display: grid; grid-template-columns: 1.4rem 1fr auto; gap: .2rem .6rem; align-items: start;
  padding: .7rem 0; border-top: 1px solid {BORDER}; }}
.exo-check:first-child {{ border-top: none; }}
.exo-check .ico {{ font-size: 1rem; line-height: 1.4rem; }}
.exo-check .name {{ color: {TEXT}; font-weight: 600; }}
.exo-check .val {{ color: {TEXT}; font-variant-numeric: tabular-nums; font-weight: 600; }}
.exo-check .hint {{ grid-column: 2 / 4; color: {TEXT_2}; font-size: .86rem; }}
.exo-kv {{ color: {TEXT_2}; font-size: .9rem; }}
.exo-kv b {{ color: {TEXT}; font-weight: 600; }}

/* step cards */
.exo-step {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 14px; padding: 1.1rem 1.1rem 1rem;
  height: 100%; }}
.exo-step .n {{ color: {PLANET}; font-weight: 700; font-size: .8rem; letter-spacing: .1em; }}
.exo-step .t {{ color: {TEXT}; font-weight: 650; margin: .3rem 0 .35rem; }}
.exo-step .d {{ color: {TEXT_2}; font-size: .9rem; line-height: 1.5; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, subtitle: str = "") -> None:
    st.markdown(
        f"<div class='exo-header'><div class='exo-eyebrow'>{html.escape(eyebrow)}</div>"
        f"<h1 class='exo-title'>{html.escape(title)}</h1>"
        + (f"<div class='exo-sub'>{subtitle}</div>" if subtitle else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def badge(text: str, color: str) -> str:
    return (f"<span class='exo-badge' style='color:{color};border-color:{color}55;background:{color}1a'>"
            f"<span class='dot' style='background:{color}'></span>{html.escape(text)}</span>")


def check_row(name: str, value: str, status: bool | None, hint: str) -> str:
    icon = {True: "✅", False: "⚠️", None: "➖"}[status]
    word = {True: "pass", False: "flag", None: "no data"}[status]
    return (f"<div class='exo-check'><span class='ico' title='{word}'>{icon}</span>"
            f"<span class='name'>{html.escape(name)}</span><span class='val'>{value}</span>"
            f"<span class='hint'>{html.escape(hint)}</span></div>")


def step_card(n: int, title: str, text: str) -> str:
    return (f"<div class='exo-step'><div class='n'>STEP {n}</div><div class='t'>{html.escape(title)}</div>"
            f"<div class='d'>{text}</div></div>")


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def style(fig: go.Figure, height: int = 380, **layout) -> go.Figure:
    fig.update_layout(height=height, **layout)
    return fig


def gauge(p: float, threshold: float) -> go.Figure:
    color = PLANET if p >= threshold else FALSE_POS
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=p * 100,
            number={"suffix": "%", "font": {"size": 46, "color": TEXT}, "valueformat": ".1f"},
            gauge={
                "shape": "angular",
                "axis": {"range": [0, 100], "ticksuffix": "%", "tickvals": [0, 25, 50, 75, 100],
                         "tickcolor": MUTED, "tickfont": {"color": MUTED, "size": 11}},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "rgba(255,255,255,0.06)",
                "borderwidth": 0,
                "threshold": {"line": {"color": TEXT, "width": 2}, "thickness": 0.9, "value": threshold * 100},
            },
        )
    )
    return style(fig, height=230, margin=dict(l=24, r=24, t=16, b=0))


def waterfall(contrib: pd.Series, base: float, values: pd.Series, top: int = 10) -> go.Figure:
    """SHAP waterfall shown in probability space, with human-readable feature names."""
    order = contrib.abs().sort_values(ascending=False).index
    shown, rest = list(order[:top]), order[top:]
    steps = contrib[shown].tolist()
    names = [f"{short(f)}  <span style='color:{MUTED}'>{fmt(values[f], f)}</span>" for f in shown]
    hover = [f"{label(f)}<br>value: {fmt(values[f], f)}" for f in shown]
    if len(rest):
        steps.append(contrib[rest].sum())
        names.append(f"<span style='color:{MUTED}'>{len(rest)} other features</span>")
        hover.append("Sum of the remaining features")
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    cum, deltas = base, []
    for s in steps:
        deltas.append(sig(cum + s) - sig(cum))
        cum += s
    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            y=["Average TOI"] + names + ["<b>This object</b>"],
            x=[sig(base)] + deltas + [0],
            measure=["absolute"] + ["relative"] * len(deltas) + ["total"],
            text=[f"{sig(base):.0%}"] + [f"{d:+.1%}" for d in deltas] + [f"<b>{sig(cum):.0%}</b>"],
            textposition="outside",
            textfont=dict(color=TEXT_2),
            hovertext=["Share of planets among labelled TOIs (model prior)"] + hover + ["Final probability"],
            hoverinfo="text+x",
            increasing={"marker": {"color": PLANET}},
            decreasing={"marker": {"color": FALSE_POS}},
            totals={"marker": {"color": NEUTRAL}},
            connector={"line": {"color": GRID, "width": 1}},
        )
    )
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(tickformat=".0%", range=[0, 1.12], dtick=0.25)
    return style(fig, height=70 + 30 * (len(names) + 2), showlegend=False, margin=dict(l=8, r=8, t=8, b=8))


def mollweide(lon_deg, lat_deg):
    """Mollweide projection (x in [-2√2, 2√2], y in [-√2, √2]); astronomical convention, RA increasing left."""
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    theta = lat.copy()
    for _ in range(20):  # Newton iterations for 2θ + sin2θ = π sinφ
        f = 2 * theta + np.sin(2 * theta) - np.pi * np.sin(lat)
        d = 2 + 2 * np.cos(2 * theta)
        theta = theta - np.where(d > 1e-9, f / np.where(d > 1e-9, d, 1), 0)
    x = -2 * np.sqrt(2) / np.pi * lon * np.cos(theta)
    return x, np.sqrt(2) * np.sin(theta)


def sky_figure(ra, dec, color, customdata, hovertemplate) -> go.Figure:
    fig = go.Figure()
    for lat in range(-60, 61, 30):  # graticule
        x, yv = mollweide(np.linspace(-180, 180, 120), np.full(120, lat))
        fig.add_trace(go.Scatter(x=x, y=yv, mode="lines", line=dict(color=GRID, width=1), hoverinfo="skip",
                                 showlegend=False))
    for lon in range(-180, 181, 60):
        x, yv = mollweide(np.full(90, lon), np.linspace(-90, 90, 90))
        edge = abs(lon) == 180
        fig.add_trace(go.Scatter(x=x, y=yv, mode="lines", hoverinfo="skip", showlegend=False,
                                 line=dict(color=BORDER if edge else GRID, width=1.5 if edge else 1)))
    lon = (np.asarray(ra) + 180) % 360 - 180
    x, yv = mollweide(lon, dec)
    fig.add_trace(go.Scattergl(
        x=x, y=yv, mode="markers", showlegend=False, customdata=customdata, hovertemplate=hovertemplate,
        marker=dict(size=5, color=color, colorscale=PROB_SCALE, cmin=0, cmax=1, opacity=0.85,
                    colorbar=dict(title=dict(text="P(planet)", side="right"), tickformat=".0%", thickness=10,
                                  outlinewidth=0, len=0.8))))
    for ra_lbl in (0, 60, 120, 240, 300):
        x, yv = mollweide([(ra_lbl + 180) % 360 - 180], [0])
        fig.add_annotation(x=x[0], y=yv[0], text=f"{ra_lbl}°", showarrow=False, font=dict(color=MUTED, size=11),
                           yshift=-10)
    fig.update_xaxes(visible=False, range=[-2.9, 2.9])
    fig.update_yaxes(visible=False, range=[-1.48, 1.48], scaleanchor="x")
    return fig


def transit_schematic(feats: pd.Series, raw: pd.Series) -> go.Figure | None:
    """Phase-folded transit sketch: what TESS saw vs. what a planet of the catalogue size would produce."""
    depth, dur = raw.get("pl_trandep"), raw.get("pl_trandurh")
    if pd.isna(depth) or pd.isna(dur) or depth <= 0:
        return None
    exp_depth = (feats["rp_rs"] ** 2 * 1e6) if pd.notna(feats.get("rp_rs")) else np.nan
    exp_dur = dur / feats["dur_ratio"] if pd.notna(feats.get("dur_ratio")) and feats["dur_ratio"] > 0 else np.nan
    span = 1.6 * np.nanmax([dur, exp_dur if pd.notna(exp_dur) else dur])
    t = np.linspace(-span, span, 600)

    def shape(d_ppm, d_h):
        ingress = 0.12 * d_h
        x = np.abs(t)
        flux = np.where(x <= d_h / 2 - ingress, 1.0, np.clip((d_h / 2 - x) / ingress, 0, 1))
        limb = 1 + 0.15 * (1 - (2 * np.clip(x / max(d_h, 1e-9), 0, 0.5)) ** 2)  # mild limb darkening
        return -d_ppm * flux * limb / 1.15

    fig = go.Figure()
    if pd.notna(exp_depth) and pd.notna(exp_dur):
        fig.add_trace(go.Scatter(x=t, y=shape(exp_depth, exp_dur), name="Expected for a planet of this size",
                                 line=dict(color=NEUTRAL, width=2),
                                 hovertemplate="%{y:,.0f} ppm<extra>expected</extra>"))
    fig.add_trace(go.Scatter(x=t, y=shape(depth, dur), name="Measured by TESS", line=dict(color=PLANET, width=2.5),
                             fill="tozeroy", fillcolor="rgba(57,135,229,0.10)",
                             hovertemplate="%{y:,.0f} ppm<extra>measured</extra>"))
    fig.update_xaxes(title="Hours from mid-transit")
    fig.update_yaxes(title="Brightness change [ppm]")
    return style(fig, height=300, margin=dict(l=8, r=8, t=36, b=8))

