"""Shared loaders, colours and chart helpers for the Streamlit pages."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from exoplanet.features import FEATURE_DESCRIPTIONS, ID_COL
from exoplanet.model import ARTIFACT_DIR, CANDIDATES_CSV, VettingModel, load_training_data

PLANET = "#4C9AFF"
FALSE_POS = "#F28C38"
NEUTRAL = "#8B93A7"
LEGACY = "#A78BFA"


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
    return scored.sort_values("rank").reset_index(drop=True)


def label(feature: str) -> str:
    return FEATURE_DESCRIPTIONS.get(feature, feature)


def verdict(p: float, threshold: float) -> tuple[str, str]:
    if p >= max(threshold, 0.8):
        return "Strong planet candidate", PLANET
    if p >= threshold:
        return "Likely planet", PLANET
    if p >= min(threshold, 0.2):
        return "Likely false positive", FALSE_POS
    return "Strong false-positive signature", FALSE_POS


def style(fig: go.Figure, height: int = 380, **layout) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(size=13),
        **layout,
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", zeroline=False)
    return fig


def gauge(p: float, threshold: float) -> go.Figure:
    color = PLANET if p >= threshold else FALSE_POS
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=p * 100,
            number={"suffix": "%", "font": {"size": 44}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": color, "thickness": 0.35},
                "bgcolor": "rgba(255,255,255,0.05)",
                "borderwidth": 0,
                "threshold": {"line": {"color": "#E6E9F2", "width": 3}, "value": threshold * 100},
            },
            title={"text": "Planet probability"},
        )
    )
    return style(fig, height=260)


def waterfall(contrib: pd.Series, base: float, values: pd.Series, top: int = 10) -> go.Figure:
    """SHAP waterfall in probability space (log-odds contributions shown as labels)."""
    order = contrib.abs().sort_values(ascending=False).index
    shown, rest = list(order[:top]), order[top:]
    steps = contrib[shown].tolist()
    names = [f"{f} = {_fmt(values[f])}" for f in shown]
    if len(rest):
        steps.append(contrib[rest].sum())
        names.append(f"{len(rest)} other features")
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    # Convert cumulative log-odds into probability deltas for readability.
    cum = base
    deltas = []
    for s in steps:
        deltas.append(sig(cum + s) - sig(cum))
        cum += s
    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            y=["Average TOI"] + names + ["This object"],
            x=[sig(base)] + deltas + [0],
            measure=["absolute"] + ["relative"] * len(deltas) + ["total"],
            text=[f"{sig(base):.0%}"] + [f"{d:+.1%}" for d in deltas] + [f"{sig(cum):.0%}"],
            textposition="outside",
            increasing={"marker": {"color": PLANET}},
            decreasing={"marker": {"color": FALSE_POS}},
            totals={"marker": {"color": NEUTRAL}},
            connector={"line": {"color": "rgba(255,255,255,0.2)"}},
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickformat=".0%", range=[0, 1.08])
    return style(fig, height=90 + 34 * (len(names) + 2), showlegend=False)


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "missing"
    v = float(v)
    if v == 0:
        return "0"
    if abs(v) >= 1e4 or abs(v) < 1e-2:
        return f"{v:.2e}"
    return f"{v:.3g}"


__all__ = [
    "FALSE_POS",
    "ID_COL",
    "LEGACY",
    "NEUTRAL",
    "PLANET",
    "gauge",
    "get_model",
    "get_oof",
    "get_scored_candidates",
    "get_training_data",
    "label",
    "style",
    "verdict",
    "waterfall",
]
