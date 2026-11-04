"""Physics-informed feature engineering for TESS transit signals.

The raw catalogue columns are combined into quantities that a human vetter would
look at to separate real planets from false positives (eclipsing binaries,
blends, systematics):

* does the measured transit depth agree with the catalogue planet/star radii?
* is the transit duration compatible with a planet orbiting *this* star
  (Kepler's third law with the stellar density)?
* is the companion too large to be a planet?

No label information is used anywhere here, so the exact same transformation is
applied to training data and to unseen candidates. Missing values are left as
NaN: LightGBM learns the best direction for them at every split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "tfopwg_disp"
ID_COL = "tid"

# Raw catalogue columns (NASA Exoplanet Archive, TOI table)
RAW_FEATURES: dict[str, str] = {
    "ra": "Right ascension [deg]",
    "dec": "Declination [deg]",
    "st_pmra": "Proper motion RA [mas/yr]",
    "st_pmdec": "Proper motion Dec [mas/yr]",
    "pl_orbper": "Orbital period [days]",
    "pl_trandurh": "Transit duration [hours]",
    "pl_trandep": "Transit depth [ppm]",
    "pl_rade": "Planet radius [Earth radii]",
    "pl_insol": "Insolation [Earth = 1]",
    "pl_eqt": "Equilibrium temperature [K]",
    "st_tmag": "TESS magnitude",
    "st_dist": "Stellar distance [pc]",
    "st_teff": "Stellar effective temperature [K]",
    "st_logg": "Stellar surface gravity [log10(cm/s²)]",
    "st_rad": "Stellar radius [R☉]",
}

# Sky position is not a physical property of the signal: keeping it lets the
# model memorise which sectors/fields were followed up, so it is dropped.
EXCLUDED_RAW = ("ra", "dec")

ENGINEERED_FEATURES: dict[str, str] = {
    "st_mass": "Stellar mass from logg & radius [M☉]",
    "st_dens": "Stellar density [ρ☉]",
    "a_over_rs": "Scaled semi-major axis a/R★ (Kepler III)",
    "rp_rs": "Radius ratio Rp/R★ from catalogue radii",
    "rp_rs_from_depth": "Radius ratio √depth implied by the light curve",
    "depth_consistency": "Observed depth / depth expected from radii",
    "dur_ratio": "Observed duration / expected central-transit duration",
    "dur_over_period": "Transit duty cycle (duration / period)",
    "pm_total": "Total proper motion [mas/yr]",
    "depth_snr_proxy": "log10(depth) − 0.2·Tmag (photometric S/N proxy)",
}

FEATURE_DESCRIPTIONS = {**RAW_FEATURES, **ENGINEERED_FEATURES}

_G = 6.674e-11
_R_SUN = 6.957e8
_M_SUN = 1.989e30
_R_EARTH = 6.371e6


def _safe(series: pd.Series) -> pd.Series:
    return series.replace([np.inf, -np.inf], np.nan)


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Return the model matrix (raw physical columns + engineered features)."""
    missing = [c for c in RAW_FEATURES if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    df = raw[list(RAW_FEATURES)].apply(pd.to_numeric, errors="coerce").astype(float)
    out = df.drop(columns=list(EXCLUDED_RAW))

    g = 10 ** df["st_logg"] / 100.0  # cgs -> m/s²
    r_star = df["st_rad"] * _R_SUN
    m_star = g * r_star**2 / _G
    out["st_mass"] = m_star / _M_SUN
    out["st_dens"] = out["st_mass"] / df["st_rad"] ** 3

    period_s = df["pl_orbper"] * 86400.0
    a = (_G * m_star * period_s**2 / (4 * np.pi**2)) ** (1 / 3)
    out["a_over_rs"] = a / r_star

    out["rp_rs"] = df["pl_rade"] * _R_EARTH / r_star
    depth = df["pl_trandep"].where(df["pl_trandep"] > 0)
    out["rp_rs_from_depth"] = np.sqrt(depth / 1e6)
    out["depth_consistency"] = depth / (out["rp_rs"] ** 2 * 1e6)

    expected_dur_h = df["pl_orbper"] * 24.0 / (np.pi * out["a_over_rs"])
    out["dur_ratio"] = df["pl_trandurh"] / expected_dur_h
    out["dur_over_period"] = df["pl_trandurh"] / (df["pl_orbper"] * 24.0)

    out["pm_total"] = np.hypot(df["st_pmra"], df["st_pmdec"])
    out["depth_snr_proxy"] = np.log10(depth) - 0.2 * df["st_tmag"]

    out = out.apply(_safe)
    return out[feature_names()]


def feature_names() -> list[str]:
    raw = [c for c in RAW_FEATURES if c not in EXCLUDED_RAW]
    return raw + list(ENGINEERED_FEATURES)
