# 🪐 Exoplanet Hunter

NASA Space Apps Challenge 2025 – *A World Away: Hunting for Exoplanets with AI*.

A Streamlit app that ranks TESS Objects of Interest (TOIs) by how likely their transit signal is a
real planet, and explains each decision.

```bash
pip install -r requirements.txt
streamlit run app.py
```

To retrain the model (≈10 s) and regenerate `artifacts/`:

```bash
python train.py
```

## Method

| Step | What | Why |
|---|---|---|
| Features | 13 catalogue columns + 10 physics-derived ones: stellar mass & density (from log g and R★), a/R★ via Kepler III, Rp/R★, **observed vs. expected transit depth**, **observed vs. expected transit duration**, duty cycle, S/N proxy | These are the checks a human vetter makes to spot eclipsing binaries and blends |
| Missing values | Left as NaN, handled natively by LightGBM | No information from the label can leak into the inputs |
| Sky position | RA/Dec dropped | Not a property of the signal; lets the model memorise survey footprints |
| Model | LightGBM, 600 shallow trees (15 leaves), strong subsampling | Better-calibrated probabilities than deep trees on 2.5 k samples |
| Validation | Stratified 5-fold CV, every metric out-of-fold | Honest estimate of performance on unseen candidates |
| Inference | Mean of the 5 fold models; their spread = uncertainty | Free ensemble + confidence indicator |
| Explanations | Exact TreeSHAP via LightGBM `pred_contrib` | Per-candidate "why" without extra dependencies |

Out-of-fold results on 2,562 labelled TOIs (same folds for both):

| | Hackathon version | This version |
|---|---|---|
| ROC-AUC | 0.923 | **0.935** |
| Accuracy (thr 0.5) | 84.2 % | **86.9 %** |
| F1 (thr 0.5) | 0.845 | **0.870** |
| Log-loss | 0.352 | **0.323** |

The hackathon app displayed ≈98 % accuracy because it evaluated the model on its own training data and
imputed missing values with the median of the *true* class.

## Layout

```
app.py                 Streamlit entry point (navigation)
views/                 One file per page + shared helpers
exoplanet/features.py  Feature engineering
exoplanet/model.py     CV training, persistence, prediction, SHAP
train.py               Rebuilds artifacts/
data/                  Labelled TOIs and open candidates (NASA Exoplanet Archive, Oct 2025)
artifacts/             5 fold models (LightGBM text format), OOF predictions, metadata
```
