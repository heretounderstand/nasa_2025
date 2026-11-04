"""Training, persistence and inference for the exoplanet vetting model.

Method
------
* Physics-informed features (see :mod:`exoplanet.features`), NaN kept as-is.
* LightGBM with conservative, regularised hyper-parameters.
* Stratified K-fold cross-validation. Every reported metric is computed on
  out-of-fold (OOF) predictions, i.e. on objects the model never saw.
* The K fold models are kept and used as an ensemble at inference time: the
  mean is the planet probability and the spread across folds is a simple
  measure of model uncertainty.
* Explanations are exact TreeSHAP contributions computed natively by LightGBM.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from .features import RAW_FEATURES, TARGET, build_features, feature_names

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
TRAIN_CSV = DATA_DIR / "tess_labeled.csv"
CANDIDATES_CSV = DATA_DIR / "tess_candidates.csv"

DEFAULT_PARAMS: dict = {
    "objective": "binary",
    "n_estimators": 600,
    "learning_rate": 0.02,
    "num_leaves": 15,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.6,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "verbose": -1,
}

# Hyper-parameters and inputs of the original hackathon model, kept so the
# improvement can be measured on exactly the same folds.
LEGACY_PARAMS: dict = {
    "objective": "binary",
    "n_estimators": 900,
    "learning_rate": 0.0375,
    "num_leaves": 60,
    "min_child_samples": 10,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 0.005,
    "min_split_gain": 1.0,
    "verbose": -1,
}


def load_training_data(path: Path = TRAIN_CSV) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    y = df[TARGET].astype(int)
    return df.drop(columns=[TARGET]), y


def metrics_at(y_true, proba, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "log_loss": log_loss(y_true, np.clip(proba, 1e-6, 1 - 1e-6)),
        "brier": brier_score_loss(y_true, proba),
    }


def best_f1_threshold(y_true, proba) -> float:
    grid = np.linspace(0.05, 0.95, 181)
    scores = [f1_score(y_true, proba >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


@dataclass
class CVResult:
    boosters: list[lgb.Booster]
    oof: np.ndarray
    fold_ids: np.ndarray
    fold_metrics: list[dict[str, float]]
    params: dict
    features: list[str]

    @property
    def metrics(self) -> dict[str, float]:
        return metrics_at(self._y, self.oof)

    _y: np.ndarray = field(default=None, repr=False)


def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict | None = None,
    n_splits: int = 5,
    seed: int = 42,
    progress=None,
) -> CVResult:
    """Stratified K-fold training. Returns fold models and OOF predictions."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    params["random_state"] = seed
    y = np.asarray(y)
    oof = np.zeros(len(y))
    fold_ids = np.zeros(len(y), dtype=int)
    boosters, fold_metrics = [], []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (tr, va) in enumerate(skf.split(X, y)):
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X.iloc[tr], y[tr])
        oof[va] = clf.predict_proba(X.iloc[va])[:, 1]
        fold_ids[va] = k
        fold_metrics.append(metrics_at(y[va], oof[va]))
        boosters.append(clf.booster_)
        if progress:
            progress((k + 1) / n_splits)
    res = CVResult(boosters, oof, fold_ids, fold_metrics, params, list(X.columns))
    res._y = y
    return res


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
def save(
    result: CVResult,
    y: pd.Series,
    extra: dict,
    extra_columns: dict | None = None,
    out_dir: Path = ARTIFACT_DIR,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("fold_*.txt"):
        old.unlink()
    for k, b in enumerate(result.boosters):
        b.save_model(out_dir / f"fold_{k}.txt")
    oof = pd.DataFrame({"y": np.asarray(y), "oof": result.oof, "fold": result.fold_ids})
    for name, values in (extra_columns or {}).items():
        oof[name] = np.round(values, 6)
    oof.round(6).to_csv(out_dir / "oof.csv", index=False)
    meta = {
        "features": result.features,
        "params": result.params,
        "n_folds": len(result.boosters),
        "n_samples": int(len(y)),
        "n_planets": int(np.sum(y)),
        "cv_metrics": result.metrics,
        "fold_metrics": result.fold_metrics,
        "threshold": best_f1_threshold(np.asarray(y), result.oof),
        "lightgbm_version": lgb.__version__,
        **extra,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


@dataclass
class VettingModel:
    boosters: list[lgb.Booster]
    meta: dict

    @classmethod
    def load(cls, path: Path = ARTIFACT_DIR) -> "VettingModel":
        meta = json.loads((path / "meta.json").read_text())
        boosters = [lgb.Booster(model_file=str(path / f"fold_{k}.txt")) for k in range(meta["n_folds"])]
        return cls(boosters, meta)

    @property
    def features(self) -> list[str]:
        return self.meta["features"]

    @property
    def threshold(self) -> float:
        return float(self.meta["threshold"])

    def _matrix(self, raw: pd.DataFrame) -> pd.DataFrame:
        return build_features(raw)[self.features]

    def predict(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Return mean probability and fold-to-fold standard deviation."""
        X = self._matrix(raw)
        probs = np.column_stack([b.predict(X) for b in self.boosters])
        return pd.DataFrame(
            {"probability": probs.mean(1), "uncertainty": probs.std(1)}, index=raw.index
        )

    def explain(self, raw: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """TreeSHAP contributions in log-odds space, averaged over folds.

        Returns (contributions[n, features], base_values[n]).
        """
        X = self._matrix(raw)
        contrib = np.mean([b.predict(X, pred_contrib=True) for b in self.boosters], axis=0)
        return pd.DataFrame(contrib[:, :-1], columns=self.features, index=raw.index), contrib[:, -1]


__all__ = [
    "ARTIFACT_DIR",
    "CANDIDATES_CSV",
    "DEFAULT_PARAMS",
    "LEGACY_PARAMS",
    "RAW_FEATURES",
    "TRAIN_CSV",
    "VettingModel",
    "build_features",
    "cross_validate",
    "feature_names",
    "load_training_data",
    "metrics_at",
    "save",
]
