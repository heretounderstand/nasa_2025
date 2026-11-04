"""Train the exoplanet vetting model and write artifacts/.

    python train.py

Runs stratified 5-fold cross-validation for the new pipeline and, on the very
same folds, for the original hackathon setup (raw columns + original
hyper-parameters) so the two can be compared honestly.
"""
from __future__ import annotations

from exoplanet.features import RAW_FEATURES, build_features
from exoplanet.model import (
    LEGACY_PARAMS,
    cross_validate,
    load_training_data,
    metrics_at,
    save,
)


def main() -> None:
    raw, y = load_training_data()
    X = build_features(raw)

    print(f"Training on {len(X)} TOIs ({y.sum()} planets) with {X.shape[1]} features")
    result = cross_validate(X, y)
    for k, m in enumerate(result.fold_metrics):
        print(f"  fold {k}: AUC={m['roc_auc']:.4f} acc={m['accuracy']:.4f}")

    legacy = cross_validate(raw[list(RAW_FEATURES)], y, params=LEGACY_PARAMS)
    legacy_metrics = metrics_at(y, legacy.oof)

    new = metrics_at(y, result.oof)
    print("\nOut-of-fold metrics (threshold 0.5)")
    print(f"{'metric':10s} {'legacy':>8s} {'new':>8s}")
    for key in ("roc_auc", "pr_auc", "accuracy", "f1", "log_loss", "brier"):
        print(f"{key:10s} {legacy_metrics[key]:8.4f} {new[key]:8.4f}")

    save(
        result,
        y,
        extra={"legacy_cv_metrics": legacy_metrics},
        extra_columns={"legacy_oof": legacy.oof},
    )
    print("\nArtifacts written to artifacts/")


if __name__ == "__main__":
    main()
