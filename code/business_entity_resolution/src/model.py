
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .features import FEATURE_COLUMNS
from .metrics import macro_f05


def build_model(config, seed: int) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        min_child_weight=config.min_child_weight,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        reg_lambda=config.reg_lambda,
        reg_alpha=config.reg_alpha,
        gamma=config.gamma,
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
    )


def fit_model(model, X: pd.DataFrame, y: pd.Series):
    model.fit(X[FEATURE_COLUMNS], y)
    return model


def tune_thresholds(
    candidate_df: pd.DataFrame,
    scores: np.ndarray,
    truth: dict[str, set[str]],
    min_value: float,
    max_value: float,
    step: float,
):
    work = candidate_df[["source1_entity_id", "candidate_entity_id", "candidate_source"]].copy()
    work["score"] = scores

    values = np.arange(min_value, max_value + step / 2, step)

    def score_pair(t2: float, t3: float) -> float:
        preds: dict[str, set[str]] = {}
        for r in work.itertuples(index=False):
            threshold = t3 if str(r.candidate_source).startswith("S3-") else t2
            if r.score >= threshold:
                preds.setdefault(r.source1_entity_id, set()).add(r.candidate_entity_id)
        return macro_f05(truth, preds)

    # Coordinate descent over the two source-specific thresholds.
    # Repeating until stable is more reliable than a single pass because the
    # optimum for S2 can shift after S3 is changed (and vice versa).
    best2, best3 = 0.80, 0.80
    best_score = score_pair(best2, best3)

    for _ in range(4):
        changed = False
        local_best2, local_score = best2, best_score
        for t2 in values:
            s = score_pair(float(t2), best3)
            if s > local_score + 1e-12:
                local_score, local_best2 = s, float(t2)
        if local_best2 != best2:
            changed = True
            best2, best_score = local_best2, local_score

        local_best3, local_score = best3, best_score
        for t3 in values:
            s = score_pair(best2, float(t3))
            if s > local_score + 1e-12:
                local_score, local_best3 = s, float(t3)
        if local_best3 != best3:
            changed = True
            best3, best_score = local_best3, local_score

        if not changed:
            break

    return {
        "threshold_s2": float(best2),
        "threshold_s3": float(best3),
        "validation_f05": float(best_score),
    }


def save_artifacts(model, thresholds: dict, feature_columns: list[str], out_dir: str | Path):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(out / "xgb_model.json")
    (out / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    (out / "feature_columns.json").write_text(json.dumps(feature_columns, indent=2))


def load_thresholds(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
