
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import Config
from .io import read_source, read_ground_truth
from .blocking import enrich, generate_candidates, compress_candidates
from .features import make_feature_frame, FEATURE_COLUMNS
from .model import build_model, fit_model, tune_thresholds, save_artifacts, load_thresholds
from .metrics import macro_f05, precision_recall, candidate_recall


def combine_targets(s2: pd.DataFrame, s3: pd.DataFrame) -> pd.DataFrame:
    a = s2.copy()
    a["source"] = "S2"
    b = s3.copy()
    b["source"] = "S3"
    return pd.concat([a, b], ignore_index=True)


def _candidate_dict(df: pd.DataFrame, col: str) -> dict[str, set[str]]:
    if df.empty:
        return {}
    out = {}
    for sid, g in df.groupby("source1_entity_id"):
        out[sid] = set(g[col].astype(str))
    return out


def _label_candidates(candidates: pd.DataFrame, truth: dict[str, set[str]]) -> pd.Series:
    return candidates.apply(
        lambda r: int(
            str(r["candidate_entity_id"])
            in truth.get(str(r["source1_entity_id"]), set())
        ),
        axis=1,
    ).astype(np.int8)


def _select_training_pairs(
    candidates: pd.DataFrame,
    labels: pd.Series,
    config: Config,
) -> pd.DataFrame:
    if candidates.empty:
        raise ValueError("No candidates generated; improve blocking before training.")

    work = candidates.copy()
    work["_label"] = labels.to_numpy()

    selected = []
    rng = np.random.default_rng(config.random_seed)

    from rapidfuzz import fuzz

    for sid, g in work.groupby("source1_entity_id", sort=False):
        pos = g[g["_label"] == 1]
        neg = g[g["_label"] == 0].copy()

        # Hard negatives first: similar text but non-matching.
        if not neg.empty:
            neg["_hardness"] = neg.apply(
                lambda r: 0.60 * fuzz.token_ratio(
                    str(r["s1_name"]), str(r["candidate_name"])
                ) / 100.0
                + 0.40 * fuzz.token_ratio(
                    str(r["s1_address"]), str(r["candidate_address"])
                ) / 100.0,
                axis=1,
            )
            neg = neg.sort_values(
                ["_hardness", "shared_blocks", "candidate_entity_id"],
                ascending=[False, False, True],
            )

        if len(pos) > 0:
            n_neg = min(
                len(neg),
                max(config.min_training_candidates_per_s1, len(pos) * config.negatives_per_positive),
            )
        else:
            n_neg = min(len(neg), config.singleton_negatives)

        chosen_neg = neg.head(n_neg)

        # Add a small random slice of easy negatives if available.
        remaining = neg.iloc[n_neg:]
        if len(remaining) and len(chosen_neg) < n_neg + 2:
            extra_n = min(2, len(remaining))
            extra_idx = rng.choice(len(remaining), size=extra_n, replace=False)
            chosen_neg = pd.concat([chosen_neg, remaining.iloc[extra_idx]])

        selected.append(pd.concat([pos, chosen_neg], ignore_index=True))

    out = pd.concat(selected, ignore_index=True)
    return out.drop(columns=["_hardness"], errors="ignore")


def train_pipeline(
    train_dir: str | Path,
    artifact_dir: str | Path,
    config: Config,
):
    train_dir = Path(train_dir)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    s1 = enrich(read_source(train_dir / "train_source1.tsv"))
    s2 = enrich(read_source(train_dir / "train_source2.tsv"))
    s3 = enrich(read_source(train_dir / "train_source3.tsv"))
    truth_all = read_ground_truth(train_dir / "train_ground_truth.tsv")

    s1_ids = np.array(sorted(s1["entity_id"].astype(str).unique()))
    train_ids, val_ids = train_test_split(
        s1_ids,
        test_size=config.validation_fraction,
        random_state=config.random_seed,
    )
    train_id_set, val_id_set = set(train_ids), set(val_ids)

    target = combine_targets(s2, s3)

    # Build candidates for each split using only candidate-generation logic.
    train_s1 = s1[s1["entity_id"].isin(train_id_set)].copy()
    val_s1 = s1[s1["entity_id"].isin(val_id_set)].copy()

    train_cand = generate_candidates(train_s1, target, config)
    val_cand = generate_candidates(val_s1, target, config)

    # Candidate compression defines the candidate set that the final model would see.
    train_cand = compress_candidates(train_cand, config)
    val_cand = compress_candidates(val_cand, config)

    train_truth = {sid: truth_all.get(sid, set()) for sid in train_id_set}
    val_truth = {sid: truth_all.get(sid, set()) for sid in val_id_set}

    # Measure blocking recall before training labels are sampled.
    blocking_stats = {
        "train_candidate_recall": candidate_recall(train_truth, _candidate_dict(train_cand, "candidate_entity_id")),
        "val_candidate_recall": candidate_recall(val_truth, _candidate_dict(val_cand, "candidate_entity_id")),
        "train_avg_candidates": float(train_cand.groupby("source1_entity_id").size().mean()) if not train_cand.empty else 0.0,
        "val_avg_candidates": float(val_cand.groupby("source1_entity_id").size().mean()) if not val_cand.empty else 0.0,
        "val_p95_candidates": float(train_cand.groupby("source1_entity_id").size().quantile(0.95)) if not train_cand.empty else 0.0,
    }

    train_labels_all = _label_candidates(train_cand, train_truth)
    selected_train = _select_training_pairs(train_cand, train_labels_all, config)
    X_train = make_feature_frame(selected_train)
    y_train = selected_train["_label"]

    # Optional: ensure there is at least one positive. A zero-positive training
    # set means blocking is catastrophically poor and should be fixed.
    if int(y_train.sum()) == 0:
        raise ValueError("Training candidates contain no positive matches. Fix blocking first.")

    model = build_model(config, config.random_seed)
    fit_model(model, X_train, y_train)

    # Validation must use the final candidate set and no ground-truth injection.
    X_val = make_feature_frame(val_cand)
    if len(X_val):
        val_scores = model.predict_proba(X_val[FEATURE_COLUMNS])[:, 1]
    else:
        val_scores = np.array([], dtype=float)

    thresholds = tune_thresholds(
        val_cand,
        val_scores,
        val_truth,
        config.threshold_min,
        config.threshold_max,
        config.threshold_step,
    )

    save_artifacts(model, thresholds, FEATURE_COLUMNS, artifact_dir)

    summary = {
        **blocking_stats,
        **thresholds,
        "train_rows_used": int(len(selected_train)),
        "train_positive_rows": int(y_train.sum()),
        "val_candidate_rows": int(len(val_cand)),
        "val_entities": int(len(val_s1)),
    }
    (artifact_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    (artifact_dir / "validation_s1_ids.json").write_text(json.dumps(sorted(val_id_set), indent=2))

    # Save validation score file for evaluation/audit; gzip keeps it compact.
    scored = val_cand[
        ["source1_entity_id", "candidate_entity_id", "candidate_source", "shared_blocks"]
    ].copy()
    scored["score"] = val_scores
    scored["label"] = _label_candidates(val_cand, val_truth).to_numpy()
    scored.to_csv(artifact_dir / "validation_scores.tsv.gz", sep="\t", index=False, compression="gzip")

    return summary


def run_inference(
    test_dir: str | Path,
    artifact_dir: str | Path,
    output_dir: str | Path,
    config: Config,
):
    test_dir, artifact_dir, output_dir = map(Path, [test_dir, artifact_dir, output_dir])
    output_dir.mkdir(parents=True, exist_ok=True)

    s1 = enrich(read_source(test_dir / "test_source1.tsv"))
    s2 = enrich(read_source(test_dir / "test_source2.tsv"))
    s3 = enrich(read_source(test_dir / "test_source3.tsv"))
    target = combine_targets(s2, s3)

    # Load model.
    from xgboost import XGBClassifier
    model = XGBClassifier()
    model.load_model(artifact_dir / "xgb_model.json")
    thresholds = load_thresholds(artifact_dir / "thresholds.json")

    candidates = generate_candidates(s1, target, config)
    candidates = compress_candidates(candidates, config)

    X = make_feature_frame(candidates)
    if len(X):
        scores = model.predict_proba(X[FEATURE_COLUMNS])[:, 1]
    else:
        scores = np.array([], dtype=float)

    candidates = candidates.copy()
    candidates["score"] = scores

    # Save candidate_pairs.tsv exactly from the candidate set actually scored.
    candidate_out = []
    for sid, g in candidates.groupby("source1_entity_id", sort=False):
        g = g.sort_values(
            ["shared_blocks", "score", "candidate_entity_id"],
            ascending=[False, False, True],
        )
        candidate_out.append({
            "source1_entity_id": sid,
            "candidate_entity_ids": ",".join(g["candidate_entity_id"].astype(str)),
        })

    candidate_map = {x["source1_entity_id"]: x["candidate_entity_ids"] for x in candidate_out}
    matching_rows = []

    for sid in s1["entity_id"].astype(str):
        g = candidates[candidates["source1_entity_id"] == sid]
        if g.empty:
            matching_rows.append({
                "source1_entity_id": sid,
                "matched_entity_ids": "",
            })
            continue

        matched = []
        for r in g.sort_values(["score", "candidate_entity_id"], ascending=[False, True]).itertuples():
            threshold = (
                thresholds["threshold_s3"]
                if str(r.candidate_source).startswith("S3-")
                else thresholds["threshold_s2"]
            )
            if float(r.score) >= threshold:
                matched.append(str(r.candidate_entity_id))

        # De-duplicate and keep deterministic ordering.
        matched = sorted(set(matched))
        matching_rows.append({
            "source1_entity_id": sid,
            "matched_entity_ids": ",".join(matched),
        })

    matching_df = pd.DataFrame(
        matching_rows,
        columns=["source1_entity_id", "matched_entity_ids"],
    )
    candidate_df = pd.DataFrame(
        candidate_out,
        columns=["source1_entity_id", "candidate_entity_ids"],
    )

    # Ensure every S1 appears exactly once, even if it has no candidates.
    all_s1 = pd.DataFrame({"source1_entity_id": s1["entity_id"].astype(str)})
    candidate_df = (
        all_s1.merge(candidate_df, on="source1_entity_id", how="left")
        .fillna("")
    )
    matching_df = (
        all_s1.merge(matching_df, on="source1_entity_id", how="left")
        .fillna("")
    )

    matching_df.to_csv(
        output_dir / "matching_results.tsv",
        sep="\t",
        index=False,
    )
    candidate_df.to_csv(
        output_dir / "candidate_pairs.tsv",
        sep="\t",
        index=False,
    )

    return {
        "s1_entities": int(len(s1)),
        "candidate_rows": int(len(candidates)),
        "non_empty_predictions": int((matching_df["matched_entity_ids"] != "").sum()),
        "threshold_s2": thresholds["threshold_s2"],
        "threshold_s3": thresholds["threshold_s3"],
    }


def evaluate_saved_validation(artifact_dir: str | Path, train_dir: str | Path):
    artifact_dir, train_dir = Path(artifact_dir), Path(train_dir)
    scores = pd.read_csv(artifact_dir / "validation_scores.tsv.gz", sep="\t", dtype=str)
    scores["score"] = scores["score"].astype(float)

    truth_all = read_ground_truth(train_dir / "train_ground_truth.tsv")
    thresholds = load_thresholds(artifact_dir / "thresholds.json")

    preds = {}
    for r in scores.itertuples(index=False):
        threshold = (
            thresholds["threshold_s3"]
            if str(r.candidate_source).startswith("S3-")
            else thresholds["threshold_s2"]
        )
        if r.score >= threshold:
            preds.setdefault(r.source1_entity_id, set()).add(r.candidate_entity_id)

    ids_path = artifact_dir / "validation_s1_ids.json"
    if ids_path.exists():
        val_s1_ids = json.loads(ids_path.read_text())
    else:
        val_s1_ids = sorted(scores["source1_entity_id"].unique())
    truth = {sid: truth_all.get(sid, set()) for sid in val_s1_ids}
    f05 = macro_f05(truth, preds)
    precision, recall = precision_recall(truth, preds)
    return {
        "macro_f05": f05,
        "micro_like_precision": precision,
        "micro_like_recall": recall,
        "threshold_s2": thresholds["threshold_s2"],
        "threshold_s3": thresholds["threshold_s3"],
    }
