
from __future__ import annotations

from typing import Iterable


def entity_f05(true_ids: set[str], pred_ids: set[str]) -> float:
    if not true_ids and not pred_ids:
        return 1.0
    if not true_ids and pred_ids:
        return 0.0
    if true_ids and not pred_ids:
        return 0.0

    tp = len(true_ids & pred_ids)
    precision = tp / len(pred_ids) if pred_ids else 0.0
    recall = tp / len(true_ids) if true_ids else 0.0

    if precision == 0.0 and recall == 0.0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def macro_f05(
    truth: dict[str, set[str]],
    predictions: dict[str, set[str]],
) -> float:
    s1_ids = list(truth.keys())
    if not s1_ids:
        return 0.0
    return sum(
        entity_f05(truth[sid], predictions.get(sid, set()))
        for sid in s1_ids
    ) / len(s1_ids)


def precision_recall(
    truth: dict[str, set[str]],
    predictions: dict[str, set[str]],
) -> tuple[float, float]:
    tp = fp = fn = 0
    for sid, true_ids in truth.items():
        pred_ids = predictions.get(sid, set())
        tp += len(true_ids & pred_ids)
        fp += len(pred_ids - true_ids)
        fn += len(true_ids - pred_ids)

    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return precision, recall


def candidate_recall(
    truth: dict[str, set[str]],
    candidates: dict[str, set[str]],
) -> float:
    total = found = 0
    for sid, true_ids in truth.items():
        total += len(true_ids)
        found += len(true_ids & candidates.get(sid, set()))
    return found / total if total else 1.0
