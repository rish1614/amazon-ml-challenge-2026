
from __future__ import annotations

from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def read_source(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns: {missing}")
    return df[REQUIRED_COLUMNS].copy()


def read_ground_truth(path: str | Path) -> dict[str, set[str]]:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    expected = {"source1_entity_id", "matched_entity_ids"}
    if set(df.columns) != expected:
        raise ValueError(
            f"{path}: expected columns {sorted(expected)}, got {list(df.columns)}"
        )

    truth: dict[str, set[str]] = {}
    for row in df.itertuples(index=False):
        raw = (row.matched_entity_ids or "").strip()
        ids = {x.strip() for x in raw.split(",") if x.strip()}
        truth[str(row.source1_entity_id)] = ids
    return truth


def read_ground_truth_for_s1s(
    path: str | Path, required_s1_ids: set[str]
) -> dict[str, set[str]]:
    truth = read_ground_truth(path)
    return {sid: truth.get(sid, set()) for sid in required_s1_ids}
