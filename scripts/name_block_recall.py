from __future__ import annotations

from collections import defaultdict
import pandas as pd
import re
import unicodedata


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    text = text.replace("&", " and ")
    text = re.sub(r"[^0-9a-zA-Z]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


s1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False,
    usecols=["entity_id", "business_name"],
)

s2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False,
    usecols=["entity_id", "business_name"],
)

s3 = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False,
    usecols=["entity_id", "business_name"],
)

gt = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False,
)

s2["name_norm"] = s2["business_name"].map(normalize)
s3["name_norm"] = s3["business_name"].map(normalize)
s1["name_norm"] = s1["business_name"].map(normalize)

index2 = defaultdict(set)
index3 = defaultdict(set)

for row in s2.itertuples(index=False):
    if row.name_norm:
        index2[row.name_norm].add(row.entity_id)

for row in s3.itertuples(index=False):
    if row.name_norm:
        index3[row.name_norm].add(row.entity_id)

truth = dict(zip(gt["source1_entity_id"], gt["matched_entity_ids"]))

total_true = 0
found_true = 0

for row in s1.itertuples(index=False):
    true_ids = {
        x.strip()
        for x in truth.get(row.entity_id, "").split(",")
        if x.strip()
    }

    candidates = index2.get(row.name_norm, set()) | index3.get(row.name_norm, set())

    total_true += len(true_ids)
    found_true += len(true_ids & candidates)

print("True matches:", total_true)
print("Recovered by exact normalized name:", found_true)
print(
    "Pair recall:",
    found_true / total_true if total_true else 0.0
)
