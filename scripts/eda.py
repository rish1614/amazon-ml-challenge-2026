#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution"))

from src.io import read_source, read_ground_truth


def main():
    p = argparse.ArgumentParser(description="Fast challenge EDA summary.")
    p.add_argument("--train-dir", default="dataset/train")
    args = p.parse_args()
    d = Path(args.train_dir)

    s1 = read_source(d / "train_source1.tsv")
    s2 = read_source(d / "train_source2.tsv")
    s3 = read_source(d / "train_source3.tsv")
    gt = read_ground_truth(d / "train_ground_truth.tsv")

    print("=== Row counts ===")
    print(f"S1: {len(s1):,}")
    print(f"S2: {len(s2):,}")
    print(f"S3: {len(s3):,}")

    empty = sum(not v for v in gt.values())
    match_counts = [len(gt.get(sid, set())) for sid in s1["entity_id"].astype(str)]
    print("\n=== Ground-truth distribution ===")
    print(f"S1 with zero matches: {empty:,} ({empty / max(1, len(s1)):.2%})")
    print(f"Mean matches/S1: {sum(match_counts) / max(1, len(match_counts)):.3f}")
    print(f"Max matches/S1: {max(match_counts) if match_counts else 0}")
    print(f"S1 with >=1 match: {sum(x > 0 for x in match_counts):,}")

    for label, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
        print(f"\n=== {label} missingness ===")
        for c in ["business_name", "business_address", "country"]:
            missing = (df[c].astype(str).str.strip() == "").sum()
            print(f"{c}: {missing:,} ({missing / max(1, len(df)):.2%})")

    print("\n=== Country values ===")
    for label, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
        print(label, sorted(df["country"].astype(str).str.strip().unique())[:50])

    print("\n=== Exact duplicates ===")
    for label, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
        dup = df.duplicated(subset=["business_name", "business_address", "country"]).sum()
        print(f"{label}: {dup:,}")


if __name__ == "__main__":
    main()
