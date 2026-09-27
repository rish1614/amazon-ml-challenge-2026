from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def read_ids_from_gt(gt_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        gt_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="dataset/train")
    parser.add_argument("--out-dir", default="dataset/mini_train")
    parser.add_argument("--n-s1", type=int, default=10000)
    parser.add_argument("--random-negatives-per-source", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    out_dir = Path(args.out_dir)

    rng = np.random.default_rng(args.seed)

    print("Loading S1...")
    s1 = pd.read_csv(
        train_dir / "train_source1.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print("Loading ground truth...")
    gt = read_ids_from_gt(train_dir / "train_ground_truth.tsv")

    # Sample S1 entities.
    n = min(args.n_s1, len(s1))
    sampled_s1 = (
        s1.sample(n=n, random_state=args.seed)
        .copy()
    )

    sampled_ids = set(sampled_s1["entity_id"])

    sampled_gt = gt[
        gt["source1_entity_id"].isin(sampled_ids)
    ].copy()

    # Collect true S2/S3 IDs referenced by the sampled S1 entities.
    positive_s2 = set()
    positive_s3 = set()

    for raw in sampled_gt["matched_entity_ids"]:
        for entity_id in raw.split(","):
            entity_id = entity_id.strip()
            if not entity_id:
                continue

            if entity_id.startswith("S2-"):
                positive_s2.add(entity_id)
            elif entity_id.startswith("S3-"):
                positive_s3.add(entity_id)

    print(f"Sampled S1: {len(sampled_s1):,}")
    print(f"Positive S2 IDs: {len(positive_s2):,}")
    print(f"Positive S3 IDs: {len(positive_s3):,}")

    def make_target(
        source_path: Path,
        positive_ids: set[str],
        source_name: str,
    ) -> pd.DataFrame:

        print(f"Reading {source_name} in chunks...")

        positives = []

        # We collect positives first.
        for chunk in pd.read_csv(
            source_path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=200_000,
        ):
            hit = chunk[
                chunk["entity_id"].isin(positive_ids)
            ]

            if not hit.empty:
                positives.append(hit)

        if positives:
            positive_df = pd.concat(
                positives,
                ignore_index=True,
            )
        else:
            positive_df = pd.DataFrame(
                columns=[
                    "entity_id",
                    "business_name",
                    "business_address",
                    "country",
                ]
            )

        # Add random negatives from target source.
        negative_pool = []

        for chunk in pd.read_csv(
            source_path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=200_000,
        ):
            negative_pool.append(
                chunk[
                    ~chunk["entity_id"].isin(positive_ids)
                ]
            )

            if sum(len(x) for x in negative_pool) >= (
                max(len(positive_df) * args.random_negatives_per_source, 1000)
            ):

                break

        if negative_pool:
            negative_df = pd.concat(
                negative_pool,
                ignore_index=True,
            )

            target_negative_n = min(
                len(negative_df),
                max(
                    len(positive_df) * args.random_negatives_per_source,
                    1000,
                ),
            )

            negative_df = negative_df.sample(
                n=target_negative_n,
                random_state=args.seed,
            )
        else:
            negative_df = pd.DataFrame(
                columns=positive_df.columns
            )

        result = pd.concat(
            [positive_df, negative_df],
            ignore_index=True,
        )

        result = result.drop_duplicates(
            subset=["entity_id"]
        )

        return result

    out_dir.mkdir(parents=True, exist_ok=True)

    mini_s2 = make_target(
        train_dir / "train_source2.tsv",
        positive_s2,
        "S2",
    )

    mini_s3 = make_target(
        train_dir / "train_source3.tsv",
        positive_s3,
        "S3",
    )

    sampled_s1.to_csv(
        out_dir / "train_source1.tsv",
        sep="\t",
        index=False,
    )

    mini_s2.to_csv(
        out_dir / "train_source2.tsv",
        sep="\t",
        index=False,
    )

    mini_s3.to_csv(
        out_dir / "train_source3.tsv",
        sep="\t",
        index=False,
    )

    # Keep exactly one ground-truth row for each sampled S1.
    sampled_gt = (
    sampled_s1[["entity_id"]]
    .merge(
        gt,
        left_on="entity_id",
        right_on="source1_entity_id",
        how="left",
    )[["source1_entity_id", "matched_entity_ids"]]
    .copy()
    )

    sampled_gt["matched_entity_ids"] = (
    sampled_gt["matched_entity_ids"]
    .fillna(""))

    sampled_gt = sampled_s1[
        ["entity_id"]
    ].merge(
        gt,
        left_on="entity_id",
        right_on="source1_entity_id",
        how="left",
    )[
        ["source1_entity_id", "matched_entity_ids"]
    ]

    sampled_gt["matched_entity_ids"] = (
        sampled_gt["matched_entity_ids"]
        .fillna("")
    )

    sampled_gt.to_csv(
        out_dir / "train_ground_truth.tsv",
        sep="\t",
        index=False,
    )

    print()
    print("Mini dataset created:")
    print(f"S1: {len(sampled_s1):,}")
    print(f"S2: {len(mini_s2):,}")
    print(f"S3: {len(mini_s3):,}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
