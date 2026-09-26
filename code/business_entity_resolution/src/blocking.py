
from __future__ import annotations

from collections import defaultdict
import pandas as pd
from rapidfuzz import fuzz

from .normalize import (
    normalize_text,
    tokens,
    digit_tokens,
    postal_like_tokens,
    name_without_legal_suffix,
)
from .indexing import BlockingIndex


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["name_norm"] = out["business_name"].map(normalize_text)
    out["name_core"] = out["business_name"].map(name_without_legal_suffix)
    out["name_tokens"] = out["business_name"].map(tokens)
    out["address_norm"] = out["business_address"].map(normalize_text)
    out["address_tokens"] = out["business_address"].map(tokens)
    out["digit_tokens"] = out["business_address"].map(digit_tokens)
    out["postal_tokens"] = out["business_address"].map(postal_like_tokens)
    out["country_norm"] = out["country"].map(normalize_text)
    return out


def make_keys(row, name_freq, addr_freq, prefix_len=5, min_token_len=3, max_token_freq=50):
    country = row["country_norm"] or "__MISSING_COUNTRY__"
    keys: set[str] = set()

    if row["name_norm"]:
        keys.add(f"n_exact|{country}|{row['name_norm']}")
        prefix = row["name_core"].replace(" ", "")[:prefix_len]
        if len(prefix) >= 3:
            keys.add(f"n_prefix|{country}|{prefix}")

    for t in row["name_tokens"]:
        if len(t) >= min_token_len and name_freq.get(t, 0) <= max_token_freq:
            keys.add(f"n_tok|{country}|{t}")

    for t in row["address_tokens"]:
        if len(t) >= min_token_len and addr_freq.get(t, 0) <= max_token_freq:
            keys.add(f"a_tok|{country}|{t}")

    for p in row["postal_tokens"]:
        keys.add(f"postal|{country}|{p}")

    digits = row["digit_tokens"]
    if digits:
        keys.add(f"house|{country}|{digits[0]}")

    return keys


def build_target_index(target_df: pd.DataFrame, config) -> BlockingIndex:
    rows = target_df.to_dict("records")
    return __build(rows, config)


def __build(rows, config):
    def maker(row, nf, af):
        return make_keys(
            row, nf, af,
            prefix_len=config.name_prefix_len,
            min_token_len=config.min_token_len,
            max_token_freq=config.max_token_freq,
        )

    from .indexing import build_index
    return build_index(rows, maker, config.max_block_size)


def candidate_rows_for_s1(
    s1_row,
    target_lookup: dict[str, dict],
    index: BlockingIndex,
    config,
) -> list[dict]:
    def fake_freq():
        return index.name_token_freq, index.address_token_freq

    name_freq, addr_freq = fake_freq()
    keys = make_keys(
        s1_row, name_freq, addr_freq,
        prefix_len=config.name_prefix_len,
        min_token_len=config.min_token_len,
        max_token_freq=config.max_token_freq,
    )

    counts = defaultdict(int)
    for key in keys:
        for target_id in index.key_to_ids.get(key, []):
            counts[target_id] += 1

    if not counts:
        return []

    # Deterministic ordering: more shared blocks first, then ID.
    ordered = sorted(counts.items(), key=lambda x: (-x[1], x[0]))

    rows = []
    for target_id, shared_blocks in ordered:
        target = target_lookup[target_id]
        rows.append({
            "source1_entity_id": s1_row["entity_id"],
            "candidate_entity_id": target_id,
            "candidate_source": target["source"],
            "shared_blocks": shared_blocks,
            "s1_name": s1_row["business_name"],
            "s1_address": s1_row["business_address"],
            "s1_country": s1_row["country"],
            "candidate_name": target["business_name"],
            "candidate_address": target["business_address"],
            "candidate_country": target["country"],
        })
    return rows


def cheap_score(row) -> float:
    name = fuzz.token_ratio(
        normalize_text(str(row["s1_name"])),
        normalize_text(str(row["candidate_name"])),
    ) / 100.0
    addr = fuzz.token_ratio(
        normalize_text(str(row["s1_address"])),
        normalize_text(str(row["candidate_address"])),
    ) / 100.0
    country = normalize_text(str(row["s1_country"])) == normalize_text(str(row["candidate_country"]))
    if country:
        return 0.60 * name + 0.40 * addr
    return 0.45 * name + 0.55 * addr


def compress_candidates(candidate_df: pd.DataFrame, config) -> pd.DataFrame:
    if candidate_df.empty or not config.compression_enabled:
        return candidate_df

    work = candidate_df.copy()
    work["_cheap"] = work.apply(cheap_score, axis=1)

    chunks = []
    for _, g in work.groupby("source1_entity_id", sort=False):
        if len(g) <= config.max_candidates_per_s1:
            chunks.append(g)
            continue

        strong = g[g["_cheap"] >= config.keep_if_cheap_score_at_least]
        if len(strong) > config.max_candidates_per_s1:
            keep = (
                g.sort_values(
                    ["_cheap", "shared_blocks", "candidate_entity_id"],
                    ascending=[False, False, True],
                )
                .head(config.max_candidates_per_s1)
            )
        else:
            rest = (
                g.drop(index=strong.index)
                .sort_values(
                    ["_cheap", "shared_blocks", "candidate_entity_id"],
                    ascending=[False, False, True],
                )
            )
            budget = config.max_candidates_per_s1 - len(strong)
            keep = pd.concat([strong, rest.head(max(0, budget))], ignore_index=True)

        chunks.append(keep)

    out = pd.concat(chunks, ignore_index=True)
    return out.drop(columns=["_cheap"], errors="ignore")


def generate_candidates(s1_df: pd.DataFrame, target_df: pd.DataFrame, config) -> pd.DataFrame:
    index = build_target_index(target_df, config)
    target_lookup = {
        r["entity_id"]: r for r in target_df.to_dict("records")
    }
    all_rows = []
    for row in s1_df.to_dict("records"):
        all_rows.extend(candidate_rows_for_s1(row, target_lookup, index, config))
    cols = [
        "source1_entity_id", "candidate_entity_id", "candidate_source",
        "shared_blocks", "s1_name", "s1_address", "s1_country",
        "candidate_name", "candidate_address", "candidate_country",
    ]
    return pd.DataFrame(all_rows, columns=cols)
