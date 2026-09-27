
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable


@dataclass
class BlockingIndex:
    key_to_ids: dict[str, list[str]]
    name_token_freq: Counter
    address_token_freq: Counter


def token_frequencies(records: Iterable[tuple[str, list[str]]]) -> Counter:
    c = Counter()
    for _, ts in records:
        c.update(set(ts))
    return c


def build_index(
    target_rows,
    make_keys_for_target,
    max_block_size: int,
) -> BlockingIndex:
    name_freq = Counter()
    addr_freq = Counter()

    for row in target_rows:
        name_freq.update(set(row["name_tokens"]))
        addr_freq.update(set(row["address_tokens"]))

    mapping: dict[str, list[str]] = defaultdict(list)

    for row in target_rows:
        for key in make_keys_for_target(row, name_freq, addr_freq):
            bucket = mapping[key]
            if len(bucket) <= max_block_size:
                bucket.append(row["entity_id"])

    compact = {
        k: v for k, v in mapping.items()
        if 0 < len(v) <= max_block_size
    }
    return BlockingIndex(compact, name_freq, addr_freq)
