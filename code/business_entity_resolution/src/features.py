
from __future__ import annotations

import re
from collections import Counter
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance

from .normalize import normalize_text, tokens, digit_tokens, postal_like_tokens


FEATURE_COLUMNS = [
    "name_ratio",
    "name_partial_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_jaro_winkler",
    "name_lev_similarity",
    "name_jaccard",
    "name_exact",
    "name_length_ratio",
    "address_ratio",
    "address_partial_ratio",
    "address_token_sort_ratio",
    "address_token_set_ratio",
    "address_jaro_winkler",
    "address_lev_similarity",
    "address_jaccard",
    "address_exact",
    "address_length_ratio",
    "digit_jaccard",
    "digit_exact",
    "postal_exact",
    "house_exact",
    "country_exact",
    "country_missing_either",
    "name_missing_either",
    "address_missing_either",
    "shared_blocks",
    "is_source3",
]


def _safe_ratio(a: str, b: str, fn) -> float:
    a = str(a or "")
    b = str(b or "")
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(fn(a, b)) / 100.0


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _length_ratio(a: str, b: str) -> float:
    la, lb = len(str(a or "")), len(str(b or ""))
    if la == 0 and lb == 0:
        return 1.0
    return min(la, lb) / max(la, lb)


def make_feature_frame(candidate_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in candidate_df.to_dict("records"):
        n1 = normalize_text(r["s1_name"])
        n2 = normalize_text(r["candidate_name"])
        a1 = normalize_text(r["s1_address"])
        a2 = normalize_text(r["candidate_address"])
        nt1, nt2 = tokens(n1), tokens(n2)
        at1, at2 = tokens(a1), tokens(a2)
        d1, d2 = digit_tokens(a1), digit_tokens(a2)
        p1, p2 = postal_like_tokens(a1), postal_like_tokens(a2)
        row = {
            "name_ratio": _safe_ratio(n1, n2, fuzz.ratio),
            "name_partial_ratio": _safe_ratio(n1, n2, fuzz.partial_ratio),
            "name_token_sort_ratio": _safe_ratio(n1, n2, fuzz.token_sort_ratio),
            "name_token_set_ratio": _safe_ratio(n1, n2, fuzz.token_set_ratio),
            "name_jaro_winkler": float(distance.JaroWinkler.normalized_similarity(n1, n2)),
            "name_lev_similarity": float(distance.Levenshtein.normalized_similarity(n1, n2)),
            "name_jaccard": _jaccard(nt1, nt2),
            "name_exact": float(n1 == n2 and n1 != ""),
            "name_length_ratio": _length_ratio(n1, n2),
            "address_ratio": _safe_ratio(a1, a2, fuzz.ratio),
            "address_partial_ratio": _safe_ratio(a1, a2, fuzz.partial_ratio),
            "address_token_sort_ratio": _safe_ratio(a1, a2, fuzz.token_sort_ratio),
            "address_token_set_ratio": _safe_ratio(a1, a2, fuzz.token_set_ratio),
            "address_jaro_winkler": float(distance.JaroWinkler.normalized_similarity(a1, a2)),
            "address_lev_similarity": float(distance.Levenshtein.normalized_similarity(a1, a2)),
            "address_jaccard": _jaccard(at1, at2),
            "address_exact": float(a1 == a2 and a1 != ""),
            "address_length_ratio": _length_ratio(a1, a2),
            "digit_jaccard": _jaccard(d1, d2),
            "digit_exact": float(bool(d1) and bool(d2) and d1 == d2),
            "postal_exact": float(bool(set(p1) & set(p2))),
            "house_exact": float(bool(d1) and bool(d2) and d1[0] == d2[0]),
            "country_exact": float(normalize_text(r["s1_country"]) == normalize_text(r["candidate_country"])),
            "country_missing_either": float(
                not normalize_text(r["s1_country"]) or not normalize_text(r["candidate_country"])
            ),
            "name_missing_either": float(not n1 or not n2),
            "address_missing_either": float(not a1 or not a2),
            "shared_blocks": float(r.get("shared_blocks", 0)),
            "is_source3": float(str(r["candidate_source"]).startswith("S3-")),
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS).astype(np.float32)
