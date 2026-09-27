from __future__ import annotations

import pandas as pd
from rapidfuzz import fuzz
from .normalize import normalize_text


def exact_normalized_name(candidates: pd.DataFrame) -> dict[str, set[str]]:
    """High-precision baseline: normalized name + country equality + exact address when available."""
    out: dict[str, set[str]] = {}
    for r in candidates.to_dict("records"):
        name_ok = normalize_text(r["s1_name"]) == normalize_text(r["candidate_name"]) and normalize_text(r["s1_name"]) != ""
        country_ok = normalize_text(r["s1_country"]) == normalize_text(r["candidate_country"])
        addr1, addr2 = normalize_text(r["s1_address"]), normalize_text(r["candidate_address"])
        address_ok = bool(addr1 and addr2 and addr1 == addr2)
        if name_ok and country_ok and (address_ok or not addr1 or not addr2):
            out.setdefault(r["source1_entity_id"], set()).add(r["candidate_entity_id"])
    return out


def fuzzy_rule(candidates: pd.DataFrame, threshold: float = 0.90) -> dict[str, set[str]]:
    """Simple fuzzy baseline; intentionally conservative because F_0.5 is precision-heavy."""
    out: dict[str, set[str]] = {}
    for r in candidates.to_dict("records"):
        n1, n2 = normalize_text(r["s1_name"]), normalize_text(r["candidate_name"])
        a1, a2 = normalize_text(r["s1_address"]), normalize_text(r["candidate_address"])
        name = fuzz.token_ratio(n1, n2) / 100.0
        addr = fuzz.token_ratio(a1, a2) / 100.0 if a1 and a2 else 0.0
        country = normalize_text(r["s1_country"]) == normalize_text(r["candidate_country"])
        score = 0.65 * name + 0.35 * addr
        if country and score >= threshold and (name >= threshold - 0.03):
            out.setdefault(r["source1_entity_id"], set()).add(r["candidate_entity_id"])
    return out
