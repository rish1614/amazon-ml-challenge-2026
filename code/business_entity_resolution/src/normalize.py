
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache


LEGAL_SUFFIXES = {
    "private": "",
    "pvt": "",
    "limited": "",
    "ltd": "",
    "llp": "",
    "llc": "",
    "incorporated": "",
    "inc": "",
    "corporation": "",
    "corp": "",
    "company": "",
    "co": "",
}


@lru_cache(maxsize=500_000)
def normalize_text(value: str) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = text.replace("&", " and ")
    text = re.sub(r"[\u2018\u2019`]", "'", text)
    text = re.sub(r"[^0-9a-zA-Z\u00C0-\u024F\u0400-\u04FF]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokens(value: str) -> list[str]:
    return [t for t in normalize_text(value).split() if t]


def sorted_tokens(value: str) -> str:
    return " ".join(sorted(tokens(value)))


def alnum(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", normalize_text(value))


def name_without_legal_suffix(value: str) -> str:
    ts = tokens(value)
    filtered = [t for t in ts if t not in LEGAL_SUFFIXES]
    return " ".join(filtered)


def digit_tokens(value: str) -> list[str]:
    return re.findall(r"\d+", str(value or ""))


def postal_like_tokens(value: str) -> list[str]:
    out: list[str] = []
    for x in digit_tokens(value):
        if len(x) in (5, 6, 9):
            out.append(x)
        elif len(x) > 6:
            out.append(x[:6])
    return out
