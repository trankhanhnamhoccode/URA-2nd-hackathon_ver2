from __future__ import annotations

import re
import unicodedata


def safe_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return str(value)


def normalize_whitespace(text: str) -> str:
    text = safe_str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_for_csv(text: str) -> str:
    """
    Keep CSV cells single-line and safe.
    Do not aggressively remove product symbols like +, /, -, %, dots.
    """
    text = safe_str(text)
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return normalize_whitespace(text)


def remove_accents(text: str) -> str:
    text = safe_str(text)
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d").replace("Đ", "D")
    return unicodedata.normalize("NFC", text)


def norm_key(text: str) -> str:
    """
    Normalized key for fuzzy/eval. Keeps alphanumeric content,
    lowercases, removes accents, normalizes spaces.
    """
    text = remove_accents(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return normalize_whitespace(text)