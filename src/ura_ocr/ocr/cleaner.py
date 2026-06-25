from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List


# Keep Vietnamese, Latin letters, digits, spaces, and product-useful symbols.
# Important symbols for product names:
# +  e.g. K2 + D3, 24h+
# /  e.g. P/S
# -  e.g. B-Complex
# %  e.g. 0%, 100%
# .  e.g. 1.5L
# &  e.g. Johnson & Johnson
# () e.g. variant notes
# :  sometimes separates title / product
_ALLOWED_SYMBOLS = set(" +-/%.,:&()[]")


def is_cjk_or_emoji_or_symbol_noise(ch: str) -> bool:
    """
    Return True for characters that are usually noise in this dataset:
    - CJK / Japanese / Korean characters
    - emoji / pictographs
    - control characters

    Do not remove Vietnamese accents.
    """
    if not ch:
        return True

    code = ord(ch)
    category = unicodedata.category(ch)

    # Control chars
    if category.startswith("C"):
        return True

    # CJK Unified Ideographs
    if 0x4E00 <= code <= 0x9FFF:
        return True

    # CJK Extension A
    if 0x3400 <= code <= 0x4DBF:
        return True

    # Hiragana / Katakana
    if 0x3040 <= code <= 0x30FF:
        return True

    # Hangul
    if 0xAC00 <= code <= 0xD7AF:
        return True

    # Common emoji / symbols blocks
    if 0x1F300 <= code <= 0x1FAFF:
        return True

    return False


def normalize_line(text: str) -> str:
    """
    Normalize one OCR line:
    - Unicode normalize
    - remove CJK/emoji/control noise
    - keep Vietnamese accents
    - keep useful product symbols
    - collapse spaces
    """
    if text is None:
        return ""

    text = str(text)
    text = unicodedata.normalize("NFC", text)

    chars: List[str] = []

    for ch in text:
        if is_cjk_or_emoji_or_symbol_noise(ch):
            chars.append(" ")
            continue

        if ch.isalnum() or ch.isspace() or ch in _ALLOWED_SYMBOLS:
            chars.append(ch)
        else:
            # Other punctuation becomes space, not deletion,
            # to avoid joining unrelated tokens.
            chars.append(" ")

    line = "".join(chars)

    # Normalize repeated spaces
    line = re.sub(r"\s+", " ", line).strip()

    # Normalize slash and hyphen first.
    line = re.sub(r"\s*/\s*", "/", line)
    line = re.sub(r"\s*-\s*", "-", line)

    # Protect suffix-plus patterns before normalizing binary plus.
    # Examples: SPF50+, 24h+, 24h + -> SPF50+, 24h+
    line = re.sub(r"\b([A-Za-zÀ-ỹ0-9]+)\s*\+\s*(?=$|\s)", r"\1+", line)

    # Normalize binary plus between two tokens.
    # Example: K2+D3, K2 +D3, K2+ D3 -> K2 + D3
    line = re.sub(
        r"(?<=[A-Za-zÀ-ỹ0-9])\s*\+\s*(?=[A-Za-zÀ-ỹ0-9])",
        " + ",
        line,
    )

    # Re-collapse spaces after symbol rules
    line = re.sub(r"\s+", " ", line).strip()

    return line


def deduplicate_lines(lines: Iterable[str]) -> List[str]:
    """
    Remove exact duplicate lines after normalization.
    Keep order.
    """
    seen = set()
    out: List[str] = []

    for line in lines:
        norm = normalize_line(line)

        if not norm:
            continue

        key = norm.lower()

        if key in seen:
            continue

        seen.add(key)
        out.append(norm)

    return out


def clean_ocr_lines(lines: Iterable[str]) -> List[str]:
    """
    Clean OCR lines and remove duplicates.
    """
    return deduplicate_lines(lines)


def clean_ocr_text(text: str) -> str:
    """
    Clean a raw OCR text block.

    This function is intentionally conservative:
    it removes obvious noise but preserves product-useful symbols.
    """
    if text is None:
        return ""

    raw_lines = str(text).replace("\r", "\n").split("\n")
    lines = clean_ocr_lines(raw_lines)

    return " ".join(lines).strip()