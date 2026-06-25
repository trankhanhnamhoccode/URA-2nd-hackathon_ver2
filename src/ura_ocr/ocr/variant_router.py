from __future__ import annotations

import re
from typing import Iterable, List

import numpy as np
import pandas as pd


DEFAULT_VARIANTS = [
    "raw",
    "bottom_45_resize_960",
    "top_45_resize_960",
    "center_60_resize_960",
]


VIETNAMESE_ACCENT_RE = re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵ"
    r"èéẹẻẽêềếệểễ"
    r"ìíịỉĩ"
    r"òóọỏõôồốộổỗơờớợởỡ"
    r"ùúụủũưừứựửữ"
    r"ỳýỵỷỹ"
    r"đ"
    r"ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴ"
    r"ÈÉẸẺẼÊỀẾỆỂỄ"
    r"ÌÍỊỈĨ"
    r"ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ"
    r"ÙÚỤỦŨƯỪỨỰỬỮ"
    r"ỲÝỴỶỸ"
    r"Đ]"
)

JUNK_RE = re.compile(r"[^0-9A-Za-zÀ-ỹ\s\.\,\-\+\/\:\%\&\(\)]")


def _safe_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value)


def _count_words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _ratio(n: int, d: int) -> float:
    return float(n / d) if d else 0.0


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add OCR text quality features for each variant row.

    These features are available at inference time after OCR has run.
    """
    out = df.copy()

    texts = out["ocr_text"].fillna("").astype(str)

    out["text_len"] = texts.str.len()
    out["word_count"] = texts.map(_count_words)
    out["line_count_text"] = texts.map(lambda x: len([ln for ln in x.splitlines() if ln.strip()]))

    out["digit_count"] = texts.map(lambda x: sum(ch.isdigit() for ch in x))
    out["alpha_count"] = texts.map(lambda x: sum(ch.isalpha() for ch in x))
    out["space_count"] = texts.map(lambda x: sum(ch.isspace() for ch in x))
    out["upper_count"] = texts.map(lambda x: sum(ch.isupper() for ch in x))
    out["accent_count"] = texts.map(lambda x: len(VIETNAMESE_ACCENT_RE.findall(x)))
    out["junk_count"] = texts.map(lambda x: len(JUNK_RE.findall(x)))

    out["digit_ratio"] = [
        _ratio(n, d) for n, d in zip(out["digit_count"], out["text_len"])
    ]
    out["alpha_ratio"] = [
        _ratio(n, d) for n, d in zip(out["alpha_count"], out["text_len"])
    ]
    out["upper_ratio"] = [
        _ratio(n, d) for n, d in zip(out["upper_count"], out["alpha_count"])
    ]
    out["accent_ratio"] = [
        _ratio(n, d) for n, d in zip(out["accent_count"], out["alpha_count"])
    ]
    out["junk_ratio"] = [
        _ratio(n, d) for n, d in zip(out["junk_count"], out["text_len"])
    ]

    out["avg_word_len"] = [
        _ratio(text_len, word_count)
        for text_len, word_count in zip(out["text_len"], out["word_count"])
    ]

    out["is_blank_pred"] = texts.str.strip().eq("").astype(int)

    return out


def add_raw_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features comparing each variant to raw for the same image.

    These are useful because raw is usually stable, while crop variants
    are fallback candidates.
    """
    out = df.copy()

    raw_cols = [
        "image_id",
        "avg_score",
        "num_lines",
        "text_len",
        "word_count",
        "line_count_text",
        "is_blank_pred",
    ]

    raw_df = out[out["variant"] == "raw"][raw_cols].copy()

    raw_df = raw_df.rename(
        columns={
            "avg_score": "raw_avg_score",
            "num_lines": "raw_num_lines",
            "text_len": "raw_text_len",
            "word_count": "raw_word_count",
            "line_count_text": "raw_line_count_text",
            "is_blank_pred": "raw_is_blank_pred",
        }
    )

    out = out.merge(raw_df, on="image_id", how="left")

    for col in [
        "raw_avg_score",
        "raw_num_lines",
        "raw_text_len",
        "raw_word_count",
        "raw_line_count_text",
        "raw_is_blank_pred",
    ]:
        out[col] = out[col].fillna(0)

    out["score_minus_raw"] = out["avg_score"] - out["raw_avg_score"]
    out["num_lines_minus_raw"] = out["num_lines"] - out["raw_num_lines"]
    out["text_len_minus_raw"] = out["text_len"] - out["raw_text_len"]
    out["word_count_minus_raw"] = out["word_count"] - out["raw_word_count"]

    return out


def build_router_features(
    report_df: pd.DataFrame,
    variants: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, List[str]]:
    """
    Build features for OCR variant routing from an ablation report.

    Input report must contain at least:
    image_id, variant, ocr_text, avg_score, num_lines

    Optional image quality columns are used if present.
    """
    variants = list(variants or DEFAULT_VARIANTS)

    df = report_df.copy()
    df = df[df["variant"].isin(variants)].copy()

    required = ["image_id", "variant", "ocr_text", "avg_score", "num_lines"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for router features: {missing}")

    df["ocr_text"] = df["ocr_text"].fillna("").astype(str)
    df["avg_score"] = pd.to_numeric(df["avg_score"], errors="coerce").fillna(0.0)
    df["num_lines"] = pd.to_numeric(df["num_lines"], errors="coerce").fillna(0.0)

    df = add_text_features(df)
    df = add_raw_relative_features(df)

    numeric_features = [
        "avg_score",
        "num_lines",
        "text_len",
        "word_count",
        "line_count_text",
        "digit_ratio",
        "alpha_ratio",
        "upper_ratio",
        "accent_ratio",
        "junk_ratio",
        "avg_word_len",
        "is_blank_pred",
        "raw_avg_score",
        "raw_num_lines",
        "raw_text_len",
        "raw_word_count",
        "raw_line_count_text",
        "raw_is_blank_pred",
        "score_minus_raw",
        "num_lines_minus_raw",
        "text_len_minus_raw",
        "word_count_minus_raw",
    ]

    image_quality_features = [
        "width",
        "height",
        "long_side",
        "short_side",
        "mean_brightness",
        "contrast_std",
        "blur_laplacian_var",
        "dark_pixel_ratio",
    ]

    for col in image_quality_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            numeric_features.append(col)

    # One-hot variant indicators.
    for variant in variants:
        col = f"variant__{variant}"
        df[col] = (df["variant"] == variant).astype(int)
        numeric_features.append(col)

    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], 0).fillna(0.0)

    return df, numeric_features


def select_best_variant_from_predictions(
    pred_df: pd.DataFrame,
    pred_col: str = "pred_cer",
) -> pd.DataFrame:
    """
    Select the lowest predicted CER row per image.
    """
    if pred_col not in pred_df.columns:
        raise ValueError(f"Missing prediction column: {pred_col}")

    return (
        pred_df.sort_values(["image_id", pred_col])
        .groupby("image_id", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )