from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


W_OCR = 0.35


CAT_COLS = [
    "candidate_name",
    "candidate_type",
    "source_variant",
]


BASE_NUM_COLS = [
    "num_lines",
    "avg_line_score",
    "median_line_score",
    "avg_area_ratio",
    "text_len",
    "word_count",
    "char_count",
    "digit_count",
    "alpha_count",
    "upper_count",
    "accent_count",
    "digit_ratio",
    "alpha_ratio",
    "upper_ratio",
    "accent_ratio",
    "junk_ratio",
    "is_blank",
    "has_digit",
    "has_accent",
    "has_dot",
    "has_plus",
    "line_score_x_num_lines",
    "area_x_num_lines",
    "raw_text_len",
    "raw_word_count",
    "raw_num_lines",
    "raw_avg_line_score",
    "raw_digit_ratio",
    "raw_alpha_ratio",
    "raw_junk_ratio",
    "rel_text_len_vs_raw",
    "rel_word_count_vs_raw",
    "rel_num_lines_vs_raw",
    "delta_score_vs_raw",
    "delta_junk_vs_raw",
]


def clean_val(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def is_blank_text(x: Any) -> bool:
    s = clean_val(x)
    return s == "" or s.lower() in {"none", "nan", "null"}


def btc_cer(gt: str, pred: str) -> float:
    gt = "" if is_blank_text(gt) else clean_val(gt)
    pred = "" if is_blank_text(pred) else clean_val(pred)

    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0

    m, n = len(gt), len(pred)
    dp = list(range(n + 1))

    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if gt[i - 1] == pred[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp

    return min(dp[n] / len(gt), 1.0)


def add_candidate_text_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["image_id", "candidate_name", "candidate_type", "source_variant", "ocr_text"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.strip()

    for col in ["num_lines", "avg_line_score", "median_line_score", "avg_area_ratio"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    text = df["ocr_text"].fillna("").astype(str)

    df["text_len"] = text.str.len()
    df["word_count"] = text.apply(lambda s: len(clean_val(s).split()))
    df["char_count"] = text.apply(lambda s: len(clean_val(s)))

    df["digit_count"] = text.apply(lambda s: sum(ch.isdigit() for ch in clean_val(s)))
    df["alpha_count"] = text.apply(lambda s: sum(ch.isalpha() for ch in clean_val(s)))
    df["upper_count"] = text.apply(lambda s: sum(ch.isupper() for ch in clean_val(s)))
    df["accent_count"] = text.apply(lambda s: sum(1 for ch in clean_val(s) if "À" <= ch <= "ỹ"))

    denom = df["char_count"].clip(lower=1)
    df["digit_ratio"] = df["digit_count"] / denom
    df["alpha_ratio"] = df["alpha_count"] / denom
    df["upper_ratio"] = df["upper_count"] / denom
    df["accent_ratio"] = df["accent_count"] / denom

    def junk_ratio(s: str) -> float:
        s = clean_val(s)
        if not s:
            return 0.0
        good = sum(ch.isalnum() or ch.isspace() or ("À" <= ch <= "ỹ") for ch in s)
        return 1.0 - good / max(len(s), 1)

    df["junk_ratio"] = text.apply(junk_ratio)
    df["is_blank"] = text.apply(lambda s: int(is_blank_text(s)))

    df["has_digit"] = (df["digit_count"] > 0).astype(int)
    df["has_accent"] = (df["accent_count"] > 0).astype(int)
    df["has_dot"] = text.str.contains(r"\.", regex=True, na=False).astype(int)
    df["has_plus"] = text.str.contains(r"\+", regex=True, na=False).astype(int)

    df["line_score_x_num_lines"] = df["avg_line_score"].fillna(0.0) * df["num_lines"].fillna(0.0)
    df["area_x_num_lines"] = df["avg_area_ratio"].fillna(0.0) * df["num_lines"].fillna(0.0)

    return df


def add_raw_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    raw = df[df["candidate_name"] == "variant::raw"].copy()

    raw_feats = raw[
        [
            "image_id",
            "text_len",
            "word_count",
            "num_lines",
            "avg_line_score",
            "digit_ratio",
            "alpha_ratio",
            "junk_ratio",
        ]
    ].rename(
        columns={
            "text_len": "raw_text_len",
            "word_count": "raw_word_count",
            "num_lines": "raw_num_lines",
            "avg_line_score": "raw_avg_line_score",
            "digit_ratio": "raw_digit_ratio",
            "alpha_ratio": "raw_alpha_ratio",
            "junk_ratio": "raw_junk_ratio",
        }
    )

    raw_feats = raw_feats.drop_duplicates("image_id")

    df = df.merge(raw_feats, on="image_id", how="left")

    df["rel_text_len_vs_raw"] = df["text_len"] / df["raw_text_len"].fillna(0.0).add(1.0)
    df["rel_word_count_vs_raw"] = df["word_count"] / df["raw_word_count"].fillna(0.0).add(1.0)
    df["rel_num_lines_vs_raw"] = df["num_lines"] / df["raw_num_lines"].fillna(0.0).add(1.0)
    df["delta_score_vs_raw"] = df["avg_line_score"] - df["raw_avg_line_score"]
    df["delta_junk_vs_raw"] = df["junk_ratio"] - df["raw_junk_ratio"]

    return df


def prepare_candidate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_candidate_text_features(df)
    df = add_raw_relative_features(df)

    for col in BASE_NUM_COLS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in CAT_COLS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    return df


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    cat_cols = [c for c in CAT_COLS if c in df.columns]
    num_cols = [c for c in BASE_NUM_COLS if c in df.columns and df[c].notna().any()]
    return cat_cols, num_cols


def ensure_feature_columns(df: pd.DataFrame, cat_cols: list[str], num_cols: list[str]) -> pd.DataFrame:
    df = df.copy()

    for col in cat_cols:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    for col in num_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def make_image_blank_features(cand_df: pd.DataFrame) -> pd.DataFrame:
    df = prepare_candidate_features(cand_df)

    df["is_nonblank"] = df["ocr_text"].fillna("").astype(str).str.strip().ne("").astype(int)

    agg = df.groupby("image_id").agg(
        cand_count=("candidate_name", "size"),
        nonblank_cand_count=("is_nonblank", "sum"),
        max_text_len=("text_len", "max"),
        mean_text_len=("text_len", "mean"),
        max_word_count=("word_count", "max"),
        mean_word_count=("word_count", "mean"),
        max_score=("avg_line_score", "max"),
        mean_score=("avg_line_score", "mean"),
        max_num_lines=("num_lines", "max"),
        mean_num_lines=("num_lines", "mean"),
        min_junk_ratio=("junk_ratio", "min"),
        mean_junk_ratio=("junk_ratio", "mean"),
    ).reset_index()

    piv = df[df["candidate_type"] == "existing_variant"].copy()
    piv["variant_short"] = piv["source_variant"].astype(str)

    keep_cols = [
        "image_id",
        "variant_short",
        "text_len",
        "word_count",
        "num_lines",
        "avg_line_score",
        "junk_ratio",
    ]

    piv = piv[keep_cols]

    wide = piv.pivot_table(
        index="image_id",
        columns="variant_short",
        values=["text_len", "word_count", "num_lines", "avg_line_score", "junk_ratio"],
        aggfunc="first",
    )

    wide.columns = [f"{a}__{b}" for a, b in wide.columns]
    wide = wide.reset_index()

    out = agg.merge(wide, on="image_id", how="left")

    if "gt_ocr_text" in df.columns:
        gt = df[["image_id", "gt_ocr_text"]].drop_duplicates("image_id").copy()
        gt["gt_blank"] = gt["gt_ocr_text"].apply(lambda s: int(is_blank_text(s)))
        out = out.merge(gt[["image_id", "gt_blank"]], on="image_id", how="left")

    return out


def get_blank_feature_columns(img_feat: pd.DataFrame) -> list[str]:
    drop_cols = {"image_id", "gt_blank", "blank_pred", "blank_pred_v1"}
    cols = [
        c for c in img_feat.columns
        if c not in drop_cols
        and pd.api.types.is_numeric_dtype(img_feat[c])
        and img_feat[c].notna().any()
    ]
    return cols
