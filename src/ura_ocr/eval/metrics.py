from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from ura_ocr.utils.text_norm import norm_key, safe_str, strip_for_csv


def is_blank(value) -> bool:
    return norm_key(safe_str(value)) == ""


def token_f1(pred: str, target: str) -> float:
    """
    Token-level F1 after normalization.

    If both are blank, return 1.0.
    If only one is blank, return 0.0.
    """
    pred_tokens = norm_key(pred).split()
    target_tokens = norm_key(target).split()

    if not pred_tokens and not target_tokens:
        return 1.0

    if not pred_tokens or not target_tokens:
        return 0.0

    pred_counts = {}
    target_counts = {}

    for tok in pred_tokens:
        pred_counts[tok] = pred_counts.get(tok, 0) + 1

    for tok in target_tokens:
        target_counts[tok] = target_counts.get(tok, 0) + 1

    overlap = 0
    for tok, count in pred_counts.items():
        overlap += min(count, target_counts.get(tok, 0))

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(target_tokens)

    return 2 * precision * recall / (precision + recall)


def levenshtein_distance(a: str, b: str) -> int:
    """
    Simple Levenshtein edit distance.
    Used for CER, no external dependency required.
    """
    a = safe_str(a)
    b = safe_str(b)

    if a == b:
        return 0

    if not a:
        return len(b)

    if not b:
        return len(a)

    prev = list(range(len(b) + 1))

    for i, ca in enumerate(a, start=1):
        curr = [i]

        for j, cb in enumerate(b, start=1):
            insert_cost = curr[j - 1] + 1
            delete_cost = prev[j] + 1
            replace_cost = prev[j - 1] + (0 if ca == cb else 1)

            curr.append(min(insert_cost, delete_cost, replace_cost))

        prev = curr

    return prev[-1]


def cer(pred: str, target: str) -> float:
    """
    Character Error Rate.
    If target is blank:
    - pred blank => 0.0
    - pred nonblank => 1.0
    """
    pred = safe_str(pred)
    target = safe_str(target)

    if target == "":
        return 0.0 if pred == "" else 1.0

    return levenshtein_distance(pred, target) / max(1, len(target))


def classify_field_error(pred: str, target: str) -> str:
    """
    Classify one field prediction.
    Useful for audit/error analysis.
    """
    pred_blank = is_blank(pred)
    target_blank = is_blank(target)

    if pred_blank and target_blank:
        return "true_blank"

    if pred_blank and not target_blank:
        return "miss"

    if not pred_blank and target_blank:
        return "false_fill"

    if norm_key(pred) == norm_key(target):
        return "correct"

    pred_tokens = set(norm_key(pred).split())
    target_tokens = set(norm_key(target).split())

    if pred_tokens and target_tokens and pred_tokens.issubset(target_tokens):
        return "partial_too_short"

    if pred_tokens and target_tokens and target_tokens.issubset(pred_tokens):
        return "partial_too_long"

    if pred_tokens & target_tokens:
        return "wrong_partial_overlap"

    return "wrong_no_overlap"


def evaluate_product_field(
    merged_df: pd.DataFrame,
    pred_col: str = "pred_product_name",
    gt_col: str = "gt_product_name",
) -> Dict[str, float]:
    total = len(merged_df)

    if total == 0:
        raise ValueError("Cannot evaluate empty dataframe.")

    pred_values = merged_df[pred_col].map(safe_str)
    gt_values = merged_df[gt_col].map(safe_str)

    pred_blank = pred_values.map(is_blank)
    gt_blank = gt_values.map(is_blank)

    exact = [
        strip_for_csv(p) == strip_for_csv(g)
        for p, g in zip(pred_values, gt_values)
    ]

    exact_no_accent = [
        norm_key(p) == norm_key(g)
        for p, g in zip(pred_values, gt_values)
    ]

    token_scores = [
        token_f1(p, g)
        for p, g in zip(pred_values, gt_values)
    ]

    error_types = [
        classify_field_error(p, g)
        for p, g in zip(pred_values, gt_values)
    ]

    false_fill_count = sum(e == "false_fill" for e in error_types)
    miss_count = sum(e == "miss" for e in error_types)
    wrong_count = sum(
        e in {
            "partial_too_short",
            "partial_too_long",
            "wrong_partial_overlap",
            "wrong_no_overlap",
        }
        for e in error_types
    )

    nonblank_gt_count = int((~gt_blank).sum())
    blank_gt_count = int(gt_blank.sum())

    return {
        "rows": total,
        "pred_blank_rate": float(pred_blank.mean()),
        "gt_blank_rate": float(gt_blank.mean()),
        "pred_fill_rate": float((~pred_blank).mean()),
        "gt_fill_rate": float((~gt_blank).mean()),
        "exact_match": float(sum(exact) / total),
        "exact_match_no_accent": float(sum(exact_no_accent) / total),
        "token_f1_macro": float(sum(token_scores) / total),
        "false_fill_rate_all": float(false_fill_count / total),
        "miss_rate_all": float(miss_count / total),
        "wrong_rate_all": float(wrong_count / total),
        "false_fill_rate_on_blank_gt": float(false_fill_count / max(1, blank_gt_count)),
        "miss_rate_on_nonblank_gt": float(miss_count / max(1, nonblank_gt_count)),
    }


def evaluate_ocr_field(
    merged_df: pd.DataFrame,
    pred_col: str = "pred_ocr_text",
    gt_col: str = "gt_ocr_text",
) -> Dict[str, float]:
    total = len(merged_df)

    if total == 0:
        raise ValueError("Cannot evaluate empty dataframe.")

    pred_values = merged_df[pred_col].map(safe_str)
    gt_values = merged_df[gt_col].map(safe_str)

    cer_values = [
        cer(p, g)
        for p, g in zip(pred_values, gt_values)
    ]

    pred_blank = pred_values.map(is_blank)
    gt_blank = gt_values.map(is_blank)

    return {
        "ocr_rows": total,
        "ocr_pred_blank_rate": float(pred_blank.mean()),
        "ocr_gt_blank_rate": float(gt_blank.mean()),
        "cer_macro": float(sum(cer_values) / total),
        "one_minus_cer_macro": float(1.0 - sum(cer_values) / total),
    }


def prepare_merged_prediction_gt(
    pred_df: pd.DataFrame,
    gt_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge prediction and ground truth by image_id.

    Expected pred columns:
    - image_id
    - ocr_text
    - product_name
    Optional:
    - brand_name

    Expected gt columns:
    - image_id
    - ocr_text
    - product_name
    """
    if "image_id" not in pred_df.columns:
        raise ValueError("Prediction dataframe missing image_id.")

    if "image_id" not in gt_df.columns:
        raise ValueError("Ground truth dataframe missing image_id.")

    pred = pred_df.copy().fillna("")
    gt = gt_df.copy().fillna("")

    for col in ["ocr_text", "product_name", "brand_name"]:
        if col not in pred.columns:
            pred[col] = ""

    for col in ["ocr_text", "product_name", "brand_name"]:
        if col not in gt.columns:
            gt[col] = ""

    pred = pred.rename(
        columns={
            "ocr_text": "pred_ocr_text",
            "product_name": "pred_product_name",
            "brand_name": "pred_brand_name",
        }
    )

    gt = gt.rename(
        columns={
            "ocr_text": "gt_ocr_text",
            "product_name": "gt_product_name",
            "brand_name": "gt_brand_name",
        }
    )

    keep_pred = [
        "image_id",
        "pred_ocr_text",
        "pred_product_name",
        "pred_brand_name",
    ]

    keep_gt = [
        "image_id",
        "gt_ocr_text",
        "gt_product_name",
        "gt_brand_name",
    ]

    merged = pred[keep_pred].merge(
        gt[keep_gt],
        on="image_id",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) == 0:
        raise ValueError("No overlapping image_id between prediction and ground truth.")

    return merged


def build_error_cases(merged_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in merged_df.iterrows():
        product_error = classify_field_error(
            row["pred_product_name"],
            row["gt_product_name"],
        )

        ocr_cer = cer(row["pred_ocr_text"], row["gt_ocr_text"])

        if product_error in {"correct", "true_blank"} and ocr_cer <= 0.2:
            continue

        rows.append(
            {
                "image_id": row["image_id"],
                "product_error": product_error,
                "ocr_cer": ocr_cer,
                "pred_ocr_text": row["pred_ocr_text"],
                "gt_ocr_text": row["gt_ocr_text"],
                "pred_product_name": row["pred_product_name"],
                "gt_product_name": row["gt_product_name"],
            }
        )

    return pd.DataFrame(rows)


def evaluate_submission_dataframe(
    pred_df: pd.DataFrame,
    gt_df: pd.DataFrame,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    merged = prepare_merged_prediction_gt(pred_df, gt_df)

    product_metrics = evaluate_product_field(merged)
    ocr_metrics = evaluate_ocr_field(merged)
    error_cases = build_error_cases(merged)

    report = {
        "rows_pred": int(len(pred_df)),
        "rows_gt": int(len(gt_df)),
        "rows_overlap": int(len(merged)),
        "product": product_metrics,
        "ocr": ocr_metrics,
        "error_cases": {
            "count": int(len(error_cases)),
        },
    }

    return report, error_cases


def save_eval_outputs(
    report: Dict[str, object],
    error_cases: pd.DataFrame,
    out_dir: str | Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "eval_report.json"
    error_path = out_dir / "error_cases.csv"

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    error_cases.to_csv(error_path, index=False)