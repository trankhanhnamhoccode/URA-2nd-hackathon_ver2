from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from ura_ocr.io.csv_io import write_csv
from ura_ocr.utils.text_norm import strip_for_csv


PHASE1_COLUMNS = ["image_id", "ocr_text", "product_name"]
PHASE2_COLUMNS = ["image_id", "ocr_text", "brand_name", "product_name"]


def get_submission_columns(phase: str) -> List[str]:
    if phase == "phase1":
        return PHASE1_COLUMNS
    if phase == "phase2":
        return PHASE2_COLUMNS
    raise ValueError(f"Unsupported phase: {phase}")


def make_empty_submission(input_df: pd.DataFrame, phase: str = "phase2") -> pd.DataFrame:
    if "image_id" not in input_df.columns:
        raise ValueError("Input dataframe must contain image_id column.")

    rows = []

    for image_id in input_df["image_id"].astype(str).tolist():
        if phase == "phase1":
            rows.append(
                {
                    "image_id": image_id,
                    "ocr_text": "",
                    "product_name": "",
                }
            )
        else:
            rows.append(
                {
                    "image_id": image_id,
                    "ocr_text": "",
                    "brand_name": "",
                    "product_name": "",
                }
            )

    return pd.DataFrame(rows, columns=get_submission_columns(phase))


def clean_submission_cells(df: pd.DataFrame, phase: str = "phase2") -> pd.DataFrame:
    columns = get_submission_columns(phase)
    out = df.copy()

    for col in columns:
        if col not in out.columns:
            out[col] = ""

    out = out[columns].fillna("")

    for col in columns:
        out[col] = out[col].map(strip_for_csv)

    return out


def validate_submission(
    submission_df: pd.DataFrame,
    input_df: pd.DataFrame,
    phase: str = "phase2",
) -> Dict[str, object]:
    columns = get_submission_columns(phase)
    errors = []

    if list(submission_df.columns) != columns:
        errors.append(
            {
                "type": "bad_columns",
                "expected": columns,
                "actual": list(submission_df.columns),
            }
        )

    if len(submission_df) != len(input_df):
        errors.append(
            {
                "type": "bad_row_count",
                "expected": len(input_df),
                "actual": len(submission_df),
            }
        )

    if "image_id" in submission_df.columns:
        duplicate_count = int(submission_df["image_id"].duplicated().sum())
        if duplicate_count > 0:
            errors.append(
                {
                    "type": "duplicate_image_id",
                    "count": duplicate_count,
                }
            )

    if "image_id" in input_df.columns and "image_id" in submission_df.columns:
        expected_ids = input_df["image_id"].astype(str).tolist()
        actual_ids = submission_df["image_id"].astype(str).tolist()

        if expected_ids != actual_ids:
            errors.append(
                {
                    "type": "image_id_order_mismatch",
                    "message": "Submission image_id order does not match input CSV.",
                }
            )

    null_count = int(submission_df.isna().sum().sum())
    if null_count > 0:
        errors.append(
            {
                "type": "null_cells",
                "count": null_count,
            }
        )

    newline_issues = {}
    for col in submission_df.columns:
        count = int(
            submission_df[col]
            .astype(str)
            .str.contains(r"[\n\r\t]", regex=True)
            .sum()
        )
        if count > 0:
            newline_issues[col] = count

    if newline_issues:
        errors.append(
            {
                "type": "newline_or_tab_cells",
                "columns": newline_issues,
            }
        )

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "rows": len(submission_df),
        "columns": list(submission_df.columns),
    }


def save_submission(
    submission_df: pd.DataFrame,
    input_df: pd.DataFrame,
    path: str | Path,
    phase: str = "phase2",
) -> Dict[str, object]:
    submission_df = clean_submission_cells(submission_df, phase=phase)
    report = validate_submission(submission_df, input_df, phase=phase)

    if not report["ok"]:
        raise ValueError(f"Invalid submission: {report}")

    write_csv(submission_df, path)
    return report