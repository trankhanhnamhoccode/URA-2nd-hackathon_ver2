from __future__ import annotations

import argparse
from pathlib import Path

from ura_ocr.config import load_config
from ura_ocr.io.csv_io import read_csv_keep_empty
from ura_ocr.io.submission import save_submission
from ura_ocr.pipeline.predict_batch import predict_batch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    input_csv = cfg["input"]["test_csv"]
    submission_path = cfg["output"]["submission_path"]
    phase = cfg.get("phase", "phase2")

    print(f"[INFO] Loading input CSV: {input_csv}")
    input_df = read_csv_keep_empty(input_csv)

    if "image_id" not in input_df.columns:
        raise ValueError("Input CSV must contain image_id column.")

    print(f"[INFO] Rows: {len(input_df)}")
    print(f"[INFO] Phase: {phase}")

    pred_df = predict_batch(input_df, cfg)

    # Important: for sliced debug runs, validation should compare with sliced input.
    runtime_cfg = cfg.get("runtime", {})
    start_index = int(runtime_cfg.get("start_index") or 0)
    limit = runtime_cfg.get("limit", None)

    if limit is not None:
        validate_input_df = input_df.iloc[start_index:start_index + int(limit)].copy()
    else:
        validate_input_df = input_df.iloc[start_index:].copy()

    report = save_submission(
        submission_df=pred_df,
        input_df=validate_input_df,
        path=submission_path,
        phase=phase,
    )

    print(f"[INFO] Saved submission: {submission_path}")
    print(f"[INFO] Validation report: {report}")


if __name__ == "__main__":
    main()