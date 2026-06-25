from __future__ import annotations

import argparse
from pathlib import Path

from ura_ocr.io.csv_io import read_csv_keep_empty
from ura_ocr.eval.metrics import evaluate_submission_dataframe, save_eval_outputs


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pred",
        type=str,
        required=True,
        help="Prediction/submission CSV path.",
    )

    parser.add_argument(
        "--gt",
        type=str,
        required=True,
        help="Ground truth CSV path.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Directory to write eval_report.json and error_cases.csv.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    pred_path = Path(args.pred)
    gt_path = Path(args.gt)
    out_dir = Path(args.out_dir)

    print(f"[INFO] Loading prediction: {pred_path}")
    pred_df = read_csv_keep_empty(pred_path)

    print(f"[INFO] Loading ground truth: {gt_path}")
    gt_df = read_csv_keep_empty(gt_path)

    report, error_cases = evaluate_submission_dataframe(pred_df, gt_df)

    save_eval_outputs(
        report=report,
        error_cases=error_cases,
        out_dir=out_dir,
    )

    print("[INFO] Evaluation complete.")
    print(f"[INFO] rows_pred: {report['rows_pred']}")
    print(f"[INFO] rows_gt: {report['rows_gt']}")
    print(f"[INFO] rows_overlap: {report['rows_overlap']}")
    print(f"[INFO] product token_f1_macro: {report['product']['token_f1_macro']:.4f}")
    print(f"[INFO] product exact_no_accent: {report['product']['exact_match_no_accent']:.4f}")
    print(f"[INFO] OCR CER macro: {report['ocr']['cer_macro']:.4f}")
    print(f"[INFO] Error cases: {len(error_cases)}")
    print(f"[INFO] Saved to: {out_dir}")


if __name__ == "__main__":
    main()