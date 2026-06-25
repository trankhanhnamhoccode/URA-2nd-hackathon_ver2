from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


W_OCR = 0.35


def _clean(val) -> str:
    return "" if pd.isna(val) else str(val).strip()


def btc_cer(gt: str, pred: str) -> float:
    """
    CER exactly following BTC private metric behavior:
    - strip NaN/space
    - if GT blank and pred blank: 0
    - if GT blank and pred nonblank: 1
    - otherwise edit_distance / len(gt), clamped to max 1
    """
    gt = _clean(gt)
    pred = _clean(pred)

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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--report",
        type=str,
        required=True,
        help="Path to ocr_ablation_report.csv.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory.",
    )

    parser.add_argument(
        "--raw-variant",
        type=str,
        default="raw",
    )

    return parser.parse_args()


def summarize_variant(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for variant, group in df.groupby("variant"):
        avg_cer = group["btc_cer"].mean()
        rows.append(
            {
                "method": variant,
                "rows": int(len(group)),
                "avg_btc_cer": float(avg_cer),
                "median_btc_cer": float(group["btc_cer"].median()),
                "ocr_score_1_minus_cer": float(1.0 - avg_cer),
                "ocr_contribution_035": float(W_OCR * (1.0 - avg_cer)),
                "runtime_mean": float(group["runtime_sec"].mean()) if "runtime_sec" in group else 0.0,
                "num_lines_mean": float(group["num_lines"].mean()) if "num_lines" in group else 0.0,
                "avg_score_mean": float(group["avg_score"].mean()) if "avg_score" in group else 0.0,
                "blank_rate": float(group["ocr_text"].fillna("").astype(str).str.strip().eq("").mean()),
            }
        )

    return pd.DataFrame(rows).sort_values("ocr_contribution_035", ascending=False)


def main():
    args = parse_args()

    report_path = Path(args.report)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(report_path)

    required = {"image_id", "variant", "ocr_text", "gt_ocr_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["btc_cer"] = df.apply(
        lambda r: btc_cer(r["gt_ocr_text"], r["ocr_text"]),
        axis=1,
    )

    variant_summary = summarize_variant(df)

    oracle = (
        df.groupby("image_id", as_index=False)["btc_cer"]
        .min()
        .rename(columns={"btc_cer": "oracle_btc_cer"})
    )

    oracle_avg = oracle["oracle_btc_cer"].mean()
    oracle_median = oracle["oracle_btc_cer"].median()

    raw_df = df[df["variant"] == args.raw_variant].copy()
    raw_avg = raw_df["btc_cer"].mean() if len(raw_df) else None
    raw_median = raw_df["btc_cer"].median() if len(raw_df) else None

    overall_rows = []

    if raw_avg is not None:
        overall_rows.append(
            {
                "method": args.raw_variant,
                "avg_btc_cer": float(raw_avg),
                "median_btc_cer": float(raw_median),
                "ocr_score_1_minus_cer": float(1.0 - raw_avg),
                "ocr_contribution_035": float(W_OCR * (1.0 - raw_avg)),
            }
        )

    overall_rows.append(
        {
            "method": "oracle",
            "avg_btc_cer": float(oracle_avg),
            "median_btc_cer": float(oracle_median),
            "ocr_score_1_minus_cer": float(1.0 - oracle_avg),
            "ocr_contribution_035": float(W_OCR * (1.0 - oracle_avg)),
        }
    )

    overall = pd.DataFrame(overall_rows)

    variant_summary_path = out_dir / "btc_variant_summary.csv"
    overall_path = out_dir / "btc_overall_summary.csv"
    json_path = out_dir / "btc_summary.json"
    report_with_btc_path = out_dir / "ocr_ablation_report_with_btc_cer.csv"

    variant_summary.to_csv(variant_summary_path, index=False, encoding="utf-8-sig")
    overall.to_csv(overall_path, index=False, encoding="utf-8-sig")
    df.to_csv(report_with_btc_path, index=False, encoding="utf-8-sig")

    payload = {
        "report": str(report_path),
        "rows": int(len(df)),
        "num_images": int(df["image_id"].nunique()),
        "raw_variant": args.raw_variant,
        "overall": overall.to_dict(orient="records"),
        "top_variants": variant_summary.head(20).to_dict(orient="records"),
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("[INFO] BTC OCR summary saved.")
    print("variant_summary:", variant_summary_path)
    print("overall:", overall_path)
    print("json:", json_path)
    print("report_with_btc:", report_with_btc_path)

    print("\n[OVERALL]")
    print(overall)

    print("\n[TOP VARIANTS]")
    print(variant_summary.head(20))


if __name__ == "__main__":
    main()