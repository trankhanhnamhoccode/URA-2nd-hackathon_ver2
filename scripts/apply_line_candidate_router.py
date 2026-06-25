from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ura_ocr.ocr.line_candidate_features import (
    btc_cer,
    clean_val,
    ensure_feature_columns,
    make_image_blank_features,
    prepare_candidate_features,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--candidates", type=str, required=True, help="Path to line_candidate_rows.csv-like file.")
    parser.add_argument("--selector-pkl", type=str, required=True, help="Path to line_candidate_selector.pkl.")
    parser.add_argument("--out-csv", type=str, required=True, help="Output selected CSV.")

    parser.add_argument("--blank-pkl", type=str, default="", help="Optional path to blank_classifier.pkl.")
    parser.add_argument("--blank-threshold", type=float, default=None, help="Override blank threshold.")

    return parser.parse_args()


def main():
    args = parse_args()

    cand_raw = pd.read_csv(args.candidates)
    cand = prepare_candidate_features(cand_raw)

    selector_payload = joblib.load(args.selector_pkl)

    selector = selector_payload["model"]
    cat_cols = selector_payload["cat_cols"]
    num_cols = selector_payload["num_cols"]

    cand = ensure_feature_columns(cand, cat_cols, num_cols)

    x_cols = cat_cols + num_cols

    cand["pred_btc_cer"] = selector.predict(cand[x_cols])
    cand["pred_btc_cer"] = cand["pred_btc_cer"].clip(0.0, 1.0)

    selected = (
        cand.sort_values(["image_id", "pred_btc_cer"], kind="mergesort")
        .groupby("image_id", as_index=False)
        .first()
    )

    selected["blank_proba"] = 0.0
    selected["blank_gate"] = 0

    if args.blank_pkl:
        blank_payload = joblib.load(args.blank_pkl)
        blank_model = blank_payload["model"]
        blank_feat_cols = blank_payload["blank_feat_cols"]

        threshold = args.blank_threshold
        if threshold is None:
            threshold = float(blank_payload.get("threshold", 0.7))

        img_feat = make_image_blank_features(cand)

        for col in blank_feat_cols:
            if col not in img_feat.columns:
                img_feat[col] = np.nan
            img_feat[col] = pd.to_numeric(img_feat[col], errors="coerce")

        blank_proba = blank_model.predict_proba(img_feat[blank_feat_cols])[:, 1]

        blank_df = pd.DataFrame({
            "image_id": img_feat["image_id"].values,
            "blank_proba": blank_proba,
        })

        selected = selected.drop(columns=["blank_proba", "blank_gate"], errors="ignore")
        selected = selected.merge(blank_df, on="image_id", how="left")
        selected["blank_proba"] = selected["blank_proba"].fillna(0.0)
        selected["blank_gate"] = (selected["blank_proba"] >= threshold).astype(int)

    selected["final_ocr_text"] = np.where(
        selected["blank_gate"] == 1,
        "",
        selected["ocr_text"].fillna("").astype(str),
    )

    if "gt_ocr_text" in selected.columns:
        selected["final_btc_cer"] = selected.apply(
            lambda r: btc_cer(r["gt_ocr_text"], r["final_ocr_text"]),
            axis=1,
        )
        selected["final_ocr_contribution_035"] = 0.35 * (1.0 - selected["final_btc_cer"])

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    selected.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("[INFO] Done.")
    print(f"[INFO] Selected rows: {len(selected)}")
    print(f"[INFO] Output: {out_path}")

    if "final_btc_cer" in selected.columns:
        avg = selected["final_btc_cer"].mean()
        print(f"[INFO] avg final BTC CER: {avg}")
        print(f"[INFO] OCR contribution 0.35: {0.35 * (1.0 - avg)}")


if __name__ == "__main__":
    main()
