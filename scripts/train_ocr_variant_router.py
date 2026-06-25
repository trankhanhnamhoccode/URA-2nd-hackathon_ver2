from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

from ura_ocr.ocr.variant_router import (
    DEFAULT_VARIANTS,
    build_router_features,
    select_best_variant_from_predictions,
)


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
        help="Output directory for trained router and eval report.",
    )

    parser.add_argument(
        "--variants",
        type=str,
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated variants to use.",
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default="gbr",
        choices=["gbr", "rf"],
        help="Router model type.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Image-level test split size.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    return parser.parse_args()


def make_model(model_type: str, random_state: int):
    if model_type == "rf":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
        )

    return GradientBoostingRegressor(
        n_estimators=250,
        learning_rate=0.04,
        max_depth=3,
        min_samples_leaf=3,
        random_state=random_state,
    )


def evaluate_router(df_feat: pd.DataFrame, y_true_col: str = "cer") -> dict:
    """
    Evaluate selected variant actual CER vs raw and oracle.
    """
    selected = select_best_variant_from_predictions(df_feat, pred_col="pred_cer")

    raw = (
        df_feat[df_feat["variant"] == "raw"][["image_id", y_true_col]]
        .rename(columns={y_true_col: "raw_cer"})
        .copy()
    )

    oracle = (
        df_feat.groupby("image_id", as_index=False)[y_true_col]
        .min()
        .rename(columns={y_true_col: "oracle_cer"})
    )

    selected_eval = (
        selected[["image_id", "variant", y_true_col, "pred_cer"]]
        .rename(columns={"variant": "selected_variant", y_true_col: "selected_cer"})
        .merge(raw, on="image_id", how="left")
        .merge(oracle, on="image_id", how="left")
    )

    result = {
        "num_images": int(selected_eval["image_id"].nunique()),
        "selected_mean_cer": float(selected_eval["selected_cer"].mean()),
        "selected_median_cer": float(selected_eval["selected_cer"].median()),
        "raw_mean_cer": float(selected_eval["raw_cer"].mean()),
        "raw_median_cer": float(selected_eval["raw_cer"].median()),
        "oracle_mean_cer": float(selected_eval["oracle_cer"].mean()),
        "oracle_median_cer": float(selected_eval["oracle_cer"].median()),
        "mean_gain_vs_raw": float(selected_eval["raw_cer"].mean() - selected_eval["selected_cer"].mean()),
        "median_gain_vs_raw": float(selected_eval["raw_cer"].median() - selected_eval["selected_cer"].median()),
        "gap_to_oracle_mean": float(selected_eval["selected_cer"].mean() - selected_eval["oracle_cer"].mean()),
        "selected_variant_counts": selected_eval["selected_variant"].value_counts().to_dict(),
    }

    return result, selected_eval


def main():
    args = parse_args()

    report_path = Path(args.report)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    print(f"[INFO] report: {report_path}")
    print(f"[INFO] out_dir: {out_dir}")
    print(f"[INFO] variants: {variants}")

    report_df = pd.read_csv(report_path)

    if "cer" not in report_df.columns:
        raise ValueError("Report must contain cer column.")

    feat_df, feature_cols = build_router_features(report_df, variants=variants)

    feat_df["cer"] = pd.to_numeric(feat_df["cer"], errors="coerce").fillna(1.0)

    groups = feat_df["image_id"].astype(str)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    train_idx, test_idx = next(splitter.split(feat_df, feat_df["cer"], groups=groups))

    train_df = feat_df.iloc[train_idx].copy()
    test_df = feat_df.iloc[test_idx].copy()

    model = make_model(args.model_type, args.random_state)

    X_train = train_df[feature_cols]
    y_train = train_df["cer"]

    X_test = test_df[feature_cols]
    y_test = test_df["cer"]

    print(f"[INFO] train rows: {len(train_df)}, test rows: {len(test_df)}")
    print(f"[INFO] train images: {train_df['image_id'].nunique()}, test images: {test_df['image_id'].nunique()}")
    print(f"[INFO] feature count: {len(feature_cols)}")

    model.fit(X_train, y_train)

    test_df["pred_cer"] = model.predict(X_test)
    test_df["pred_cer"] = test_df["pred_cer"].clip(lower=0.0)

    mae = mean_absolute_error(y_test, test_df["pred_cer"])

    eval_result, selected_eval = evaluate_router(test_df)

    eval_result["row_level_mae"] = float(mae)
    eval_result["model_type"] = args.model_type
    eval_result["variants"] = variants
    eval_result["feature_cols"] = feature_cols

    print("[INFO] Router eval:")
    print(json.dumps(eval_result, ensure_ascii=False, indent=2))

    # Train final model on all rows.
    final_model = make_model(args.model_type, args.random_state)
    final_model.fit(feat_df[feature_cols], feat_df["cer"])

    artifact = {
        "model": final_model,
        "feature_cols": feature_cols,
        "variants": variants,
        "model_type": args.model_type,
    }

    model_path = out_dir / "ocr_variant_router.pkl"
    with model_path.open("wb") as f:
        pickle.dump(artifact, f)

    eval_path = out_dir / "router_eval.json"
    with eval_path.open("w", encoding="utf-8") as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)

    selected_path = out_dir / "router_selected_eval.csv"
    selected_eval.to_csv(selected_path, index=False)

    feature_importance_path = out_dir / "feature_importance.csv"

    if hasattr(final_model, "feature_importances_"):
        fi = pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": final_model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        fi.to_csv(feature_importance_path, index=False)
        print(f"[INFO] feature importance: {feature_importance_path}")

    print(f"[INFO] saved model: {model_path}")
    print(f"[INFO] saved eval: {eval_path}")
    print(f"[INFO] saved selected eval: {selected_path}")


if __name__ == "__main__":
    main()