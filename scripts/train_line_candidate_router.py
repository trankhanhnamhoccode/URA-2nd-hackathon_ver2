from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ura_ocr.ocr.line_candidate_features import (
    W_OCR,
    btc_cer,
    clean_val,
    ensure_feature_columns,
    get_blank_feature_columns,
    get_feature_columns,
    make_image_blank_features,
    prepare_candidate_features,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--candidates", type=str, required=True, help="Path to line_candidate_rows.csv.")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory.")

    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--eval-seeds", type=str, default="0,1,2,3,42,123")

    parser.add_argument("--selector-trees", type=int, default=400)
    parser.add_argument("--selector-min-leaf", type=int, default=2)

    parser.add_argument("--blank-trees", type=int, default=300)
    parser.add_argument("--blank-min-leaf", type=int, default=3)
    parser.add_argument("--blank-threshold", type=float, default=0.7)

    parser.add_argument("--random-state", type=int, default=42)

    return parser.parse_args()


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_selector_pipeline(cat_cols: list[str], num_cols: list[str], random_state: int, n_estimators: int, min_samples_leaf: int):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("onehot", make_ohe()),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )

    model = ExtraTreesRegressor(
        random_state=random_state,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
    )

    return Pipeline([
        ("pre", pre),
        ("model", model),
    ])


def build_blank_pipeline(random_state: int, n_estimators: int, min_samples_leaf: int):
    model = ExtraTreesClassifier(
        random_state=random_state,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        n_jobs=-1,
    )

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])


def evaluate_baselines(test_df: pd.DataFrame) -> dict[str, float]:
    raw = test_df[test_df["candidate_name"] == "variant::raw"].copy()
    raw_best = raw.groupby("image_id", as_index=False).first()
    raw_avg = raw_best["btc_cer"].mean()

    existing = test_df[test_df["candidate_type"] == "existing_variant"].copy()
    existing_best = (
        existing.sort_values(["image_id", "btc_cer"], kind="mergesort")
        .groupby("image_id", as_index=False)
        .first()
    )
    existing_avg = existing_best["btc_cer"].mean()

    all_best = (
        test_df.sort_values(["image_id", "btc_cer"], kind="mergesort")
        .groupby("image_id", as_index=False)
        .first()
    )
    all_avg = all_best["btc_cer"].mean()

    return {
        "raw_avg_btc_cer": float(raw_avg),
        "existing_oracle_avg_btc_cer": float(existing_avg),
        "all_candidate_oracle_avg_btc_cer": float(all_avg),
        "raw_ocr_contribution_035": float(W_OCR * (1.0 - raw_avg)),
        "existing_oracle_ocr_contribution_035": float(W_OCR * (1.0 - existing_avg)),
        "all_candidate_oracle_ocr_contribution_035": float(W_OCR * (1.0 - all_avg)),
    }


def select_by_prediction(test_df: pd.DataFrame) -> pd.DataFrame:
    selected = (
        test_df.sort_values(["image_id", "pred_btc_cer"], kind="mergesort")
        .groupby("image_id", as_index=False)
        .first()
    )

    return selected


def apply_blank_gate_to_selected(selected: pd.DataFrame, blank_pred: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = selected.copy()

    out = out.merge(
        blank_pred[["image_id", "blank_proba"]],
        on="image_id",
        how="left",
    )

    out["blank_proba"] = out["blank_proba"].fillna(0.0)
    out["blank_gate"] = (out["blank_proba"] >= threshold).astype(int)

    out["final_ocr_text"] = np.where(
        out["blank_gate"] == 1,
        "",
        out["ocr_text"].fillna("").astype(str),
    )

    if "gt_ocr_text" in out.columns:
        out["final_btc_cer"] = out.apply(
            lambda r: btc_cer(r["gt_ocr_text"], r["final_ocr_text"]),
            axis=1,
        )
    else:
        out["final_btc_cer"] = np.nan

    return out


def eval_selected(selected: pd.DataFrame, final_col: str = "btc_cer") -> dict[str, float]:
    avg = selected[final_col].mean()
    med = selected[final_col].median()

    return {
        "selected_avg_btc_cer": float(avg),
        "selected_median_btc_cer": float(med),
        "selected_ocr_contribution_035": float(W_OCR * (1.0 - avg)),
        "selected_rows": int(len(selected)),
    }


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(s.strip()) for s in args.eval_seeds.split(",") if s.strip()]

    cand_raw = pd.read_csv(args.candidates)
    cand = prepare_candidate_features(cand_raw)

    if "btc_cer" not in cand.columns:
        raise ValueError("Candidate file must contain btc_cer for training.")

    cand["btc_cer"] = pd.to_numeric(cand["btc_cer"], errors="coerce")
    cand = cand.dropna(subset=["btc_cer"]).reset_index(drop=True)

    cat_cols, num_cols = get_feature_columns(cand)
    cand = ensure_feature_columns(cand, cat_cols, num_cols)

    x_cols = cat_cols + num_cols
    y_col = "btc_cer"

    img_feat = make_image_blank_features(cand)
    if "gt_blank" not in img_feat.columns:
        raise ValueError("Need gt_ocr_text in candidates to train blank classifier.")

    blank_feat_cols = get_blank_feature_columns(img_feat)
    img_feat[blank_feat_cols] = img_feat[blank_feat_cols].apply(pd.to_numeric, errors="coerce")

    eval_rows = []
    selected_rows = []

    for seed in seeds:
        print(f"[INFO] eval seed={seed}", flush=True)

        splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(cand, groups=cand["image_id"]))

        train_ids = set(cand.iloc[train_idx]["image_id"].unique())
        test_ids = set(cand.iloc[test_idx]["image_id"].unique())

        train_df = cand[cand["image_id"].isin(train_ids)].copy()
        test_df = cand[cand["image_id"].isin(test_ids)].copy()

        selector = build_selector_pipeline(
            cat_cols=cat_cols,
            num_cols=num_cols,
            random_state=seed,
            n_estimators=args.selector_trees,
            min_samples_leaf=args.selector_min_leaf,
        )

        selector.fit(train_df[x_cols], train_df[y_col])

        test_df["pred_btc_cer"] = selector.predict(test_df[x_cols])
        test_df["pred_btc_cer"] = test_df["pred_btc_cer"].clip(0.0, 1.0)

        selected = select_by_prediction(test_df)

        train_img = img_feat[img_feat["image_id"].isin(train_ids)].copy()
        test_img = img_feat[img_feat["image_id"].isin(test_ids)].copy()

        blank_pipe = build_blank_pipeline(
            random_state=seed,
            n_estimators=args.blank_trees,
            min_samples_leaf=args.blank_min_leaf,
        )

        blank_pipe.fit(train_img[blank_feat_cols], train_img["gt_blank"].astype(int))

        blank_proba = blank_pipe.predict_proba(test_img[blank_feat_cols])[:, 1]
        blank_pred = pd.DataFrame({
            "image_id": test_img["image_id"].values,
            "blank_proba": blank_proba,
        })

        selected_gate = apply_blank_gate_to_selected(
            selected=selected,
            blank_pred=blank_pred,
            threshold=args.blank_threshold,
        )

        baseline = evaluate_baselines(test_df)
        selected_metric = eval_selected(selected, final_col="btc_cer")
        gated_metric = eval_selected(selected_gate, final_col="final_btc_cer")

        eval_rows.append({
            "seed": seed,
            **baseline,
            "selected_avg_btc_cer": selected_metric["selected_avg_btc_cer"],
            "selected_ocr_contribution_035": selected_metric["selected_ocr_contribution_035"],
            "gated_avg_btc_cer": gated_metric["selected_avg_btc_cer"],
            "gated_ocr_contribution_035": gated_metric["selected_ocr_contribution_035"],
            "gain_selected_vs_raw": selected_metric["selected_ocr_contribution_035"] - baseline["raw_ocr_contribution_035"],
            "gain_gated_vs_raw": gated_metric["selected_ocr_contribution_035"] - baseline["raw_ocr_contribution_035"],
            "gain_gate_vs_selector": gated_metric["selected_ocr_contribution_035"] - selected_metric["selected_ocr_contribution_035"],
            "gap_gated_to_all_oracle": baseline["all_candidate_oracle_ocr_contribution_035"] - gated_metric["selected_ocr_contribution_035"],
            "blank_forced": int(selected_gate["blank_gate"].sum()),
        })

        keep = selected_gate.copy()
        keep["seed"] = seed
        selected_rows.append(keep)

    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(out_dir / "line_router_eval.csv", index=False, encoding="utf-8-sig")

    if selected_rows:
        pd.concat(selected_rows, ignore_index=True).to_csv(
            out_dir / "line_router_selected_eval.csv",
            index=False,
            encoding="utf-8-sig",
        )

    summary = {
        "num_candidate_rows": int(len(cand)),
        "num_images": int(cand["image_id"].nunique()),
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "blank_feat_cols": blank_feat_cols,
        "blank_threshold": float(args.blank_threshold),
        "eval": {
            "mean_selected_ocr_contribution_035": float(eval_df["selected_ocr_contribution_035"].mean()),
            "mean_gated_ocr_contribution_035": float(eval_df["gated_ocr_contribution_035"].mean()),
            "mean_raw_ocr_contribution_035": float(eval_df["raw_ocr_contribution_035"].mean()),
            "mean_existing_oracle_ocr_contribution_035": float(eval_df["existing_oracle_ocr_contribution_035"].mean()),
            "mean_all_candidate_oracle_ocr_contribution_035": float(eval_df["all_candidate_oracle_ocr_contribution_035"].mean()),
            "mean_gain_gate_vs_selector": float(eval_df["gain_gate_vs_selector"].mean()),
        },
    }

    with (out_dir / "line_router_eval_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[INFO] Training final selector on all candidates...", flush=True)

    final_selector = build_selector_pipeline(
        cat_cols=cat_cols,
        num_cols=num_cols,
        random_state=args.random_state,
        n_estimators=args.selector_trees,
        min_samples_leaf=args.selector_min_leaf,
    )
    final_selector.fit(cand[x_cols], cand[y_col])

    print("[INFO] Training final blank classifier on all images...", flush=True)

    final_blank = build_blank_pipeline(
        random_state=args.random_state,
        n_estimators=args.blank_trees,
        min_samples_leaf=args.blank_min_leaf,
    )
    final_blank.fit(img_feat[blank_feat_cols], img_feat["gt_blank"].astype(int))

    selector_payload = {
        "model": final_selector,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "feature_version": "line_candidate_features_v1",
        "target": "btc_cer",
    }

    blank_payload = {
        "model": final_blank,
        "blank_feat_cols": blank_feat_cols,
        "threshold": float(args.blank_threshold),
        "feature_version": "image_blank_features_v1",
    }

    joblib.dump(selector_payload, out_dir / "line_candidate_selector.pkl")
    joblib.dump(blank_payload, out_dir / "blank_classifier.pkl")

    print("[INFO] Done.")
    print(f"[INFO] Saved selector: {out_dir / 'line_candidate_selector.pkl'}")
    print(f"[INFO] Saved blank classifier: {out_dir / 'blank_classifier.pkl'}")
    print(f"[INFO] Eval summary: {out_dir / 'line_router_eval_summary.json'}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
