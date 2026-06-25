from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


W_OCR = 0.35


DEFAULT_VARIANTS = [
    "raw",
    "center_70_resize_960",
    "bottom_60_resize_960",
    "bottom_50_resize_960",
    "bottom_45_resize_960",
    "middle_bottom_70_resize_960",
    "center_60_resize_960",
    "upper_60_resize_960",
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--report",
        type=str,
        required=True,
        help="Path to ocr_ablation_report.csv.",
    )

    parser.add_argument(
        "--lines",
        type=str,
        required=True,
        help="Path to ocr_ablation_lines.csv.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory.",
    )

    parser.add_argument(
        "--variants",
        type=str,
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated variants to use.",
    )

    return parser.parse_args()


def clean_val(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def normalize_spaces(text: str) -> str:
    text = clean_val(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def final_join(lines: Iterable[str]) -> str:
    vals = []
    for line in lines:
        line = clean_val(line)
        if line:
            vals.append(line)
    return normalize_spaces(" ".join(vals))


def btc_cer(gt: str, pred: str) -> float:
    gt = clean_val(gt)
    pred = clean_val(pred)

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


def text_key(text: str) -> str:
    text = clean_val(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^0-9a-zà-ỹ]+", "", text)
    return text


def alnum_ratio(text: str) -> float:
    text = clean_val(text)
    if not text:
        return 0.0
    return sum(ch.isalnum() for ch in text) / max(len(text), 1)


def accent_count(text: str) -> int:
    text = clean_val(text)
    return sum(1 for ch in text if "À" <= ch <= "ỹ")


def prepare_lines(lines: pd.DataFrame) -> pd.DataFrame:
    df = lines.copy()

    for col in ["line_text", "variant", "image_id"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    for col in [
        "line_score",
        "box_xmin",
        "box_ymin",
        "box_xmax",
        "box_ymax",
        "box_width",
        "box_height",
        "box_area_ratio",
        "box_cx_ratio",
        "box_cy_ratio",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    if "line_idx" in df.columns:
        df["line_idx"] = pd.to_numeric(df["line_idx"], errors="coerce").fillna(0).astype(int)
    else:
        df["line_idx"] = 0

    df["text_len"] = df["line_text"].str.len()
    df["word_count"] = df["line_text"].apply(lambda s: len(clean_val(s).split()))
    df["alnum_ratio"] = df["line_text"].apply(alnum_ratio)
    df["accent_count"] = df["line_text"].apply(accent_count)
    df["line_key"] = df["line_text"].apply(text_key)
    df["is_blank"] = df["line_text"].eq("")

    # Sort by detected reading-ish order.
    # Fallback to line_idx when bbox missing.
    df["sort_y"] = df["box_ymin"].fillna(df["line_idx"].astype(float) * 10000.0)
    df["sort_x"] = df["box_xmin"].fillna(0.0)

    return df


def dedupe_keep_order(group: pd.DataFrame) -> pd.DataFrame:
    keep_idx = []
    seen = set()

    for idx, row in group.iterrows():
        key = clean_val(row.get("line_key"))
        text = clean_val(row.get("line_text"))

        if not text:
            continue

        if not key:
            key = text.lower()

        if key in seen:
            continue

        seen.add(key)
        keep_idx.append(idx)

    return group.loc[keep_idx].copy()


def sort_for_reading(group: pd.DataFrame) -> pd.DataFrame:
    return group.sort_values(["sort_y", "sort_x", "line_idx"], kind="mergesort")


def build_candidate(
    image_id: str,
    gt_ocr_text: str,
    candidate_name: str,
    candidate_type: str,
    source_variant: str,
    line_df: pd.DataFrame,
    note: str = "",
) -> Optional[Dict[str, Any]]:
    line_df = line_df.copy()

    if line_df.empty:
        text = ""
    else:
        line_df = sort_for_reading(line_df)
        line_df = dedupe_keep_order(line_df)
        text = final_join(line_df["line_text"].tolist())

    if not text:
        return None

    c = btc_cer(gt_ocr_text, text)

    return {
        "image_id": image_id,
        "candidate_name": candidate_name,
        "candidate_type": candidate_type,
        "source_variant": source_variant,
        "ocr_text": text,
        "gt_ocr_text": gt_ocr_text,
        "btc_cer": c,
        "ocr_score_035": W_OCR * (1.0 - c),
        "num_lines": int(len(line_df)),
        "avg_line_score": float(line_df["line_score"].mean()) if len(line_df) else 0.0,
        "median_line_score": float(line_df["line_score"].median()) if len(line_df) else 0.0,
        "avg_area_ratio": float(line_df["box_area_ratio"].mean()) if len(line_df) else 0.0,
        "note": note,
    }


def add_existing_variant_candidates(
    rows: List[Dict[str, Any]],
    report: pd.DataFrame,
    variants: List[str],
):
    keep = report[report["variant"].isin(variants)].copy()

    for _, r in keep.iterrows():
        image_id = clean_val(r["image_id"])
        variant = clean_val(r["variant"])
        gt = clean_val(r["gt_ocr_text"])
        text = clean_val(r["ocr_text"])

        c = btc_cer(gt, text)

        rows.append(
            {
                "image_id": image_id,
                "candidate_name": f"variant::{variant}",
                "candidate_type": "existing_variant",
                "source_variant": variant,
                "ocr_text": text,
                "gt_ocr_text": gt,
                "btc_cer": c,
                "ocr_score_035": W_OCR * (1.0 - c),
                "num_lines": int(r.get("num_lines", 0)) if not pd.isna(r.get("num_lines", 0)) else 0,
                "avg_line_score": float(r.get("avg_score", 0.0)) if not pd.isna(r.get("avg_score", 0.0)) else 0.0,
                "median_line_score": np.nan,
                "avg_area_ratio": np.nan,
                "note": "original ablation output",
            }
        )


def candidate_policies_for_variant(
    g: pd.DataFrame,
    image_id: str,
    gt: str,
    variant: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    base = g[
        (~g["is_blank"])
        & (g["text_len"] >= 2)
        & (g["alnum_ratio"] >= 0.35)
    ].copy()

    if base.empty:
        return out

    policies = []

    # Conservative line filters.
    policies.append(("all_clean", base, "nonblank, text_len>=2, alnum>=0.35"))

    for th in [0.70, 0.80, 0.85, 0.90, 0.93]:
        policies.append(
            (
                f"score_ge_{str(th).replace('.', '')}",
                base[base["line_score"].fillna(0.0) >= th],
                f"line_score>={th}",
            )
        )

    # Larger text boxes often correspond to main headline/product text.
    for q in [0.50, 0.65, 0.75]:
        if base["box_area_ratio"].notna().any():
            area_th = base["box_area_ratio"].quantile(q)
            policies.append(
                (
                    f"area_ge_q{int(q * 100)}",
                    base[base["box_area_ratio"].fillna(0.0) >= area_th],
                    f"box_area_ratio >= per-image variant q{q}",
                )
            )

    # Keep top-k by box area, then restore reading order.
    for k in [1, 2, 3, 5]:
        if base["box_area_ratio"].notna().any():
            sub = base.sort_values("box_area_ratio", ascending=False).head(k)
            policies.append((f"top{k}_area", sub, f"top {k} lines by area"))

    # Keep top-k by score.
    for k in [1, 2, 3, 5]:
        sub = base.sort_values("line_score", ascending=False).head(k)
        policies.append((f"top{k}_score", sub, f"top {k} lines by score"))

    # Region filters inside current variant crop coordinates.
    # These are not original image coords, but still useful for line layout inside crop.
    policies.append(("upper_half_lines", base[base["box_cy_ratio"].fillna(0.5) <= 0.55], "cy<=0.55"))
    policies.append(("lower_half_lines", base[base["box_cy_ratio"].fillna(0.5) >= 0.45], "cy>=0.45"))
    policies.append(("middle_lines", base[base["box_cy_ratio"].fillna(0.5).between(0.25, 0.80)], "0.25<=cy<=0.80"))

    # Text-ish policies.
    policies.append(("with_accent", base[base["accent_count"] > 0], "lines containing Vietnamese accents"))
    policies.append(("long_lines", base[base["text_len"] >= 8], "text_len>=8"))

    for name, sub, note in policies:
        if sub.empty:
            continue

        cand = build_candidate(
            image_id=image_id,
            gt_ocr_text=gt,
            candidate_name=f"line::{variant}::{name}",
            candidate_type="single_variant_line_policy",
            source_variant=variant,
            line_df=sub,
            note=note,
        )

        if cand is not None:
            out.append(cand)

    return out


def add_single_variant_line_candidates(
    rows: List[Dict[str, Any]],
    lines: pd.DataFrame,
    report: pd.DataFrame,
    variants: List[str],
):
    gt_map = (
        report[["image_id", "gt_ocr_text"]]
        .drop_duplicates("image_id")
        .set_index("image_id")["gt_ocr_text"]
        .to_dict()
    )

    use_lines = lines[lines["variant"].isin(variants)].copy()

    for (image_id, variant), g in use_lines.groupby(["image_id", "variant"]):
        gt = clean_val(gt_map.get(image_id, ""))
        rows.extend(candidate_policies_for_variant(g, image_id, gt, variant))


def make_merged_candidate(
    image_id: str,
    gt: str,
    name: str,
    source_variant: str,
    groups: List[pd.DataFrame],
    note: str,
) -> Optional[Dict[str, Any]]:
    parts = []

    for g in groups:
        if g is None or g.empty:
            continue
        parts.append(g)

    if not parts:
        return None

    merged = pd.concat(parts, ignore_index=True)

    # When same/similar line appears in multiple variants, prefer:
    # higher score, larger area, then earlier row.
    merged = merged.sort_values(
        ["line_score", "box_area_ratio"],
        ascending=[False, False],
        kind="mergesort",
    )

    merged = dedupe_keep_order(merged)
    merged = sort_for_reading(merged)

    return build_candidate(
        image_id=image_id,
        gt_ocr_text=gt,
        candidate_name=name,
        candidate_type="merged_line_policy",
        source_variant=source_variant,
        line_df=merged,
        note=note,
    )


def add_merged_line_candidates(
    rows: List[Dict[str, Any]],
    lines: pd.DataFrame,
    report: pd.DataFrame,
    variants: List[str],
):
    gt_map = (
        report[["image_id", "gt_ocr_text"]]
        .drop_duplicates("image_id")
        .set_index("image_id")["gt_ocr_text"]
        .to_dict()
    )

    priority_groups = [
        ["raw", "bottom_50_resize_960"],
        ["raw", "bottom_60_resize_960"],
        ["raw", "center_70_resize_960"],
        ["bottom_50_resize_960", "center_70_resize_960"],
        ["bottom_60_resize_960", "center_70_resize_960"],
        ["raw", "bottom_50_resize_960", "center_70_resize_960"],
        ["raw", "bottom_60_resize_960", "center_70_resize_960"],
    ]

    priority_groups = [
        [v for v in group if v in variants]
        for group in priority_groups
    ]
    priority_groups = [group for group in priority_groups if len(group) >= 2]

    for image_id, img_lines in lines.groupby("image_id"):
        gt = clean_val(gt_map.get(image_id, ""))

        for group in priority_groups:
            group_lines = img_lines[img_lines["variant"].isin(group)].copy()

            if group_lines.empty:
                continue

            base = group_lines[
                (~group_lines["is_blank"])
                & (group_lines["text_len"] >= 2)
                & (group_lines["alnum_ratio"] >= 0.35)
            ].copy()

            if base.empty:
                continue

            source = "+".join(group)

            # Merge all clean lines from chosen variants.
            cand = make_merged_candidate(
                image_id=image_id,
                gt=gt,
                name=f"merge::{source}::all_clean",
                source_variant=source,
                groups=[base],
                note="dedupe all clean lines from selected variants",
            )
            if cand is not None:
                rows.append(cand)

            # Merge only high score lines.
            for th in [0.80, 0.85, 0.90]:
                sub = base[base["line_score"].fillna(0.0) >= th].copy()
                cand = make_merged_candidate(
                    image_id=image_id,
                    gt=gt,
                    name=f"merge::{source}::score_ge_{str(th).replace('.', '')}",
                    source_variant=source,
                    groups=[sub],
                    note=f"dedupe lines with score>={th}",
                )
                if cand is not None:
                    rows.append(cand)

            # Merge top area lines across variants.
            for k in [3, 5, 8]:
                sub = base.sort_values("box_area_ratio", ascending=False).head(k)
                cand = make_merged_candidate(
                    image_id=image_id,
                    gt=gt,
                    name=f"merge::{source}::top{k}_area",
                    source_variant=source,
                    groups=[sub],
                    note=f"dedupe top {k} largest-area lines across variants",
                )
                if cand is not None:
                    rows.append(cand)


def summarize_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for name, g in candidates.groupby("candidate_name"):
        avg_cer = float(g["btc_cer"].mean())
        rows.append(
            {
                "candidate_name": name,
                "candidate_type": g["candidate_type"].iloc[0],
                "source_variant": g["source_variant"].iloc[0],
                "rows": int(len(g)),
                "avg_btc_cer": avg_cer,
                "median_btc_cer": float(g["btc_cer"].median()),
                "ocr_score_1_minus_cer": float(1.0 - avg_cer),
                "ocr_contribution_035": float(W_OCR * (1.0 - avg_cer)),
                "avg_num_lines": float(g["num_lines"].mean()),
                "avg_line_score": float(g["avg_line_score"].mean()),
                "avg_area_ratio": float(g["avg_area_ratio"].mean()),
            }
        )

    return pd.DataFrame(rows).sort_values("ocr_contribution_035", ascending=False)


def oracle_summary(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Existing variant oracle.
    existing = candidates[candidates["candidate_type"] == "existing_variant"].copy()
    existing_winners = (
        existing.sort_values(["image_id", "btc_cer"], kind="mergesort")
        .groupby("image_id", as_index=False)
        .first()
    )
    existing_winners["oracle_group"] = "existing_variant_oracle"

    # All candidate oracle.
    all_winners = (
        candidates.sort_values(["image_id", "btc_cer"], kind="mergesort")
        .groupby("image_id", as_index=False)
        .first()
    )
    all_winners["oracle_group"] = "all_line_candidate_oracle"

    # Raw baseline.
    raw = candidates[candidates["candidate_name"] == "variant::raw"].copy()
    raw["oracle_group"] = "raw"

    blocks = []

    if len(raw):
        blocks.append(raw)

    blocks.append(existing_winners)
    blocks.append(all_winners)

    winner_df = pd.concat(blocks, ignore_index=True)

    summary_rows = []
    for group_name, g in winner_df.groupby("oracle_group"):
        avg_cer = float(g["btc_cer"].mean())
        summary_rows.append(
            {
                "method": group_name,
                "rows": int(len(g)),
                "avg_btc_cer": avg_cer,
                "median_btc_cer": float(g["btc_cer"].median()),
                "ocr_score_1_minus_cer": float(1.0 - avg_cer),
                "ocr_contribution_035": float(W_OCR * (1.0 - avg_cer)),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("ocr_contribution_035", ascending=False)

    return summary_df, winner_df


def main():
    args = parse_args()

    report_path = Path(args.report)
    lines_path = Path(args.lines)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    print(f"[INFO] report: {report_path}")
    print(f"[INFO] lines : {lines_path}")
    print(f"[INFO] out_dir: {out_dir}")
    print(f"[INFO] variants: {variants}")

    report = pd.read_csv(report_path)
    lines = pd.read_csv(lines_path)

    required_report = {"image_id", "variant", "ocr_text", "gt_ocr_text"}
    missing_report = required_report - set(report.columns)
    if missing_report:
        raise ValueError(f"Missing report columns: {missing_report}")

    required_lines = {"image_id", "variant", "line_text"}
    missing_lines = required_lines - set(lines.columns)
    if missing_lines:
        raise ValueError(f"Missing line columns: {missing_lines}")

    for col in ["image_id", "variant", "ocr_text", "gt_ocr_text"]:
        if col in report.columns:
            report[col] = report[col].fillna("").astype(str).str.strip()

    lines = prepare_lines(lines)

    report = report[report["variant"].isin(variants)].copy()
    lines = lines[lines["variant"].isin(variants)].copy()

    candidate_rows: List[Dict[str, Any]] = []

    add_existing_variant_candidates(candidate_rows, report, variants)
    add_single_variant_line_candidates(candidate_rows, lines, report, variants)
    add_merged_line_candidates(candidate_rows, lines, report, variants)

    candidates = pd.DataFrame(candidate_rows)

    if candidates.empty:
        raise RuntimeError("No candidates were generated. Check report/lines input.")

    candidate_summary = summarize_candidates(candidates)
    oracle_df, winners = oracle_summary(candidates)

    candidates_path = out_dir / "line_candidate_rows.csv"
    candidate_summary_path = out_dir / "line_candidate_summary.csv"
    oracle_path = out_dir / "line_candidate_oracle_summary.csv"
    winners_path = out_dir / "line_candidate_winners.csv"
    json_path = out_dir / "line_candidate_summary.json"

    candidates.to_csv(candidates_path, index=False, encoding="utf-8-sig")
    candidate_summary.to_csv(candidate_summary_path, index=False, encoding="utf-8-sig")
    oracle_df.to_csv(oracle_path, index=False, encoding="utf-8-sig")
    winners.to_csv(winners_path, index=False, encoding="utf-8-sig")

    payload = {
        "report": str(report_path),
        "lines": str(lines_path),
        "variants": variants,
        "num_candidate_rows": int(len(candidates)),
        "num_images": int(candidates["image_id"].nunique()),
        "oracle_summary": oracle_df.to_dict(orient="records"),
        "top_candidates": candidate_summary.head(25).to_dict(orient="records"),
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("[INFO] Done.")
    print(f"[INFO] candidates: {candidates_path}")
    print(f"[INFO] candidate summary: {candidate_summary_path}")
    print(f"[INFO] oracle summary: {oracle_path}")
    print(f"[INFO] winners: {winners_path}")

    print("\n[ORACLE SUMMARY]")
    print(oracle_df)

    print("\n[TOP CANDIDATES]")
    print(candidate_summary.head(25))


if __name__ == "__main__":
    main()