from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def clean_text(x):
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def norm_space(s):
    return " ".join(clean_text(s).split())


def join_lines(lines):
    out = []
    seen = set()

    for s in lines:
        s = norm_space(s)
        if not s:
            continue

        key = s.lower()
        if key in seen:
            continue

        seen.add(key)
        out.append(s)

    return " ".join(out).strip()


def candidate_row(
    image_id,
    candidate_name,
    candidate_type,
    source_variant,
    ocr_text,
    num_lines=np.nan,
    avg_line_score=np.nan,
    median_line_score=np.nan,
    avg_area_ratio=np.nan,
):
    return {
        "image_id": image_id,
        "candidate_name": candidate_name,
        "candidate_type": candidate_type,
        "source_variant": source_variant,
        "ocr_text": norm_space(ocr_text),
        "num_lines": num_lines,
        "avg_line_score": avg_line_score,
        "median_line_score": median_line_score,
        "avg_area_ratio": avg_area_ratio,
    }


def ensure_line_text_column(lines: pd.DataFrame) -> pd.DataFrame:
    lines = lines.copy()

    if "text" in lines.columns:
        lines["text"] = lines["text"].fillna("").astype(str)
        return lines

    for alt in ["line_text", "ocr_text", "det_text", "rec_text"]:
        if alt in lines.columns:
            lines["text"] = lines[alt].fillna("").astype(str)
            return lines

    lines["text"] = ""
    return lines


def summarize_lines(g):
    if len(g) == 0:
        return 0, np.nan, np.nan, np.nan

    scores = pd.to_numeric(g.get("score", pd.Series([], dtype=float)), errors="coerce")
    areas = pd.to_numeric(g.get("area_ratio", pd.Series([], dtype=float)), errors="coerce")

    return (
        int(len(g)),
        float(scores.mean()) if scores.notna().any() else np.nan,
        float(scores.median()) if scores.notna().any() else np.nan,
        float(areas.mean()) if areas.notna().any() else np.nan,
    )


def line_policy_candidates(image_id, variant, g):
    rows = []

    if g.empty:
        return rows

    g = g.copy()
    g["text"] = g["text"].fillna("").astype(str).map(norm_space)
    g = g[g["text"].str.len() > 0].copy()

    if g.empty:
        return rows

    for col in ["score", "area_ratio", "cy_ratio"]:
        if col not in g.columns:
            g[col] = np.nan
        g[col] = pd.to_numeric(g[col], errors="coerce")

    policies = []

    policies.append(("all_clean", g))

    for th_name, th in [
        ("score_ge_07", 0.70),
        ("score_ge_08", 0.80),
        ("score_ge_085", 0.85),
        ("score_ge_09", 0.90),
        ("score_ge_093", 0.93),
    ]:
        policies.append((th_name, g[g["score"].fillna(0) >= th]))

    if g["area_ratio"].notna().any():
        for q_name, q in [
            ("area_ge_q50", 0.50),
            ("area_ge_q65", 0.65),
            ("area_ge_q75", 0.75),
        ]:
            cut = g["area_ratio"].quantile(q)
            policies.append((q_name, g[g["area_ratio"].fillna(0) >= cut]))

        for k in [1, 2, 3, 5]:
            policies.append((f"top{k}_area", g.sort_values("area_ratio", ascending=False).head(k)))

    if g["score"].notna().any():
        for k in [1, 2, 3, 5]:
            policies.append((f"top{k}_score", g.sort_values("score", ascending=False).head(k)))

    if g["cy_ratio"].notna().any():
        policies.append(("upper_half_lines", g[g["cy_ratio"] <= 0.50]))
        policies.append(("lower_half_lines", g[g["cy_ratio"] >= 0.50]))
        policies.append(("middle_lines", g[(g["cy_ratio"] >= 0.25) & (g["cy_ratio"] <= 0.75)]))

    accent_mask = g["text"].str.contains(r"[À-ỹ]", regex=True, na=False)
    policies.append(("with_accent", g[accent_mask]))

    long_mask = g["text"].str.len() >= 8
    policies.append(("long_lines", g[long_mask]))

    for pname, pg in policies:
        if pg.empty:
            continue

        text = join_lines(pg["text"].tolist())
        if not text:
            continue

        n, avg_s, med_s, avg_a = summarize_lines(pg)

        rows.append(candidate_row(
            image_id=image_id,
            candidate_name=f"line::{variant}::{pname}",
            candidate_type="line_policy",
            source_variant=variant,
            ocr_text=text,
            num_lines=n,
            avg_line_score=avg_s,
            median_line_score=med_s,
            avg_area_ratio=avg_a,
        ))

    return rows


def merged_policy_candidates(image_id, lines_df, variants):
    rows = []

    groups = [
        ("raw+bottom_50_resize_960", ["raw", "bottom_50_resize_960"]),
        ("raw+bottom_60_resize_960", ["raw", "bottom_60_resize_960"]),
        ("raw+center_70_resize_960", ["raw", "center_70_resize_960"]),
        ("bottom_50_resize_960+center_70_resize_960", ["bottom_50_resize_960", "center_70_resize_960"]),
        ("bottom_60_resize_960+center_70_resize_960", ["bottom_60_resize_960", "center_70_resize_960"]),
        ("raw+bottom_50_resize_960+center_70_resize_960", ["raw", "bottom_50_resize_960", "center_70_resize_960"]),
        ("raw+bottom_60_resize_960+center_70_resize_960", ["raw", "bottom_60_resize_960", "center_70_resize_960"]),
    ]

    for gname, vars_ in groups:
        use_vars = [v for v in vars_ if v in variants]
        if not use_vars:
            continue

        g = lines_df[lines_df["variant"].isin(use_vars)].copy()
        if g.empty:
            continue

        g["text"] = g["text"].fillna("").astype(str).map(norm_space)
        g = g[g["text"].str.len() > 0].copy()

        if g.empty:
            continue

        for col in ["score", "area_ratio"]:
            if col not in g.columns:
                g[col] = np.nan
            g[col] = pd.to_numeric(g[col], errors="coerce")

        policies = [("all_clean", g)]

        for th_name, th in [
            ("score_ge_08", 0.80),
            ("score_ge_085", 0.85),
            ("score_ge_09", 0.90),
        ]:
            policies.append((th_name, g[g["score"].fillna(0) >= th]))

        if g["area_ratio"].notna().any():
            for k in [3, 5, 8]:
                policies.append((f"top{k}_area", g.sort_values("area_ratio", ascending=False).head(k)))

        for pname, pg in policies:
            if pg.empty:
                continue

            text = join_lines(pg["text"].tolist())
            if not text:
                continue

            n, avg_s, med_s, avg_a = summarize_lines(pg)

            rows.append(candidate_row(
                image_id=image_id,
                candidate_name=f"merge::{gname}::{pname}",
                candidate_type="merged_line_policy",
                source_variant=gname,
                ocr_text=text,
                num_lines=n,
                avg_line_score=avg_s,
                median_line_score=med_s,
                avg_area_ratio=avg_a,
            ))

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, help="Path to ocr_ablation_report.csv.")
    parser.add_argument("--lines", required=True, help="Path to ocr_ablation_lines.csv.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    args = parser.parse_args()

    report_path = Path(args.report)
    lines_path = Path(args.lines)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = pd.read_csv(report_path)
    lines = pd.read_csv(lines_path)

    for col in ["image_id", "variant", "ocr_text"]:
        if col not in report.columns:
            report[col] = ""

    for col in ["image_id", "variant"]:
        if col not in lines.columns:
            lines[col] = ""

    lines = ensure_line_text_column(lines)

    rows = []

    total_images = report["image_id"].nunique()
    print("[INFO] report rows:", len(report), flush=True)
    print("[INFO] line rows:", len(lines), flush=True)
    print("[INFO] images:", total_images, flush=True)

    # Existing variant candidates.
    for _, r in report.iterrows():
        image_id = r["image_id"]
        variant = r["variant"]
        text = r.get("ocr_text", "")

        num_lines = r.get("num_lines", np.nan)
        avg_score = r.get("avg_score", np.nan)

        rows.append(candidate_row(
            image_id=image_id,
            candidate_name=f"variant::{variant}",
            candidate_type="existing_variant",
            source_variant=variant,
            ocr_text=text,
            num_lines=num_lines,
            avg_line_score=avg_score,
            median_line_score=np.nan,
            avg_area_ratio=np.nan,
        ))

    # Single-variant line policies.
    for idx, ((image_id, variant), g) in enumerate(lines.groupby(["image_id", "variant"], sort=False), start=1):
        rows.extend(line_policy_candidates(image_id, variant, g))

        if idx % 1000 == 0:
            print(
                "[INFO] processed image-variant groups:",
                idx,
                "candidate rows:",
                len(rows),
                flush=True,
            )

    # Merged line policies per image.
    variants_by_image = report.groupby("image_id")["variant"].apply(lambda s: set(s.astype(str))).to_dict()

    for idx, (image_id, g) in enumerate(lines.groupby("image_id", sort=False), start=1):
        rows.extend(merged_policy_candidates(image_id, g, variants_by_image.get(image_id, set())))

        if idx % 100 == 0:
            print(
                "[INFO] processed images for merge:",
                idx,
                "candidate rows:",
                len(rows),
                flush=True,
            )

    out = pd.DataFrame(rows)

    out = out.drop_duplicates(["image_id", "candidate_name", "ocr_text"]).reset_index(drop=True)

    out_path = out_dir / "line_candidate_rows.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    summary = {
        "rows": int(len(out)),
        "images": int(out["image_id"].nunique()) if len(out) else 0,
        "candidate_names": int(out["candidate_name"].nunique()) if len(out) else 0,
    }

    with (out_dir / "line_candidate_summary_private.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[INFO] saved:", out_path, flush=True)
    print("[INFO] summary:", summary, flush=True)


if __name__ == "__main__":
    main()
