
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

from ura_ocr.product.rulebase_clean_v3 import CleanRulebaseV3, clean_text


def build_evidence_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "evidence_text" in out.columns:
        out["evidence_text"] = out["evidence_text"].fillna("").astype(str)
        return out

    text_cols = [
        c for c in out.columns
        if c in {"final_ocr_text", "submission_ocr_text", "ocr_text"}
        or c.startswith("ocr_")
        or c.endswith("ocr_text")
    ]

    if not text_cols:
        raise ValueError(
            "evidence CSV must contain evidence_text or at least one OCR text column"
        )

    def unique_join(row):
        parts = []
        seen = set()

        for c in text_cols:
            s = clean_text(row.get(c, ""))
            if not s:
                continue

            k = s.lower()
            if k in seen:
                continue

            seen.add(k)
            parts.append(s)

        return " || ".join(parts)

    out["evidence_text"] = out.apply(unique_join, axis=1)
    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--evidence-csv", required=True)
    parser.add_argument("--sub-base-csv", required=True)
    parser.add_argument("--train-labels-csv", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--audit-csv", required=True)
    parser.add_argument("--phase", default="phase2", choices=["phase1", "phase2"])
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    evidence = pd.read_csv(
        args.evidence_csv,
        dtype=str,
        keep_default_na=False,
    ).fillna("")

    evidence = build_evidence_if_missing(evidence)

    sub_base = pd.read_csv(
        args.sub_base_csv,
        dtype=str,
        keep_default_na=False,
    ).fillna("")

    train = pd.read_csv(
        args.train_labels_csv,
        dtype=str,
        keep_default_na=False,
    ).fillna("")

    if args.limit is not None:
        evidence = evidence.head(args.limit).copy()
        keep_ids = set(evidence["image_id"].astype(str))
        sub_base = sub_base[sub_base["image_id"].astype(str).isin(keep_ids)].copy()

    rb = CleanRulebaseV3(train)

    pred_rows = []

    for _, row in evidence.iterrows():
        image_id = clean_text(row.get("image_id", ""))
        evidence_text = clean_text(row.get("evidence_text", ""))
        final_ocr = clean_text(row.get("final_ocr_text", ""))

        p = rb.predict(evidence_text)

        pred_rows.append({
            "image_id": image_id,
            "brand_name_pred": p.brand_name,
            "product_name_pred": p.product_name,
            "brand_reason": p.brand_reason,
            "product_reason": p.product_reason,
            "final_ocr_text": final_ocr,
            "evidence_text": evidence_text,
        })

    pred = pd.DataFrame(pred_rows)

    if not {"image_id", "ocr_text"}.issubset(sub_base.columns):
        raise ValueError(
            f"sub base must contain image_id and ocr_text; got {sub_base.columns.tolist()}"
        )

    out = sub_base[["image_id", "ocr_text"]].copy()

    out = out.merge(
        pred[["image_id", "brand_name_pred", "product_name_pred"]],
        on="image_id",
        how="left",
    )

    out["brand_name"] = out["brand_name_pred"].fillna("").astype(str)
    out["product_name"] = out["product_name_pred"].fillna("").astype(str)

    if args.phase == "phase1":
        out = out[["image_id", "ocr_text", "product_name"]].copy()
    else:
        out = out[["image_id", "ocr_text", "brand_name", "product_name"]].copy()

    for col in out.columns:
        out[col] = out[col].fillna(" ").astype(str)
        out.loc[out[col].str.strip().eq(""), col] = " "

    out_path = Path(args.out_csv)
    audit_path = Path(args.audit_csv)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    pred.to_csv(audit_path, index=False, encoding="utf-8-sig")

    print("Saved submission:", out_path)
    print("Saved audit:", audit_path)
    print("rows:", len(out))
    print("blank OCR:", int(out["ocr_text"].astype(str).str.strip().eq("").sum()))

    if args.phase == "phase2":
        print("brand fill:", int(out["brand_name"].astype(str).str.strip().ne("").sum()))

    print("product fill:", int(out["product_name"].astype(str).str.strip().ne("").sum()))


if __name__ == "__main__":
    main()
