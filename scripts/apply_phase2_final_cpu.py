from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ura_ocr.product.rulebase_clean_v3 import CleanRulebaseV3
from ura_ocr.product.open_phrase_layer_v2_conservative import predict_product_phrase_conservative
from ura_ocr.product.brand_expander_v1 import build_brand_alias_map, predict_brand_name
from ura_ocr.product.curated_rules_v1 import apply_curated_rules


REQUIRED_COLS = ["image_id", "ocr_text", "brand_name", "product_name"]


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False).fillna("")


def ensure_submission_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "image_id" not in out.columns:
        raise ValueError("Missing image_id")

    if "ocr_text" not in out.columns:
        out["ocr_text"] = ""

    if "brand_name" not in out.columns:
        out["brand_name"] = ""

    if "product_name" not in out.columns:
        out["product_name"] = ""

    return out[REQUIRED_COLS].copy()


def normalize_submission(sub: pd.DataFrame) -> pd.DataFrame:
    out = sub[REQUIRED_COLS].copy()

    for c in REQUIRED_COLS:
        out[c] = out[c].astype(str).replace({"nan": "", "None": ""})
        out[c] = out[c].apply(lambda x: " " if str(x).strip() == "" else str(x).strip())

    return out


def build_evidence_text(evi: pd.DataFrame) -> pd.DataFrame:
    evi = evi.copy()

    if "evidence_text" not in evi.columns:
        text_cols = [
            c for c in evi.columns
            if c != "image_id"
            and (
                "ocr" in c.lower()
                or "text" in c.lower()
                or "variant" in c.lower()
                or "selected" in c.lower()
                or "evidence" in c.lower()
            )
        ]

        if not text_cols:
            text_cols = [c for c in evi.columns if c != "image_id"]

        evi["evidence_text"] = evi[text_cols].astype(str).agg(" || ".join, axis=1)
    else:
        evi["evidence_text"] = evi["evidence_text"].astype(str)

    return evi


def get_variant_cols(evi: pd.DataFrame) -> list[str]:
    cols = []

    for c in evi.columns:
        if c in {"image_id", "evidence_text"}:
            continue

        lc = c.lower()
        if (
            "ocr" in lc
            or "text" in lc
            or "variant" in lc
            or "selected" in lc
            or "evidence" in lc
        ):
            cols.append(c)

    if not cols:
        cols = [c for c in evi.columns if c not in {"image_id", "evidence_text"}]

    return cols


def get_evidence_row(evi_idx, image_id: str):
    if image_id not in evi_idx.index:
        return None

    row = evi_idx.loc[image_id]

    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    return row


def run_v3_rulebase(sub_base: pd.DataFrame, evi: pd.DataFrame, train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rb = CleanRulebaseV3(train)

    sub = ensure_submission_cols(sub_base)
    evi_idx = evi.set_index("image_id", drop=False)

    out = sub.copy()
    rows = []

    for i, row in out.iterrows():
        image_id = str(row.get("image_id", "")).strip()
        ocr_text = str(row.get("ocr_text", "")).strip()

        old_brand = str(row.get("brand_name", "")).strip()
        old_product = str(row.get("product_name", "")).strip()

        evi_row = get_evidence_row(evi_idx, image_id)
        evidence_text = ""

        if evi_row is not None:
            evidence_text = str(evi_row.get("evidence_text", "")).strip()

        if not evidence_text:
            evidence_text = ocr_text

        pred = rb.predict(evidence_text)

        brand_layer = str(pred.brand_name or "").strip()
        product_layer = str(pred.product_name or "").strip()

        brand_final = old_brand
        product_final = old_product

        if not old_brand and brand_layer:
            brand_final = brand_layer
            out.at[i, "brand_name"] = brand_final

        if not old_product and product_layer:
            product_final = product_layer
            out.at[i, "product_name"] = product_final

        rows.append({
            "image_id": image_id,
            "ocr_text": ocr_text,
            "brand_before": old_brand,
            "product_before": old_product,
            "brand_layer": brand_layer,
            "product_layer": product_layer,
            "brand_final": brand_final,
            "product_final": product_final,
            "brand_reason": getattr(pred, "brand_reason", ""),
            "product_reason": getattr(pred, "product_reason", ""),
            "evidence_text": evidence_text[:1000],
        })

    return out, pd.DataFrame(rows)


def run_v4a_product(sub: pd.DataFrame, product_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = sub.copy()
    rows = []

    for i, row in out.iterrows():
        image_id = str(row.get("image_id", "")).strip()
        ocr_text = str(row.get("ocr_text", "")).strip()
        brand_name = str(row.get("brand_name", "")).strip()
        old_product = str(row.get("product_name", "")).strip()

        final_product = old_product
        layer_product = ""
        layer_score = 0.0
        layer_reason = "skip_existing_product" if old_product else ""

        if not old_product and ocr_text:
            pred = predict_product_phrase_conservative(
                ocr_text=ocr_text,
                brand_name=brand_name,
                threshold=product_threshold,
            )

            layer_product = str(pred.product_name or "").strip()
            layer_score = float(pred.product_score)
            layer_reason = pred.product_reason

            if layer_product:
                final_product = layer_product
                out.at[i, "product_name"] = final_product

        rows.append({
            "image_id": image_id,
            "ocr_text": ocr_text,
            "brand_name": brand_name,
            "product_name_before": old_product,
            "product_name_layer": layer_product,
            "product_name_final": final_product,
            "product_score": layer_score,
            "product_reason": layer_reason,
        })

    return out, pd.DataFrame(rows)


def run_v5_brand(
    sub: pd.DataFrame,
    evi: pd.DataFrame,
    train: pd.DataFrame,
    brand_threshold: float,
    brand_margin: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = sub.copy()

    alias_map = build_brand_alias_map(train)
    variant_cols = get_variant_cols(evi)
    evi_idx = evi.set_index("image_id", drop=False)

    rows = []

    for i, row in out.iterrows():
        image_id = str(row.get("image_id", "")).strip()
        ocr_text = str(row.get("ocr_text", "")).strip()
        old_brand = str(row.get("brand_name", "")).strip()
        product_name = str(row.get("product_name", "")).strip()

        final_brand = old_brand
        layer_brand = ""
        score = 0.0
        reason = "skip_existing_brand" if old_brand else ""
        alias = ""
        hits = 0
        debug_top = ""

        if not old_brand:
            variant_texts = []
            evi_row = get_evidence_row(evi_idx, image_id)

            if evi_row is not None:
                for c in variant_cols:
                    val = str(evi_row.get(c, "")).strip()
                    if val:
                        variant_texts.append(val)

                ev_text = str(evi_row.get("evidence_text", "")).strip()
                if ev_text:
                    variant_texts.append(ev_text)

            pred = predict_brand_name(
                ocr_text=ocr_text,
                variant_texts=variant_texts,
                alias_map=alias_map,
                threshold=brand_threshold,
                margin=brand_margin,
            )

            layer_brand = str(pred.brand_name or "").strip()
            score = float(pred.brand_score)
            reason = pred.brand_reason
            alias = pred.matched_alias
            hits = int(pred.variant_hits)
            debug_top = pred.debug_top

            if layer_brand:
                final_brand = layer_brand
                out.at[i, "brand_name"] = final_brand

        rows.append({
            "image_id": image_id,
            "ocr_text": ocr_text,
            "brand_name_before": old_brand,
            "brand_name_layer": layer_brand,
            "brand_name_final": final_brand,
            "product_name": product_name,
            "brand_score": score,
            "brand_reason": reason,
            "matched_alias": alias,
            "variant_hits": hits,
            "debug_top": debug_top,
        })

    return out, pd.DataFrame(rows)


def run(args):
    sub_base = read_csv(args.sub_base_csv)
    evi = build_evidence_text(read_csv(args.evidence_csv))
    train = read_csv(args.train_labels_csv)

    stage_v3, audit_v3 = run_v3_rulebase(sub_base, evi, train)

    stage_v4a, audit_v4a = run_v4a_product(
        stage_v3,
        product_threshold=args.product_threshold,
    )

    stage_v5, audit_v5 = run_v5_brand(
        stage_v4a,
        evi=evi,
        train=train,
        brand_threshold=args.brand_threshold,
        brand_margin=args.brand_margin,
    )

    final_sub, audit_curated = apply_curated_rules(stage_v5, evi)

    final_sub = normalize_submission(final_sub)

    out_path = Path(args.out_csv)
    audit_dir = Path(args.audit_dir)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    final_sub.to_csv(out_path, index=False, encoding="utf-8-sig")
    audit_v3.to_csv(audit_dir / "audit_v3_rulebase.csv", index=False, encoding="utf-8-sig")
    audit_v4a.to_csv(audit_dir / "audit_v4a_product.csv", index=False, encoding="utf-8-sig")
    audit_v5.to_csv(audit_dir / "audit_v5_brand.csv", index=False, encoding="utf-8-sig")
    audit_curated.to_csv(audit_dir / "audit_v5_curated_rules.csv", index=False, encoding="utf-8-sig")

    print("Saved submission:", out_path)
    print("Saved audits:", audit_dir)
    print("Rows:", len(final_sub))
    print("Blank OCR:", final_sub["ocr_text"].astype(str).str.strip().eq("").sum())
    print("Blank brand:", final_sub["brand_name"].astype(str).str.strip().eq("").sum())
    print("Blank product:", final_sub["product_name"].astype(str).str.strip().eq("").sum())
    print("Brand fill:", final_sub["brand_name"].astype(str).str.strip().ne("").sum())
    print("Product fill:", final_sub["product_name"].astype(str).str.strip().ne("").sum())


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sub-base-csv", required=True)
    parser.add_argument("--evidence-csv", required=True)
    parser.add_argument("--train-labels-csv", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--audit-dir", required=True)

    parser.add_argument("--product-threshold", type=float, default=6.3)
    parser.add_argument("--brand-threshold", type=float, default=5.0)
    parser.add_argument("--brand-margin", type=float, default=0.5)

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
