from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from ura_ocr.product.rulebase_clean_v3 import clean_text, norm_key


def compact_norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", norm_key(s))


def evidence_has(evidence: str, alias: str) -> bool:
    ne = norm_key(clean_text(evidence))
    na = norm_key(clean_text(alias))

    if not ne or not na:
        return False

    pat = r"(?<![a-z0-9])" + re.escape(na) + r"(?![a-z0-9])"
    if re.search(pat, ne):
        return True

    ce = compact_norm(evidence)
    ca = compact_norm(alias)

    if len(ca) >= 6 and ca in ce:
        return True

    return False


BRAND_RULES = [
    ("Fresh Lady", ["fresh lady", "fresht lady", "freshlady"]),
    ("APPIR SKIN", ["appir skin", "appirskin"]),
    ("Round Lab", ["round lab", "roundlab"]),
    ("PhePhaFood", ["phephafood", "phe pha food", "phe pha"]),
    ("Trà Sữa SHARE", ["trà sữa share", "tra sua share", "share trà sữa", "share tra sua"]),
    ("Mật Tea & Coffee", ["mật tea coffee", "mat tea coffee", "mật tea & coffee", "mat tea & coffee"]),
    ("Capdir Skin", ["capdir skin", "capdirskin"]),
    ("Lucas+", ["lucas+", "lucas plus", "lucas"]),
    ("Rosehot", ["rosehot", "rose hot"]),
    ("Care Skin", ["care skin", "careskin"]),
    ("WHISIS", ["whisis"]),
    ("THE WHOO", ["the whoo", "whoo"]),
    ("OHUI", ["ohui", "o hui"]),
    ("Centellian24", ["centellian24", "centellian 24"]),
    ("Huxley", ["huxley"]),
    ("Finegrow", ["finegrow", "fine grow"]),
    ("NUTA", ["nuta"]),
    ("RASO", ["raso"]),
    ("Edifier", ["edifier"]),
    ("Yes Mom", ["yes mom", "yesmom"]),
    ("Bảo Vy", ["bảo vy", "bao vy"]),

    # EXTRA CURATED RULES 0443-0413
    ("Molfix Việt Nam", ["molfix việt nam", "molfix viet nam", "molfix"]),
    ("d’Alba", ["d’alba", "d'alba", "dalba"]),
    ("Top Gia", ["top gia"]),
    ("Panasonic", ["panasonic"]),
    ("Organika", ["organika"]),
    ("Cỏ mềm", ["cỏ mềm", "co mem"]),
    ("MeTiS", ["metis"]),
    ("ORINGER", ["oringer"]),
    ("Lộc Lạc", ["lộc lạc", "loc lac"]),
    ("KINGBABY", ["kingbaby", "king baby"]),
    ("Black Bag Chia", ["black bag chia", "blackbag chia"]),
    ("Comfort", ["comfort"]),
    ("DABACHA", ["dabacha"]),
    ("CORTIS", ["cortis"]),
    ("Torriden", ["torriden"]),
    ("Rainbow Bee", ["rainbow bee", "rainbowbee"]),
]


PRODUCT_RULES = [
    ("Facial Cleansing Mousse", ["facial cleansing mousse", "cleansing mousse"]),
    ("Day Shield Perfect Sun Black", ["day shield perfect sun black", "perfect sun black"]),
    ("Sun Cream Lucas", ["sun cream lucas", "lucas sun cream"]),
    ("Rose Oil", ["rose oil"]),
    ("Nepro 1 Gold", ["nepro 1 gold", "nepro gold"]),
    ("Toddler Nutritional Drink", ["toddler nutritional drink"]),
    ("Ultimate Plant Protein", ["ultimate plant protein", "plant protein"]),
    ("ThermaPlex Warming Gel", ["thermaplex warming gel", "warming gel"]),
    ("Turmeric Cream", ["turmeric cream"]),
    ("Paradoxe", ["paradoxe"]),
    ("QS30 Soundbar", ["qs30 soundbar", "qs 30 soundbar"]),
    ("Solaire SPF50", ["solaire spf50", "solaire 50"]),
    ("Hyper Curl Waterproof Mascara", ["hyper curl waterproof mascara", "hyper curl waterproof"]),
    ("Sensibio H2O", ["sensibio h2o"]),
    ("Gong Jin Hyang Clarifying Cleansing Balm", ["gong jin hyang clarifying cleansing balm", "clarifying cleansing balm"]),
    ("Hydrating Cleanser", ["hydrating cleanser"]),
    ("Bio-Placental Scalp Serum", ["bio placental scalp serum", "bio-placental scalp serum", "scalp serum"]),
    ("Gel Moussant", ["gel moussant"]),
    ("MelaB3 Serum", ["melab3 serum", "mela b3 serum"]),
    ("Gentle Skin Cleanser", ["gentle skin cleanser"]),
    ("MaxFresh", ["maxfresh", "max fresh"]),
    ("BoneMax", ["bonemax", "bone max"]),
    ("PediaSure", ["pediasure", "pedia sure"]),
    ("Nem chua Huế", ["nem chua huế", "nem chua hue"]),
    ("Mắm Ruốc Huế", ["mắm ruốc huế", "mam ruoc hue"]),
    ("Tiramisu", ["tiramisu"]),
    ("Mì Trộn Cô Xúc Xích", ["mì trộn cô xúc xích", "mi tron co xuc xich", "mì trộn xúc xích", "mi tron xuc xich"]),
    ("Bánh mì kebab", ["bánh mì kebab", "banh mi kebab", "kebab"]),
    ("Mì dao cắt Sơn Tây", ["mì dao cắt sơn tây", "mi dao cat son tay"]),
    ("Bún nước tương", ["bún nước tương", "bun nuoc tuong"]),
    ("Cá viên thập cẩm sốt HQ", ["cá viên thập cẩm sốt hq", "ca vien thap cam sot hq"]),

    # EXTRA CURATED RULES 0443-0413
    ("Băng keo chống thấm siêu dính", [
        "băng keo chống thấm siêu dính",
        "bang keo chong tham sieu dinh",
        "waterpoof tinh paste",
        "waterproof tinh paste",
    ]),
    ("Yến mạch cán mỏng", [
        "yến mạch cán mỏng",
        "yen mach can mong",
    ]),
    ("Vital Spray Serum", [
        "vital spray serum",
    ]),
    ("Quả bóng nước", [
        "quả bóng nước",
        "qua bong nuoc",
    ]),
    ("Tủ lạnh", [
        "tủ lạnh",
        "tu lanh",
        "công nghệ blue ag",
        "cong nghe blue ag",
    ]),
    ("Premium Liga-Joint", [
        "premium liga-joint",
        "premium liga joint",
        "liga-joint",
        "liga joint",
    ]),
    ("Chelated Calcium Magnesium Zinc With Vitamin D3", [
        "chelated calcium magnesium zinc with vitamin d3",
        "calcium magnesium zinc with vitamin d3",
        "calcium chelate magnesium zinc",
    ]),
    ("Cacao latte trân châu dẻo", [
        "cacao latte trân châu dẻo",
        "cacao latte tran chau deo",
    ]),
    ("Bông Đất Trắng", [
        "bông đất trắng",
        "bong dat trang",
    ]),
    ("Ca D3 K2 + Arginine", [
        "ca d3 k2 arginine",
        "ca d3 k2 + arginine",
        "d3 k2 arginine",
    ]),
    ("Nepro", [
        "nepro",
    ]),
    ("Protein ADE Ca+ Bổ sung B12", [
        "protein ade ca",
        "protein ade ca+",
        "bổ sung b12",
        "bo sung b12",
    ]),
    ("Dầu tràm", [
        "dầu tràm",
        "dau tram",
    ]),
    ("Kem đánh răng KINGBABY", [
        "kem đánh răng kingbaby",
        "kem danh rang kingbaby",
        "kingbaby freshmint",
    ]),
    ("Whole Dark Chia Seeds", [
        "whole dark chia seeds",
        "dark chia seeds",
    ]),
    ("Dịu nhẹ", [
        "comfort dịu nhẹ",
        "comfort diu nhe",
    ]),
    ("Xúc xích phô mai mozzarella", [
        "xúc xích phô mai mozzarella",
        "xuc xich pho mai mozzarella",
    ]),
    ("Profutura 3", [
        "aptamil profutura 3",
        "profutura 3",
    ]),
    ("3X DHA+", [
        "3x dha",
        "3x dha+",
    ]),
    ("Son dưỡng môi Blue Lips", [
        "son dưỡng môi blue lips",
        "son duong moi blue lips",
        "blue lips",
    ]),
    ("Mật ong hoa cà phê", [
        "mật ong hoa cà phê",
        "mat ong hoa ca phe",
        "mật ong cà phê lâm đồng",
        "mat ong ca phe lam dong",
    ]),
]


@dataclass(frozen=True)
class CuratedRulePrediction:
    brand_name: str = ""
    product_name: str = ""
    brand_reason: str = ""
    product_reason: str = ""


def predict_curated_rules(evidence: str, existing_brand: str = "", existing_product: str = "") -> CuratedRulePrediction:
    evidence = clean_text(evidence)

    brand = ""
    product = ""
    brand_reason = ""
    product_reason = ""

    if not str(existing_brand or "").strip():
        for canonical, aliases in BRAND_RULES:
            if any(evidence_has(evidence, a) for a in aliases):
                brand = canonical
                brand_reason = "v5_3_curated_brand_rule"
                break

    if not str(existing_product or "").strip():
        for canonical, aliases in PRODUCT_RULES:
            if any(evidence_has(evidence, a) for a in aliases):
                product = canonical
                product_reason = "v5_3_curated_product_rule"
                break

    return CuratedRulePrediction(
        brand_name=brand,
        product_name=product,
        brand_reason=brand_reason,
        product_reason=product_reason,
    )


def apply_curated_rules(sub: pd.DataFrame, evidence: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "evidence_text" not in evidence.columns:
        text_cols = [
            c for c in evidence.columns
            if c != "image_id" and (
                "ocr" in c.lower()
                or "text" in c.lower()
                or "variant" in c.lower()
                or "selected" in c.lower()
            )
        ]
        evidence = evidence.copy()
        evidence["evidence_text"] = evidence[text_cols].astype(str).agg(" || ".join, axis=1)
    else:
        evidence = evidence.copy()
        evidence["evidence_text"] = evidence["evidence_text"].astype(str)

    work = sub.merge(evidence[["image_id", "evidence_text"]], on="image_id", how="left")
    out = sub.copy()
    rows = []

    for i, row in work.iterrows():
        old_brand = str(row.get("brand_name", "")).strip()
        old_product = str(row.get("product_name", "")).strip()

        ev = clean_text(str(row.get("ocr_text", "")) + " || " + str(row.get("evidence_text", "")))
        pred = predict_curated_rules(ev, existing_brand=old_brand, existing_product=old_product)

        final_brand = old_brand
        final_product = old_product

        if pred.brand_name:
            final_brand = pred.brand_name
            out.at[i, "brand_name"] = final_brand

        if pred.product_name:
            final_product = pred.product_name
            out.at[i, "product_name"] = final_product

        rows.append({
            "image_id": row.get("image_id", ""),
            "ocr_text": row.get("ocr_text", ""),
            "brand_before": old_brand,
            "product_before": old_product,
            "brand_rule": pred.brand_name,
            "product_rule": pred.product_name,
            "brand_final": final_brand,
            "product_final": final_product,
            "brand_reason": pred.brand_reason,
            "product_reason": pred.product_reason,
            "evidence_text": str(row.get("evidence_text", ""))[:800],
        })

    return out, pd.DataFrame(rows)
