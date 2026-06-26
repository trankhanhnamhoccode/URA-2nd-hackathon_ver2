from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from ura_ocr.product.rulebase_clean_v3 import clean_text, norm_key, split_plus_raw, plus_join


PRODUCT_CUE_RE = re.compile(
    r"(?i)\b("
    r"serum|scrub|mascara|sunscreen|cleanser|cleansing|mousse|cream|wash|mask|spf|"
    r"body\s*scrub|sữa\s*rửa\s*mặt|sua\s*rua\s*mat|tẩy\s*da\s*chết|tay\s*da\s*chet|"
    r"k2|d3|dha|colos|baby|kid|gold|grow|profutura|lactoferrin|"
    r"trà\s*sữa|tra\s*sua|bánh|banh|pate|kebab|matcha|roll|"
    r"sữa|sua|milk|tã|ta|bỉm|bim"
    r")\b"
)

BAD_BRAND_NORMS = {
    "",
    "review", "tiktok", "sale", "deal", "voucher", "freeship",
    "official", "shop", "store", "chinh hang", "hang moi", "mau moi",
    "san pham", "thuc pham", "my pham", "do an", "mon ngon",
    "serum", "scrub", "mascara", "sunscreen", "cleanser", "cream", "wash",
    "body scrub", "tra sua", "banh", "pate", "combo", "gold", "baby", "kid",
    "k2", "d3", "dha",
}

# Brand list tổng quát/known từ public domain + những brand đã thấy qua train/rule.
# Không lấy từ manual private label CSV.
BUILTIN_BRANDS: Dict[str, List[str]] = {
    "Dove": ["dove"],
    "Klairs": ["klairs", "dear klairs"],
    "Perspirex": ["perspirex"],
    "Maybelline": ["maybelline", "maybeline", "maybellne"],
    "LineaBon": ["lineabon", "linea bon"],
    "Moony": ["moony"],
    "Huggies": ["huggies"],
    "VitaDairy": ["vitadairy", "vita dairy"],
    "Aptamil": ["aptamil"],
    "L'Oréal": ["loreal", "l oreal", "l'oreal", "loreal paris"],
    "Garnier": ["garnier"],
    "Fresh Lady": ["fresh lady", "fresht lady", "freshlady"],
    "Ha Long Canfoco": ["ha long canfoco", "halong canfoco", "canfoco"],
    "Nescafé": ["nescafe", "nescafé"],
    "Abbott": ["abbott"],
    "Nutifood": ["nutifood", "nuti food"],
    "Vinamilk": ["vinamilk"],
    "TH true MILK": ["th true milk", "th true", "true milk"],
    "Dielac": ["dielac"],
    "Ensure": ["ensure"],
    "Glucerna": ["glucerna"],
    "ColosBaby": ["colosbaby", "colos baby"],
    "Rohto": ["rohto"],
    "Sunplay": ["sunplay"],
    "Cocoon": ["cocoon"],
    "La Roche-Posay": ["la roche posay", "laroche posay", "la roche-posay"],
    "CeraVe": ["cerave", "cera ve"],
    "Bioderma": ["bioderma"],
    "Eucerin": ["eucerin"],
    "Anessa": ["anessa"],
    "Skin1004": ["skin1004", "skin 1004"],
    "Senka": ["senka"],
    "Hada Labo": ["hada labo", "hadalabo"],
    "Simple": ["simple"],
    "Cetaphil": ["cetaphil"],
    "Olay": ["olay"],
    "Vaseline": ["vaseline"],
    "Nivea": ["nivea"],
    "Pond's": ["ponds", "pond's"],
    "Innisfree": ["innisfree"],
    "The Ordinary": ["the ordinary"],
    "Some By Mi": ["some by mi", "somebymi"],
    "Lifebuoy": ["lifebuoy"],
    "Sunsilk": ["sunsilk"],
    "Clear": ["clear"],
    "Head & Shoulders": ["head shoulders", "head and shoulders", "head&shoulders"],
    "P/S": ["ps", "p/s"],
    "Closeup": ["closeup", "close up"],
    "Colgate": ["colgate"],
    "Sensodyne": ["sensodyne"],
}


@dataclass(frozen=True)
class BrandPrediction:
    brand_name: str
    brand_score: float
    brand_reason: str
    matched_alias: str
    variant_hits: int
    debug_top: str


def _brand_valid(name: str) -> bool:
    name = clean_text(name)
    if not name:
        return False

    n = norm_key(name)

    if n in BAD_BRAND_NORMS:
        return False

    if len(n) < 3:
        return False

    if len(n) > 35:
        return False

    if not re.search(r"[a-zA-ZÀ-ỹ]", name):
        return False

    return True


def _add_alias(alias_map: Dict[str, List[str]], canonical: str, alias: str):
    canonical = clean_text(canonical)
    alias = clean_text(alias)

    if not _brand_valid(canonical):
        return

    if not alias:
        return

    na = norm_key(alias)
    if na in BAD_BRAND_NORMS:
        return

    # Alias quá ngắn dễ bắn rác. Không nhận alias <=2.
    if len(na.replace(" ", "")) <= 2:
        return

    alias_map.setdefault(canonical, [])
    if alias not in alias_map[canonical]:
        alias_map[canonical].append(alias)


def build_brand_alias_map(train_df=None) -> Dict[str, List[str]]:
    alias_map: Dict[str, List[str]] = {}

    for canonical, aliases in BUILTIN_BRANDS.items():
        _add_alias(alias_map, canonical, canonical)
        for a in aliases:
            _add_alias(alias_map, canonical, a)

    # Nếu train_labels có cột brand_name thì dùng thêm brand public train.
    if train_df is not None and "brand_name" in getattr(train_df, "columns", []):
        for raw in train_df["brand_name"].astype(str).fillna(""):
            for b in split_plus_raw(raw):
                b = clean_text(b)
                if _brand_valid(b):
                    _add_alias(alias_map, b, b)

    return alias_map


def _norm_contains(text: str, alias: str) -> bool:
    nt = norm_key(text)
    na = norm_key(alias)

    if not nt or not na:
        return False

    # Word-boundary trên text đã normalize.
    pat = r"(?<![a-z0-9])" + re.escape(na) + r"(?![a-z0-9])"
    if re.search(pat, nt):
        return True

    # Compact match cho brand có dấu cách/gạch.
    ca = re.sub(r"[^a-z0-9]+", "", na)
    ct = re.sub(r"[^a-z0-9]+", "", nt)

    if len(ca) >= 6 and ca in ct:
        return True

    return False


def _position_bonus(text: str, alias: str) -> float:
    nt = norm_key(text)
    na = norm_key(alias)

    pos = nt.find(na)
    if pos < 0:
        ca = re.sub(r"[^a-z0-9]+", "", na)
        ct = re.sub(r"[^a-z0-9]+", "", nt)
        pos = ct.find(ca) if len(ca) >= 6 else -1

    if pos < 0:
        return 0.0

    if pos <= 40:
        return 1.0
    if pos <= 100:
        return 0.5
    return 0.0


def _has_product_context(text: str) -> bool:
    return bool(PRODUCT_CUE_RE.search(text or ""))


def _score_alias(
    canonical: str,
    alias: str,
    ocr_text: str,
    variant_texts: List[str],
) -> Tuple[float, int, str]:
    score = 0.0
    reasons = []

    alias_norm = norm_key(alias)
    alias_compact_len = len(re.sub(r"[^a-z0-9]+", "", alias_norm))

    # Alias ngắn phải cực kỳ rõ.
    short_alias = alias_compact_len <= 3

    base_hit = _norm_contains(ocr_text, alias)
    variant_hits = sum(1 for t in variant_texts if _norm_contains(t, alias))
    all_text = " || ".join([ocr_text] + variant_texts)

    if base_hit:
        score += 4.0
        reasons.append("base_hit")
        score += _position_bonus(ocr_text, alias)

    if variant_hits:
        score += min(variant_hits, 4) * 0.75
        reasons.append(f"variant_hits={variant_hits}")

    # Brand gần ngữ cảnh product thì đáng tin hơn.
    if (base_hit or variant_hits >= 2) and _has_product_context(all_text):
        score += 0.6
        reasons.append("product_context")

    # Canonical từ builtin/public brand.
    score += 0.4

    # Alias dài đáng tin hơn alias ngắn.
    if alias_compact_len >= 6:
        score += 0.3

    if short_alias:
        # Ví dụ CP, PS rất dễ rác. Bắt buộc base hit + product context + repeated.
        if not (base_hit and variant_hits >= 1 and _has_product_context(all_text)):
            return -999.0, variant_hits, "short_alias_reject"
        score -= 1.0

    # Nếu chỉ hit đúng 1 OCR variant mà không có base hit thì khá rủi ro.
    if (not base_hit) and variant_hits < 2:
        score -= 1.2
        reasons.append("weak_variant_only")

    return round(score, 4), variant_hits, "+".join(reasons) if reasons else "no_hit"


def predict_brand_name(
    ocr_text: str,
    variant_texts: List[str],
    alias_map: Dict[str, List[str]],
    threshold: float = 5.6,
    margin: float = 0.5,
) -> BrandPrediction:
    ocr_text = clean_text(ocr_text)
    variant_texts = [clean_text(t) for t in variant_texts if clean_text(t)]

    scored = []

    for canonical, aliases in alias_map.items():
        best_score = -999.0
        best_alias = ""
        best_hits = 0
        best_reason = ""

        for alias in aliases:
            score, hits, reason = _score_alias(
                canonical=canonical,
                alias=alias,
                ocr_text=ocr_text,
                variant_texts=variant_texts,
            )

            if score > best_score:
                best_score = score
                best_alias = alias
                best_hits = hits
                best_reason = reason

        if best_score > -100:
            scored.append((canonical, best_score, best_alias, best_hits, best_reason))

    scored.sort(key=lambda x: x[1], reverse=True)

    debug_top = " | ".join(
        f"{c}:{s}:{a}:{r}" for c, s, a, h, r in scored[:5]
    )

    if not scored:
        return BrandPrediction("", 0.0, "no_candidate", "", 0, "")

    top = scored[0]
    canonical, score, alias, hits, reason = top

    if score < threshold:
        return BrandPrediction("", score, f"below_threshold:{reason}", alias, hits, debug_top)

    if len(scored) >= 2:
        second = scored[1]
        if score - second[1] < margin:
            return BrandPrediction("", score, f"low_margin_vs:{second[0]}:{reason}", alias, hits, debug_top)

    return BrandPrediction(canonical, score, f"v5_brand_expander:{reason}", alias, hits, debug_top)
