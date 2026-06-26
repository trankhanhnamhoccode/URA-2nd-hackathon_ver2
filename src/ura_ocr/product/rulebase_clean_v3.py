
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

import pandas as pd


def clean_text(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return " ".join(str(x).split()).strip()


def strip_accents(s: str) -> str:
    s = clean_text(s)
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", s)


def norm_key(s: str) -> str:
    s = strip_accents(s).lower()
    s = re.sub(r"[^a-z0-9\s\+\-&\.]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def compact(s: str) -> str:
    return re.sub(r"\s+", "", norm_key(s))


def split_plus_raw(s: str) -> list[str]:
    s = clean_text(s)
    if not s:
        return []
    parts = re.split(r"\s*\+\s*|;|,", s)
    return [clean_text(p) for p in parts if clean_text(p)]


def plus_join(items: Iterable[str]) -> str:
    out = []
    seen = set()

    for x in items:
        x = clean_text(x)
        if not x:
            continue

        k = norm_key(x)
        if not k or k in seen:
            continue

        seen.add(k)
        out.append(x)

    return " + ".join(out)


def valid_label(s: str) -> bool:
    s = clean_text(s)
    if not s:
        return False
    if s.lower() in {"nan", "none", "null"}:
        return False
    return len(norm_key(s)) >= 2


def canonical_brand(x: str) -> str:
    nk = norm_key(x)

    mapping = {
        "loreal": "L'Oréal Paris",
        "l oreal": "L'Oréal Paris",
        "l oreal paris": "L'Oréal Paris",
        "bio essence": "Bioessence",
        "bioessence": "Bioessence",
        "d alba": "d'Alba",
        "dalba": "d'Alba",
        "cappir": "Cappir Skin",
        "cappir skin": "Cappir Skin",
        "ha long canfoco": "Hạ Long Canfoco",
        "halong canfoco": "Hạ Long Canfoco",
        "cot den": "Cột Đèn",
        "linea bon": "LineaBon",
        "lineabon": "LineaBon",
        "vita dairy": "VitaDairy",
        "vitadairy": "VitaDairy",
        "x men": "X-Men",
        "xmen": "X-Men",
        "mat tea coffee": "Mat Tea&Coffee",
    }

    return mapping.get(nk, clean_text(x))


BRAND_ALIASES = [
    "VitaDairy", "LineaBon", "Huggies", "Moony", "Dove", "Maybelline",
    "Aptamil", "Nutricia", "Abbott", "Ensure", "Glucerna", "Meiji", "HiPP",
    "L'Oréal Paris", "Loreal", "Olay", "Vaseline", "Klairs", "Caryophy",
    "d'Alba", "D Alba", "Bioessence", "Bio Essence", "Cappir Skin", "Cappir",
    "Round Lab", "Skin1004", "COSRX", "Tree Hut", "Beplain", "Perspirex",
    "Comfort", "Sinocare", "Joyroom", "Browit", "Biolizin", "Vitabiotics",
    "Pregnacare", "SHARE", "Mat Tea&Coffee", "LyLy Bakery",
    "Hạ Long Canfoco", "Ha Long Canfoco", "Cột Đèn", "Cot Den",
    "An Phát", "Botanika", "X-Men",
]

BAD_BRAND_NORMS = {
    "new", "hot", "sale", "review", "video", "shop", "store", "official",
    "chinh hang", "combo", "free", "ship", "freeship",
}

JUNK_PRODUCT_NORMS = {
    "sua", "sua me", "sua bot", "sua uong", "sua cong thuc",
    "san pham do hop", "lo sua cong thuc",
    "thdt", "d en", "dien", "d ien", "cp", "ns", "60",
    "antem", "anthem",
    "kem", "serum", "mask", "mascara", "cleanser", "pate", "tra sua",
    "sua rua mat", "sua tam", "dau goi", "tay da chet", "kem chong nang",
}

PRODUCT_RULES: list[tuple[str, list[str], str]] = [
    ("K2 + D3", [
        r"\bk2\s*\+?\s*d3\b",
        r"\bd3\s*k2\b",
    ], "k2_d3"),

    ("ColosBaby", [
        r"\bcolos\s*baby\b",
        r"\bcolosbaby\b",
        r"\bcolosbab\w*\b",
    ], "colosbaby"),

    ("Lactoferrin", [
        r"\blactoferrin\b",
        r"\blactofer\w*\b",
    ], "lactoferrin"),

    ("Super Gold Kid", [
        r"\bsuper\s+gold\s+kid\b",
    ], "super_gold_kid"),

    ("Profutura", [
        r"\bprofutura\b",
    ], "profutura"),

    ("Grow Gold", [
        r"\bgrow\s+gold\b",
    ], "grow_gold"),

    ("GrowPlus+", [
        r"\bgrow\s*plus\+?\b",
    ], "growplus"),

    ("Ensure Gold", [
        r"\bensure\s+gold\b",
    ], "ensure_gold"),

    ("Glucerna", [
        r"\bglucerna\b",
    ], "glucerna"),

    ("The Hyper Curl Mascara", [
        r"\bthe\s+hyper\s+curl\s+mascara\b",
        r"\bhyper\s+curl\s+mascara\b",
    ], "hyper_curl_mascara"),

    ("Facial Cleansing Mousse", [
        r"\bfacial\s+cleansing\s+mousse\b",
        r"\bcleansing\s+mousse\b",
    ], "facial_cleansing_mousse"),

    ("Jelly Facial Wash", [
        r"\bjelly\s+facial\s+wash\b",
    ], "jelly_facial_wash"),

    ("Body scrub", [
        r"\bbody\s+scrub\b",
        r"\bgommage\s+corps\b",
    ], "body_scrub"),

    ("Smoothie tẩy da chết", [
        r"\bsmoothie\b.*\btay\s+da\s+chet\b",
        r"\btay\s+da\s+chet\b.*\bsmoothie\b",
    ], "smoothie_tay_da_chet"),

    ("Mask Madecassoside", [
        r"\bmask\b.*\bmadecassoside\b",
        r"\bmadecassoside\b.*\bmask\b",
    ], "mask_madecassoside"),

    ("Tone Up & Correcting Sunscreen", [
        r"\btone\s*up\b.*\bcorrecting\b.*\bsunscreen\b",
    ], "toneup_correcting_sunscreen"),

    ("Serum Hyaluronic Acid", [
        r"\bserum\b.*\bhyaluronic\s+acid\b",
        r"\bhyaluronic\s+acid\b.*\bserum\b",
    ], "serum_hyaluronic_acid"),

    ("Revitalift", [
        r"\brevitalift\b",
    ], "revitalift"),

    ("Pate Cột Đèn", [
        r"\bpate\b.*\bcot\s*den\b",
        r"\bcot\s*den\b.*\bpate\b",
        r"\bpat[eê]\b.*\bcot\s*den\b",
    ], "pate_cot_den"),

    ("Đồ hộp Hạ Long", [
        r"\bdo\s+hop\s+ha\s+long\b",
        r"\bha\s+long\s+canfoco\b",
        r"\bhalong\s+canfoco\b",
    ], "do_hop_ha_long"),

    ("Bánh mì kebab", [
        r"\bbanh\s+mi\s+kebab\b",
        r"\bkebab\b",
    ], "banh_mi_kebab"),

    ("Trà sữa", [
        r"\btra\s+sua\b",
        r"\bmilk\s*tea\b",
    ], "tra_sua"),

    ("Máy đo đường huyết", [
        r"\bmay\s+do\b.*\bduong\s+huyet\b",
        r"\bsafe\s*aq\b",
        r"\bsinocare\b.*\bduong\s+huyet\b",
    ], "may_do_duong_huyet"),

    ("Túi đựng rác", [
        r"\btui\s+dung\s+rac\b",
        r"\bgarbage\s+bags\b",
    ], "tui_dung_rac"),
]

PRODUCT_SUPPRESS = {
    "Đồ hộp Hạ Long": {"Pate Cột Đèn"},
}


@dataclass(frozen=True)
class RulebasePrediction:
    brand_name: str
    product_name: str
    brand_reason: str
    product_reason: str


class CleanRulebaseV3:
    def __init__(self, train_labels: pd.DataFrame | None = None):
        self.train_labels = train_labels if train_labels is not None else pd.DataFrame()
        self.brand_gaz = self._build_brand_gazetteer(self.train_labels)
        self.product_gaz = self._build_product_gazetteer(self.train_labels)

    def _build_brand_gazetteer(self, train: pd.DataFrame) -> pd.DataFrame:
        vals = []

        if "brand_name" in train.columns:
            for x in train["brand_name"].fillna("").astype(str):
                for p in split_plus_raw(x):
                    if valid_label(p):
                        vals.append(p)

        vals += BRAND_ALIASES

        gaz = pd.DataFrame({"brand": vals})
        gaz["brand"] = gaz["brand"].map(canonical_brand)
        gaz["norm"] = gaz["brand"].map(norm_key)
        gaz["compact"] = gaz["brand"].map(compact)
        gaz["char_len"] = gaz["norm"].str.len()

        gaz = gaz[
            (gaz["norm"].str.len() >= 2)
            & (~gaz["norm"].isin(BAD_BRAND_NORMS))
        ].copy()

        gaz = (
            gaz.sort_values("char_len", ascending=False)
            .drop_duplicates("norm")
            .reset_index(drop=True)
        )

        return gaz

    def _is_dirty_product(self, product: str) -> bool:
        p = clean_text(product)
        pn = norm_key(p)

        if not p or not pn:
            return True

        if pn in JUNK_PRODUCT_NORMS:
            return True

        if re.fullmatch(r"\d+", pn):
            return True

        if len(pn) <= 3 and pn not in {"k2 d3"}:
            return True

        toks = pn.split()

        if len(toks) == 1 and len(pn) < 5:
            return True

        single_char_tokens = [t for t in toks if len(t) == 1]
        if len(single_char_tokens) >= 2:
            return True

        return False

    def _build_product_gazetteer(self, train: pd.DataFrame) -> pd.DataFrame:
        vals = []

        if "product_name" in train.columns:
            for x in train["product_name"].fillna("").astype(str):
                for p in split_plus_raw(x):
                    if valid_label(p) and not self._is_dirty_product(p):
                        vals.append(p)

        if not vals:
            return pd.DataFrame(columns=["product", "norm", "compact", "char_len", "freq"])

        gaz = pd.DataFrame({"product": vals})
        gaz["product"] = gaz["product"].map(clean_text)
        gaz["norm"] = gaz["product"].map(norm_key)
        gaz["compact"] = gaz["product"].map(compact)
        gaz["char_len"] = gaz["norm"].str.len()

        gaz = (
            gaz.value_counts(["product", "norm", "compact", "char_len"])
            .reset_index(name="freq")
            .sort_values(["freq", "char_len"], ascending=False)
            .drop_duplicates("norm")
            .reset_index(drop=True)
        )

        return gaz

    def detect_brands(self, evidence: str) -> tuple[str, str]:
        nt = norm_key(evidence)
        ct = compact(evidence)

        hits = []

        for _, r in self.brand_gaz.iterrows():
            b = clean_text(r["brand"])
            bn = clean_text(r["norm"])
            bc = clean_text(r["compact"])

            if not bn:
                continue

            pos = -1
            m = re.search(r"\b" + re.escape(bn) + r"\b", nt)

            if m:
                pos = m.start()
            elif len(bc) >= 5 and bc in ct:
                pos = ct.find(bc)

            if pos >= 0:
                hits.append((pos, len(bn), b))

        if not hits:
            return "", "no_brand"

        hits = sorted(hits, key=lambda x: (x[0], -x[1]))

        return plus_join([h[2] for h in hits]), "brand_evidence"

    def detect_products_rules(self, evidence: str) -> tuple[str, str]:
        nt = norm_key(evidence)

        hits = []

        for product, patterns, reason in PRODUCT_RULES:
            for pattern in patterns:
                m = re.search(pattern, nt)
                if m:
                    hits.append({
                        "pos": m.start(),
                        "product": product,
                        "reason": reason,
                    })
                    break

        if not hits:
            return "", "no_rule_product_evidence"

        hits = sorted(hits, key=lambda x: x["pos"])

        products = []
        reasons = []
        seen = set()

        for h in hits:
            p = clean_text(h["product"])
            k = norm_key(p)

            if k in seen:
                continue

            seen.add(k)
            products.append(p)
            reasons.append(h["reason"])

        product_set = set(products)
        final_products = []
        final_reasons = []

        for p, r in zip(products, reasons):
            suppressors = PRODUCT_SUPPRESS.get(p, set())
            if suppressors and any(s in product_set for s in suppressors):
                continue

            final_products.append(p)
            final_reasons.append(r)

        return plus_join(final_products), "+".join(final_reasons)

    def detect_products_train_gaz(self, evidence: str) -> tuple[str, str]:
        nt = norm_key(evidence)
        ct = compact(evidence)

        candidates = []

        for _, r in self.product_gaz.iterrows():
            p = clean_text(r["product"])
            pn = clean_text(r["norm"])
            pc = clean_text(r["compact"])

            if not pn or not pc:
                continue

            exact_hit = re.search(r"\b" + re.escape(pn) + r"\b", nt) is not None
            compact_hit = len(pc) >= 6 and pc in ct

            if not exact_hit and not compact_hit:
                continue

            score = 0.0
            if exact_hit:
                score += 2.0
            if compact_hit:
                score += 1.7

            score += min(float(r.get("freq", 1)), 20.0) / 200.0
            score += min(int(r.get("char_len", 1)), 40) / 100.0

            candidates.append({
                "product": p,
                "score": score,
                "reason": "exact_phrase" if exact_hit else "compact_phrase",
            })

        if not candidates:
            return "", "no_train_product_exact_evidence"

        cand = pd.DataFrame(candidates).sort_values("score", ascending=False)

        selected = []
        reasons = []

        for _, row in cand.iterrows():
            p = clean_text(row["product"])
            pn = norm_key(p)

            already = [norm_key(x) for x in selected]

            skip = False
            for a in already:
                if pn == a:
                    skip = True
                    break
                if pn in a and len(pn) <= len(a):
                    skip = True
                    break

            if skip:
                continue

            selected.append(p)
            reasons.append("train_gaz_strict_" + str(row["reason"]))

            if len(selected) >= 3:
                break

        return plus_join(selected), "+".join(reasons)

    def predict(self, evidence: str) -> RulebasePrediction:
        evidence = clean_text(evidence)

        brand, brand_reason = self.detect_brands(evidence)

        product_rule, product_reason_rule = self.detect_products_rules(evidence)
        product_gaz, product_reason_gaz = self.detect_products_train_gaz(evidence)

        product_parts = []
        product_reasons = []

        if product_rule:
            product_parts += split_plus_raw(product_rule)
            product_reasons.append(product_reason_rule)

        if product_gaz:
            product_parts += split_plus_raw(product_gaz)
            product_reasons.append(product_reason_gaz)

        product = plus_join(product_parts)
        product_reason = "+".join(product_reasons) if product else "no_product_evidence"

        return RulebasePrediction(
            brand_name=brand,
            product_name=product,
            brand_reason=brand_reason,
            product_reason=product_reason,
        )
