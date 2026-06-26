from __future__ import annotations

import re
from dataclasses import dataclass

from ura_ocr.product.rulebase_clean_v3 import clean_text, norm_key, split_plus_raw


TOKEN_RE = re.compile(r"[A-Za-zÀ-ỹ0-9][A-Za-zÀ-ỹ0-9+\-/]*")

PRICE_RE = re.compile(
    r"(?i)(\b\d{1,3}([.,]\d{3})+\b|\b\d+\s*(k|đ|d|vnd|vnđ)\b)"
)

PHONE_RE = re.compile(r"\b0\d{8,10}\b")

URL_RE = re.compile(
    r"(?i)\b(www\.|http|\.com|facebook|tiktok|shopee|lazada|zalo|youtube|instagram)\b"
)

PROMO_RE = re.compile(
    r"(?i)\b("
    r"sale|deal|voucher|freeship|free\s*ship|ship|review|viral|tiktok|"
    r"giảm|giam|giá|gia|mua|bán|ban|tặng|tang|quà|qua|"
    r"chính\s*hãng|chinh\s*hang|mẫu\s*mới|mau\s*moi|hàng\s*mới|hang\s*moi|"
    r"hot|new|follow|like|share|livestream|video|clip|"
    r"giao\s*nhanh|miễn\s*phí|mien\s*phi|trả\s*góp|tra\s*gop|"
    r"cam\s*kết|camket|độc\s*quyền|doc\s*quyen"
    r")\b"
)

CAPTION_RE = re.compile(
    r"(?i)\b("
    r"tôi|toi|mình|minh|bạn|ban|nó|no|này|nay|kia|đó|do|"
    r"không|khong|ko|k|chưa|chua|rồi|roi|sao|vậy|zay|vay|"
    r"một|mot|buổi|buoi|chiều|chieu|sáng|sang|tối|toi|"
    r"hôm\s*nay|hom\s*nay|là|la|khi|lúc|luc|cần|can|thấy|thay|"
    r"ăn|an|uống|uong|ngon|quá|qua|thích|thich|"
    r"bắt\s*lực|bat\s*luc|tuyệt\s*tác|tuyet\s*tac|hoàng\s*hôn|hoang\s*hon"
    r")\b"
)

BAD_CONTEXT_RE = re.compile(
    r"(?i)\b("
    r"shop|store|official|địa\s*chỉ|dia\s*chi|liên\s*hệ|lien\s*he|"
    r"inbox|comment|ib|sdt|hotline|nguồn|source|ảnh|photo|"
    r"cửa\s*hàng|cua\s*hang|thông\s*minh|thong\s*minh"
    r")\b"
)

# Cues có khả năng là product category/name.
PRODUCT_CUE_RE = re.compile(
    r"(?i)\b("
    # food/drink
    r"bánh|banh|trà|tra|sữa|sua|milk|tea|coffee|cafe|cà\s*phê|ca\s*phe|"
    r"matcha|roll|combo|kebab|phô\s*mai|pho\s*mai|trứng|trung|"
    r"mực|muc|khô|kho|gà|ga|bò|bo|heo|pate|pat[eê]|"
    r"xúc\s*xích|xuc\s*xich|chả|cha|nem|"
    # beauty/care
    r"scrub|serum|cleanser|cleansing|mousse|mascara|sunscreen|toner|"
    r"cream|gel|wash|mask|spf|body\s*wash|shampoo|arbutin|vitamin\s*c|"
    r"tẩy\s*da\s*chết|tay\s*da\s*chet|sữa\s*rửa\s*mặt|sua\s*rua\s*mat|"
    r"kem\s*chống\s*nắng|kem\s*chong\s*nang|dầu\s*gội|dau\s*goi|sữa\s*tắm|sua\s*tam|"
    # baby/device/common product
    r"k2|d3|dha|canxi|colos|baby|kid|gold|grow|profutura|lactoferrin|"
    r"máy\s*đo|may\s*do|đường\s*huyết|duong\s*huyet|túi\s*đựng|tui\s*dung"
    r")\b"
)

GENERIC_BAD_NORMS = {
    "",
    "san pham",
    "thuc pham",
    "my pham",
    "do an",
    "mon ngon",
    "breaking news",
    "tin hot",
    "top",
    "best",
    "hang moi",
    "mau moi",
    "chinh hang",
    "gia re",
    "gia soc",
}

FILLER_NORMS = {
    "review", "tiktok", "viral", "sale", "deal", "voucher", "freeship",
    "free", "ship", "chinh", "hang", "mau", "moi", "hot", "new",
    "mua", "ban", "gia", "giam", "tang", "qua", "video", "clip",
    "shop", "store", "official", "tot", "nhat", "cua", "cho", "voi",
    "khong", "khongg", "ko", "roi", "sao", "vay", "zay",
    "toi", "minh", "ban", "mot", "buoi", "chieu", "hom", "nay",
    "ho", "tro", "tang", "cuong", "de", "khang",
}


@dataclass(frozen=True)
class ConservativePhrasePrediction:
    product_name: str
    product_score: float
    product_reason: str


def _tokens(s: str) -> list[str]:
    return TOKEN_RE.findall(clean_text(s))


def remove_brand_text(s: str, brand_name: str) -> str:
    s = clean_text(s)
    for b in split_plus_raw(brand_name or ""):
        b = clean_text(b)
        if not b:
            continue
        s = re.sub(r"(?i)\b" + re.escape(b) + r"\b", " ", s)
    return clean_text(s)


def split_segments(text: str) -> list[str]:
    text = clean_text(str(text or ""))
    if not text:
        return []

    text = text.replace("||", "\n")
    raw = re.split(r"[\n\r]+|[•✓·]+", text)

    out = []
    for p in raw:
        p = clean_text(p)
        if len(p) >= 3:
            out.append(p)

    # Nếu OCR chỉ là 1 dòng dài, chia nhẹ theo dấu câu.
    more = []
    for p in out:
        chunks = re.split(r"[-–—,:;(){}\[\]]+", p)
        for c in chunks:
            c = clean_text(c)
            if len(c) >= 3:
                more.append(c)

    return more or out


def clean_candidate(s: str) -> str:
    s = clean_text(str(s or ""))
    if not s:
        return ""

    m = URL_RE.search(s)
    if m:
        s = s[:m.start()]

    # Product thường nằm trước giá.
    m = PRICE_RE.search(s)
    if m and m.start() >= 3:
        s = s[:m.start()]

    s = PHONE_RE.sub(" ", s)
    s = re.sub(r"[@#]\S+", " ", s)
    s = re.sub(r"[|:;,_]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" -–—.,:;()[]{}")

    # Xóa promo, nhưng không đụng cue product.
    s = PROMO_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip(" -–—.,:;()[]{}")

    return s


def _vowelish_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0

    good = 0
    for t in tokens:
        nt = norm_key(t)
        if re.search(r"[aeiouyăâêôơư]", nt):
            good += 1
        elif re.search(r"\d", nt):
            good += 0.5

    return good / max(len(tokens), 1)


def is_ocr_soup(c: str) -> bool:
    toks = _tokens(c)
    ntoks = [norm_key(t) for t in toks if norm_key(t)]

    if not ntoks:
        return True

    one_char = sum(len(t) <= 1 for t in ntoks)
    two_char = sum(len(t) <= 2 for t in ntoks)

    if len(ntoks) >= 4 and one_char >= 2:
        return True

    if len(ntoks) >= 6 and two_char / len(ntoks) >= 0.55:
        return True

    if _vowelish_ratio(ntoks) < 0.55:
        return True

    weird = re.findall(r"[^A-Za-zÀ-ỹ0-9\s+\-/]", c)
    if len(weird) >= 3:
        return True

    return False


def is_bad_candidate(c: str) -> bool:
    c = clean_candidate(c)
    if not c:
        return True

    nc = norm_key(c)
    toks = nc.split()

    if nc in GENERIC_BAD_NORMS:
        return True

    if len(c) < 4 or len(c) > 70:
        return True

    if len(toks) < 2 or len(toks) > 9:
        return True

    if not re.search(r"[A-Za-zÀ-ỹ]", c):
        return True

    if PHONE_RE.search(c) or URL_RE.search(c):
        return True

    if is_ocr_soup(c):
        return True

    filler = sum(t in FILLER_NORMS for t in toks)
    if filler / max(len(toks), 1) >= 0.35:
        return True

    # Nếu giống caption/câu nói hơn product thì reject.
    caption_hits = len(CAPTION_RE.findall(c))
    if caption_hits >= 2:
        return True

    if CAPTION_RE.search(c) and not re.search(r"(?i)\b(serum|mascara|scrub|sunscreen|cleanser|mousse|body|spf|k2|d3)\b", c):
        return True

    if BAD_CONTEXT_RE.search(c) and not PRODUCT_CUE_RE.search(c):
        return True

    digit_only = sum(bool(re.fullmatch(r"\d+", t)) for t in toks)
    if digit_only >= max(2, len(toks) // 2):
        return True

    return False


def score_candidate(c: str, segment_rank: int, source: str) -> float:
    c = clean_candidate(c)
    if is_bad_candidate(c):
        return -999.0

    nc = norm_key(c)
    toks = nc.split()

    score = 0.0

    if 2 <= len(toks) <= 5:
        score += 1.6
    elif 6 <= len(toks) <= 8:
        score += 0.8
    else:
        score -= 1.0

    if PRODUCT_CUE_RE.search(c):
        score += 2.4

    # Ưu tiên text đầu.
    score += max(0.0, 0.7 - 0.07 * segment_rank)

    if source == "known_pattern":
        score += 1.3
    elif source == "cue_window":
        score += 0.8
    elif source == "clean_chunk":
        score += 0.4

    if PROMO_RE.search(c):
        score -= 1.0

    caption_hits = len(CAPTION_RE.findall(c))
    score -= caption_hits * 0.8

    filler = sum(t in FILLER_NORMS for t in toks)
    score -= filler * 0.5

    # English beauty/product phrases thường khá ổn.
    if re.search(r"(?i)\b(serum|mascara|scrub|sunscreen|cleanser|mousse|body|spf|cream|wash|mask)\b", c):
        score += 0.8

    # Food phrases chỉ nhận khi gọn, không giống câu kể.
    if re.search(r"(?i)\b(bánh|banh|trà|tra|matcha|roll|combo|kebab|pate)\b", c):
        if not CAPTION_RE.search(c):
            score += 0.6

    return round(score, 4)


def _add_candidate(candidates, seen, c: str, source: str, rank: int):
    c = clean_candidate(c)
    if is_bad_candidate(c):
        return

    key = norm_key(c)
    if key in seen:
        return

    seen.add(key)
    sc = score_candidate(c, rank, source)
    if sc > -100:
        candidates.append((c, source, rank, sc))


def generate_candidates_from_ocr_text(ocr_text: str, brand_name: str = "") -> list[tuple[str, str, int, float]]:
    segments = split_segments(ocr_text)
    candidates = []
    seen = set()

    for rank, seg in enumerate(segments):
        seg = remove_brand_text(seg, brand_name)
        seg = clean_candidate(seg)

        if not seg:
            continue

        # Không xét segment nếu hoàn toàn không có cue.
        if not PRODUCT_CUE_RE.search(seg):
            continue

        # Pattern rõ, ưu tiên hơn.
        known_patterns = [
            r"(?i)\bbody\s*scrub\b",
            r"(?i)\b[a-zA-ZÀ-ỹ0-9+\-/]{2,}\s+serum\b(?:\s+[a-zA-ZÀ-ỹ0-9+\-/]{1,}){0,4}",
            r"(?i)\bserum\b(?:\s+[a-zA-ZÀ-ỹ0-9+\-/]{1,}){1,5}",
            r"(?i)\b[a-zA-ZÀ-ỹ0-9+\-/]{2,}\s+mascara\b(?:\s+[a-zA-ZÀ-ỹ0-9+\-/]{1,}){0,4}",
            r"(?i)\bcleansing\s+mousse\b",
            r"(?i)\bfacial\s+cleansing\s+mousse\b",
            r"(?i)\bsữa\s+rửa\s+mặt\b(?:\s+[A-Za-zÀ-ỹ0-9+\-/]{1,}){0,4}",
            r"(?i)\bsua\s+rua\s+mat\b(?:\s+[A-Za-zÀ-ỹ0-9+\-/]{1,}){0,4}",
            r"(?i)\bmatcha\s+roll\b",
            r"(?i)\btrà\s+sữa\b(?:\s+[A-Za-zÀ-ỹ0-9+\-/]{1,}){0,3}",
            r"(?i)\btra\s+sua\b(?:\s+[A-Za-zÀ-ỹ0-9+\-/]{1,}){0,3}",
            r"(?i)\bbánh\b(?:\s+[A-Za-zÀ-ỹ0-9+\-/]{1,}){1,6}",
            r"(?i)\bbanh\b(?:\s+[A-Za-zÀ-ỹ0-9+\-/]{1,}){1,6}",
        ]

        for pat in known_patterns:
            for m in re.finditer(pat, seg):
                _add_candidate(candidates, seen, m.group(0), "known_pattern", rank)

        toks = _tokens(seg)
        if not toks:
            continue

        # Windows quanh cue, nhưng ngắn và bảo thủ.
        for i, tok in enumerate(toks):
            if not PRODUCT_CUE_RE.search(tok):
                continue

            for start in range(max(0, i - 2), i + 1):
                for end in range(i + 2, min(len(toks), start + 7) + 1):
                    window = " ".join(toks[start:end])
                    _add_candidate(candidates, seen, window, "cue_window", rank)

        # Clean chunk ngắn nếu cả chunk rất giống tên sản phẩm.
        if 2 <= len(toks) <= 7:
            _add_candidate(candidates, seen, seg, "clean_chunk", rank)

    candidates.sort(key=lambda x: x[3], reverse=True)
    return candidates


def predict_product_phrase_conservative(
    ocr_text: str,
    brand_name: str = "",
    threshold: float = 4.7,
) -> ConservativePhrasePrediction:
    candidates = generate_candidates_from_ocr_text(ocr_text, brand_name=brand_name)

    if not candidates:
        return ConservativePhrasePrediction("", 0.0, "no_candidate")

    best, source, rank, score = candidates[0]

    if score < threshold:
        return ConservativePhrasePrediction("", score, f"below_threshold:{source}:rank={rank}")

    return ConservativePhrasePrediction(best, score, f"v4a_conservative:{source}:rank={rank}")
