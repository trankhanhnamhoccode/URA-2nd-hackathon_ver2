from ura_ocr.ocr.cleaner import clean_ocr_text, normalize_line, clean_ocr_lines


def test_cleaner_preserves_important_product_symbols():
    text = "LineaBon   K2 + D3\nSPF50+\nP/S\n24h+"
    cleaned = clean_ocr_text(text)

    assert "K2 + D3" in cleaned
    assert "SPF50+" in cleaned
    assert "P/S" in cleaned
    assert "24h+" in cleaned


def test_cleaner_removes_emoji_and_cjk_noise():
    text = "Dove Smoothie 😭🔥 美白 こんにちは"
    cleaned = clean_ocr_text(text)

    assert "Dove" in cleaned
    assert "Smoothie" in cleaned
    assert "😭" not in cleaned
    assert "🔥" not in cleaned
    assert "美白" not in cleaned
    assert "こんにちは" not in cleaned


def test_cleaner_keeps_vietnamese_accents():
    text = "Pate Cột Đèn Hải Phòng"
    cleaned = clean_ocr_text(text)

    assert cleaned == "Pate Cột Đèn Hải Phòng"


def test_cleaner_deduplicates_lines():
    lines = [
        "LineaBon K2 + D3",
        "LineaBon   K2 + D3",
        "Dove Smoothie",
    ]

    cleaned_lines = clean_ocr_lines(lines)

    assert cleaned_lines == [
        "LineaBon K2 + D3",
        "Dove Smoothie",
    ]


def test_normalize_line_does_not_join_tokens_badly():
    line = normalize_line("Kem!!! đánh??? răng")
    assert line == "Kem đánh răng"

def test_plus_suffix_and_binary_plus_are_handled_differently():
    assert normalize_line("SPF50+") == "SPF50+"
    assert normalize_line("SPF50 +") == "SPF50+"
    assert normalize_line("24h+") == "24h+"
    assert normalize_line("24h +") == "24h+"
    assert normalize_line("K2+D3") == "K2 + D3"
    assert normalize_line("K2 + D3") == "K2 + D3"