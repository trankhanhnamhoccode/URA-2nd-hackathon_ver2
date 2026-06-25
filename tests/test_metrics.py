import pandas as pd

from ura_ocr.eval.metrics import (
    cer,
    classify_field_error,
    evaluate_submission_dataframe,
    token_f1,
)


def test_token_f1_exact_and_partial():
    assert token_f1("LineaBon K2 + D3", "LineaBon K2 + D3") == 1.0
    assert token_f1("", "") == 1.0
    assert token_f1("", "Dove") == 0.0

    score = token_f1("Dove Smoothie", "Dove Smoothie tẩy da chết")
    assert 0.0 < score < 1.0


def test_cer_basic():
    assert cer("abc", "abc") == 0.0
    assert cer("", "") == 0.0
    assert cer("abc", "") == 1.0
    assert cer("abx", "abc") > 0.0


def test_classify_field_error():
    assert classify_field_error("", "") == "true_blank"
    assert classify_field_error("", "Dove") == "miss"
    assert classify_field_error("Dove", "") == "false_fill"
    assert classify_field_error("Dove", "Dove") == "correct"
    assert classify_field_error("Dove", "Dove Smoothie") == "partial_too_short"
    assert classify_field_error("Dove Smoothie tẩy da chết", "Dove Smoothie") == "partial_too_long"


def test_evaluate_submission_dataframe():
    pred = pd.DataFrame(
        {
            "image_id": ["a", "b", "c"],
            "ocr_text": ["hello", "", "abc"],
            "product_name": ["Dove", "", "Wrong"],
        }
    )

    gt = pd.DataFrame(
        {
            "image_id": ["a", "b", "c"],
            "ocr_text": ["hello", "", "abcd"],
            "product_name": ["Dove", "", "Right"],
        }
    )

    report, error_cases = evaluate_submission_dataframe(pred, gt)

    assert report["rows_overlap"] == 3
    assert report["product"]["exact_match_no_accent"] < 1.0
    assert report["product"]["token_f1_macro"] < 1.0
    assert report["ocr"]["cer_macro"] >= 0.0
    assert len(error_cases) >= 1