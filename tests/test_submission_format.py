import pandas as pd

from ura_ocr.io.submission import (
    make_empty_submission,
    validate_submission,
    PHASE2_COLUMNS,
)


def test_make_empty_phase2_submission():
    input_df = pd.DataFrame(
        {
            "image_id": ["a.jpg", "b.jpg"],
        }
    )

    sub = make_empty_submission(input_df, phase="phase2")

    assert list(sub.columns) == PHASE2_COLUMNS
    assert len(sub) == 2
    assert sub["image_id"].tolist() == ["a.jpg", "b.jpg"]

    report = validate_submission(sub, input_df, phase="phase2")
    assert report["ok"] is True