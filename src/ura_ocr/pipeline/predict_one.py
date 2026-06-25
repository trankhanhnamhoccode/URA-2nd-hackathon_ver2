from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from ura_ocr.schema import PredictionResult


def predict_one_image(
    image_id: str,
    image_path: Optional[str | Path],
    cfg: Dict,
) -> PredictionResult:
    """
    Temporary stub.

    Later this function will call:
    preprocess -> OCR -> correction -> brand/product predictor.
    """
    debug = {
        "image_path": str(image_path) if image_path is not None else "",
        "note": "stub prediction; OCR/product modules not enabled yet",
    }

    return PredictionResult(
        image_id=image_id,
        ocr_text="",
        brand_name="",
        product_name="",
        score=0.0,
        route="stub",
        debug=debug,
    )