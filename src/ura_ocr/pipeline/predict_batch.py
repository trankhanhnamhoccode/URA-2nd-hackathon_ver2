from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from ura_ocr.io.image_loader import resolve_image_path
from ura_ocr.io.submission import get_submission_columns
from ura_ocr.pipeline.predict_one import predict_one_image


def predict_batch(input_df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    phase = cfg.get("phase", "phase2")
    columns = get_submission_columns(phase)

    images_dir = cfg["input"]["images_dir"]
    runtime_cfg = cfg.get("runtime", {})

    start_index = int(runtime_cfg.get("start_index") or 0)
    limit = runtime_cfg.get("limit", None)

    df = input_df.copy()

    if limit is not None:
        end_index = start_index + int(limit)
        df = df.iloc[start_index:end_index].copy()
    else:
        df = df.iloc[start_index:].copy()

    rows: List[dict] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
        image_id = str(row["image_id"])
        image_path = resolve_image_path(images_dir, image_id)

        pred = predict_one_image(
            image_id=image_id,
            image_path=image_path,
            cfg=cfg,
        )

        rows.append(pred.to_submission_row(phase=phase))

    out = pd.DataFrame(rows, columns=columns)
    return out