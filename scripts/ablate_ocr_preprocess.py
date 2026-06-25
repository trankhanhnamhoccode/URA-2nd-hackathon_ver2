from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from ura_ocr.eval.metrics import cer
from ura_ocr.io.csv_io import read_csv_keep_empty
from ura_ocr.io.image_loader import resolve_image_path
from ura_ocr.ocr.cleaner import clean_ocr_text
from ura_ocr.preprocess.transforms import get_image_quality, make_preprocess_variants


DEFAULT_VARIANTS = [
    "raw",
    "resize_960",
    "resize_1280",
    "clahe",
    "resize_960_clahe",
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Path to train_labels.csv containing image_id and ocr_text.",
    )

    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Directory containing train images.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for ablation report.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of images to run. Use small number first.",
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start row index in labels CSV.",
    )

    parser.add_argument(
        "--variants",
        type=str,
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated preprocess variants to run.",
    )

    parser.add_argument(
        "--lang",
        type=str,
        default="vi",
        help="PaddleOCR language. Try vi first; if it fails, try en or latin.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        choices=["gpu", "cpu"],
        help="Device hint for PaddleOCR.",
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Write partial report every N images.",
    )

    parser.add_argument(
        "--save-variant-images",
        action="store_true",
        help="Save preprocessed images for debugging. This creates many files.",
    )

    return parser.parse_args()


def create_paddle_ocr(lang: str = "vi", device: str = "gpu"):
    """
    Create PaddleOCR object with compatibility fallbacks.

    PaddleOCR APIs have changed across versions, so we try a few safe configs.
    """
    from paddleocr import PaddleOCR

    configs = [
        {
            "lang": lang,
            "device": device,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        },
        {
            "lang": lang,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        },
        {
            "lang": lang,
            "use_angle_cls": False,
            "show_log": False,
            "use_gpu": device == "gpu",
        },
        {
            "lang": lang,
            "use_angle_cls": False,
            "use_gpu": device == "gpu",
        },
        {
            "lang": lang,
        },
    ]

    last_error = None

    for cfg in configs:
        try:
            print(f"[INFO] Trying PaddleOCR config: {cfg}")
            return PaddleOCR(**cfg)
        except Exception as e:
            last_error = e
            print(f"[WARN] PaddleOCR config failed: {type(e).__name__}: {e}")

    raise RuntimeError(f"Could not initialize PaddleOCR. Last error: {last_error}")


def extract_lines_from_v2_result(result) -> Tuple[List[str], List[float]]:
    """
    Parse classic PaddleOCR result:
    [
      [
        [box, (text, score)],
        ...
      ]
    ]
    or
    [
      [box, (text, score)],
      ...
    ]
    """
    if result is None:
        return [], []

    # Sometimes result is [page_result]
    if (
        isinstance(result, list)
        and len(result) == 1
        and isinstance(result[0], list)
        and result[0]
        and isinstance(result[0][0], (list, tuple))
    ):
        maybe_page = result[0]
    else:
        maybe_page = result

    texts = []
    scores = []

    if not isinstance(maybe_page, list):
        return texts, scores

    for item in maybe_page:
        try:
            # item = [box, (text, score)]
            rec = item[1]
            text = str(rec[0])
            score = float(rec[1])
            texts.append(text)
            scores.append(score)
        except Exception:
            continue

    return texts, scores


def extract_lines_from_v3_item(item) -> Tuple[List[str], List[float]]:
    """
    Try parsing PaddleOCR v3 predict output from dict-like/object-like results.
    """
    texts = []
    scores = []

    if item is None:
        return texts, scores

    if isinstance(item, dict):
        for text_key in ["rec_texts", "texts", "text"]:
            if text_key in item:
                raw_texts = item[text_key]
                if isinstance(raw_texts, str):
                    texts = [raw_texts]
                else:
                    texts = [str(x) for x in raw_texts]
                break

        for score_key in ["rec_scores", "scores", "score"]:
            if score_key in item:
                raw_scores = item[score_key]
                if isinstance(raw_scores, (float, int)):
                    scores = [float(raw_scores)]
                else:
                    scores = [float(x) for x in raw_scores]
                break

        if texts and not scores:
            scores = [0.0] * len(texts)

        return texts, scores

    # Some PaddleOCR result objects have json/dict attributes.
    for attr in ["json", "res", "data"]:
        if hasattr(item, attr):
            try:
                value = getattr(item, attr)
                if callable(value):
                    value = value()
                return extract_lines_from_v3_item(value)
            except Exception:
                pass

    return texts, scores


def run_ocr_on_image(ocr_engine, img: Image.Image) -> Tuple[str, float, int]:
    """
    Run PaddleOCR on PIL image and return:
    - cleaned OCR text
    - average OCR score
    - number of detected lines
    """
    arr = np.array(img.convert("RGB"))

    # Try old/classic API first.
    try:
        result = ocr_engine.ocr(arr, cls=False)
        texts, scores = extract_lines_from_v2_result(result)
    except Exception:
        texts, scores = [], []

    # Try v3 predict API if classic API yields nothing.
    if not texts:
        try:
            result = ocr_engine.predict(arr)
            if isinstance(result, list):
                all_texts = []
                all_scores = []
                for item in result:
                    t, s = extract_lines_from_v3_item(item)
                    all_texts.extend(t)
                    all_scores.extend(s)
                texts, scores = all_texts, all_scores
            else:
                texts, scores = extract_lines_from_v3_item(result)
        except Exception:
            texts, scores = [], []

    raw_text = "\n".join(texts)
    cleaned_text = clean_ocr_text(raw_text)

    avg_score = float(sum(scores) / len(scores)) if scores else 0.0
    num_lines = len(texts)

    return cleaned_text, avg_score, num_lines


def summarize_report(report_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    summary = {}

    for variant, group in report_df.groupby("variant"):
        summary[variant] = {
            "rows": int(len(group)),
            "mean_cer": float(group["cer"].mean()),
            "median_cer": float(group["cer"].median()),
            "mean_one_minus_cer": float(group["one_minus_cer"].mean()),
            "mean_runtime_sec": float(group["runtime_sec"].mean()),
            "median_runtime_sec": float(group["runtime_sec"].median()),
            "mean_num_lines": float(group["num_lines"].mean()),
            "mean_avg_score": float(group["avg_score"].mean()),
            "ocr_blank_rate": float(group["ocr_text"].astype(str).str.strip().eq("").mean()),
        }

    return summary


def main():
    args = parse_args()

    labels_path = Path(args.labels)
    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants_to_run = [v.strip() for v in args.variants.split(",") if v.strip()]

    print(f"[INFO] labels: {labels_path}")
    print(f"[INFO] images_dir: {images_dir}")
    print(f"[INFO] out_dir: {out_dir}")
    print(f"[INFO] variants: {variants_to_run}")
    print(f"[INFO] lang: {args.lang}")
    print(f"[INFO] device: {args.device}")

    labels_df = read_csv_keep_empty(labels_path)

    if "image_id" not in labels_df.columns:
        raise ValueError("labels CSV must contain image_id.")

    if "ocr_text" not in labels_df.columns:
        raise ValueError("labels CSV must contain ocr_text.")

    work_df = labels_df.iloc[args.start_index : args.start_index + args.limit].copy()
    work_df = work_df.reset_index(drop=True)

    print(f"[INFO] Running rows: {len(work_df)}")

    ocr_engine = create_paddle_ocr(lang=args.lang, device=args.device)

    rows = []
    report_path = out_dir / "ocr_ablation_report.csv"
    summary_path = out_dir / "ocr_ablation_summary.json"

    variant_img_dir = out_dir / "variant_images"
    if args.save_variant_images:
        variant_img_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in tqdm(work_df.iterrows(), total=len(work_df), desc="OCR ablation"):
        image_id = str(row["image_id"])
        gt_ocr_text = str(row.get("ocr_text", ""))

        image_path = resolve_image_path(images_dir, image_id)

        if image_path is None:
            print(f"[WARN] Missing image: {image_id}")
            continue

        img = Image.open(image_path).convert("RGB")
        quality = get_image_quality(img)
        variants = make_preprocess_variants(img)

        for variant_name in variants_to_run:
            if variant_name not in variants:
                print(f"[WARN] Unknown variant skipped: {variant_name}")
                continue

            variant_img = variants[variant_name]

            if args.save_variant_images:
                save_dir = variant_img_dir / Path(image_id).stem
                save_dir.mkdir(parents=True, exist_ok=True)
                variant_img.save(save_dir / f"{variant_name}.jpg", quality=95)

            start = time.perf_counter()

            try:
                pred_ocr_text, avg_score, num_lines = run_ocr_on_image(
                    ocr_engine,
                    variant_img,
                )
                error = ""
            except Exception as e:
                pred_ocr_text = ""
                avg_score = 0.0
                num_lines = 0
                error = f"{type(e).__name__}: {e}"

            runtime_sec = time.perf_counter() - start

            row_cer = cer(pred_ocr_text, gt_ocr_text)
            one_minus = max(0.0, 1.0 - row_cer)

            rows.append(
                {
                    "image_id": image_id,
                    "variant": variant_name,
                    "ocr_text": pred_ocr_text,
                    "gt_ocr_text": gt_ocr_text,
                    "cer": row_cer,
                    "one_minus_cer": one_minus,
                    "avg_score": avg_score,
                    "num_lines": num_lines,
                    "runtime_sec": runtime_sec,
                    "width": quality.width,
                    "height": quality.height,
                    "long_side": quality.long_side,
                    "short_side": quality.short_side,
                    "mean_brightness": quality.mean_brightness,
                    "contrast_std": quality.contrast_std,
                    "blur_laplacian_var": quality.blur_laplacian_var,
                    "dark_pixel_ratio": quality.dark_pixel_ratio,
                    "error": error,
                }
            )

        if args.save_every > 0 and (idx + 1) % args.save_every == 0:
            partial_df = pd.DataFrame(rows)
            partial_df.to_csv(report_path, index=False)
            summary = summarize_report(partial_df)
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"[INFO] Partial saved at {idx + 1} images.")

    report_df = pd.DataFrame(rows)
    report_df.to_csv(report_path, index=False)

    summary = summarize_report(report_df)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[INFO] OCR ablation complete.")
    print(f"[INFO] Report: {report_path}")
    print(f"[INFO] Summary: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()