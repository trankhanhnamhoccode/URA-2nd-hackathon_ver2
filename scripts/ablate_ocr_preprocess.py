from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from ura_ocr.eval.metrics import cer
from ura_ocr.io.csv_io import read_csv_keep_empty
from ura_ocr.io.image_loader import resolve_image_path
from ura_ocr.ocr.cleaner import clean_ocr_text
from ura_ocr.ocr.line_records import (
    extract_line_records,
    line_records_avg_score,
)
from ura_ocr.preprocess.transforms import get_image_quality, make_preprocess_variants


DEFAULT_VARIANTS = [
    "raw",
    "resize_960",
    "resize_1280",
    "clahe",
    "resize_960_clahe",
]


LINE_COLUMNS = [
    "image_id",
    "variant",
    "line_idx",
    "line_text",
    "line_score",
    "variant_width",
    "variant_height",
    "parser",
    "box_x1",
    "box_y1",
    "box_x2",
    "box_y2",
    "box_x3",
    "box_y3",
    "box_x4",
    "box_y4",
    "box_xmin",
    "box_ymin",
    "box_xmax",
    "box_ymax",
    "box_width",
    "box_height",
    "box_area",
    "box_cx_ratio",
    "box_cy_ratio",
    "box_area_ratio",
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


def safe_import_paddleocr(hide_torch: bool = True):
    """
    Kaggle fix:
    PaddleOCR 3.x imports PaddleX -> ModelScope.
    ModelScope may probe/import torch during import.
    Some Kaggle runtimes have Torch CUDA/NCCL mismatch:
        libtorch_cuda.so: undefined symbol: ncclCommShrink

    For PaddleOCR inference we do not need torch, so during PaddleOCR import
    only, hide torch from importlib.util.find_spec().
    """
    import sys
    import importlib.util as importlib_util

    if "paddleocr" in sys.modules:
        import paddleocr
        from paddleocr import PaddleOCR

        return paddleocr, PaddleOCR

    if not hide_torch:
        import paddleocr
        from paddleocr import PaddleOCR

        return paddleocr, PaddleOCR

    original_find_spec = importlib_util.find_spec

    def patched_find_spec(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            return None
        return original_find_spec(name, *args, **kwargs)

    importlib_util.find_spec = patched_find_spec

    try:
        import paddleocr
        from paddleocr import PaddleOCR

        return paddleocr, PaddleOCR
    finally:
        importlib_util.find_spec = original_find_spec


def create_paddle_ocr(lang: str = "vi", device: str = "gpu"):
    """
    Create PaddleOCR object with compatibility fallbacks.

    PaddleOCR APIs have changed across versions, so we try a few safe configs.
    """
    _, PaddleOCR = safe_import_paddleocr(hide_torch=True)

    configs = [
        {
            "lang": lang,
            "device": device,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "engine": "paddle",
        },
        {
            "lang": lang,
            "device": device,
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
            reader = PaddleOCR(**cfg)
            print("[INFO] PaddleOCR loaded OK")
            return reader
        except Exception as e:
            last_error = e
            print(f"[WARN] PaddleOCR config failed: {type(e).__name__}: {e}")

    raise RuntimeError(f"Could not initialize PaddleOCR. Last error: {last_error}")


def _safe_text(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _unwrap_v3_dict(item: Any) -> Any:
    """
    PaddleOCR 3.x result objects can be dict-like or object-like.
    This helper is only for fallback text parsing. The main parser is
    extract_line_records() from ura_ocr.ocr.line_records.
    """
    if item is None:
        return None

    if isinstance(item, dict):
        if "res" in item and isinstance(item["res"], dict):
            return item["res"]
        return item

    for attr in ["json", "res", "data"]:
        if hasattr(item, attr):
            try:
                value = getattr(item, attr)
                if callable(value):
                    value = value()
                return _unwrap_v3_dict(value)
            except Exception:
                pass

    return item


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
            rec = item[1]
            text = str(rec[0]).strip()
            score = float(rec[1])
            if text:
                texts.append(text)
                scores.append(score)
        except Exception:
            continue

    return texts, scores


def extract_lines_from_v3_item(item) -> Tuple[List[str], List[float]]:
    """
    Fallback parser for PaddleOCR v3 predict output.
    The preferred parser is extract_line_records().
    """
    item = _unwrap_v3_dict(item)

    texts: List[str] = []
    scores: List[float] = []

    if item is None:
        return texts, scores

    if isinstance(item, list):
        all_texts = []
        all_scores = []
        for sub in item:
            t, s = extract_lines_from_v3_item(sub)
            all_texts.extend(t)
            all_scores.extend(s)
        return all_texts, all_scores

    if isinstance(item, dict):
        for text_key in ["rec_texts", "texts", "text"]:
            if text_key in item:
                raw_texts = item[text_key]
                if isinstance(raw_texts, str):
                    texts = [raw_texts.strip()]
                else:
                    texts = [str(x).strip() for x in raw_texts if str(x).strip()]
                break

        for score_key in ["rec_scores", "scores", "score"]:
            if score_key in item:
                raw_scores = item[score_key]
                if isinstance(raw_scores, (float, int)):
                    scores = [float(raw_scores)]
                else:
                    parsed_scores = []
                    for x in raw_scores:
                        val = _safe_float(x)
                        if val is not None:
                            parsed_scores.append(val)
                    scores = parsed_scores
                break

        if texts and not scores:
            scores = [0.0] * len(texts)

        if len(scores) < len(texts):
            scores = scores + [0.0] * (len(texts) - len(scores))

        return texts, scores[: len(texts)]

    return texts, scores


def make_fallback_line_rows(
    texts: List[str],
    scores: List[float],
    image_id: str,
    variant: str,
    variant_width: int,
    variant_height: int,
    parser: str,
) -> List[Dict[str, Any]]:
    """
    If the structured parser cannot extract bbox-level rows but the old text
    parser can still extract texts/scores, keep text-only line rows so the
    downstream line-level CSV is not empty.
    """
    line_rows: List[Dict[str, Any]] = []

    for i, text in enumerate(texts):
        text = _safe_text(text)
        if not text:
            continue

        score = scores[i] if i < len(scores) else None

        line_rows.append(
            {
                "image_id": image_id,
                "variant": variant,
                "line_idx": i,
                "line_text": text,
                "line_score": _safe_float(score),
                "variant_width": variant_width,
                "variant_height": variant_height,
                "parser": parser,
                "box_x1": None,
                "box_y1": None,
                "box_x2": None,
                "box_y2": None,
                "box_x3": None,
                "box_y3": None,
                "box_x4": None,
                "box_y4": None,
                "box_xmin": None,
                "box_ymin": None,
                "box_xmax": None,
                "box_ymax": None,
                "box_width": None,
                "box_height": None,
                "box_area": None,
                "box_cx_ratio": None,
                "box_cy_ratio": None,
                "box_area_ratio": None,
            }
        )

    return line_rows


def run_ocr_on_image(
    ocr_engine,
    img: Image.Image,
    image_id: str,
    variant: str,
) -> Tuple[str, float, int, List[Dict[str, Any]], str]:
    """
    Run PaddleOCR on PIL image and return:
    - cleaned OCR text
    - average OCR score
    - number of detected lines
    - line-level rows
    - API used: ocr / predict / none

    This keeps the old output behavior but additionally exposes line-level
    records for later line selection / candidate merge.
    """
    arr = np.array(img.convert("RGB"))
    variant_width, variant_height = img.size

    raw_result = None
    api_used = "none"
    line_rows: List[Dict[str, Any]] = []
    texts: List[str] = []
    scores: List[float] = []

    # Try old/classic API first.
    try:
        raw_result = ocr_engine.ocr(arr, cls=False)
        api_used = "ocr"

        line_rows = extract_line_records(
            raw_result=raw_result,
            image_id=image_id,
            variant=variant,
            variant_width=variant_width,
            variant_height=variant_height,
        )

        if line_rows:
            texts = [_safe_text(r.get("line_text")) for r in line_rows]
            texts = [t for t in texts if t]
            scores = [
                float(r["line_score"])
                for r in line_rows
                if _safe_float(r.get("line_score")) is not None
            ]
        else:
            texts, scores = extract_lines_from_v2_result(raw_result)
            line_rows = make_fallback_line_rows(
                texts=texts,
                scores=scores,
                image_id=image_id,
                variant=variant,
                variant_width=variant_width,
                variant_height=variant_height,
                parser="fallback_v2_text_only",
            )
    except Exception:
        raw_result = None
        api_used = "none"
        line_rows = []
        texts = []
        scores = []

    # Try v3 predict API if classic API yields nothing.
    if not texts:
        try:
            raw_result = ocr_engine.predict(arr)
            api_used = "predict"

            line_rows = extract_line_records(
                raw_result=raw_result,
                image_id=image_id,
                variant=variant,
                variant_width=variant_width,
                variant_height=variant_height,
            )

            if line_rows:
                texts = [_safe_text(r.get("line_text")) for r in line_rows]
                texts = [t for t in texts if t]
                scores = [
                    float(r["line_score"])
                    for r in line_rows
                    if _safe_float(r.get("line_score")) is not None
                ]
            else:
                texts, scores = extract_lines_from_v3_item(raw_result)
                line_rows = make_fallback_line_rows(
                    texts=texts,
                    scores=scores,
                    image_id=image_id,
                    variant=variant,
                    variant_width=variant_width,
                    variant_height=variant_height,
                    parser="fallback_v3_text_only",
                )
        except Exception:
            raw_result = None
            api_used = "none"
            line_rows = []
            texts = []
            scores = []

    raw_text = "\n".join([t for t in texts if _safe_text(t)])
    cleaned_text = clean_ocr_text(raw_text)

    if scores:
        avg_score = float(sum(scores) / len(scores))
    elif line_rows:
        avg_score = float(line_records_avg_score(line_rows))
    else:
        avg_score = 0.0

    num_lines = len([t for t in texts if _safe_text(t)])

    # Do not clean individual line_text here.
    # Keep raw line text for line-level analysis later.
    return cleaned_text, avg_score, num_lines, line_rows, api_used


def summarize_report(report_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    summary = {}

    if report_df.empty:
        return summary

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


def save_reports(
    rows: List[Dict[str, Any]],
    all_line_rows: List[Dict[str, Any]],
    report_path: Path,
    line_report_path: Path,
    summary_path: Path,
):
    report_df = pd.DataFrame(rows)
    report_df.to_csv(report_path, index=False, encoding="utf-8-sig")

    if all_line_rows:
        line_df = pd.DataFrame(all_line_rows)

        for col in LINE_COLUMNS:
            if col not in line_df.columns:
                line_df[col] = None

        ordered_cols = [c for c in LINE_COLUMNS if c in line_df.columns]
        extra_cols = [c for c in line_df.columns if c not in ordered_cols]
        line_df = line_df[ordered_cols + extra_cols]
    else:
        line_df = pd.DataFrame(columns=LINE_COLUMNS)

    line_df.to_csv(line_report_path, index=False, encoding="utf-8-sig")

    summary = summarize_report(report_df)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return report_df, line_df, summary


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

    rows: List[Dict[str, Any]] = []
    all_line_rows: List[Dict[str, Any]] = []

    report_path = out_dir / "ocr_ablation_report.csv"
    line_report_path = out_dir / "ocr_ablation_lines.csv"
    summary_path = out_dir / "ocr_ablation_summary.json"

    variant_img_dir = out_dir / "variant_images"
    if args.save_variant_images:
        variant_img_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in tqdm(work_df.iterrows(), total=len(work_df), desc="OCR ablation"):
        image_id = _safe_text(row["image_id"])
        gt_ocr_text = _safe_text(row.get("ocr_text", ""))

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
                pred_ocr_text, avg_score, num_lines, line_rows, api_used = run_ocr_on_image(
                    ocr_engine=ocr_engine,
                    img=variant_img,
                    image_id=image_id,
                    variant=variant_name,
                )
                error = ""
            except Exception as e:
                pred_ocr_text = ""
                avg_score = 0.0
                num_lines = 0
                line_rows = []
                api_used = "none"
                error = f"{type(e).__name__}: {e}"

            runtime_sec = time.perf_counter() - start

            if line_rows:
                all_line_rows.extend(line_rows)

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
                    "api_used": api_used,
                    "line_rows": len(line_rows),
                    "error": error,
                }
            )

        if args.save_every > 0 and (idx + 1) % args.save_every == 0:
            _, partial_line_df, _ = save_reports(
                rows=rows,
                all_line_rows=all_line_rows,
                report_path=report_path,
                line_report_path=line_report_path,
                summary_path=summary_path,
            )
            print(
                f"[INFO] Partial saved at {idx + 1} images. "
                f"Rows={len(rows)}, line_rows={len(partial_line_df)}"
            )

    report_df, line_df, summary = save_reports(
        rows=rows,
        all_line_rows=all_line_rows,
        report_path=report_path,
        line_report_path=line_report_path,
        summary_path=summary_path,
    )

    print("[INFO] OCR ablation complete.")
    print(f"[INFO] Report: {report_path}")
    print(f"[INFO] Line report: {line_report_path}")
    print(f"[INFO] Summary: {summary_path}")
    print(f"[INFO] Report rows: {len(report_df)}")
    print(f"[INFO] Line rows: {len(line_df)}")

    if len(line_df) == 0:
        print("[WARN] No line-level OCR rows were extracted. Check PaddleOCR output parser.")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()