from __future__ import annotations

from typing import Any


def _safe_float(x: Any):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _to_builtin(obj: Any) -> Any:
    """
    Convert PaddleOCR/PaddleX result objects into plain Python objects when possible.
    Supports common PaddleOCR 2.x and 3.x result styles.
    """
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_to_builtin(x) for x in obj]

    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass

    # PaddleX/PaddleOCR 3.x Result often exposes .json or .to_json-like data.
    for attr in ["json", "to_dict", "dict"]:
        if hasattr(obj, attr):
            try:
                val = getattr(obj, attr)
                val = val() if callable(val) else val
                return _to_builtin(val)
            except Exception:
                pass

    if hasattr(obj, "__dict__"):
        try:
            return _to_builtin(vars(obj))
        except Exception:
            pass

    return obj


def _flatten_numbers(obj: Any) -> list[float]:
    obj = _to_builtin(obj)

    if obj is None:
        return []

    if isinstance(obj, (int, float)):
        return [float(obj)]

    if isinstance(obj, dict):
        for key in ["points", "poly", "polys", "box", "bbox", "dt_poly", "rec_poly"]:
            if key in obj:
                return _flatten_numbers(obj[key])
        nums = []
        for v in obj.values():
            nums.extend(_flatten_numbers(v))
        return nums

    if isinstance(obj, (list, tuple)):
        nums = []
        for item in obj:
            nums.extend(_flatten_numbers(item))
        return nums

    return []


def _poly_stats(poly: Any, width: int | None = None, height: int | None = None) -> dict[str, Any]:
    """
    Convert either:
    - [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    - [x1,y1,x2,y2,x3,y3,x4,y4]
    - [xmin,ymin,xmax,ymax]
    into normalized polygon/stat fields.
    """
    nums = _flatten_numbers(poly)

    if len(nums) >= 8:
        pts = nums[:8]
    elif len(nums) == 4:
        xmin, ymin, xmax, ymax = nums
        pts = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
    else:
        pts = [None] * 8

    xs = [v for v in pts[0::2] if v is not None]
    ys = [v for v in pts[1::2] if v is not None]

    out = {
        "box_x1": pts[0],
        "box_y1": pts[1],
        "box_x2": pts[2],
        "box_y2": pts[3],
        "box_x3": pts[4],
        "box_y3": pts[5],
        "box_x4": pts[6],
        "box_y4": pts[7],
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

    if xs and ys:
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        bw = max(0.0, xmax - xmin)
        bh = max(0.0, ymax - ymin)
        area = bw * bh

        out.update(
            {
                "box_xmin": xmin,
                "box_ymin": ymin,
                "box_xmax": xmax,
                "box_ymax": ymax,
                "box_width": bw,
                "box_height": bh,
                "box_area": area,
            }
        )

        if width and width > 0:
            out["box_cx_ratio"] = ((xmin + xmax) / 2.0) / width

        if height and height > 0:
            out["box_cy_ratio"] = ((ymin + ymax) / 2.0) / height

        if width and height and width > 0 and height > 0:
            out["box_area_ratio"] = area / float(width * height)

    return out


def _unwrap_v3_dict(obj: Any) -> dict[str, Any] | None:
    d = _to_builtin(obj)

    if isinstance(d, list) and len(d) == 1:
        d = d[0]

    if not isinstance(d, dict):
        return None

    # PaddleOCR/PaddleX 3.x often has {"res": {...}}.
    if "res" in d and isinstance(d["res"], dict):
        d = d["res"]

    return d


def _extract_v3_records(
    raw_result: Any,
    image_id: str,
    variant: str,
    variant_width: int | None,
    variant_height: int | None,
) -> list[dict[str, Any]]:
    """
    Parse PaddleOCR 3.x / PaddleX style output with rec_texts, rec_scores, rec_polys/dt_polys.
    """
    d = _unwrap_v3_dict(raw_result)

    if not d:
        return []

    texts = d.get("rec_texts") or d.get("texts") or d.get("text")
    scores = d.get("rec_scores") or d.get("scores") or d.get("rec_score")
    polys = (
        d.get("rec_polys")
        or d.get("dt_polys")
        or d.get("polys")
        or d.get("boxes")
        or d.get("rec_boxes")
        or d.get("dt_boxes")
    )

    if isinstance(texts, str):
        texts = [texts]

    if not isinstance(texts, list):
        return []

    if not isinstance(scores, list):
        scores = [None] * len(texts)

    if not isinstance(polys, list):
        polys = [None] * len(texts)

    rows: list[dict[str, Any]] = []

    for i, text in enumerate(texts):
        text = "" if text is None else str(text).strip()
        score = scores[i] if i < len(scores) else None
        poly = polys[i] if i < len(polys) else None

        row = {
            "image_id": image_id,
            "variant": variant,
            "line_idx": i,
            "line_text": text,
            "line_score": _safe_float(score),
            "variant_width": variant_width,
            "variant_height": variant_height,
            "parser": "paddle_v3_dict",
        }
        row.update(_poly_stats(poly, variant_width, variant_height))
        rows.append(row)

    return rows


def _looks_like_v2_line(obj: Any) -> bool:
    """
    PaddleOCR 2.x line normally looks like:
    [box, (text, score)]
    """
    if not isinstance(obj, (list, tuple)):
        return False

    if len(obj) < 2:
        return False

    rec = obj[1]

    if isinstance(rec, (list, tuple)) and len(rec) >= 1:
        return isinstance(rec[0], str)

    return False


def _extract_v2_records(
    raw_result: Any,
    image_id: str,
    variant: str,
    variant_width: int | None,
    variant_height: int | None,
) -> list[dict[str, Any]]:
    """
    Parse PaddleOCR 2.x style output:
    [
      [
        [box, (text, score)],
        ...
      ]
    ]
    """
    data = _to_builtin(raw_result)

    if not isinstance(data, list):
        return []

    # Common shape: [ [line1, line2, ...] ]
    if len(data) == 1 and isinstance(data[0], list):
        candidate_lines = data[0]
    else:
        candidate_lines = data

    rows: list[dict[str, Any]] = []

    for i, item in enumerate(candidate_lines):
        if not _looks_like_v2_line(item):
            continue

        box = item[0]
        rec = item[1]

        text = ""
        score = None

        if isinstance(rec, (list, tuple)):
            text = "" if rec[0] is None else str(rec[0]).strip()
            if len(rec) > 1:
                score = rec[1]

        row = {
            "image_id": image_id,
            "variant": variant,
            "line_idx": i,
            "line_text": text,
            "line_score": _safe_float(score),
            "variant_width": variant_width,
            "variant_height": variant_height,
            "parser": "paddle_v2_list",
        }
        row.update(_poly_stats(box, variant_width, variant_height))
        rows.append(row)

    return rows


def extract_line_records(
    raw_result: Any,
    image_id: str,
    variant: str,
    variant_width: int | None = None,
    variant_height: int | None = None,
) -> list[dict[str, Any]]:
    """
    Robust line-level extractor for PaddleOCR 2.x/3.x outputs.
    Returns one row per detected/recognized text line.
    """
    rows = _extract_v3_records(
        raw_result=raw_result,
        image_id=image_id,
        variant=variant,
        variant_width=variant_width,
        variant_height=variant_height,
    )

    if rows:
        return rows

    rows = _extract_v2_records(
        raw_result=raw_result,
        image_id=image_id,
        variant=variant,
        variant_width=variant_width,
        variant_height=variant_height,
    )

    return rows


def line_records_to_text(rows: list[dict[str, Any]], sep: str = "\n") -> str:
    texts = []
    for r in rows:
        t = str(r.get("line_text", "") or "").strip()
        if t:
            texts.append(t)
    return sep.join(texts).strip()


def line_records_avg_score(rows: list[dict[str, Any]]) -> float:
    scores = []
    for r in rows:
        s = _safe_float(r.get("line_score"))
        if s is not None:
            scores.append(s)

    if not scores:
        return 0.0

    return float(sum(scores) / len(scores))