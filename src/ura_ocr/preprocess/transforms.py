from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


@dataclass
class ImageQuality:
    width: int
    height: int
    long_side: int
    short_side: int
    mean_brightness: float
    contrast_std: float
    blur_laplacian_var: float
    dark_pixel_ratio: float


def ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def pil_to_rgb_array(img: Image.Image) -> np.ndarray:
    return np.array(ensure_rgb(img))


def rgb_array_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def get_image_quality(img: Image.Image) -> ImageQuality:
    img = ensure_rgb(img)
    arr = np.array(img)

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    h, w = gray.shape[:2]
    long_side = max(w, h)
    short_side = min(w, h)

    mean_brightness = float(gray.mean())
    contrast_std = float(gray.std())
    blur_laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    dark_pixel_ratio = float((gray < 70).mean())

    return ImageQuality(
        width=w,
        height=h,
        long_side=long_side,
        short_side=short_side,
        mean_brightness=mean_brightness,
        contrast_std=contrast_std,
        blur_laplacian_var=blur_laplacian_var,
        dark_pixel_ratio=dark_pixel_ratio,
    )


def resize_long_side(
    img: Image.Image,
    long_side: int = 960,
    allow_downscale: bool = False,
) -> Image.Image:
    """
    Resize image while preserving aspect ratio.

    By default, this only upscales smaller images and does not downscale
    larger images. This avoids losing text details.
    """
    img = ensure_rgb(img)
    w, h = img.size
    current_long = max(w, h)

    if current_long == 0:
        return img

    if not allow_downscale and current_long >= long_side:
        return img.copy()

    scale = long_side / current_long
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def apply_clahe_rgb(
    img: Image.Image,
    clip_limit: float = 2.0,
    tile_grid: Tuple[int, int] = (4, 4),
) -> Image.Image:
    """
    Apply CLAHE on L channel in LAB space.

    This improves local contrast while keeping color relatively natural.
    """
    arr = pil_to_rgb_array(img)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid,
    )

    l_eq = clahe.apply(l_channel)
    lab_eq = cv2.merge([l_eq, a_channel, b_channel])

    rgb_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    return rgb_array_to_pil(rgb_eq)


def autocontrast_rgb(
    img: Image.Image,
    cutoff: int = 1,
) -> Image.Image:
    """
    Stretch global contrast using PIL autocontrast.
    """
    img = ensure_rgb(img)
    return ImageOps.autocontrast(img, cutoff=cutoff)


def sharpen_light(
    img: Image.Image,
    radius: float = 1.0,
    percent: int = 120,
    threshold: int = 3,
) -> Image.Image:
    """
    Light unsharp mask.
    """
    img = ensure_rgb(img)
    return img.filter(
        ImageFilter.UnsharpMask(
            radius=radius,
            percent=percent,
            threshold=threshold,
        )
    )


def to_grayscale_rgb(img: Image.Image) -> Image.Image:
    """
    Convert to grayscale but return RGB image.
    """
    img = ensure_rgb(img)
    gray = ImageOps.grayscale(img)
    return Image.merge("RGB", (gray, gray, gray))


def invert_rgb(img: Image.Image) -> Image.Image:
    """
    Invert RGB image.
    """
    img = ensure_rgb(img)
    return ImageOps.invert(img)


def maybe_invert_for_dark_bg(
    img: Image.Image,
    dark_ratio_threshold: float = 0.55,
    mean_brightness_threshold: float = 105.0,
) -> Image.Image:
    """
    Invert only if the image is likely dark-background dominant.
    """
    quality = get_image_quality(img)

    if (
        quality.dark_pixel_ratio >= dark_ratio_threshold
        and quality.mean_brightness <= mean_brightness_threshold
    ):
        return invert_rgb(img)

    return ensure_rgb(img).copy()


def make_preprocess_variants(
    img: Image.Image,
) -> Dict[str, Image.Image]:
    """
    Generate preprocessing variants for visual inspection and OCR ablation.

    No variant is assumed to be universally best.
    """
    img = ensure_rgb(img)

    resize_960 = resize_long_side(img, long_side=960)
    resize_1280 = resize_long_side(img, long_side=1280)

    variants = {
        "raw": img.copy(),
        "resize_960": resize_960,
        "resize_1280": resize_1280,
        "clahe": apply_clahe_rgb(img),
        "resize_960_clahe": apply_clahe_rgb(resize_960),
        "gray": to_grayscale_rgb(img),
        "autocontrast": autocontrast_rgb(img),
        "sharpen": sharpen_light(img),
        "dark_bg_invert": maybe_invert_for_dark_bg(img),
    }

    return variants