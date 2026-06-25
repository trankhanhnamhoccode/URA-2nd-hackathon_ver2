from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]


def resolve_image_path(images_dir: str | Path, image_id: str) -> Optional[Path]:
    images_dir = Path(images_dir)
    image_id = str(image_id)

    direct = images_dir / image_id
    if direct.exists():
        return direct

    stem = Path(image_id).stem

    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    return None


def load_rgb_image(path: str | Path) -> Image.Image:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")