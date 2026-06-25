from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class OCRLine:
    text: str
    score: float = 0.0
    box: Optional[list] = None


@dataclass
class OCRResult:
    raw_text: str = ""
    cleaned_text: str = ""
    corrected_text: str = ""
    lines: List[OCRLine] = field(default_factory=list)
    avg_score: float = 0.0
    route: str = "empty"

    @property
    def final_text(self) -> str:
        """
        Text used for submission. For now, use cleaned_text if available,
        otherwise raw_text.
        """
        return self.cleaned_text or self.raw_text or ""


@dataclass
class Candidate:
    text: str
    kind: str
    source: str
    score: float = 0.0
    evidence: str = ""


@dataclass
class PredictionResult:
    image_id: str
    ocr_text: str = ""
    brand_name: str = ""
    product_name: str = ""
    score: float = 0.0
    route: str = "stub"
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_submission_row(self, phase: str = "phase2") -> Dict[str, str]:
        if phase == "phase1":
            return {
                "image_id": self.image_id,
                "ocr_text": self.ocr_text or "",
                "product_name": self.product_name or "",
            }

        return {
            "image_id": self.image_id,
            "ocr_text": self.ocr_text or "",
            "brand_name": self.brand_name or "",
            "product_name": self.product_name or "",
        }