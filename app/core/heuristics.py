# app/core/heuristics.py
from typing import Any, Dict, Literal


def apply_heuristics(
    verdict: Literal["REAL", "FAKE", "UNCLEAR"],
    confidence: int,
    metadata: Dict[str, Any] | None = None,
    heuristics_info: Dict[str, Any] | None = None,
) -> tuple[Literal["REAL", "FAKE", "UNCLEAR"], int, Dict[str, Any]]:
    """
    Placeholder for heuristic adjustments.
    In the future, this will look at title, URL, platform, etc.
    For now, it just returns the input verdict/confidence.
    """
    extra: Dict[str, Any] = {}

    # Example future logic (pseudo):
    # if metadata and "title" in metadata and "deepfake" in metadata["title"].lower():
    #     if verdict == "REAL":
    #         verdict = "UNCLEAR"
    #         confidence = min(confidence, 60)
    #     extra["title_flagged"] = True

    return verdict, confidence, extra
