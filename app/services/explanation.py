# app/services/explanation.py
from typing import List, Literal, Optional


def generate_explanation(
    verdict: Literal["REAL", "FAKE", "UNCLEAR"],
    confidence: int,
    frame_scores: List[float],
    url: Optional[str] = None,
    mode: str = "standard",
) -> str:
    """
    Basic explanation builder.
    Later, this can call GPT or some rule-based text builder.
    """
    max_fake = max(frame_scores) if frame_scores else 0.0
    url_part = f" for {url}" if url else ""

    if verdict == "FAKE":
        return (
            f"The frames{url_part} show strong signs of AI-generated manipulation "
            f"(max fake probability ~{int(max_fake * 100)}%)."
        )
    elif verdict == "REAL":
        return (
            f"The frames{url_part} look consistent with a real camera recording "
            f"(fake probability stays low across all frames)."
        )
    else:
        return (
            f"The analysis{url_part} is inconclusive. The model could not confidently "
            f"classify the frames as real or AI-generated."
        )
