# app/detectors/video.py
from typing import List, Literal, Tuple


def analyze_frames(
    frames_base64: List[str],
) -> Tuple[List[float], Literal["REAL", "FAKE", "UNCLEAR"], int]:
    """
    TEMPORARY STUB:
    Pretends to run a deepfake detector.
    For now, we:
      - return 0.5 fake probability for each frame
      - always say UNCLEAR with 0 confidence

    Later, this will:
      - decode base64 → PIL Image
      - transform to tensors
      - run PyTorch model
      - aggregate per-frame scores into verdict + confidence.
    """
    num_frames = len(frames_base64)
    frame_scores = [0.5] * num_frames

    verdict: Literal["REAL", "FAKE", "UNCLEAR"] = "UNCLEAR"
    confidence = 0

    return frame_scores, verdict, confidence
