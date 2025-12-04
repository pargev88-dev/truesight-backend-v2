# app/detectors/video.py

from typing import List, Literal, Tuple
import base64
import io

from PIL import Image, ImageFilter
import torch
import torchvision.transforms as T

from app.detectors.model_loader import get_video_model


# -----------------------------
# QUALITY FILTERS
# -----------------------------
def _is_low_quality(img: Image.Image) -> bool:
    """
    Returns True if the frame is too blurry / low-detail.
    Uses variance of Laplacian approximation.
    """
    gray = img.convert("L")
    lap = gray.filter(ImageFilter.FIND_EDGES)
    variance = torch.tensor(lap).float().var().item()

    # Very low detail → skip this frame
    return variance < 5.0  # tuned threshold


# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def _decode_base64_image(data_url: str) -> Image.Image:
    if "," in data_url:
        _, b64data = data_url.split(",", 1)
    else:
        b64data = data_url

    b64data = "".join(b64data.split())
    missing_padding = len(b64data) % 4
    if missing_padding:
        b64data += "=" * (4 - missing_padding)

    img_bytes = base64.b64decode(b64data)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


# -----------------------------
# SMART AGGREGATION (UPGRADED)
# -----------------------------
def _aggregate_scores(scores: List[float]) -> Tuple[Literal["REAL", "FAKE", "UNCLEAR"], int]:
    """
    Improved logic:
      - FAKE requires:
            max_fake >= 0.93 AND at least 2 frames >= 0.90
      - REAL requires:
            all frames <= 0.20
      - Otherwise UNCLEAR
    """
    if not scores:
        return "UNCLEAR", 0

    max_fake = max(scores)
    strong_fakes = [s for s in scores if s >= 0.90]

    # --- FAKE RULE (more conservative)
    if max_fake >= 0.93 and len(strong_fakes) >= 2:
        verdict: Literal["REAL", "FAKE", "UNCLEAR"] = "FAKE"
        confidence = int(max_fake * 100)
        return verdict, confidence

    # --- REAL RULE
    if all(s <= 0.20 for s in scores):
        verdict = "REAL"
        confidence = int((1.0 - max_fake) * 100)  # strong real → high confidence
        return verdict, confidence

    # --- UNCLEAR RULE
    # Build a balanced confidence score.
    # If max_fake ~0.50 → ~50% confidence
    # If max_fake ~0.80 → ~70% confidence
    # If max_fake ~0.30 → ~40% confidence
    confidence = int(40 + (max_fake - 0.20) / 0.73 * 40)
    confidence = max(0, min(confidence, 100))
    return "UNCLEAR", confidence


# -----------------------------
# MAIN ANALYSIS FUNCTION
# -----------------------------
def analyze_frames(frames_base64: List[str]) -> Tuple[List[float], Literal["REAL", "FAKE", "UNCLEAR"], int]:
    """
    Steps:
      1. Decode frames
      2. Filter low-quality frames
      3. Preprocess
      4. Model inference
      5. Smart aggregation
    """
    if not frames_base64:
        return [], "UNCLEAR", 0

    model = get_video_model()
    device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    tensors: List[torch.Tensor] = []

    for data_url in frames_base64:
        try:
            img = _decode_base64_image(data_url)

            # -----------------------------
            # APPLY QUALITY FILTER
            # -----------------------------
            if _is_low_quality(img):
                continue  # skip blurry/low-detail frames

            tensors.append(_transform(img))
        except Exception:
            continue  # skip bad frames

    if not tensors:
        return [], "UNCLEAR", 0

    batch = torch.stack(tensors).to(device, non_blocking=True)

    with torch.inference_mode():
        outputs = model(batch)          # shape [batch, 1]
        probs_fake = outputs.view(-1).detach().cpu().numpy().tolist()

    frame_scores = [float(p) for p in probs_fake]

    verdict, confidence = _aggregate_scores(frame_scores)
    return frame_scores, verdict, confidence
