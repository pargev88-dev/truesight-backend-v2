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
    return False



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
    Pass-through mode:
    - We do NOT label as REAL or FAKE.
    - We just return the max fake probability as the "confidence".
    - verdict is always 'UNCLEAR' so the UI shows e.g.:
        'Unclear (67% confidence)'
      where 67% is the model's estimated fake probability.
    """
    if not scores:
        return "UNCLEAR", 0

    max_fake = max(scores)          # in [0, 1]
    confidence = int(round(max_fake * 100))  # 0–100

    # Always return UNCLEAR; UI just displays the % number.
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
