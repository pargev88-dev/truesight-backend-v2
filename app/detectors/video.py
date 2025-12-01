# app/detectors/video.py

from typing import List, Literal, Tuple
import base64
import io

from PIL import Image
import torch
import torchvision.transforms as T

from app.detectors.model_loader import get_video_model


# Image preprocessing (adjust sizes/means if you swap models later)
_transform = T.Compose([
    T.Resize((224, 224)),      # common input size; change if your model needs different
    T.ToTensor(),              # [0, 1]
    T.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet stats
        std=[0.229, 0.224, 0.225],
    ),
])


def _decode_base64_image(data_url: str) -> Image.Image:
    """
    Handles both full data URLs and raw base64 strings.
    Fixes whitespace + padding automatically.
    """
    # Extract base64 if data URL
    if "," in data_url:
        _, b64data = data_url.split(",", 1)
    else:
        b64data = data_url

    # Remove whitespace/newlines
    b64data = "".join(b64data.split())

    # Fix base64 padding if needed
    missing_padding = len(b64data) % 4
    if missing_padding:
        b64data += "=" * (4 - missing_padding)

    img_bytes = base64.b64decode(b64data)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return img


def _aggregate_scores(scores: List[float]) -> Tuple[Literal["REAL", "FAKE", "UNCLEAR"], int]:
    """
    Convert per-frame fake probabilities into a global verdict.

    Policy:
      - If any frame > 0.80 → FAKE
      - Else if all frames < 0.30 → REAL
      - Else → UNCLEAR
    """
    if not scores:
        return "UNCLEAR", 0

    max_fake = max(scores)

    if max_fake > 0.80:
        verdict: Literal["REAL", "FAKE", "UNCLEAR"] = "FAKE"
        confidence = int(max_fake * 100)
    elif max_fake < 0.30:
        verdict = "REAL"
        # strongest frame still looks real; confidence based on “not fake”
        confidence = int((1.0 - max_fake) * 100)
    else:
        verdict = "UNCLEAR"
        # mid-range → lower confidence (roughly 40–70)
        confidence = int(40 + (max_fake - 0.3) / 0.5 * 30)
        confidence = max(0, min(confidence, 100))

    return verdict, confidence


def analyze_frames(
    frames_base64: List[str],
) -> Tuple[List[float], Literal["REAL", "FAKE", "UNCLEAR"], int]:
    """
    Main entry point used by the /api/v1/scan endpoint.

    Steps:
      1. Decode base64 frames to PIL images.
      2. Preprocess to tensors.
      3. Run through the PyTorch model.
      4. Aggregate per-frame fake probabilities into verdict + confidence.
    """
    if not frames_base64:
        return [], "UNCLEAR", 0

    model = get_video_model()
    device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    tensors: List[torch.Tensor] = []
    for data_url in frames_base64:
        try:
            img = _decode_base64_image(data_url)
            tensors.append(_transform(img))
        except Exception:
            # If decoding fails for a frame, skip it
            continue

    if not tensors:
        # If every frame failed to decode, bail out gracefully
        return [], "UNCLEAR", 0

    batch = torch.stack(tensors).to(device)  # [batch, 3, 224, 224]

    with torch.no_grad():
        outputs = model(batch)          # expect shape [batch, 1]
        probs_fake = outputs.view(-1).detach().cpu().numpy().tolist()

    # Ensure python floats
    frame_scores = [float(p) for p in probs_fake]

    verdict, confidence = _aggregate_scores(frame_scores)
    return frame_scores, verdict, confidence
